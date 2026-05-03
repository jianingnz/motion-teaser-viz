#!/usr/bin/env python3
"""Build a HOT3D scene PC from MonST3R 2-frame inference.

Pipeline
--------
1. Read first frame (clip-1995, t=84 — the existing anchor) and last
   frame (clip-1996, t=39) of the stitched 106-frame sequence from
   `tmp/rgbs/{clip}_rgb.mp4`.
2. Decode each frame's per-object modal mask, warp fisheye→pinhole,
   rotate into display orientation, OR the masks together → "moving
   foreground" mask. We feed MonST3R with the masked-out regions
   suppressed so dynamic objects don't pollute its feature matching.
3. Run MonST3R (`Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt`)
   on the pair, `square_ok=True` so the 1408² input lands at 512² and
   the full FoV is preserved.
4. Sample MonST3R's view-1 pointmap at the GT 2D-track pixels of the
   anchor frame (display coords, scaled to the 512² inference grid).
   Pair each with its corresponding GT 3D position transformed into
   anchor display coords.
5. Filter GT correspondences by MonST3R confidence ≥ MIN_GT_CONF;
   solve a 7-DOF Umeyama similarity (R, t, s) MonST3R → anchor.
6. Apply the similarity to the FULL pair's pointmaps + colours,
   filter by MonST3R confidence ≥ MIN_PC_CONF, and (optionally)
   stride-subsample.
7. Build per-point min-pixel-distance to nearest visible 2D-track at
   the anchor frame (re-projected into 1408² display via the known
   pinhole intrinsics) — same `min_pix_dist` side-channel the viewer
   already uses for the runtime mask.
8. Write `*_pc_monst3r.bin` and `*_pc_monst3r_dist.bin` next to the
   existing scene PC. Run with `--swap` to move them onto the
   canonical `*_pc.bin` / `*_pc_dist.bin` slots so the deployed
   bundle picks them up.

Run inside the `monst3r` conda env at
`/weka/prior-default/jianingz/home/anaconda3/envs/monst3r`.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REF_ROOT = "/weka/prior-default/jianingz/home/project/_GenTraj/reference_code/monst3r"
sys.path.insert(0, REF_ROOT)
sys.path.insert(0, "/weka/prior-default/jianingz/home/dataset/hot3d_repo/hot3d/clips")

from dust3r.model import AsymmetricCroCo3DStereo                # noqa: E402
from dust3r.utils.image import load_images                       # noqa: E402
from dust3r.image_pairs import make_pairs                        # noqa: E402
from dust3r.inference import inference                           # noqa: E402

# ---------- constants ----------
ANCHOR_FRAME = 84              # clip-1995 frame index of the anchor
LAST_FRAME   = 39              # clip-1996 frame index of the last frame
CLIP_A       = "clip-001995"
CLIP_B       = "clip-001996"

RGB_TPL    = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"
TRACKS_DIR = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/hot3d_dense_16k"
ARIA_TARS  = "/weka/prior-default/jianingz/home/dataset/hot3d/train_aria"

# Display intrinsics (1408×1408) — from visual_example/build_scene.py.
FX = 608.9346
FY = 608.9346
CX = 701.7970
CY = 705.7216
IMG_W = 1408
IMG_H = 1408

# Display-from-world R for the anchor frame: rot90(k=3)-equivalent rotation
# applied at the end of vis_clips_backproject's export_clip_tracks.
R_ROT90 = np.array([[0, -1, 0],
                    [ 1, 0, 0],
                    [ 0, 0, 1]], dtype=np.float64)

# MoGe-class confidence thresholds in MonST3R's logspace.
MIN_GT_CONF = 1.5
MIN_PC_CONF = 1.5

# Default HF weights id for MonST3R.
MONST3R_HF = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"


# ---------- pose helpers (mirror visual_example/build_scene.py) ----------
def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]
    ], dtype=np.float64)


def load_T_we(path, stream="214-1"):
    d = json.load(open(path))
    e = d[stream]["T_world_from_camera"]
    R = quat_to_R(e["quaternion_wxyz"])
    t = np.array(e["translation_xyz"], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_T_we(tag, frame):
    return load_T_we(f"/tmp/{tag}_cams/{frame:06d}.cameras.json")


def diag4(R):
    T = np.eye(4)
    T[:3, :3] = R
    return T


def transform_pts(M, P):
    Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    return (M @ Ph.T).T[:, :3]


# ---------- Umeyama similarity ----------
def umeyama(src, dst):
    """Closed-form similarity (R, t, s) such that s*R*src + t ≈ dst.

    Returns the homogeneous 4×4 matrix `M` and the (R, s, t) parts.
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    n   = len(src)
    cs  = src.mean(axis=0)
    cd  = dst.mean(axis=0)
    src_c = src - cs
    dst_c = dst - cd
    H = src_c.T @ dst_c / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = Vt.T @ S @ U.T
    var_src = (src_c ** 2).sum() / n
    s = (D * np.diag(S)).sum() / var_src
    t = cd - s * R @ cs
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3]  = t
    return M, R, s, t


# ---------- moving-foreground mask (per frame) ----------
def fg_mask_for_frame(clip, frame, warp_x=None, warp_y=None):
    """Decode every modal mask in `{clip}_object.json` for `frame`,
    warp fisheye→pinhole, rotate (k=3) to display, OR all together.
    Returns a (H, W) uint8 mask in display coords (1408×1408)."""
    import tarfile
    import clip_util
    tar_path = f"{ARIA_TARS}/{clip}.tar"
    tar = tarfile.open(tar_path, mode="r")
    objs = clip_util.load_object_annotations(tar, f"{frame:06d}")
    if objs is None:
        tar.close()
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8), warp_x, warp_y
    if warp_x is None or warp_y is None:
        cameras, _ = clip_util.load_cameras(tar, f"{frame:06d}")
        camera_pinhole = clip_util.convert_to_pinhole_camera(cameras["214-1"])
        # Reuse vis_clips_backproject's warp utility.
        sys.path.insert(0, "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp")
        import vis_clips_backproject as VCB
        warp_x, warp_y = VCB.compute_warp_maps(cameras["214-1"], camera_pinhole)

    sys.path.insert(0, "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp")
    import vis_clips_backproject as VCB
    fg_pin = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for bid_str, inst_list in objs.items():
        for inst in inst_list:
            mm = inst.get("masks_modal", {})
            if "214-1" not in mm:
                continue
            mask_fish = clip_util.decode_binary_mask_rle(mm["214-1"]).astype(np.uint8)
            mask_pin  = VCB.fast_warp(mask_fish * 255, warp_x, warp_y, interpolation=cv2.INTER_NEAREST) > 128
            fg_pin |= mask_pin.astype(np.uint8)
    tar.close()
    # rot90(k=3) in display orientation (matches the rgb mp4 + the
    # final 2D track rotation in export_clip_tracks).
    fg_disp = np.ascontiguousarray(np.rot90(fg_pin, k=3))
    return fg_disp, warp_x, warp_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/weka/prior-default/jianingz/home/visual/motion-teaser-viz/static/data")
    ap.add_argument("--name", default="hot3d_clip1995_clip1996")
    ap.add_argument("--mask-fg", action="store_true",
                    help="Suppress moving-object regions in BOTH input frames before "
                         "running MonST3R, so dynamic content doesn't pollute feature matching.")
    ap.add_argument("--no-mask-fg", dest="mask_fg", action="store_false")
    ap.set_defaults(mask_fg=True)
    ap.add_argument("--swap", action="store_true",
                    help="After writing *_pc_monst3r.bin / *_pc_monst3r_dist.bin, also overwrite "
                         "the canonical *_pc.bin / *_pc_dist.bin so the bundle picks up MonST3R's PC.")
    ap.add_argument("--max-pts", type=int, default=80_000,
                    help="Final point count after random sub-sample. Default 80k matches the MoGe scene PC size.")
    ap.add_argument("--gt-conf", type=float, default=MIN_GT_CONF)
    ap.add_argument("--pc-conf", type=float, default=MIN_PC_CONF)
    ap.add_argument("--single-view", action="store_true",
                    help="Use ONLY pred1 (anchor view's MonST3R lift); skip pred2 entirely. "
                         "Avoids the MonST3R-predicted view-1→view-2 relative-pose error showing up as "
                         "a visibly offset second cloud. Loses the wider FoV from view 2 — same coverage "
                         "as the MoGe single-frame backprojection, just with MonST3R's depth.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Camera-pose dependencies ─────────────────────────────────
    T_we_a0       = get_T_we("c1995", 0)
    T_we_a_anchor = get_T_we("c1995", ANCHOR_FRAME)
    M_anchor_from_world = diag4(R_ROT90) @ np.linalg.inv(T_we_a_anchor)
    M_anchor_from_a0    = M_anchor_from_world @ T_we_a0 @ diag4(R_ROT90.T)

    # ── 2. Anchor-frame GT 2D ↔ 3D correspondences ─────────────────
    d2A = np.load(f"{TRACKS_DIR}/{CLIP_A}_2d.npz")
    d3A = np.load(f"{TRACKS_DIR}/{CLIP_A}_3d.npz")
    tracks_uv = d2A["tracks"][ANCHOR_FRAME]              # (N, 2) display
    vis_a     = d2A["visibility"][ANCHOR_FRAME]          # (N,)
    pts_a0    = d3A["points_3d"][:, ANCHOR_FRAME]        # (N, 3) clipA-frame-0 display
    pts_anchor = transform_pts(M_anchor_from_a0, pts_a0).astype(np.float64)
    valid = (
        vis_a
        & np.isfinite(tracks_uv).all(axis=1)
        & np.isfinite(pts_anchor).all(axis=1)
        & (tracks_uv[:, 0] >= 0) & (tracks_uv[:, 0] < IMG_W)
        & (tracks_uv[:, 1] >= 0) & (tracks_uv[:, 1] < IMG_H)
    )
    track_uv  = tracks_uv[valid]
    track_xyz = pts_anchor[valid]
    print(f"[gt] anchor-frame visible 2D-track corresps: {len(track_uv)}")

    # ── 3. Read the two RGB frames ────────────────────────────────
    def read_frame(clip, frame):
        cap = cv2.VideoCapture(RGB_TPL.format(clip=clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not read {clip}:{frame}")
        return bgr

    anchor_bgr = read_frame(CLIP_A, ANCHOR_FRAME)
    last_bgr   = read_frame(CLIP_B, LAST_FRAME)
    print(f"[rgb] anchor {anchor_bgr.shape}  last {last_bgr.shape}")

    # ── 4. Optionally suppress moving-object pixels ────────────────
    if args.mask_fg:
        fg_a, wx, wy = fg_mask_for_frame(CLIP_A, ANCHOR_FRAME)
        fg_b, _, _   = fg_mask_for_frame(CLIP_B, LAST_FRAME, warp_x=wx, warp_y=wy)
        anchor_bgr = anchor_bgr.copy()
        last_bgr   = last_bgr.copy()
        anchor_bgr[fg_a > 0] = 0
        last_bgr[fg_b > 0]   = 0
        print(f"[mask] foreground pixels: anchor={(fg_a > 0).sum()}, last={(fg_b > 0).sum()}")

    work = Path("/tmp/monst3r_in")
    work.mkdir(exist_ok=True)
    a_path = work / "a_anchor.png"
    b_path = work / "b_last.png"
    cv2.imwrite(str(a_path), anchor_bgr)
    cv2.imwrite(str(b_path), last_bgr)

    # ── 5. MonST3R inference ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] loading {MONST3R_HF} on {device}")
    model = AsymmetricCroCo3DStereo.from_pretrained(MONST3R_HF).to(device).eval()
    imgs = load_images([str(a_path), str(b_path)], size=512, square_ok=True, verbose=True)
    H_in, W_in = int(imgs[0]["true_shape"][0, 0]), int(imgs[0]["true_shape"][0, 1])
    print(f"[input] true_shape = {H_in}x{W_in}")
    pairs = make_pairs(imgs, scene_graph="complete", symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # ── 6. Extract pointmaps + colours ────────────────────────────
    # `inference()` collates across pairs: each tensor in output['view1'/
    # 'view2'/'pred1'/'pred2'] has a leading batch dimension. With
    # symmetrize=True we get B=2 (one row each for the 1→2 and 2→1
    # pairs). We want the row whose view1.idx == 0 and view2.idx == 1
    # — that's the pair where pred1 is the anchor frame and
    # pred2_in_other_view places the last frame in the anchor's coords.
    v1_idx = output["view1"]["idx"]
    v2_idx = output["view2"]["idx"]
    if isinstance(v1_idx, torch.Tensor):
        v1_idx = v1_idx.detach().cpu().tolist()
        v2_idx = v2_idx.detach().cpu().tolist()
    pair_idx = None
    for i, (a, b) in enumerate(zip(v1_idx, v2_idx)):
        if int(a) == 0 and int(b) == 1:
            pair_idx = i
            break
    if pair_idx is None:
        raise RuntimeError(f"could not locate the (1,2) pair: v1={v1_idx}, v2={v2_idx}")

    pts1  = output["pred1"]["pts3d"][pair_idx].detach().cpu().numpy().astype(np.float64)
    pts2  = output["pred2"]["pts3d_in_other_view"][pair_idx].detach().cpu().numpy().astype(np.float64)
    conf1 = output["pred1"]["conf"][pair_idx].detach().cpu().numpy()
    conf2 = output["pred2"]["conf"][pair_idx].detach().cpu().numpy()
    rgb1  = imgs[0]["img"][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    rgb2  = imgs[1]["img"][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    rgb1  = rgb1.clip(0, 1)
    rgb2  = rgb2.clip(0, 1)
    print(f"[infer] pts1 {pts1.shape}  conf1 [{conf1.min():.2f}, {conf1.max():.2f}, mean {conf1.mean():.2f}]")
    print(f"[infer] pts2 {pts2.shape}  conf2 [{conf2.min():.2f}, {conf2.max():.2f}, mean {conf2.mean():.2f}]")

    # ── 7. Sample at GT 2D-track pixels ───────────────────────────
    sx = W_in / IMG_W
    sy = H_in / IMG_H
    gx = np.round(track_uv[:, 0] * sx).astype(int)
    gy = np.round(track_uv[:, 1] * sy).astype(int)
    in_b = (gx >= 0) & (gx < W_in) & (gy >= 0) & (gy < H_in)
    gx, gy = gx[in_b], gy[in_b]
    track_xyz_gt = track_xyz[in_b]
    src = pts1[gy, gx]
    cf  = conf1[gy, gx]
    print(f"[align] in-bounds corresps: {len(src)}; conf median {np.median(cf):.2f}")
    keep = cf > args.gt_conf
    src = src[keep]
    dst = track_xyz_gt[keep]
    print(f"[align] kept after conf>{args.gt_conf}: {len(src)}")
    if len(src) < 100:
        raise RuntimeError("too few high-confidence GT correspondences for Umeyama")

    M, R, s, t = umeyama(src, dst)
    resid = transform_pts(M, src) - dst
    err   = np.linalg.norm(resid, axis=1)
    print(f"[align] s = {s:.5f}  |t| = {np.linalg.norm(t):.4f} m")
    print(f"[align] residual: median {np.median(err) * 1000:.2f} mm,  90% {np.percentile(err, 90) * 1000:.2f} mm")

    # ── 8. Apply transform + assemble fused PC ────────────────────
    if args.single_view:
        Pall = pts1.reshape(-1, 3)
        Call = rgb1.reshape(-1, 3)
        Cfa  = conf1.reshape(-1)
        print("[fuse] single-view mode: using pred1 only")
    else:
        Pall = np.concatenate([pts1.reshape(-1, 3), pts2.reshape(-1, 3)], axis=0)
        Call = np.concatenate([rgb1.reshape(-1, 3), rgb2.reshape(-1, 3)], axis=0)
        Cfa  = np.concatenate([conf1.reshape(-1),    conf2.reshape(-1)],   axis=0)
    Pal  = transform_pts(M, Pall)
    keep = (Cfa > args.pc_conf) & np.isfinite(Pal).all(axis=1)
    Pal, Call, Cfa = Pal[keep], Call[keep], Cfa[keep]
    print(f"[fuse] post-conf PC: {len(Pal)}")

    # Optional sub-sample.
    if args.max_pts > 0 and len(Pal) > args.max_pts:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(Pal), args.max_pts, replace=False)
        Pal, Call, Cfa = Pal[idx], Call[idx], Cfa[idx]
    print(f"[fuse] final PC: {len(Pal)}")

    # ── 9. Re-project final PC to 1408² display pixels for min_pix_dist ──
    # Pal is now in anchor display coords. Project with the same
    # pinhole intrinsics the rest of the pipeline uses.
    proj_z = Pal[:, 2]
    in_front = proj_z > 0.05
    u = np.full(len(Pal), -1.0, dtype=np.float32)
    v = np.full(len(Pal), -1.0, dtype=np.float32)
    u[in_front] = (FX * Pal[in_front, 0] / proj_z[in_front] + CX).astype(np.float32)
    v[in_front] = (FY * Pal[in_front, 1] / proj_z[in_front] + CY).astype(np.float32)

    # min_pix_dist via cKDTree against the anchor's GT 2D-track pixels.
    from scipy.spatial import cKDTree
    track_pix = track_uv.astype(np.float32)
    tree = cKDTree(track_pix)
    pix_uv = np.stack([u, v], axis=-1)
    md, _ = tree.query(pix_uv, k=1)
    md[~in_front] = 1e6   # behind-camera: treat as "very far" so the runtime mask never hides them
    md = md.astype(np.float32)
    print(f"[mpd] median {np.median(md):.1f}px  frac<30 {(md < 30).mean() * 100:.1f}%")

    # ── 10. Write binaries ────────────────────────────────────────
    pc_path  = out_dir / f"{args.name}_pc_monst3r.bin"
    md_path  = out_dir / f"{args.name}_pc_monst3r_dist.bin"
    rgb_u8 = (Call * 255.0).clip(0, 255).astype(np.uint8)
    with open(pc_path, "wb") as f:
        f.write(np.uint32(len(Pal)).tobytes())
        f.write(Pal.astype(np.float32).tobytes())
        f.write(rgb_u8.tobytes())
    md_path.write_bytes(md.astype(np.float32).tobytes())
    print(f"[out] {pc_path}  ({pc_path.stat().st_size // 1024} KB, {len(Pal)} pts)")
    print(f"[out] {md_path}  ({md_path.stat().st_size // 1024} KB)")

    if args.swap:
        canon_pc = out_dir / f"{args.name}_pc.bin"
        canon_md = out_dir / f"{args.name}_pc_dist.bin"
        canon_pc.write_bytes(pc_path.read_bytes())
        canon_md.write_bytes(md_path.read_bytes())
        print(f"[swap] overwrote canonical {canon_pc.name} + {canon_md.name}")

        # The bundle JSON also carries n_points — patch it so the loader
        # doesn't allocate a stale-sized buffer.
        bundle_json = out_dir / f"{args.name}.json"
        if bundle_json.exists():
            bundle = json.loads(bundle_json.read_text())
            bundle["pc_bin"]["n_points"] = int(len(Pal))
            bundle["pc_dist_bin"]["n_points"] = int(len(Pal))
            bundle["pc_bin"]["sampling"]  = "monst3r-pair"
            bundle["pc_bin"]["subsample"] = 1
            bundle_json.write_text(json.dumps(bundle, indent=2))
            print(f"[swap] patched {bundle_json.name} (n_points={len(Pal)})")


if __name__ == "__main__":
    main()
