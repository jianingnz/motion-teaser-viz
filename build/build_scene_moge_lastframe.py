#!/usr/bin/env python3
"""Build a HOT3D scene PC from MoGe applied to the LAST frame of the
stitched 106-frame sequence (clip-001996 t=39).

Same shape as `regen_hot3d_dense_scene_pc.py` (pixel-stride lift +
per-point min-pix-dist side-channel) but anchored at the final
frame:

  1. Read last-frame RGB from `tmp/rgbs/clip-001996_rgb.mp4`.
  2. Run MoGe-1 to get an affine-invariant pointmap + valid mask.
  3. Solve `(a, b)` from THIS frame's GT 2D-track ↔ 3D corresps —
     the visible 3D tracks at clip-1996 t=39 give the metric-depth
     anchors. Independent of the anchor-frame solve in
     visual_example/build_scene.py.
  4. Lift the MoGe depth at stride-5 → 3D in last-frame display
     (camera) coords, then transform to ANCHOR display coords via
     `M_anchor_from_world @ T_we_b_last`. The result lives in the
     same coord system as the existing scene PC and the GT object
     cloud, so trails / prediction-mode / mask radius all keep
     working unchanged.
  5. Compute per-point min-pix-dist to the nearest visible
     ANCHOR-frame 2D-track (after re-projecting the PC into the
     anchor's pinhole) — this is the runtime mask source so it
     must be computed against the anchor frame, not the last
     frame.

Run inside the `moge` conda env at
`/weka/prior-default/jianingz/home/anaconda3/envs/moge`.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------- constants ----------
CLIP_A         = "clip-001995"
CLIP_B         = "clip-001996"
ANCHOR_FRAME   = 84
LAST_FRAME     = 39   # in CLIP_B
RGB_TPL    = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"
TRACKS_DIR = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/hot3d_dense_16k"

FX = 608.9346
FY = 608.9346
CX = 701.7970
CY = 705.7216
W  = 1408
H  = 1408

R_ROT90 = np.array([[0, -1, 0],
                    [ 1, 0, 0],
                    [ 0, 0, 1]], dtype=np.float64)


# ---------- pose / transform helpers ----------
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
    T[:3, 3]  = t
    return T


def get_T_we(tag, frame):
    return load_T_we(f"/tmp/{tag}_cams/{frame:06d}.cameras.json")


def diag4(R):
    T = np.eye(4); T[:3, :3] = R; return T


def transform_pts(M, P):
    Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    return (M @ Ph.T).T[:, :3]


def write_pc_binary(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray):
    N = int(xyz.shape[0])
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.astype(np.uint8).tobytes())


# ---------- MoGe (a, b) solve ----------
def solve_ab_at_lastframe(depth, mask, tracks_uv, pts_lastframe_camera):
    """gt_z = a * pred_depth + b, solved on bilinearly-sampled
    MoGe depth at the visible LAST-frame 2D-track pixels."""
    valid = (
        np.isfinite(tracks_uv).all(axis=1)
        & np.isfinite(pts_lastframe_camera).all(axis=1)
        & (tracks_uv[:, 0] >= 0) & (tracks_uv[:, 0] < W - 1)
        & (tracks_uv[:, 1] >= 0) & (tracks_uv[:, 1] < H - 1)
        & (pts_lastframe_camera[:, 2] > 0.05)
    )
    uv = tracks_uv[valid]
    gt_z = pts_lastframe_camera[valid, 2]
    u  = uv[:, 0]; v = uv[:, 1]
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    du = u - u0; dv = v - v0
    in_mask = mask[v0, u0] & mask[v0 + 1, u0] & mask[v0, u0 + 1] & mask[v0 + 1, u0 + 1]
    valid_mask = np.isfinite(depth[v0, u0]) & in_mask
    if valid_mask.sum() < 200:
        raise RuntimeError(f"too few valid 2D-track samples at last frame ({valid_mask.sum()})")
    u0 = u0[valid_mask]; v0 = v0[valid_mask]; du = du[valid_mask]; dv = dv[valid_mask]
    gt_z = gt_z[valid_mask]
    pred_d = (
        depth[v0, u0]         * (1 - du) * (1 - dv) +
        depth[v0, u0 + 1]     *      du  * (1 - dv) +
        depth[v0 + 1, u0]     * (1 - du) *      dv  +
        depth[v0 + 1, u0 + 1] *      du  *      dv
    )
    A = np.stack([pred_d, np.ones_like(pred_d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, gt_z, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    res = a * pred_d + b - gt_z
    return a, b, res, len(gt_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/weka/prior-default/jianingz/home/visual/motion-teaser-viz/static/data")
    ap.add_argument("--name",    default="hot3d_clip1995_clip1996")
    ap.add_argument("--out-tag", default="lastframe")
    ap.add_argument("--stride",  type=int, default=5)
    ap.add_argument("--moge-resolution", type=int, default=9)
    ap.add_argument("--depth-min", type=float, default=0.05)
    ap.add_argument("--depth-max", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # ── 1. GT camera poses (anchor + last frame + clip-B frame 0) ────
    T_we_a_anchor = get_T_we("c1995", ANCHOR_FRAME)
    T_we_b_last   = get_T_we("c1996", LAST_FRAME)
    T_we_b0       = get_T_we("c1996", 0)
    M_anchor_from_world      = diag4(R_ROT90) @ np.linalg.inv(T_we_a_anchor)
    M_lastframe_from_world   = diag4(R_ROT90) @ np.linalg.inv(T_we_b_last)
    # last-frame display ↔ anchor display.
    M_anchor_from_lastframe  = M_anchor_from_world @ T_we_b_last @ diag4(R_ROT90.T)
    # clip-B-frame-0 display → last-frame display (used to put the GT
    # 3D tracks into last-frame camera coords for the (a, b) solve).
    M_lastframe_from_b0      = M_lastframe_from_world @ T_we_b0 @ diag4(R_ROT90.T)

    # ── 2. Last-frame RGB ──────────────────────────────────────────
    cap = cv2.VideoCapture(RGB_TPL.format(clip=CLIP_B))
    cap.set(cv2.CAP_PROP_POS_FRAMES, LAST_FRAME)
    ok, last_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read {CLIP_B}:{LAST_FRAME}")
    last_rgb = cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB)
    print(f"[rgb] last frame: {last_rgb.shape}")

    # ── 3. MoGe inference ─────────────────────────────────────────
    from moge.model.v1 import MoGeModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[moge] loading on {device}")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    img = torch.from_numpy(last_rgb).permute(2, 0, 1).float() / 255.0
    fov_x_deg = 2 * np.degrees(np.arctan2(W / 2, FX))
    print(f"[moge] inference at resolution_level={args.moge_resolution}, fov_x={fov_x_deg:.2f}°")
    with torch.inference_mode():
        out = moge.infer(img, fov_x=fov_x_deg,
                         resolution_level=args.moge_resolution,
                         apply_mask=True, force_projection=True, use_fp16=True)
    depth = out["depth"].cpu().numpy()
    mask  = out["mask"].cpu().numpy().astype(bool)
    print(f"[moge] valid mask frac: {mask.mean():.3f}")

    # ── 4. (a, b) solve from LAST-frame GT 2D ↔ 3D ────────────────
    d2B = np.load(f"{TRACKS_DIR}/{CLIP_B}_2d.npz")
    d3B = np.load(f"{TRACKS_DIR}/{CLIP_B}_3d.npz")
    tracks_uv_b = d2B["tracks"][LAST_FRAME]
    vis_b       = d2B["visibility"][LAST_FRAME].astype(bool)
    pts_b0      = d3B["points_3d"][:, LAST_FRAME]
    # Move to last-frame camera coords.
    pts_lastframe = transform_pts(M_lastframe_from_b0, pts_b0)
    # Filter to the visible-and-finite-and-in-bounds-and-in-front set.
    keep = (
        vis_b
        & np.isfinite(tracks_uv_b).all(axis=1)
        & np.isfinite(pts_lastframe).all(axis=1)
        & (tracks_uv_b[:, 0] >= 0) & (tracks_uv_b[:, 0] < W - 1)
        & (tracks_uv_b[:, 1] >= 0) & (tracks_uv_b[:, 1] < H - 1)
        & (pts_lastframe[:, 2] > 0.05)
    )
    a, b, res, n_used = solve_ab_at_lastframe(
        depth, mask, tracks_uv_b[keep], pts_lastframe[keep])
    print(f"[ab] solved a={a:.5f}  b={b:.5f}  on N={n_used}  |res| median={np.median(np.abs(res)) * 1000:.1f} mm")
    metric_d = a * depth + b
    metric_d[~mask] = np.nan
    metric_d[(metric_d < args.depth_min) | (metric_d > args.depth_max)] = np.nan

    # ── 5. Stride-5 grid lift in last-frame display coords ────────
    s = int(args.stride)
    yy, xx = np.mgrid[0:H:s, 0:W:s]
    valid_pix = mask[yy, xx] & np.isfinite(metric_d[yy, xx])
    print(f"[grid] stride={s}: {yy.size} pre-mask  →  {valid_pix.sum()} kept")
    u = xx[valid_pix].astype(np.float32)
    v = yy[valid_pix].astype(np.float32)
    z = metric_d[yy, xx][valid_pix].astype(np.float32)
    x = (u - CX) / FX * z
    y = (v - CY) / FY * z
    xyz_lastframe = np.stack([x, y, z], axis=-1)
    rgb_u8 = last_rgb[yy, xx][valid_pix].astype(np.uint8)
    # Move to anchor display coords.
    xyz_anchor = transform_pts(M_anchor_from_lastframe, xyz_lastframe).astype(np.float32)
    print(f"[lift] PC in anchor coords: {xyz_anchor.shape[0]} pts")

    # ── 6. min_pix_dist against MOVING-OBJECT-only last-frame tracks ──
    # The static objects in HOT3D are already 'in' the last-frame RGB
    # we lifted, so the lastframe-MoGe scene PC contains them at their
    # real last-frame appearance + position. We do NOT want to carve
    # them away with the runtime mask — we want the scene PC ALONE to
    # represent them. So min_pix_dist is computed against only the
    # MOVING object's last-frame tracks, leaving the static patches of
    # the scene PC untouched. The viewer then skips its static-object
    # GT-cloud layer (via viewer_defaults.skipStaticObjCloud) so the
    # static objects don't get rendered twice.
    pts3_b   = d3B["points_3d"]                         # (N, F, 3) clip-B-frame-0 display
    vis3_b   = d3B["visibility"].squeeze()              # (N, F)
    N_total  = pts3_b.shape[0]
    N_OBJ    = 6                                        # HOT3D-aria fixed
    N_PER    = N_total // N_OBJ
    # Per-track motion magnitude = bbox diagonal across visible frames.
    track_motion = np.zeros(N_total, dtype=np.float64)
    for k in range(N_total):
        visk = vis3_b[k]
        if visk.sum() < 2:
            continue
        ptsk = pts3_b[k][visk]
        bbox = ptsk.max(axis=0) - ptsk.min(axis=0)
        track_motion[k] = float(np.linalg.norm(bbox))
    obj_motion = np.array([track_motion[oi*N_PER:(oi+1)*N_PER].mean() for oi in range(N_OBJ)])
    moving_obj_id = int(np.argmax(obj_motion))
    print(f"[mpd] per-object mean motion (m): {obj_motion.round(3)}  →  moving = {moving_obj_id}")

    moving_idx  = np.arange(moving_obj_id * N_PER, (moving_obj_id + 1) * N_PER)
    tracks_uv_b = d2B["tracks"][LAST_FRAME][moving_idx]
    vis_b       = d2B["visibility"][LAST_FRAME][moving_idx].astype(bool)
    keep_b = (
        vis_b
        & np.isfinite(tracks_uv_b).all(axis=1)
        & (tracks_uv_b[:, 0] >= 0) & (tracks_uv_b[:, 0] < W)
        & (tracks_uv_b[:, 1] >= 0) & (tracks_uv_b[:, 1] < H)
    )
    track_pix = tracks_uv_b[keep_b].astype(np.float32)
    print(f"[mpd] moving-object last-frame visible tracks: {len(track_pix)}")

    from scipy.spatial import cKDTree
    if len(track_pix) == 0:
        # No moving-object tracks visible at the last frame. Set every
        # min_pix_dist to a huge sentinel so the runtime mask never
        # culls anything — gracefully degrades to "scene PC for
        # everything" if the moving object happens to be off-camera at
        # the last frame.
        md = np.full(xyz_anchor.shape[0], 1e6, dtype=np.float32)
    else:
        tree = cKDTree(track_pix)
        src_pix_uv = np.stack([u.astype(np.float32), v.astype(np.float32)], axis=-1)
        md, _ = tree.query(src_pix_uv, k=1)
        md = md.astype(np.float32)
    print(f"[mpd] median {np.median(md):.1f}px  frac<30 {(md < 30).mean() * 100:.1f}%")

    # ── 7. Write outputs ──────────────────────────────────────────
    pc_path = out_dir / f"{args.name}_pc_{args.out_tag}.bin"
    md_path = out_dir / f"{args.name}_pc_{args.out_tag}_dist.bin"
    write_pc_binary(pc_path, xyz_anchor, rgb_u8)
    md_path.write_bytes(md.astype(np.float32).tobytes())
    print(f"[out] {pc_path}  ({pc_path.stat().st_size // 1024} KB, {xyz_anchor.shape[0]} pts)")
    print(f"[out] {md_path}  ({md_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
