#!/usr/bin/env python3
"""Build a SINGLE-CLIP HOT3D motion-teaser-viz bundle (one clip, one object,
arbitrary [start, end] frame range) — the simplified counterpart of
prepare_hot3d.py (which is hard-coded for the 1995→1996 cross-clip stitch).

Pipeline (mirrors prepare_hot3d.py + regen_hot3d_dense_scene_pc.py
collapsed into one pass):

  1. Untar the clip's per-frame `*.cameras.json` if not already extracted.
  2. Load 2D + 3D dense surface tracks from
     `/weka/.../hot3d_repo/tmp/all_tracks/{clip}_2d|3d.npz`. Dense tracks
     come as 12000 × F × {2,3}; the layout is six concatenated objects of
     2000 pts each, so obj N = tracks[N*2000 : (N+1)*2000].
  3. Anchor frame = `--frame-start`. Pose conversion follows the canonical
     HOT3D convention:
         M_anchor_from_world = R_ROT90 @ inv(T_we_anchor)
         M_anchor_from_clip0 = M_anchor_from_world @ T_we_clip0 @ R_ROT90.T
     so tracks (stored in clip0-display coords) → anchor-display coords.
  4. Run MoGe-1 on the anchor RGB to get a per-pixel affine-invariant depth
     map. Solve (a, b) such that `gt_depth = a * pred_depth + b` via LSQ on
     visible 2D-track pixels at the anchor frame (where we know GT depth
     from `clip_3d.npz` after transforming the 3D points into the anchor
     CAMERA frame, NOT the rotated display frame — depth is along +Z of
     the camera).
  5. Stride-sample the metric depth map to a scene point cloud in the
     anchor's CAMERA frame (no R_ROT90 — matches build_scene.py /
     regen_hot3d_dense_scene_pc.py so the viewer renders consistently).
  6. Compute per-frame camera position in anchor display coords + a
     synthesized c2w that always points at the per-frame mean object centre
     (the existing build_c2w_per_frame from prepare_hot3d.py).
  7. Sample per-track RGB at the first frame (in the chosen [start, end]
     window) where the track is visible AND in-bounds at the source RGB.
  8. Write { _gt3d.bin, _cam.bin, _pc.bin, _pc_dist.bin, .mp4, _chrono.jpg,
     .json } using the same binary layouts as prepare_hot3d.py.

Inputs (all default-resolved, override with CLI flags):
  * Source dense tracks: /weka/.../hot3d_repo/tmp/all_tracks/{clip}_*.npz
  * Source RGB mp4:      /weka/.../hot3d_repo/tmp/rgbs/{clip}_rgb.mp4
  * Source HOT3D tar:    /weka/.../hot3d/train_aria/{clip}.tar  (for cams)

Run inside the `moge` conda env so MoGe-1 + torch + cuda are available:
  source /weka/prior-default/jianingz/home/anaconda3/etc/profile.d/conda.sh
  conda activate moge
  python3 build/prepare_hot3d_singleclip.py \
      --clip clip-003020 --frame-start 23 --frame-end 56 --obj 0 \
      --out-dir . --name hot3d_clip3020_obj0_s023_e056
"""
import argparse, json, struct, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

# All_tracks (12k pts/clip) + RGB + HOT3D source tars
ALL_TRACKS_DEFAULT = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/all_tracks"
RGB_TPL            = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"
HOT3D_TAR_TPL      = "/weka/prior-default/jianingz/home/dataset/hot3d/train_aria/{clip}.tar"
CAM_DIR_TPL        = "/tmp/{tag}_cams"

# Aria 214-1 RGB intrinsics (constant across clips at native 1408×1408).
FX = 608.9346; FY = 608.9346; CX = 701.7970; CY = 705.7216
W_SRC = H_SRC = 1408
N_OBJ_TOTAL = 6
PTS_PER_OBJ = 2000

# Display rotation. Same convention as prepare_hot3d.py / build_scene.py:
# rotate the anchor's camera frame by 90° about Z so +Y points display-up.
R_ROT90 = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]], dtype=np.float64)


# ──────────────────────────── camera-pose io ────────────────────────────

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]],
        dtype=np.float64)


def load_T_we(path, stream="214-1"):
    d = json.load(open(path))
    e = d[stream]["T_world_from_camera"]
    R = quat_to_R(e["quaternion_wxyz"])
    t = np.array(e["translation_xyz"], dtype=np.float64)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def get_T_we(cam_dir, frame):
    return load_T_we(f"{cam_dir}/{frame:06d}.cameras.json")


def diag4(R):
    T = np.eye(4); T[:3, :3] = R; return T


def transform(M, P):
    Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    return (M @ Ph.T).T[:, :3]


# ──────────────────────────── camera synth ────────────────────────────

def build_c2w_per_frame(cam_pos, mean_obj_pos):
    """Synthesise per-frame c2w (camera-to-world in anchor coords) that puts
    the camera at `cam_pos[fi]` and points it at `mean_obj_pos[fi]`. World-up
    is +Y. Mirrors prepare_hot3d.build_c2w_per_frame."""
    F = cam_pos.shape[0]
    c2w = np.zeros((F, 4, 4), dtype=np.float32)
    up_world = np.array([0, 1, 0], dtype=np.float64)
    for fi in range(F):
        eye = cam_pos[fi].astype(np.float64)
        target = mean_obj_pos[fi].astype(np.float64)
        fwd = target - eye
        n = np.linalg.norm(fwd)
        if n < 1e-6:
            fwd = np.array([0, 0, 1.0])
        else:
            fwd /= n
        right = np.cross(up_world, fwd)
        nr = np.linalg.norm(right)
        right = right / nr if nr > 1e-6 else np.array([1.0, 0, 0])
        up = np.cross(fwd, right)
        R = np.column_stack([right, up, fwd])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = eye
        c2w[fi] = T.astype(np.float32)
    return c2w


# ──────────────────────────── binary writers ────────────────────────────

def write_pc_binary(path: Path, xyz, rgb_u8):
    N = int(xyz.shape[0])
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.astype(np.uint8).tobytes())


def write_gt3d_binary(path, positions, vis, annot, obj_ids, rgb_u8, n_obj):
    """GT3D v2 layout — must match prepare_hot3d.write_gt3d_binary EXACTLY,
    since both are parsed by the same viewer-side `loadGT3DBinary`. Field
    order matters; the loader reads sequentially via offset arithmetic.

    Layout (little-endian):
      MAGIC 'GT3D' (4) | version u32 = 2 | N u32 | F u32 | nObj u32
      | f32 axisMin[3] | f32 axisMax[3]
      | u8  obj_ids[N]
      | u8  rgb[N, 3]
      | i16 quant[F, N, 3]      (FRAME-major; sentinel -32768 = un-annotated)
      | u8  visBytes[F * N]     (1 = visible/cloud-eligible, 0 = occluded)
    """
    N, F, _ = positions.shape
    assert vis.shape == (N, F)
    assert annot.shape == (N, F)
    assert obj_ids.shape == (N,)
    assert rgb_u8.shape == (N, 3)

    valid_mask = annot & np.isfinite(positions).all(axis=-1)     # (N, F)
    if not valid_mask.any():
        raise RuntimeError("no annotated positions to encode")
    valid_pts = positions[valid_mask]
    axis_min = valid_pts.min(axis=0).astype(np.float32)
    axis_max = valid_pts.max(axis=0).astype(np.float32)
    center = (axis_max + axis_min) / 2.0
    half = (axis_max - axis_min) / 2.0
    half = np.where(half < 1e-9, 1e-9, half)

    quant_nft = np.full((N, F, 3), -32768, dtype=np.int16)
    normed = (positions[valid_mask] - center) / half
    normed = np.clip(np.round(normed * 32767.0), -32767, 32767).astype(np.int16)
    quant_nft[valid_mask] = normed
    # Frame-major for the on-disk layout (the loader reads it as F*N*3).
    quant_fnt = np.ascontiguousarray(quant_nft.transpose(1, 0, 2))   # (F, N, 3)
    # Visibility is stored frame-major too: byte index = f * N + p.
    vis_fn = np.ascontiguousarray(vis.T.astype(np.uint8))            # (F, N)

    with open(path, 'wb') as f:
        f.write(b"GT3D")
        f.write(struct.pack("<I", 2))      # version = 2
        f.write(struct.pack("<I", N))
        f.write(struct.pack("<I", F))
        f.write(struct.pack("<I", n_obj))
        for v in axis_min: f.write(struct.pack("<f", float(v)))
        for v in axis_max: f.write(struct.pack("<f", float(v)))
        f.write(obj_ids.astype(np.uint8).tobytes())
        f.write(rgb_u8.astype(np.uint8).tobytes())
        f.write(quant_fnt.tobytes())
        f.write(vis_fn.tobytes())
    return {
        "format":   "GT3D v2",
        "n_tracks": int(N),
        "n_frames": int(F),
        "n_objects": int(n_obj),
        "axis_min":  [float(v) for v in axis_min],
        "axis_max":  [float(v) for v in axis_max],
    }


def write_cam_binary(path, c2w, intrinsics):
    F = c2w.shape[0]
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    with open(path, "wb") as f:
        f.write(b"CAM4")
        f.write(np.uint32(1).tobytes())
        f.write(np.uint32(F).tobytes())
        f.write(np.array([fx, fy, cx, cy], dtype=np.float32).tobytes())
        # Row-major 4×4 c2w per frame.
        f.write(c2w.astype(np.float32).tobytes())


# ──────────────────────────── mp4 + chrono ────────────────────────────

def trim_mp4(src_mp4, out_path, fr_start, fr_end, target_height, fps):
    """Trim [fr_start, fr_end] inclusive from src_mp4 (native 30fps), 1 src
    frame → 1 output frame, output at `target_height` (square aspect for
    HOT3D 1408×1408)."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fc = (
        f"trim=start_frame={fr_start}:end_frame={fr_end + 1},"
        f"setpts=PTS-STARTPTS,scale=-2:{target_height},fps={fps}"
    )
    cmd = [ffmpeg, "-y", "-i", str(src_mp4), "-filter_complex", fc,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-an", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


# ──────────────────────────── camera tar extract ────────────────────────────

def ensure_cams_extracted(clip, tag, frame_start, frame_end):
    """Untar `*.cameras.json` for [frame_start, frame_end] from the HOT3D tar
    if /tmp/{tag}_cams/ is missing or empty. Idempotent."""
    cam_dir = Path(CAM_DIR_TPL.format(tag=tag))
    cam_dir.mkdir(parents=True, exist_ok=True)
    # Quick sentinel: if the anchor + last frame are present, skip extract.
    needed = [cam_dir / f"{frame_start:06d}.cameras.json",
              cam_dir / f"{frame_end:06d}.cameras.json"]
    if all(p.exists() for p in needed):
        return cam_dir
    tar_path = HOT3D_TAR_TPL.format(clip=clip)
    if not Path(tar_path).exists():
        raise RuntimeError(f"HOT3D source tar missing: {tar_path}")
    print(f"untaring camera files from {tar_path} → {cam_dir}/ ...")
    cmd = ["tar", "xf", tar_path, "-C", str(cam_dir),
           "--wildcards", "*.cameras.json"]
    subprocess.run(cmd, check=True)
    return cam_dir


# ──────────────────────────── MoGe scene PC ────────────────────────────

def lift_scene_pc(anchor_rgb, gt_uv_pix, gt_depth_m, stride=3,
                  resolution_level=9, depth_min=0.05, depth_max=5.0):
    """Run MoGe-1 on `anchor_rgb` (H×W×3 uint8), solve (a, b) so MoGe pred
    depth aligns with `gt_depth_m` at `gt_uv_pix`, and stride-sample the
    metric depth map into a scene point cloud in CAMERA-frame coords.

    Returns (xyz, rgb_u8, uv_scene, a, b).
    """
    import torch
    from moge.model.v1 import MoGeModel

    H, W = anchor_rgb.shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MoGe] loading model on {device}...")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    img = torch.from_numpy(anchor_rgb).permute(2, 0, 1).float() / 255.0
    fov_x_deg = 2 * np.degrees(np.arctan2(W / 2, FX))
    print(f"[MoGe] running fov_x={fov_x_deg:.2f}° resolution_level={resolution_level} ...")
    with torch.inference_mode():
        out = moge.infer(img, fov_x=fov_x_deg, resolution_level=resolution_level,
                         apply_mask=True, force_projection=True, use_fp16=True)
    pred_depth = out["depth"].cpu().numpy()        # (H, W) MoGe-units
    mask       = out["mask"].cpu().numpy().astype(bool)
    print(f"[MoGe] valid mask frac: {mask.mean():.3f}")

    # Sample MoGe depth at GT track pixels.
    u = gt_uv_pix[:, 0]; v = gt_uv_pix[:, 1]
    in_b = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_b]; v = v[in_b]
    gt_d = gt_depth_m[in_b]
    iu = np.clip(u.astype(np.int64), 0, W - 1)
    iv = np.clip(v.astype(np.int64), 0, H - 1)
    pd = pred_depth[iv, iu]
    keep = mask[iv, iu] & np.isfinite(pd) & np.isfinite(gt_d) & (gt_d > 0.05) & (gt_d < 5.0)
    pd = pd[keep]; gt_d = gt_d[keep]
    if len(pd) < 50:
        raise RuntimeError(f"too few valid MoGe/GT depth pairs ({len(pd)}) — calibration unreliable")
    A = np.stack([pd, np.ones_like(pd)], axis=1)
    coef, _resid, _rk, _sv = np.linalg.lstsq(A, gt_d, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = gt_d - (a * pd + b)
    print(f"[MoGe] solved a={a:.5f}  b={b:.5f}  median|resid|={np.median(np.abs(resid))*1000:.1f} mm "
          f"(n={len(pd)})")

    metric_d = a * pred_depth + b

    # Stride-sample.
    s = max(1, int(stride))
    yy, xx = np.mgrid[0:H:s, 0:W:s]
    valid = (mask[yy, xx]
             & np.isfinite(metric_d[yy, xx])
             & (metric_d[yy, xx] > depth_min)
             & (metric_d[yy, xx] < depth_max))
    print(f"[MoGe] stride-{s} grid: {yy.size} pre-mask, {valid.sum()} post-mask")

    u_g = xx[valid].astype(np.float32)
    v_g = yy[valid].astype(np.float32)
    z_g = metric_d[yy, xx][valid].astype(np.float32)
    x_g = (u_g - CX) / FX * z_g
    y_g = (v_g - CY) / FY * z_g
    xyz = np.stack([x_g, y_g, z_g], axis=-1)
    rgb_u8 = anchor_rgb[yy, xx][valid].astype(np.uint8)
    uv_scene = np.stack([u_g, v_g], axis=-1)
    return xyz, rgb_u8, uv_scene, a, b


# ──────────────────────────── per-track RGB ────────────────────────────

def sample_per_track_rgb(rgb_frames, tracks_2d, vis_2d, K_total):
    """For each of K_total tracks, walk the `rgb_frames` list (one per
    stitched frame, frame index = list index) and return the bilinearly
    sampled RGB at the FIRST visible+in-bounds frame. Tracks never visible
    → neutral grey 128."""
    F = len(rgb_frames)
    rgb_u8 = np.full((K_total, 3), 128, dtype=np.uint8)
    for k in range(K_total):
        for f in range(F):
            if not vis_2d[f, k]:
                continue
            u_raw, v_raw = float(tracks_2d[f, k, 0]), float(tracks_2d[f, k, 1])
            if not (np.isfinite(u_raw) and np.isfinite(v_raw)):
                continue
            if not (0 <= u_raw < W_SRC - 1 and 0 <= v_raw < H_SRC - 1):
                continue
            u0 = int(np.floor(u_raw)); v0 = int(np.floor(v_raw))
            du = u_raw - u0; dv = v_raw - v0
            img = rgb_frames[f].astype(np.float32)
            c = (img[v0,   u0  ] * (1 - du) * (1 - dv)
               + img[v0,   u0+1] *      du  * (1 - dv)
               + img[v0+1, u0  ] * (1 - du) *      dv
               + img[v0+1, u0+1] *      du  *      dv)
            rgb_u8[k] = np.clip(c, 0, 255).astype(np.uint8)
            break
    return rgb_u8


# ──────────────────────────── main ────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True,
                    help="HOT3D clip id, e.g. 'clip-003020'.")
    ap.add_argument("--frame-start", type=int, required=True,
                    help="First source frame to include (inclusive).")
    ap.add_argument("--frame-end", type=int, required=True,
                    help="Last source frame to include (inclusive).")
    ap.add_argument("--obj", type=int, required=True,
                    help="Object index in [0, 5] (each object = 2000 tracks).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="motion-teaser-viz repo root (containing static/).")
    ap.add_argument("--name", default=None,
                    help="Bundle name. Defaults to "
                         "'hot3d_{clipShort}_obj{N}_s{start:03d}_e{end:03d}'.")
    ap.add_argument("--all-tracks-dir", default=ALL_TRACKS_DEFAULT)
    ap.add_argument("--target-height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--moge-stride", type=int, default=3,
                    help="Pixel stride for the scene PC (3 → ~155k pre-mask "
                         "at 1408². Lower = denser cloud, larger pc.bin).")
    ap.add_argument("--moge-resolution", type=int, default=9)
    ap.add_argument("--caption", default=None,
                    help="Bundle caption shown in the page header. "
                         "Defaults to a generated description.")
    args = ap.parse_args()

    if not (0 <= args.obj < N_OBJ_TOTAL):
        raise ValueError(f"--obj must be in [0, {N_OBJ_TOTAL - 1}]")
    fr_s, fr_e = int(args.frame_start), int(args.frame_end)
    if fr_e <= fr_s:
        raise ValueError("--frame-end must be > --frame-start")
    F = fr_e - fr_s + 1
    if args.name is None:
        clip_short = args.clip.replace("clip-", "clip")
        args.name = f"hot3d_{clip_short}_obj{args.obj}_s{fr_s:03d}_e{fr_e:03d}"
    if args.caption is None:
        args.caption = (f"HOT3D · {args.clip} · obj {args.obj} · "
                        f"frames {fr_s}–{fr_e} ({F} frames)")
    print(f"=== {args.name} ===")
    print(f"  clip={args.clip}  obj={args.obj}  frames=[{fr_s}, {fr_e}] "
          f"({F} frames)")

    out_data  = args.out_dir / "static" / "data";   out_data.mkdir(parents=True, exist_ok=True)
    out_video = args.out_dir / "static" / "videos"; out_video.mkdir(parents=True, exist_ok=True)

    # ── Load tracks + subset to obj N ──
    tracks_root = Path(args.all_tracks_dir)
    d3 = np.load(tracks_root / f"{args.clip}_3d.npz")
    d2 = np.load(tracks_root / f"{args.clip}_2d.npz")
    pts3_full = d3["points_3d"]                    # (12000, F_src, 3)
    vis3_full = d3["visibility"].squeeze()         # (12000, F_src)
    tracks2_full = d2["tracks"]                    # (F_src, 12000, 2)
    vis2_full   = d2["visibility"]                 # (F_src, 12000)
    obj_slice = slice(args.obj * PTS_PER_OBJ, (args.obj + 1) * PTS_PER_OBJ)
    pts3   = pts3_full[obj_slice]                  # (2000, F_src, 3)
    vis3   = vis3_full[obj_slice]                  # (2000, F_src)
    tracks2 = tracks2_full[:, obj_slice]           # (F_src, 2000, 2)
    vis2    = vis2_full[:, obj_slice]              # (F_src, 2000)
    K_total = pts3.shape[0]
    print(f"  loaded obj{args.obj} tracks: {K_total} pts × {pts3.shape[1]} src frames")

    # ── Camera poses for [fr_s, fr_e] ──
    tag = args.clip.replace("clip-", "c")          # 'clip-003020' → 'c003020'
    cam_dir = ensure_cams_extracted(args.clip, tag, fr_s, fr_e)
    T_we = [get_T_we(str(cam_dir), t) for t in range(fr_s, fr_e + 1)]
    T_we_clip0 = get_T_we(str(cam_dir), 0)         # tracks live in clip-frame-0 display coords
    T_we_anchor = T_we[0]                          # frame fr_s = anchor

    M_anchor_from_world = diag4(R_ROT90) @ np.linalg.inv(T_we_anchor)
    M_anchor_from_clip0 = M_anchor_from_world @ T_we_clip0 @ diag4(R_ROT90.T)
    # CAMERA-frame transform (no R_ROT90) — needed for depth calibration.
    M_anchorCam_from_clip0 = np.linalg.inv(T_we_anchor) @ T_we_clip0 @ diag4(R_ROT90.T)

    # ── Per-frame stitched arrays ──
    obj_arr  = np.zeros((K_total, F, 3), dtype=np.float32)   # anchor display
    cam_arr  = np.zeros((K_total, F, 3), dtype=np.float32)   # anchor camera (depth = +Z)
    vis_arr  = np.zeros((K_total, F), dtype=bool)
    annot_arr = np.zeros((K_total, F), dtype=bool)
    cam_pos  = np.zeros((F, 3), dtype=np.float32)            # in anchor display
    for fi, t in enumerate(range(fr_s, fr_e + 1)):
        P = pts3[:, t, :]
        obj_arr[:, fi] = transform(M_anchor_from_clip0,    P).astype(np.float32)
        cam_arr[:, fi] = transform(M_anchorCam_from_clip0, P).astype(np.float32)
        vis_arr[:, fi] = vis3[:, t]
        annot_arr[:, fi] = (P != 0).any(axis=1)
        cam_h = np.append(T_we[fi][:3, 3], 1.0)
        cam_pos[fi] = (M_anchor_from_world @ cam_h)[:3]

    # ── Read RGB frames for [fr_s, fr_e] ──
    print(f"  loading RGB frames {fr_s}..{fr_e} from {RGB_TPL.format(clip=args.clip)} ...")
    cap = cv2.VideoCapture(RGB_TPL.format(clip=args.clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {RGB_TPL.format(clip=args.clip)}")
    rgb_frames = []
    for t in range(fr_s, fr_e + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to read frame {t}")
        rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  loaded {len(rgb_frames)} frames at {rgb_frames[0].shape}")

    # ── Per-track RGB (bilinear at first visible frame inside window) ──
    print("  sampling per-track RGB...")
    tracks2_window = tracks2[fr_s:fr_e + 1]        # (F, 2000, 2)
    vis2_window    = vis2[fr_s:fr_e + 1]           # (F, 2000)
    rgb_u8 = sample_per_track_rgb(rgb_frames, tracks2_window, vis2_window, K_total)
    fallback = (rgb_u8 == 128).all(axis=1).sum()
    print(f"  per-track RGB sampled (fallback grey for {fallback} never-visible tracks)")

    # ── MoGe scene PC + (a, b) calibration ──
    # Anchor-frame visible tracks for calibration: GT depth = +Z in CAMERA frame.
    anchor_rgb = rgb_frames[0]
    anchor_uv  = tracks2_window[0]                 # (2000, 2)
    anchor_vis = vis2_window[0]                    # (2000,)
    anchor_z   = cam_arr[:, 0, 2]                  # CAMERA-frame Z
    anchor_keep = anchor_vis & np.isfinite(anchor_z) & (anchor_z > 0.05) & (anchor_z < 5.0) \
                  & np.isfinite(anchor_uv).all(axis=1)
    print(f"  anchor 2D-track sample for MoGe calibration: "
          f"{int(anchor_keep.sum())} / {len(anchor_keep)} visible+in-range")
    xyz_scene, rgb_scene, uv_scene, mo_a, mo_b = lift_scene_pc(
        anchor_rgb,
        anchor_uv[anchor_keep],
        anchor_z[anchor_keep],
        stride=args.moge_stride,
        resolution_level=args.moge_resolution,
    )
    print(f"  scene PC: {xyz_scene.shape[0]} pts (camera-frame)")

    # ── Per-scene-pt min-pix-dist to anchor-frame visible 2D tracks ──
    from scipy.spatial import cKDTree
    valid_anchor_uv = anchor_uv[anchor_keep]
    if len(valid_anchor_uv) > 0:
        tree = cKDTree(valid_anchor_uv.astype(np.float32))
        min_pix_dist, _ = tree.query(uv_scene.astype(np.float32), k=1)
    else:
        min_pix_dist = np.full((xyz_scene.shape[0],), 1e6, dtype=np.float32)
    min_pix_dist = min_pix_dist.astype(np.float32)
    print(f"  min_pix_dist median={np.median(min_pix_dist):.1f}px "
          f"frac<30px={(min_pix_dist<30).mean()*100:.1f}%")

    # ── Per-frame mean object centre (drives synth c2w fwd vector) ──
    mean_obj = np.zeros((F, 3), dtype=np.float32)
    for fi in range(F):
        m = vis_arr[:, fi]
        mean_obj[fi] = obj_arr[m, fi].mean(axis=0) if m.any() else cam_pos[fi]
    c2w_per_frame = build_c2w_per_frame(cam_pos, mean_obj)

    # ── Trim mp4 + chrono ──
    mp4_dst = out_video / f"{args.name}.mp4"
    print(f"  trimming mp4 → {mp4_dst}")
    trim_mp4(RGB_TPL.format(clip=args.clip), mp4_dst,
             fr_s, fr_e, args.target_height, args.fps)
    cap = cv2.VideoCapture(str(mp4_dst))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    ok, bg = cap.read(); cap.release()
    chrono_path = out_video / f"{args.name}_chrono.jpg"
    cv2.imwrite(str(chrono_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # ── Write binaries ──
    pc_path = out_data / f"{args.name}_pc.bin"
    write_pc_binary(pc_path, xyz_scene, rgb_scene)
    pc_dist_path = out_data / f"{args.name}_pc_dist.bin"
    pc_dist_path.write_bytes(min_pix_dist.tobytes())

    obj_ids = np.full((K_total,), 0, dtype=np.uint8)         # single object → all 0
    gt3d_path = out_data / f"{args.name}_gt3d.bin"
    gt3d_meta = write_gt3d_binary(gt3d_path, obj_arr, vis_arr, annot_arr,
                                   obj_ids, rgb_u8, n_obj=1)
    gt3d_urls = [f"static/data/{args.name}_gt3d.bin"]
    if gt3d_path.stat().st_size > 49_000_000:
        data = gt3d_path.read_bytes()
        mid  = len(data) // 2
        a_path = out_data / f"{args.name}_gt3d_a.bin"
        b_path = out_data / f"{args.name}_gt3d_b.bin"
        a_path.write_bytes(data[:mid])
        b_path.write_bytes(data[mid:])
        gt3d_path.unlink()
        gt3d_urls = [f"static/data/{args.name}_gt3d_a.bin",
                     f"static/data/{args.name}_gt3d_b.bin"]
        print(f"  split gt3d into {a_path.name} + {b_path.name}")

    cam_path = out_data / f"{args.name}_cam.bin"
    write_cam_binary(cam_path, c2w_per_frame, [FX, FY, CX, CY])

    # ── Bundle JSON ──
    bundle = {
        "configs": [{
            "obj_name": f"hot3d_obj{args.obj}",
            "t": 0,
            "hist_frames":   list(range(0, F // 2)),
            "future_frames": list(range(F // 2, F)),
            "all_frames":    list(range(F)),
            "n_hist": F // 2,
            "l2": 0.0,
            "gt_3d": None, "pred_3d": None, "vis": None,
            "gt_2d": None, "pred_2d": None,
            "pt_colors_rgb": None,
            "color_sample_frame": 0,
        }],
        "mse": 0.0, "l2": 0.0, "n_configs": 1,
        "num_frames": F, "fps": float(args.fps),
        "video_fps_mult": 1,
        "caption": args.caption,
        "raw_meta": {
            "n_points": K_total, "n_frames": F,
            "video_dim_HW": [args.target_height, args.target_height],
            "src_clip_start": 0, "src_clip_end": F - 1,
            "source_3d": str(tracks_root),
            "source_clip": args.clip,
            "source_frame_range": [fr_s, fr_e],
            "obj_index": args.obj,
            "note": (f"HOT3D single-clip; {args.clip} obj {args.obj} "
                     f"({K_total} pts), frames [{fr_s}, {fr_e}]."),
        },
        "chrono": {
            "image_url": f"static/videos/{args.name}_chrono.jpg",
            "frame_indices": [F - 1], "mode": "last_frame_only", "dilate_px": 0,
        },
        "pc_bin": {
            "url": f"static/data/{args.name}_pc.bin",
            "n_points": int(xyz_scene.shape[0]),
            "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
            "n_concat_frames": 1, "subsample": int(args.moge_stride),
            "frame_indices_original": [fr_s],
            "sampling": "pixel-stride",
            "moge": {"a": float(mo_a), "b": float(mo_b),
                     "resolution_level": int(args.moge_resolution)},
        },
        "pc_dist_bin": {
            "url": f"static/data/{args.name}_pc_dist.bin",
            "n_points": int(xyz_scene.shape[0]),
            "format": "float32 N",
        },
        "gt3d_bin": {
            "url": gt3d_urls[0] if len(gt3d_urls) == 1 else None,
            "urls": gt3d_urls,
            **gt3d_meta,
        },
        "cam_bin": {
            "url": f"static/data/{args.name}_cam.bin",
            "format": ("CAM4 v1: magic | u32 v | u32 F | f32×4 intrinsics "
                       "| f32 F*16 c2w"),
            "n_frames": F,
        },
        "camera": {
            "intrinsics_frame0": [FX, FY, CX, CY],
            "video_stem": args.name,
            "video_dim_HW": [args.target_height, args.target_height],
        },
        "viewer_defaults": {
            # Same defaults as the cross-clip HOT3D bundle.
            "objMaskRadiusPx": 30,
            "objectCloud": True,
            "objCloudPointPx": 12,
            "showHist": False, "showPred": False,
            "showBalls": False, "showEndpoints": False,
            "trackSmoothWindow": 1,
            "maxSceneDepth": 0.85,
        },
    }
    out_json = out_data / f"{args.name}.json"
    out_json.write_text(json.dumps(bundle, indent=2))

    print()
    print(f"wrote {out_json}     ({out_json.stat().st_size//1024} KB)")
    if gt3d_path.exists():
        print(f"wrote {gt3d_path}     ({gt3d_path.stat().st_size//1024//1024} MB)")
    print(f"wrote {cam_path}     ({cam_path.stat().st_size//1024} KB)")
    print(f"wrote {pc_path}     ({pc_path.stat().st_size//1024//1024} MB, "
          f"{xyz_scene.shape[0]} pts)")
    print(f"wrote {pc_dist_path}     ({pc_dist_path.stat().st_size//1024} KB)")
    print(f"wrote {mp4_dst}     ({mp4_dst.stat().st_size//1024} KB)")
    print(f"wrote {chrono_path}     ({chrono_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
