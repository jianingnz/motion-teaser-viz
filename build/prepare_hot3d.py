#!/usr/bin/env python3
"""Build a HOT3D cross-clip motion-teaser-viz bundle, per-object subsampled.

Pipeline (mirrors /weka/prior-default/jianingz/home/dataset/hot3d_repo/visual_example/build_scene.py
but operates DIRECTLY on the per-clip all_tracks npz so we can split tracks by
object — the visual_example/data/obj.bin output mixes the 6 objects together
into one anchor-visible pool, which loses object identity and biases the
sampler toward whichever object had the most anchor-visible points.)

  * Anchor frame = clip-001995 frame 84 (first frame of the stitched video)
  * Source:
      - all_tracks/clip-001995_{2d,3d}.npz   12000 = 6 obj × 2000 per object
      - all_tracks/clip-001996_{2d,3d}.npz   same layout
      - /tmp/c1995_cams/{frame:06d}.cameras.json   per-frame world<->cam pose
      - /tmp/c1996_cams/{frame:06d}.cameras.json
      - hot3d_repo/visual_example/data/scene.bin           (MoGe scene cloud)
      - hot3d_repo/visual_example/data/min_pix_dist.bin    (per-scene-point px-dist)
  * Per object 0..5: pick K_per_object visible-at-anchor tracks from the
    object's own 2000-point slice. Concatenate across objects so the GT
    cloud gets even per-object representation. Object IDs are recorded
    in cfg.obj_ids (parallel to gt_3d's per-track axis) for downstream
    debugging.
  * Per-track RGB sampled bilinearly from the anchor frame's RGB.
  * Stitched mp4: clip-001995[84..149] + clip-001996[0..39] = 106 frames,
    re-encoded at libx264 yuv420p.

The MoGe scene PC + min_pix_dist are reused from visual_example/data/ so we
don't need to re-run MoGe inference here.
"""
import argparse, json, subprocess
from pathlib import Path

import cv2
import numpy as np

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

ALL_TRACKS = Path("/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/all_tracks")
RGB_TPL    = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"
SCENE_DIR  = Path("/weka/prior-default/jianingz/home/dataset/hot3d_repo/visual_example/data")
CAM_DIR    = "/tmp/{tag}_cams"  # pre-staged cameras.json per-frame

CLIP_A = "clip-001995"
CLIP_B = "clip-001996"
T_A_START, T_A_END = 84, 149
T_B_START, T_B_END = 0,  39

FX = 608.9346; FY = 608.9346; CX = 701.7970; CY = 705.7216
W_SRC = H_SRC = 1408
N_OBJ = 6
PTS_PER_OBJ = 2000

R_ROT90 = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]], dtype=np.float64)


def quat_to_R(qwxyz):
    w, x, y, z = qwxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]],
        dtype=np.float64)


def load_T_we(cam_json_path, stream="214-1"):
    d = json.load(open(cam_json_path))
    e = d[stream]["T_world_from_camera"]
    R = quat_to_R(e["quaternion_wxyz"])
    t = np.array(e["translation_xyz"], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R; T[:3, 3] = t
    return T


def get_T_we(tag, frame):
    return load_T_we(f"{CAM_DIR.format(tag=tag)}/{frame:06d}.cameras.json")


def diag4(R):
    T = np.eye(4); T[:3, :3] = R; return T


def transform(M, P):
    Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    return (M @ Ph.T).T[:, :3]


def write_pc_binary(path: Path, xyz, rgb_norm):
    """uint32 N | float32 N*3 xyz | uint8 N*3 rgb"""
    N = int(xyz.shape[0])
    rgb_u8 = np.clip(rgb_norm * 255.0, 0, 255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.tobytes())


def stitch_mp4(out_path, target_height=480, fps=30):
    """Concat clip-A[T_A_START..T_A_END] + clip-B[T_B_START..T_B_END] preserving
    1-frame-in-source = 1-frame-in-output so the GT track arrays (one entry per
    source frame) line up with mp4 frames exactly. The earlier `fps=15` filter
    halved that mapping and caused the video to freeze halfway while the JSON
    playhead kept going."""
    pa, pb = RGB_TPL.format(clip=CLIP_A), RGB_TPL.format(clip=CLIP_B)
    n_a = T_A_END - T_A_START + 1
    fc = (
        f"[0:v]trim=start_frame={T_A_START}:end_frame={T_A_END + 1},"
        f"setpts=PTS-STARTPTS,scale=-2:{target_height}[a];"
        f"[1:v]trim=start_frame={T_B_START}:end_frame={T_B_END + 1},"
        f"setpts=PTS-STARTPTS,scale=-2:{target_height}[b];"
        f"[a][b]concat=n=2:v=1:a=0,fps={fps}[v]"
    )
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-i", pa, "-i", pb,
           "-filter_complex", fc, "-map", "[v]",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
           "-vsync", "cfr",
           str(out_path)]
    subprocess.run(cmd, check=True)


def build_c2w_per_frame(cam_pos, mean_obj_pos):
    F = cam_pos.shape[0]
    out = np.zeros((F, 4, 4), dtype=np.float32); out[:, 3, 3] = 1.0
    for t in range(F):
        fwd = mean_obj_pos[t] - cam_pos[t]
        n = np.linalg.norm(fwd)
        fwd = (fwd / n).astype(np.float32) if n > 1e-6 else np.array([0,0,1], dtype=np.float32)
        wup = np.array([0, -1, 0], dtype=np.float32)
        right = np.cross(fwd, wup)
        if np.linalg.norm(right) < 1e-3: right = np.array([1,0,0], dtype=np.float32)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        out[t, :3, 0] = right; out[t, :3, 1] = up; out[t, :3, 2] = fwd
        out[t, :3, 3] = cam_pos[t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--name", default="hot3d_clip1995_clip1996")
    ap.add_argument("--caption", default="HOT3D cross-clip stitch · clip-001995 → clip-001996 (per-object)")
    ap.add_argument("--per-object", type=int, default=2000,
                    help="Per-object cap on the GT cloud. Default 2000 keeps every "
                         "anchor-visible track per object (×6 objects = up to 12k).")
    ap.add_argument("--target-height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30,
                    help="Output mp4 fps. Source is 30 fps; the GT track arrays "
                         "are 1-per-source-frame, so the mp4 must stay at 30 fps "
                         "(else ffmpeg's fps filter drops every other frame and "
                         "the video freezes at frame ~53 while the playhead keeps "
                         "going to 106).")
    args = ap.parse_args()

    out_data  = args.out_dir / "static" / "data";  out_data.mkdir(parents=True, exist_ok=True)
    out_video = args.out_dir / "static" / "videos"; out_video.mkdir(parents=True, exist_ok=True)

    # ── Load source tracks (12k = 6×2000 per clip, 150 frames each) ──
    d3A = np.load(ALL_TRACKS / f"{CLIP_A}_3d.npz", allow_pickle=True)
    d2A = np.load(ALL_TRACKS / f"{CLIP_A}_2d.npz", allow_pickle=True)
    d3B = np.load(ALL_TRACKS / f"{CLIP_B}_3d.npz", allow_pickle=True)
    d2B = np.load(ALL_TRACKS / f"{CLIP_B}_2d.npz", allow_pickle=True)
    pts3_A = d3A["points_3d"]; vis3_A = d3A["visibility"].squeeze()
    tracks_A = d2A["tracks"];   vis2_A = d2A["visibility"]
    pts3_B = d3B["points_3d"]; vis3_B = d3B["visibility"].squeeze()
    print("source: A", pts3_A.shape, "B", pts3_B.shape)
    assert pts3_A.shape[0] == N_OBJ * PTS_PER_OBJ, pts3_A.shape

    # ── Camera poses → anchor-frame transforms ──
    T_we_A0       = get_T_we("c1995", 0)
    T_we_A_anchor = get_T_we("c1995", T_A_START)
    T_we_A = [get_T_we("c1995", t) for t in range(T_A_START, T_A_END + 1)]
    T_we_B0 = get_T_we("c1996", 0)
    T_we_B  = [get_T_we("c1996", t) for t in range(T_B_START, T_B_END + 1)]
    M_anchor_from_world = diag4(R_ROT90) @ np.linalg.inv(T_we_A_anchor)
    M_anchor_from_A0    = M_anchor_from_world @ T_we_A0 @ diag4(R_ROT90.T)
    M_anchor_from_B0    = M_anchor_from_world @ T_we_B0 @ diag4(R_ROT90.T)

    # ── Anchor RGB (used for per-track RGB sampling) ──
    cap = cv2.VideoCapture(RGB_TPL.format(clip=CLIP_A))
    cap.set(cv2.CAP_PROP_POS_FRAMES, T_A_START)
    ok, anchor_bgr = cap.read(); cap.release()
    assert ok, "anchor frame read failed"
    anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)

    # ── Per-object index assembly (no random subsampling) ──────────────
    # Take every track in each object's 2000-point slice, in source order
    # (mirrors how prepare_hdepic.py iterates the dict-per-object format).
    # Visibility is encoded per-frame in `vis_arr`, so per-object emptiness
    # at a given frame is intrinsic to the data — not introduced by
    # sampling. obj_ids[k] records which object track k belongs to so the
    # viewer can colour-code or filter by object later.
    sub_idx = np.arange(N_OBJ * PTS_PER_OBJ, dtype=np.int64)
    obj_ids = np.repeat(np.arange(N_OBJ, dtype=np.int32), PTS_PER_OBJ)
    if args.per_object < PTS_PER_OBJ:
        # Optional cap (e.g. for a faster preview bundle): keep the FIRST
        # `per_object` indices of each slice — deterministic, source-order.
        keep = []
        for obj in range(N_OBJ):
            s = obj * PTS_PER_OBJ
            keep.extend(range(s, s + args.per_object))
        sub_idx = np.array(keep, dtype=np.int64)
        obj_ids = obj_ids[sub_idx]
    K_total = len(sub_idx)
    for obj in range(N_OBJ):
        n = (obj_ids == obj).sum()
        print(f"  obj {obj}: kept {n} tracks (source-order)")

    # ── Build per-frame anchor-coord points (66 from A, 40 from B) ──
    F = (T_A_END - T_A_START + 1) + (T_B_END - T_B_START + 1)
    obj_arr = np.zeros((F, K_total, 3), dtype=np.float32)
    vis_arr = np.zeros((F, K_total), dtype=bool)
    cam_pos = np.zeros((F, 3), dtype=np.float32)
    for fi, t in enumerate(range(T_A_START, T_A_END + 1)):
        P = pts3_A[sub_idx, t, :]  # (K, 3) in clip-A-frame-0-display
        obj_arr[fi] = transform(M_anchor_from_A0, P).astype(np.float32)
        vis_arr[fi] = vis3_A[sub_idx, t]
        cam_h = np.append(T_we_A[fi][:3, 3], 1.0)
        cam_pos[fi] = (M_anchor_from_world @ cam_h)[:3]
    for j, t in enumerate(range(T_B_START, T_B_END + 1)):
        fi = (T_A_END - T_A_START + 1) + j
        P = pts3_B[sub_idx, t, :]
        obj_arr[fi] = transform(M_anchor_from_B0, P).astype(np.float32)
        vis_arr[fi] = vis3_B[sub_idx, t]
        cam_h = np.append(T_we_B[j][:3, 3], 1.0)
        cam_pos[fi] = (M_anchor_from_world @ cam_h)[:3]

    # ── Per-track RGB sampled at anchor 2D-track pixels ──
    # Some tracks are not visible at the anchor frame (NaN uv). For those
    # we don't have a meaningful colour, so emit a neutral mid-gray —
    # they'll only show up at later frames where they ARE visible.
    uv_anchor = tracks_A[T_A_START][sub_idx]  # (K, 2)
    u_raw, v_raw = uv_anchor[:, 0], uv_anchor[:, 1]
    valid_uv = np.isfinite(u_raw) & np.isfinite(v_raw)
    u = np.where(valid_uv, u_raw, 0.0)
    v = np.where(valid_uv, v_raw, 0.0)
    u0 = np.clip(np.floor(u).astype(int), 0, W_SRC - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, H_SRC - 2)
    u1, v1 = u0 + 1, v0 + 1
    du = (u - u0)[:, None]; dv = (v - v0)[:, None]
    rgbf = anchor_rgb.astype(np.float32)
    obj_color = (
        rgbf[v0, u0] * (1 - du) * (1 - dv)
      + rgbf[v0, u1] *      du  * (1 - dv)
      + rgbf[v1, u0] * (1 - du) *      dv
      + rgbf[v1, u1] *      du  *      dv
    ) / 255.0
    # Neutral grey for not-anchor-visible tracks.
    obj_color[~valid_uv] = 0.5

    # ── Build cfg.gt_3d / vis / pt_colors_rgb ──
    # Round to 4 decimals (≈ 0.1 mm precision in metres) to keep the
    # bundle JSON readable and ~40 % smaller than the default 17-sig-fig
    # serialization that float64 would produce.
    gt_3d = []
    for fi in range(F):
        frame_pts = []
        for k in range(K_total):
            if not vis_arr[fi, k]:
                frame_pts.append(None); continue
            x, y, z = obj_arr[fi, k]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                frame_pts.append(None); continue
            frame_pts.append([round(float(x), 4), round(float(y), 4), round(float(z), 4)])
        gt_3d.append(frame_pts)
    vis_list = [[bool(vis_arr[fi, k]) for k in range(K_total)] for fi in range(F)]
    obj_color_list = obj_color.tolist()  # numpy float32 → Python floats
    pt_colors_rgb = [[int(c[0]*255), int(c[1]*255), int(c[2]*255)] for c in obj_color_list]

    # 2D anchor projection (only the anchor frame is meaningful)
    gt_2d = [None] * F
    proj0 = []
    for k in range(K_total):
        if not vis_arr[0, k]: proj0.append(None); continue
        x, y, z = float(obj_arr[0, k, 0]), float(obj_arr[0, k, 1]), float(obj_arr[0, k, 2])
        if z <= 0.05: proj0.append(None); continue
        u = (FX * x / z + CX) / W_SRC
        v = (FY * y / z + CY) / H_SRC
        proj0.append([u, v] if 0 <= u <= 1 and 0 <= v <= 1 else None)
    gt_2d[0] = proj0

    # ── Camera per-frame c2w (positions + synthesized fwd to mean obj) ──
    mean_obj = np.zeros((F, 3), dtype=np.float32)
    for fi in range(F):
        m = vis_arr[fi]
        mean_obj[fi] = obj_arr[fi][m].mean(axis=0) if m.any() else cam_pos[fi]
    c2w_per_frame = build_c2w_per_frame(cam_pos, mean_obj)

    # ── Stitch + downsample mp4 ──
    mp4_dst = out_video / f"{args.name}.mp4"
    stitch_mp4(mp4_dst, target_height=args.target_height, fps=args.fps)

    # ── Chrono = last frame of stitched video (stamps without 2D mask not meaningful) ──
    cap = cv2.VideoCapture(str(mp4_dst))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    ok, bg = cap.read(); cap.release()
    chrono_path = out_video / f"{args.name}_chrono.jpg"
    cv2.imwrite(str(chrono_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # ── Scene PC + per-point px-distance side-channel (reused from visual_example) ──
    scene = np.fromfile(SCENE_DIR / "scene.bin", dtype=np.float32).reshape(-1, 6)
    pc_path = out_data / f"{args.name}_pc.bin"
    write_pc_binary(pc_path, scene[:, :3], scene[:, 3:6])
    min_pix_dist = np.fromfile(SCENE_DIR / "min_pix_dist.bin", dtype=np.float32)
    assert min_pix_dist.shape[0] == scene.shape[0]
    pc_dist_path = out_data / f"{args.name}_pc_dist.bin"
    pc_dist_path.write_bytes(min_pix_dist.astype(np.float32).tobytes())

    # ── Bundle ──
    # pred_3d is intentionally null in HOT3D — the viewer hides the pred
    # layer by default (viewer_defaults.showPred = false) so we save 50 %
    # of the bundle size by not duplicating gt_3d.
    config = {
        "gt_3d": gt_3d, "pred_3d": None, "vis": vis_list,
        "obj_name": "hot3d_objects",
        "t": 0,
        "hist_frames":   list(range(0, F // 2)),
        "future_frames": list(range(F // 2, F)),
        "all_frames":    list(range(F)),
        "n_hist": F // 2,
        "l2": 0.0,
        "gt_2d": gt_2d, "pred_2d": None,
        "pt_colors_rgb": pt_colors_rgb,
        "color_sample_frame": 0,
        "obj_ids": obj_ids.tolist(),
    }
    bundle = {
        "configs": [config],
        "mse": 0.0, "l2": 0.0, "n_configs": 1,
        "num_frames": F, "fps": float(args.fps),
        "video_fps_mult": 1,
        "caption": args.caption,
        "raw_meta": {
            "n_points": K_total, "n_frames": F,
            "video_dim_HW": [args.target_height, args.target_height],
            "src_clip_start": 0, "src_clip_end": F - 1,
            "source_2d": str(ALL_TRACKS),
            "source_3d": str(ALL_TRACKS),
            "note": f"HOT3D cross-clip; per-object subsample (K={args.per_object} × {N_OBJ} objects).",
        },
        "chrono": {
            "image_url": f"static/videos/{args.name}_chrono.jpg",
            "frame_indices": [F - 1], "mode": "last_frame_only", "dilate_px": 0,
        },
        "pc_bin": {
            "url": f"static/data/{args.name}_pc.bin",
            "n_points": int(scene.shape[0]),
            "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
            "n_concat_frames": 1, "subsample": 1,
            "frame_indices_original": [T_A_START],
        },
        "camera": {
            "c2w_frame0": c2w_per_frame[0].tolist(),
            "intrinsics_frame0": [FX, FY, CX, CY],
            "video_stem": args.name,
            "video_dim_HW": [args.target_height, args.target_height],
            "c2w_per_frame": c2w_per_frame.tolist(),
            "intrinsics_per_frame": [[FX, FY, CX, CY]] * F,
        },
        "pc_dist_bin": {
            "url": f"static/data/{args.name}_pc_dist.bin",
            "n_points": int(scene.shape[0]),
            "format": "float32 N",
        },
        "viewer_defaults": {
            "objMaskRadiusPx": 30,
            "objectCloud": True,
            "objCloudPointPx": 7,
            "showHist": False, "showPred": False,
            "trackSmoothWindow": 1,
            "maxSceneDepth": 0.85,
        },
    }
    out_json = out_data / f"{args.name}.json"
    out_json.write_text(json.dumps(bundle))
    # Versioned snapshot for convenience — keeps prior bundles around so
    # the user can A/B compare without re-running this script. Filename
    # encodes the most consequential build flag (per-object cap) so older
    # versions don't get clobbered by re-runs.
    snapshot = out_data / f"{args.name}__perObj{args.per_object}.json"
    snapshot.write_text(json.dumps(bundle))

    # ── Debug: per-frame visibility breakdown by object ──
    print()
    print(f"VISIBILITY DEBUG ({K_total} total tracks across {N_OBJ} objects):")
    for fi in [0, 1, 2, 5, 10, 30, 60, 65, 66, 100, F-1]:
        per_obj = np.zeros(N_OBJ, dtype=int)
        for k in range(K_total):
            if vis_arr[fi, k]:
                per_obj[obj_ids[k]] += 1
        print(f"  frame {fi:3d}: total={vis_arr[fi].sum():4d}  per-obj={per_obj.tolist()}")

    print()
    print(f"wrote {out_json}  ({out_json.stat().st_size//1024} KB)")
    print(f"wrote {mp4_dst}   ({mp4_dst.stat().st_size//1024} KB, {F} frames)")
    print(f"wrote {chrono_path}")
    print(f"wrote {pc_path}   ({scene.shape[0]} pts)")


if __name__ == "__main__":
    main()
