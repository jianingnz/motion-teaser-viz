#!/usr/bin/env python3
"""
Augment an existing EgoDex clip bundle (already emitted by prepare_clip.py)
with the *full source video* + the *full-video object trajectory* + a
*scene point cloud at video frame 0*.

The motivation is to make the visualization tell the story of the pipeline:
the full untrimmed video is shown alongside the cut clip, with a visible
clip-cut bracket on the full-video timeline. The new big PC panel can show
the scene PC at the first frame of the full video instead of the first
frame of the clip — and the viewer's UI exposes both as switchable.

Inputs (all expected to exist for an EgoDex clip whose JSON has
`camera.video_stem` set):
    /weka/prior-default/jianingz/home/project/_GenTraj/vipe/
        vipe_results/rgb/<stem>.mp4
        vipe_results/pose/<stem>.npz
        vipe_results/intrinsics/<stem>.npz
        vipe_results/depth/<stem>.zip
        final_tracks/<stem>_2d.npz   (filtered + smoothed full-video tracks)

Outputs (paths relative to --out-dir):
    static/videos/<clip-id>_full.mp4         (h264 re-encode of the full src)
    static/data/<clip-id>_full_pc.bin        (depth-backprojected PC @ frame 0)
    static/data/<clip-id>.json               (in-place: + full_video, + full_pc_bin, + full_2d_tracks)
"""
import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


VIPE_ROOT = Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe/vipe_results")
TRACK_ROOT = Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe")


def load_exr_depth(raw_bytes: bytes) -> np.ndarray:
    import OpenEXR
    import Imath
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        f.write(raw_bytes)
        fname = f.name
    try:
        exr = OpenEXR.InputFile(fname)
        dw = exr.header()["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        buf = exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(buf, dtype=np.float32).reshape(H, W).copy()
        exr.close()
    finally:
        os.unlink(fname)
    return depth


def backproject_frame_to_pc(stem: str, frame_idx: int, subsample: int = 3):
    """Backproject a single (RGB, depth, pose, intrinsics) frame to (xyz, rgb)."""
    pose = np.load(VIPE_ROOT / "pose" / f"{stem}.npz")["data"].astype(np.float32)
    intr = np.load(VIPE_ROOT / "intrinsics" / f"{stem}.npz")["data"].astype(np.float32)
    depth_zip = VIPE_ROOT / "depth" / f"{stem}.zip"
    rgb_path = VIPE_ROOT / "rgb" / f"{stem}.mp4"

    cap = cv2.VideoCapture(str(rgb_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read RGB frame {frame_idx} from {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with zipfile.ZipFile(depth_zip) as zf:
        names = sorted(zf.namelist())
        depth = load_exr_depth(zf.read(names[int(frame_idx)]))

    c2w = pose[int(frame_idx)]
    fx, fy, cx, cy = intr[int(frame_idx)]

    H, W = depth.shape
    s = max(1, int(subsample))
    us = np.arange(0, W, s, dtype=np.int32)
    vs = np.arange(0, H, s, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    z = depth[vv, uu]
    valid = (z > 0) & np.isfinite(z)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    xc = (uu.astype(np.float32) - cx) / fx * z
    yc = (vv.astype(np.float32) - cy) / fy * z
    pts_cam = np.stack([xc, yc, z, np.ones_like(z)], axis=1)
    xyz = (c2w @ pts_cam.T).T[:, :3].astype(np.float32)
    colors = rgb[vv, uu].astype(np.uint8)
    return xyz, colors, (H, W), pose.shape[0]


def write_pc_binary(path: Path, xyz: np.ndarray, colors_u8: np.ndarray):
    N = int(xyz.shape[0])
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(colors_u8.astype(np.uint8).tobytes())


def reencode_full_mp4(src: Path, dst: Path, out_fps: float):
    """h264 re-encode of the full source mp4 (no trimming)."""
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", str(src),
        "-r", str(out_fps),
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def load_full_2d_tracks(stem: str):
    """Filtered + smoothed full-video 2D tracks → list-of-frames [[u,v]|None]."""
    p = TRACK_ROOT / "final_tracks" / f"{stem}_2d.npz"
    if not p.exists():
        raise RuntimeError(f"missing final_tracks 2d: {p}")
    d = np.load(p, allow_pickle=True)
    tracks = d["tracks"]            # (T, N, 2) px
    vis = d["visibility"]           # (T, N) bool
    H, W = [int(x) for x in d["dim"]]
    if tracks.dtype == object:
        # HD-EPIC dict format — concatenate every named object's tracks
        tracks_d = tracks[()]
        vis_d = vis[()]
        all_t, all_v = [], []
        for k in tracks_d.keys():
            all_t.append(tracks_d[k])
            all_v.append(vis_d[k])
        tracks = np.concatenate(all_t, axis=1)
        vis = np.concatenate(all_v, axis=1)
    T, N, _ = tracks.shape
    out = []
    for t in range(T):
        frame = []
        for n in range(N):
            if not bool(vis[t, n]):
                frame.append(None)
                continue
            u = float(tracks[t, n, 0]) / max(W, 1)
            v = float(tracks[t, n, 1]) / max(H, 1)
            frame.append([u, v])
        out.append(frame)
    return out, T, N, (H, W)


def load_full_2d_raw_tracks(stem: str):
    """Pre-filter, pre-smoothing 2D tracks (cotracker output, full video).

    These are the noisy raw tracks demonstrating what the filtering +
    smoothing stage cleans up. Indexed by their own P_total (different
    cardinality than the kept-point set in final_tracks).
    """
    p = TRACK_ROOT / "track_output" / stem / f"{stem}_merged.npz"
    if not p.exists():
        raise RuntimeError(f"missing track_output merged: {p}")
    d = np.load(p, allow_pickle=True)
    tracks = d["tracks"]            # (T, P, 2) px
    vis = d["visibility"]           # (T, P) bool
    H, W = [int(x) for x in d["dim"]]
    T, P, _ = tracks.shape
    out = []
    for t in range(T):
        frame = []
        for n in range(P):
            if not bool(vis[t, n]):
                frame.append(None)
                continue
            u = float(tracks[t, n, 0]) / max(W, 1)
            v = float(tracks[t, n, 1]) / max(H, 1)
            frame.append([u, v])
        out.append(frame)
    return out, T, P, (H, W)


def load_full_3d_tracks(stem: str):
    """Filtered + smoothed full-video 3D tracks → list-of-frames [[x,y,z]|None]."""
    p = TRACK_ROOT / "final_tracks" / f"{stem}_3d.npz"
    if not p.exists():
        raise RuntimeError(f"missing final_tracks 3d: {p}")
    d = np.load(p, allow_pickle=True)
    points = d["points_3d"]         # (N, T, 3) flat / dict
    vis = d["visibility"]           # (N, T, 1) bool
    if points.dtype == object:
        points_d = points[()]
        vis_d = vis[()]
        all_p, all_v = [], []
        for k in points_d.keys():
            all_p.append(points_d[k])
            all_v.append(vis_d[k])
        points = np.concatenate(all_p, axis=0)
        vis = np.concatenate(all_v, axis=0)
    if vis.ndim == 3:
        vis = vis.squeeze(-1)
    N, T = points.shape[:2]
    out = []
    for t in range(T):
        frame = []
        for n in range(N):
            if not bool(vis[n, t]):
                frame.append(None)
                continue
            x = float(points[n, t, 0]); y = float(points[n, t, 1]); z = float(points[n, t, 2])
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                frame.append(None)
                continue
            frame.append([x, y, z])
        out.append(frame)
    return out, T, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True,
                    help="Output bundle id (matches the existing files in static/data and static/videos).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="motion-teaser-viz repo root (the dir containing static/).")
    ap.add_argument("--pc-subsample", type=int, default=3)
    ap.add_argument("--out-fps", type=float, default=15.0,
                    help="Output mp4 fps. EgoDex source is 15fps, so default re-uses native rate.")
    args = ap.parse_args()

    json_path = args.out_dir / "static" / "data" / f"{args.clip_id}.json"
    if not json_path.exists():
        raise RuntimeError(f"clip JSON not found: {json_path}")
    clip = json.loads(json_path.read_text())
    cam = clip.get("camera") or {}
    stem = cam.get("video_stem")
    if not stem:
        raise RuntimeError(f"clip JSON has no camera.video_stem — not an EgoDex clip we can extend: {json_path}")

    # 1. Re-encode the full source mp4
    src_mp4 = VIPE_ROOT / "rgb" / f"{stem}.mp4"
    if not src_mp4.exists():
        raise RuntimeError(f"missing source rgb mp4: {src_mp4}")
    full_mp4 = args.out_dir / "static" / "videos" / f"{args.clip_id}_full.mp4"
    full_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] re-encoding full mp4 → {full_mp4}")
    reencode_full_mp4(src_mp4, full_mp4, args.out_fps)

    # 2. Frame-0 scene PC of the full video
    print(f"[2/3] backprojecting depth @ frame 0 (subsample={args.pc_subsample}) …")
    xyz0, col0, (H, W), T_video = backproject_frame_to_pc(
        stem, frame_idx=0, subsample=args.pc_subsample)
    full_pc = args.out_dir / "static" / "data" / f"{args.clip_id}_full_pc.bin"
    write_pc_binary(full_pc, xyz0, col0)

    # 3. Full-video 2D + 3D tracks (filtered + smoothed; same kept-point set)
    print(f"[3/3] loading full-video filtered 2D + 3D tracks …")
    full_2d, T_tracks, N_pts, (Ht, Wt) = load_full_2d_tracks(stem)
    full_3d, T_tracks_3d, N_pts_3d = load_full_3d_tracks(stem)
    if N_pts_3d != N_pts:
        raise RuntimeError(f"2D/3D track point counts disagree: "
                           f"2d N={N_pts}, 3d N={N_pts_3d} — pipeline mismatch.")
    if T_tracks_3d != T_tracks:
        T_use = min(T_tracks, T_tracks_3d)
        full_2d = full_2d[:T_use]; full_3d = full_3d[:T_use]
        T_tracks = T_use
    if T_tracks != T_video:
        T_use = min(T_tracks, T_video)
        full_2d = full_2d[:T_use]; full_3d = full_3d[:T_use]
        T_tracks = T_use

    # Clip-cut window in original-video frame indices comes from raw_meta
    # (set by prepare_clip.py before remapping). For DROID/DAVIS clips we
    # don't have it; this script is EgoDex-only so it must be present.
    raw_meta = clip.get("raw_meta") or {}
    src_clip_start = raw_meta.get("src_clip_start")
    src_clip_end = raw_meta.get("src_clip_end")
    if src_clip_start is None or src_clip_end is None:
        raise RuntimeError("clip JSON has no raw_meta.src_clip_start/src_clip_end "
                           "— required to draw the clip-cut bracket on the full video.")

    clip["full_video"] = {
        "url": f"static/videos/{args.clip_id}_full.mp4",
        "n_frames": int(T_video),
        "fps": float(args.out_fps),
        "src_clip_start": int(src_clip_start),
        "src_clip_end": int(src_clip_end),
        "video_dim_HW": [int(H), int(W)],
    }
    clip["full_pc_bin"] = {
        "url": f"static/data/{args.clip_id}_full_pc.bin",
        "n_points": int(xyz0.shape[0]),
        "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
        "frame_index_original": 0,
        "subsample": int(args.pc_subsample),
    }
    clip["full_video_2d"] = {
        "tracks": full_2d,                 # (T_video, N) of [u,v] | null
        "n_points": int(N_pts),
        "n_frames": int(T_tracks),
        "video_dim_HW": [int(Ht), int(Wt)],
        "source": str(TRACK_ROOT / "final_tracks" / f"{stem}_2d.npz"),
        "note": "Filtered + smoothed full-video object tracks — same point set as the cut-clip's gt_2d/gt_3d.",
    }
    clip["full_video_3d"] = {
        "tracks": full_3d,                 # (T_video, N) of [x,y,z] | null
        "n_points": int(N_pts),
        "n_frames": int(T_tracks),
        "source": str(TRACK_ROOT / "final_tracks" / f"{stem}_3d.npz"),
        "note": "Filtered + smoothed full-video 3D tracks; aligned 1:1 with full_video_2d points.",
    }

    # Pre-filter, pre-smoothing 2D tracks (cotracker raw output) — used by
    # the full-video panel's "Raw" overlay layer to demonstrate the
    # filter+smooth cleanup. Indexed by its own P_total (≠ N_pts).
    print("       loading full-video raw (pre-filter) 2D tracks …")
    raw_2d_full, T_raw, P_raw, (Hr, Wr) = load_full_2d_raw_tracks(stem)
    if T_raw != T_tracks:
        T_use = min(T_raw, T_tracks)
        raw_2d_full = raw_2d_full[:T_use]
        T_raw = T_use
    clip["full_video_raw_2d"] = {
        "tracks": raw_2d_full,             # (T_video, P_raw) of [u,v] | null
        "n_points": int(P_raw),
        "n_frames": int(T_raw),
        "video_dim_HW": [int(Hr), int(Wr)],
        "source": str(TRACK_ROOT / "track_output" / stem / f"{stem}_merged.npz"),
        "note": "Pre-filter, pre-smoothing 2D tracks — the noisy raw cotracker output.",
    }

    json_path.write_text(json.dumps(clip))
    print(f"wrote {full_mp4}   ({full_mp4.stat().st_size//1024} KB, {T_video} frames @ {args.out_fps:.1f}fps)")
    print(f"wrote {full_pc}     ({full_pc.stat().st_size//1024} KB, {xyz0.shape[0]} PC points)")
    print(f"wrote {json_path}   (+ full_video, full_pc_bin, full_video_2d: {N_pts} pts × {T_tracks} frames)")


if __name__ == "__main__":
    main()
