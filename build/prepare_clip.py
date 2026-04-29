#!/usr/bin/env python3
"""
Take a source motion5 clip JSON + its mp4 and produce a self-contained
single-clip bundle for motion-teaser-viz:

  - mp4 trimmed to ONLY the clip frames (hist[0] .. future[-1])
  - JSON enriched with per-point object RGB sampled from frame 0
  - JSON's video-frame indices remapped to be 0-based within the trimmed mp4
  - Chronophotography image: sharp last-frame background + opaque
    object-region stamps from each sampled frame on top
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
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"  # rely on PATH


# ────────────────────── dense PC from depth (visual.py style) ──────────────────────

def load_exr_depth(raw_bytes: bytes) -> np.ndarray:
    """Decode a single-channel ('Z') EXR byte string to float32 (H, W)."""
    import OpenEXR
    import Imath
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
        f.write(raw_bytes); fname = f.name
    try:
        exr = OpenEXR.InputFile(fname)
        dw  = exr.header()['dataWindow']
        W   = dw.max.x - dw.min.x + 1
        H   = dw.max.y - dw.min.y + 1
        buf = exr.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(buf, dtype=np.float32).reshape(H, W).copy()
        exr.close()
    finally:
        os.unlink(fname)
    return depth


def backproject_depth_to_world(depth: np.ndarray, rgb: np.ndarray,
                               c2w: np.ndarray, intrinsics: np.ndarray,
                               subsample: int = 2):
    """Regular-grid sample of depth → world-space (xyz, colors). visual.py logic."""
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    us = np.arange(0, W, subsample, dtype=np.int32)
    vs = np.arange(0, H, subsample, dtype=np.int32)
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
    return xyz, colors


def build_dense_concat_pc(vipe_root: Path, video_stem: str,
                          frame_indices: list, subsample: int = 3):
    """Concatenate depth-backprojected PCs from `frame_indices` (raw video frame ids)."""
    pose_path  = vipe_root / 'pose' / f'{video_stem}.npz'
    intr_path  = vipe_root / 'intrinsics' / f'{video_stem}.npz'
    depth_zip  = vipe_root / 'depth' / f'{video_stem}.zip'
    rgb_path   = vipe_root / 'rgb' / f'{video_stem}.mp4'
    if not all(p.exists() for p in [pose_path, intr_path, depth_zip, rgb_path]):
        raise RuntimeError(f"vipe artifacts missing for {video_stem}")

    poses = np.load(pose_path)['data'].astype(np.float32)
    intrs = np.load(intr_path)['data'].astype(np.float32)

    cap = cv2.VideoCapture(str(rgb_path))
    rgb_frames = {}
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read RGB frame {fi} from {rgb_path}")
        rgb_frames[int(fi)] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()

    all_xyz, all_col = [], []
    with zipfile.ZipFile(depth_zip) as zf:
        names = sorted(zf.namelist())
        for fi in frame_indices:
            depth = load_exr_depth(zf.read(names[int(fi)]))
            c2w   = poses[int(fi)]
            intr  = intrs[int(fi)]
            rgb   = rgb_frames[int(fi)]
            xyz, col = backproject_depth_to_world(depth, rgb, c2w, intr, subsample)
            all_xyz.append(xyz); all_col.append(col)

    xyz_all = np.concatenate(all_xyz, axis=0)
    col_all = np.concatenate(all_col, axis=0)
    return xyz_all, col_all, poses, intrs


def write_pc_binary(path: Path, xyz: np.ndarray, colors_u8: np.ndarray):
    """Compact binary: 4-byte LE uint32 N | N*12 bytes float32 xyz | N*3 bytes uint8 rgb."""
    N = int(xyz.shape[0])
    with open(path, 'wb') as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(colors_u8.astype(np.uint8).tobytes())


# ────────────────────── color sampling ──────────────────────

def sample_color(img: np.ndarray, u: float, v: float, half: int = 3) -> list:
    H, W = img.shape[:2]
    cx = int(round(u * (W - 1)))
    cy = int(round(v * (H - 1)))
    x0 = max(0, cx - half); x1 = min(W, cx + half + 1)
    y0 = max(0, cy - half); y1 = min(H, cy + half + 1)
    patch = img[y0:y1, x0:x1].reshape(-1, 3)
    mean_bgr = patch.mean(axis=0)
    return [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]


def grab_frame(mp4_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {mp4_path}")
    return frame


def grab_frames(mp4_path: Path, frame_indices: list) -> list:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4_path}")
    out = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {fi} from {mp4_path}")
        out.append(frame)
    cap.release()
    return out


# ────────────────────── chronophotography (object stamps) ──────────────────────

def build_object_stamps_chrono(
    bg: np.ndarray,
    stamp_frames: list,                 # list of BGR images, one per sampled time
    stamp_pts2d:   list,                # parallel list of [P, 2] normalized point arrays
    dilate_px:     int = 14,            # margin around the convex hull (px)
    edge_blur:     int = 7,             # small soft edge to avoid jagged seams
) -> np.ndarray:
    """Composite an opaque object cutout per stamp_frame onto `bg`.

    The mask for each stamp is the **convex hull** of the query points,
    optionally dilated by a small margin. This tightly hugs the object
    rather than the union-of-discs approach (which would include lots of
    background between query points). Hard-blended so the object reads as
    a solid stamp, not a translucent ghost; later stamps occlude earlier
    ones where they overlap.
    """
    H, W = bg.shape[:2]
    out = bg.astype(np.float32).copy()

    for frame, pts in zip(stamp_frames, stamp_pts2d):
        coords = []
        for pt in pts:
            if pt is None or pt[0] is None:
                continue
            u, v = pt[0], pt[1]
            if not (0 <= u <= 1 and 0 <= v <= 1):
                continue
            coords.append([int(round(u * (W - 1))), int(round(v * (H - 1)))])
        if len(coords) < 3:
            continue
        coords_np = np.array(coords, dtype=np.int32)
        hull = cv2.convexHull(coords_np)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel)
        if edge_blur > 1:
            soft = cv2.GaussianBlur(mask, (edge_blur, edge_blur), 0).astype(np.float32) / 255.0
        else:
            soft = mask.astype(np.float32) / 255.0
        soft = soft[..., None]
        out = out * (1.0 - soft) + frame.astype(np.float32) * soft

    return out.clip(0, 255).astype(np.uint8)


# ────────────────────── mp4 trim ──────────────────────

def trim_mp4_ffmpeg(src: Path, dst: Path, start_frame: int, end_frame: int, fps: float):
    """Use ffmpeg `select` filter to extract exactly frames [start_frame, end_frame] inclusive."""
    n_frames = end_frame - start_frame + 1
    duration = n_frames / fps
    # Use frame-accurate seek + select; re-encode h264 yuv420p crf 22.
    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=PTS-STARTPTS,fps={fps}",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return n_frames


# ────────────────────── frame-index remap ──────────────────────

def remap_clip_indices(clip: dict, start_frame: int):
    """Subtract start_frame from every video-frame index so the trimmed mp4 starts at 0."""
    for cfg in clip["configs"]:
        cfg["hist_frames"]   = [f - start_frame for f in cfg["hist_frames"]]
        cfg["future_frames"] = [f - start_frame for f in cfg["future_frames"]]
        cfg["all_frames"]    = [f - start_frame for f in cfg["all_frames"]]
        if "color_sample_frame" in cfg:
            cfg["color_sample_frame"] = cfg["color_sample_frame"] - start_frame
    if "chrono" in clip:
        clip["chrono"]["frame_indices"] = [f - start_frame for f in clip["chrono"]["frame_indices"]]
    new_total = clip["configs"][0]["all_frames"][-1] + 1
    clip["num_frames"] = new_total


# ────────────────────── main ──────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-json", required=True, type=Path)
    ap.add_argument("--src-mp4", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--n-stamps", type=int, default=4,
                    help="Number of object stamps in the chronophoto (last frame is the bg + this many earlier stamps)")
    ap.add_argument("--dilate-px", type=int, default=2,
                    help="Margin (px) added around the strict convex hull of query points")
    ap.add_argument("--edge-blur", type=int, default=3,
                    help="Tiny gaussian blur on the mask edge (odd kernel size; 1 = none)")
    ap.add_argument("--vipe-root", type=Path,
                    default=Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe/vipe_results"),
                    help="vipe_results dir containing depth/pose/intrinsics/rgb subdirs")
    ap.add_argument("--video-stem", required=True,
                    help="Video stem (e.g. part4_pick_place_food_2372) — used to locate vipe artifacts")
    ap.add_argument("--pc-subsample", type=int, default=3,
                    help="Pixel stride when backprojecting depth (lower = denser; visual.py default = 2)")
    ap.add_argument("--pc-n-frames", type=int, default=5,
                    help="How many depth frames to backproject and concatenate")
    args = ap.parse_args()

    clip = json.loads(args.src_json.read_text())

    out_data_dir = args.out_dir / "static" / "data"
    out_video_dir = args.out_dir / "static" / "videos"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_video_dir.mkdir(parents=True, exist_ok=True)

    cfg0 = clip["configs"][0]
    af = cfg0["all_frames"]
    clip_start = af[0]
    clip_end   = af[-1]
    n_clip     = clip_end - clip_start + 1
    fps        = float(clip.get("fps", 30))

    # ── 1. Sample per-point colors from each config's first hist frame (using ORIGINAL indices) ──
    sampled = {}
    for cfg in clip["configs"]:
        hist0 = cfg["hist_frames"][0]
        gt2d_0 = cfg["gt_2d"][0]
        if gt2d_0 is None:
            raise RuntimeError(f"config has no gt_2d: obj={cfg['obj_name']}")
        frame = grab_frame(args.src_mp4, hist0)
        colors = []
        for pt in gt2d_0:
            if pt is None or pt[0] is None or not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
                colors.append(None)
            else:
                colors.append(sample_color(frame, pt[0], pt[1]))
        cfg["pt_colors_rgb"] = colors
        cfg["color_sample_frame"] = hist0
        sampled[cfg["obj_name"]] = sum(1 for c in colors if c is not None)

    # ── 2. Build object-stamp chronophoto (bg = LAST clip frame, stamps from earlier frames) ──
    n_stamps = min(args.n_stamps, n_clip)
    # Pick stamp positions evenly across the clip-window indices (in cfg0.all_frames space)
    stamp_idxs_local = [int(round(i)) for i in np.linspace(0, len(af) - 1, n_stamps)]
    # Drop duplicates while preserving order
    seen = set(); stamp_idxs_local_dedup = []
    for k in stamp_idxs_local:
        if k not in seen: seen.add(k); stamp_idxs_local_dedup.append(k)
    stamp_idxs_local = stamp_idxs_local_dedup
    stamp_video_idxs = [af[k] for k in stamp_idxs_local]

    # Drop the LAST one (it would be identical to the bg)
    if stamp_video_idxs and stamp_video_idxs[-1] == clip_end:
        stamp_idxs_local = stamp_idxs_local[:-1]
        stamp_video_idxs = stamp_video_idxs[:-1]

    bg_frame = grab_frame(args.src_mp4, clip_end)
    stamp_frames = grab_frames(args.src_mp4, stamp_video_idxs) if stamp_video_idxs else []
    # Per stamp, the 2D point list comes from cfg0.gt_2d at the corresponding all_frames index
    stamp_pts2d = [cfg0["gt_2d"][k] for k in stamp_idxs_local]

    chrono_img = build_object_stamps_chrono(
        bg=bg_frame,
        stamp_frames=stamp_frames,
        stamp_pts2d=stamp_pts2d,
        dilate_px=args.dilate_px,
        edge_blur=args.edge_blur,
    )
    chrono_path = out_video_dir / f"{args.clip_id}_chrono.jpg"
    cv2.imwrite(str(chrono_path), chrono_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    clip["chrono"] = {
        "image_url": f"static/videos/{args.clip_id}_chrono.jpg",
        "frame_indices": stamp_video_idxs + [clip_end],   # original indices
        "mode": "object_stamps_on_last_frame_convex_hull",
        "dilate_px": args.dilate_px,
    }

    # ── 3. Trim mp4 to [clip_start, clip_end] inclusive ──
    mp4_dst = out_video_dir / f"{args.clip_id}.mp4"
    n_written = trim_mp4_ffmpeg(args.src_mp4, mp4_dst, clip_start, clip_end, fps=fps)
    if n_written != n_clip:
        print(f"  WARN: ffmpeg wrote {n_written} frames, expected {n_clip}")

    # ── 4. Build dense REGULAR-GRID concatenated PC from depth ──
    out_pc_dir = args.out_dir / "static" / "data"
    out_pc_dir.mkdir(parents=True, exist_ok=True)
    n_concat = max(1, min(args.pc_n_frames, n_clip))
    pc_frame_idxs = np.linspace(clip_start, clip_end, n_concat).astype(int)
    print(f"Backprojecting depth at frames {list(pc_frame_idxs)} "
          f"(subsample={args.pc_subsample}) …")
    xyz_d, col_d, poses_full, intrs_full = build_dense_concat_pc(
        args.vipe_root, args.video_stem, list(pc_frame_idxs),
        subsample=args.pc_subsample,
    )
    pc_bin_path = out_pc_dir / f"{args.clip_id}_pc.bin"
    write_pc_binary(pc_bin_path, xyz_d, col_d)
    # Drop the bulky pc_xyz / pc_colors arrays from JSON; viewer fetches the bin.
    clip.pop("pc_xyz", None)
    clip.pop("pc_colors", None)
    clip["pc_bin"] = {
        "url": f"static/data/{args.clip_id}_pc.bin",
        "n_points": int(xyz_d.shape[0]),
        "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
        "n_concat_frames": n_concat,
        "subsample": args.pc_subsample,
        "frame_indices_original": [int(x) for x in pc_frame_idxs],
    }

    # Stash camera params + frame-0 ego camera so the static panels can use them
    f0 = clip_start
    clip["camera"] = {
        "c2w_frame0": poses_full[f0].astype(float).tolist(),
        "intrinsics_frame0": intrs_full[f0].astype(float).tolist(),
        "video_stem": args.video_stem,
    }

    # ── 5. Remap all video-frame indices in JSON to be 0-based within trimmed mp4 ──
    remap_clip_indices(clip, clip_start)

    # ── 6. Write enriched JSON ──
    out_json = out_data_dir / f"{args.clip_id}.json"
    out_json.write_text(json.dumps(clip))

    print(f"wrote {out_json}    ({out_json.stat().st_size//1024} KB)")
    print(f"wrote {mp4_dst}     ({mp4_dst.stat().st_size//1024} KB, {n_clip} frames @ {fps:.1f}fps)")
    print(f"wrote {chrono_path}  ({chrono_path.stat().st_size//1024} KB, bg=last frame + {len(stamp_video_idxs)} object stamps)")
    print(f"wrote {pc_bin_path}   ({pc_bin_path.stat().st_size//1024} KB, {xyz_d.shape[0]} dense PC points from {n_concat} frames @ subsample={args.pc_subsample})")
    for obj, n in sampled.items():
        print(f"  sampled {n} colors for obj={obj}")


if __name__ == "__main__":
    main()
