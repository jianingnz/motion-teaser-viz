#!/usr/bin/env python3
"""
Lightweight clip-bundle builder for datasets that already ship `pc_xyz` /
`pc_colors` in the source JSON (DAVIS, DROID — anything that motion5-viz's
preprocess_modeling.py has already PC-baked). Differences vs prepare_clip.py:

  - No vipe artifact dependency (depth zip / pose / intrinsics / raw tracks)
  - No raw_2d / raw_3d → those datasets don't have cotracker outputs at the
    paths prepare_clip.py expects, so we just skip the raw layer.
  - Re-uses pc_xyz / pc_colors from the source JSON to write the binary PC.
  - Handles `video_fps_mult > 1` clips (e.g. DROID at 60fps native, tracks at
    15fps): trims AND subsamples the mp4 to one video frame per track index,
    then resets video_fps_mult to 1 so the viewer plays it 1:1 with track
    indices.

Usage:
    python build/prepare_clip_simple.py \\
        --src-json motion5-viz/static/data/modeling_json/davis/test/<id>.json \\
        --src-mp4  motion5-viz/static/videos/modeling/davis/<stem>.mp4 \\
        --out-dir  . \\
        --clip-id  <id>

Outputs (paths relative to --out-dir):
    static/data/<clip-id>.json           (enriched, no pc_xyz/pc_colors)
    static/data/<clip-id>_pc.bin
    static/videos/<clip-id>.mp4          (trimmed, one frame per track index)
    static/videos/<clip-id>_chrono.jpg
"""
import argparse, json, subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


def grab_frame(mp4_path: Path, fi: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read video frame {fi} from {mp4_path}")
    return frame


def sample_color(img: np.ndarray, u: float, v: float, half: int = 3) -> list:
    H, W = img.shape[:2]
    cx = int(round(u * (W - 1)))
    cy = int(round(v * (H - 1)))
    x0 = max(0, cx - half); x1 = min(W, cx + half + 1)
    y0 = max(0, cy - half); y1 = min(H, cy + half + 1)
    patch = img[y0:y1, x0:x1].reshape(-1, 3)
    mean_bgr = patch.mean(axis=0)
    return [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]


def write_pc_binary(path: Path, xyz: np.ndarray, colors_u8: np.ndarray):
    N = int(xyz.shape[0])
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(colors_u8.astype(np.uint8).tobytes())


def build_object_stamps_chrono(bg, stamp_frames, stamp_pts2d,
                               dilate_px=2, edge_blur=3):
    """Composite per-stamp object cutouts (convex hull of tracked pts) onto bg."""
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
        cv2.fillConvexPoly(mask, hull, 255)
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * dilate_px + 1,) * 2)
            mask = cv2.dilate(mask, kernel)
        if edge_blur > 1:
            soft = cv2.GaussianBlur(mask, (edge_blur, edge_blur), 0).astype(np.float32) / 255.0
        else:
            soft = mask.astype(np.float32) / 255.0
        soft = soft[..., None]
        out = out * (1.0 - soft) + frame.astype(np.float32) * soft
    return out.clip(0, 255).astype(np.uint8)


def remap_indices(clip: dict, start_track: int):
    """Subtract start_track from every track-frame index so the trimmed mp4 is 0-based."""
    for cfg in clip["configs"]:
        cfg["hist_frames"]   = [f - start_track for f in cfg["hist_frames"]]
        cfg["future_frames"] = [f - start_track for f in cfg["future_frames"]]
        cfg["all_frames"]    = [f - start_track for f in cfg["all_frames"]]
        if "color_sample_frame" in cfg:
            cfg["color_sample_frame"] = cfg["color_sample_frame"] - start_track
    if "chrono" in clip:
        clip["chrono"]["frame_indices"] = [f - start_track for f in clip["chrono"]["frame_indices"]]
    new_total = clip["configs"][0]["all_frames"][-1] + 1
    clip["num_frames"] = new_total


def trim_subsample_mp4(src: Path, dst: Path, start_v: int, end_v: int,
                       stride: int, out_fps: float):
    """Select every `stride`-th frame in [start_v, end_v] from `src`; emit at out_fps."""
    if stride <= 1:
        vf = (f"select=between(n\\,{start_v}\\,{end_v}),"
              f"setpts=PTS-STARTPTS,fps={out_fps}")
    else:
        vf = (f"select='between(n\\,{start_v}\\,{end_v})*not(mod(n-{start_v}\\,{stride}))',"
              f"setpts=PTS-STARTPTS,fps={out_fps}")
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-i", str(src),
           "-vf", vf, "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
           str(dst)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-json", required=True, type=Path)
    ap.add_argument("--src-mp4",  required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--clip-id",  required=True)
    ap.add_argument("--n-stamps", type=int, default=4)
    ap.add_argument("--dilate-px", type=int, default=2)
    args = ap.parse_args()

    clip = json.loads(args.src_json.read_text())
    cfg0 = clip["configs"][0]
    af   = cfg0["all_frames"]
    clip_start_t = af[0]
    clip_end_t   = af[-1]
    n_clip = len(af)
    fps    = float(clip.get("fps", 15))
    mult   = int(clip.get("video_fps_mult", 1))

    out_data_dir  = args.out_dir / "static" / "data"
    out_video_dir = args.out_dir / "static" / "videos"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_video_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Sample per-tracked-point RGB at hist[0] (using video frame index) ──
    sampled = {}
    for cfg in clip["configs"]:
        hist0_t = cfg["hist_frames"][0]
        hist0_v = hist0_t * mult
        gt2d_0  = cfg["gt_2d"][0]
        if gt2d_0 is None:
            raise RuntimeError(f"config has no gt_2d: obj={cfg['obj_name']}")
        frame = grab_frame(args.src_mp4, hist0_v)
        colors = []
        for pt in gt2d_0:
            if pt is None or pt[0] is None or not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
                colors.append(None)
            else:
                colors.append(sample_color(frame, pt[0], pt[1]))
        cfg["pt_colors_rgb"] = colors
        cfg["color_sample_frame"] = hist0_t
        sampled[cfg.get("obj_name", "?")] = sum(1 for c in colors if c is not None)

    # ── 2. Object-stamp chrono: bg = LAST clip frame; stamps = earlier ──
    n_stamps = min(args.n_stamps, n_clip)
    local_idxs = [int(round(i)) for i in np.linspace(0, len(af) - 1, n_stamps)]
    seen = set(); dedup = []
    for k in local_idxs:
        if k not in seen: seen.add(k); dedup.append(k)
    local_idxs = dedup
    stamp_track_idxs = [af[k] for k in local_idxs]
    if stamp_track_idxs and stamp_track_idxs[-1] == clip_end_t:
        local_idxs = local_idxs[:-1]
        stamp_track_idxs = stamp_track_idxs[:-1]

    bg_v = clip_end_t * mult
    bg_frame = grab_frame(args.src_mp4, bg_v)
    stamp_v = [k * mult for k in stamp_track_idxs]
    stamp_frames = [grab_frame(args.src_mp4, fi) for fi in stamp_v]
    stamp_pts2d = [cfg0["gt_2d"][k] for k in local_idxs]

    chrono_img = build_object_stamps_chrono(
        bg=bg_frame, stamp_frames=stamp_frames, stamp_pts2d=stamp_pts2d,
        dilate_px=args.dilate_px, edge_blur=3,
    )
    chrono_path = out_video_dir / f"{args.clip_id}_chrono.jpg"
    cv2.imwrite(str(chrono_path), chrono_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    clip["chrono"] = {
        "image_url": f"static/videos/{args.clip_id}_chrono.jpg",
        "frame_indices": stamp_track_idxs + [clip_end_t],
        "mode": "object_stamps_on_last_frame_convex_hull",
        "dilate_px": args.dilate_px,
    }

    # ── 3. Trim + (optionally) subsample mp4 to one frame per track index ──
    mp4_dst = out_video_dir / f"{args.clip_id}.mp4"
    start_v = clip_start_t * mult
    end_v   = clip_end_t   * mult
    trim_subsample_mp4(args.src_mp4, mp4_dst, start_v, end_v, mult, fps)

    # ── 4. PC bin from pc_xyz/pc_colors (reuse motion5-viz's pre-baked PC) ──
    pc_xyz_list   = clip.pop("pc_xyz", None)
    pc_color_list = clip.pop("pc_colors", None)
    if pc_xyz_list is None or pc_color_list is None:
        raise RuntimeError("source JSON missing pc_xyz / pc_colors — this script "
                           "expects motion5-viz pre-baked PCs (e.g. davis/droid).")
    xyz = np.asarray(pc_xyz_list,   dtype=np.float32)
    col = np.asarray(pc_color_list, dtype=np.uint8)
    pc_path = out_data_dir / f"{args.clip_id}_pc.bin"
    write_pc_binary(pc_path, xyz, col)
    clip["pc_bin"] = {
        "url": f"static/data/{args.clip_id}_pc.bin",
        "n_points": int(xyz.shape[0]),
        "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
        "source": "pre-baked from motion5-viz pc_xyz/pc_colors",
    }

    # ── 5. Remap track indices → 0-based; output mp4 is now 1:1 with tracks ──
    remap_indices(clip, clip_start_t)
    clip["video_fps_mult"] = 1

    # ── 6. Write enriched JSON ──
    out_json = out_data_dir / f"{args.clip_id}.json"
    out_json.write_text(json.dumps(clip))

    print(f"wrote {out_json}    ({out_json.stat().st_size//1024} KB)")
    print(f"wrote {mp4_dst}     ({mp4_dst.stat().st_size//1024} KB, {n_clip} frames @ {fps:.1f} fps)")
    print(f"wrote {chrono_path}  ({chrono_path.stat().st_size//1024} KB, bg + {len(stamp_v)} stamps)")
    print(f"wrote {pc_path}      ({pc_path.stat().st_size//1024} KB, {xyz.shape[0]} PC points)")
    for obj, n in sampled.items():
        print(f"  sampled {n} colors for obj={obj}")


if __name__ == "__main__":
    main()
