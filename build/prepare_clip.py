#!/usr/bin/env python3
"""
Take a source motion5 clip JSON + its mp4, sample per-point object RGB colors
from the first history frame, and write an enriched JSON + a copy of the mp4
into the new motion-teaser-viz site.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def sample_color(img: np.ndarray, u: float, v: float, half: int = 3) -> list:
    """Sample mean RGB color in a (2*half+1)x(2*half+1) window centered at (u,v).

    `img` is HxWx3 in BGR (OpenCV). `u,v` are normalized [0,1] image coords.
    Returns [R,G,B] ints in 0..255.
    """
    H, W = img.shape[:2]
    cx = int(round(u * (W - 1)))
    cy = int(round(v * (H - 1)))
    x0 = max(0, cx - half)
    x1 = min(W, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(H, cy + half + 1)
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
    """Read a set of frames from the same mp4 in one pass."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4_path}")
    frames = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {fi} from {mp4_path}")
        frames.append(frame)
    cap.release()
    return frames


def build_chronophoto(frames: list, mode: str = "min") -> np.ndarray:
    """Composite a list of BGR frames into a single chronophotography image.

    `mode='min'` keeps the darkest pixel across the stack at every position —
    great for highlighting moving foreground objects (a la Marey).
    `mode='mean'` simple alpha-blend averaging.
    `mode='max'` keeps the brightest — good for bright objects on dark bg.
    """
    stack = np.stack(frames, axis=0).astype(np.float32)  # (N,H,W,3)
    if mode == "min":
        out = stack.min(axis=0)
    elif mode == "max":
        out = stack.max(axis=0)
    elif mode == "mean":
        out = stack.mean(axis=0)
    else:
        raise ValueError(mode)
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-json", required=True, type=Path)
    ap.add_argument("--src-mp4", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="root of motion-teaser-viz site")
    ap.add_argument("--clip-id", required=True, help="short id used in output filenames")
    args = ap.parse_args()

    clip = json.loads(args.src_json.read_text())

    out_data_dir = args.out_dir / "static" / "data"
    out_video_dir = args.out_dir / "static" / "videos"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_video_dir.mkdir(parents=True, exist_ok=True)

    # --- Sample per-point colors from each config's first hist frame ---
    sampled = {}
    for cfg in clip["configs"]:
        hist0 = cfg["hist_frames"][0]
        gt2d_0 = cfg["gt_2d"][0] if cfg.get("gt_2d") else None
        if gt2d_0 is None:
            raise RuntimeError(f"config has no gt_2d, cannot sample colors: obj={cfg['obj_name']}")
        frame = grab_frame(args.src_mp4, hist0)
        colors = []
        for pt in gt2d_0:
            if pt is None or pt[0] is None or not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
                colors.append(None)
            else:
                colors.append(sample_color(frame, pt[0], pt[1]))
        cfg["pt_colors_rgb"] = colors  # length P, each [R,G,B] or null

        # also stash the sampled-frame index so the viewer can show it
        cfg["color_sample_frame"] = hist0
        sampled[cfg["obj_name"]] = sum(1 for c in colors if c is not None)

    # --- Build chronophotography image (frames spanning history+future) ---
    cfg0 = clip["configs"][0]
    af = cfg0["all_frames"]
    # pick ~16 frames evenly across the full history+future window
    n_chrono = min(16, len(af))
    chrono_idxs = [int(round(i)) for i in np.linspace(0, len(af) - 1, n_chrono)]
    chrono_frame_ids = [af[i] for i in chrono_idxs]
    chrono_bgrs = grab_frames(args.src_mp4, chrono_frame_ids)
    chrono_img = build_chronophoto(chrono_bgrs, mode="min")
    chrono_path = out_video_dir / f"{args.clip_id}_chrono.jpg"
    cv2.imwrite(str(chrono_path), chrono_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    clip["chrono"] = {
        "image_url": f"static/videos/{args.clip_id}_chrono.jpg",
        "frame_indices": chrono_frame_ids,
        "all_frames_indices": chrono_idxs,
        "mode": "min",
    }

    # --- Copy mp4 ---
    mp4_dst = out_video_dir / f"{args.clip_id}.mp4"
    shutil.copy(args.src_mp4, mp4_dst)

    # --- Write enriched JSON ---
    out_json = out_data_dir / f"{args.clip_id}.json"
    out_json.write_text(json.dumps(clip))

    print(f"wrote {out_json}  ({out_json.stat().st_size//1024} KB)")
    print(f"wrote {mp4_dst}    ({mp4_dst.stat().st_size//1024} KB)")
    print(f"wrote {chrono_path} ({chrono_path.stat().st_size//1024} KB, {n_chrono} frames composited)")
    for obj, n in sampled.items():
        print(f"  sampled {n} colors for obj={obj}")


if __name__ == "__main__":
    main()
