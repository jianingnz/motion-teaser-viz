#!/usr/bin/env python3
"""Adapt the HOT3D cross-clip example at
   /weka/prior-default/jianingz/home/dataset/hot3d_repo/visual_example/
into a motion-teaser-viz bundle.

Source format (compact binaries, one shot — stitched 1995[84..149] + 1996[0..39]):
  data/meta.json         intrinsics, frame ranges, MoGe a/b
  data/scene.bin         (Nscene, 6) float32: XYZ + RGB-normalized [0,1]   — anchor-frame coords
  data/obj.bin           (F, K, 3) float32  : per-frame anchor-frame XYZ
  data/vis.bin           (F, K)    uint8    : per-frame visibility
  data/obj_color.bin     (K, 3)    float32  : per-track RGB-normalized (anchor-frame sampled)
  data/cam.bin           (F, 3)    float32  : per-frame camera centre in anchor-frame coords
  data/min_pix_dist.bin  (Nscene,) float32  : px-distance to nearest 2D-track (unused here)

This converter writes:
  static/data/{name}.json      single-config bundle, gt_3d/pred_3d in anchor frame
  static/data/{name}_pc.bin    scene PC repacked into our binary format
  static/videos/{name}.mp4     stitched + downsampled mp4 from the two source clips
  static/videos/{name}_chrono.jpg

Limitations vs HD-EPIC/EgoDex bundles:
  * No 2D tracks for non-anchor frames (only frame 0 has gt_2d projected from
    anchor-frame intrinsics; other frames are null). The 2D RGB-overlay panel
    will look sparse outside frame 0.
  * Camera frustum uses cam.bin positions + a synthesized forward direction
    (per-frame velocity, or "look at mean object" at the boundaries). No full
    per-frame rotation is shipped in the source bundle.
"""
import argparse, json, struct, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SRC = Path("/weka/prior-default/jianingz/home/dataset/hot3d_repo/visual_example/data")
RGB_TPL = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"


def load_source():
    meta = json.load(open(SRC / "meta.json"))
    F = meta["obj"]["n_frames"]
    K = meta["obj"]["n_pts"]
    Nscene = meta["scene"]["n_pts"]

    scene = np.fromfile(SRC / "scene.bin", dtype=np.float32).reshape(-1, 6)
    obj   = np.fromfile(SRC / "obj.bin",   dtype=np.float32).reshape(F, K, 3)
    vis   = np.fromfile(SRC / "vis.bin",   dtype=np.uint8 ).reshape(F, K).astype(bool)
    color = np.fromfile(SRC / "obj_color.bin", dtype=np.float32).reshape(K, 3)
    cam   = np.fromfile(SRC / "cam.bin",   dtype=np.float32).reshape(F, 3)
    assert scene.shape[0] == Nscene, (scene.shape, Nscene)
    return meta, scene, obj, vis, color, cam


def write_pc_binary(path: Path, xyz: np.ndarray, rgb_norm: np.ndarray):
    """uint32 N | float32 N*3 xyz | uint8 N*3 rgb (matches loadPCBinary in index.html)."""
    N = int(xyz.shape[0])
    rgb_u8 = np.clip(rgb_norm * 255.0, 0, 255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.tobytes())


def stitch_mp4(clip_a, frames_a, clip_b, frames_b, out_path: Path,
               target_height=480, fps=15):
    """Concatenate clip_a[frames_a[0]..frames_a[1]] + clip_b[frames_b[0]..frames_b[1]],
    downsampled to target_height (square crop preserved), libx264 yuv420p so the
    browser plays it. The viewer will see one continuous 106-frame video."""
    pa, pb = RGB_TPL.format(clip=clip_a), RGB_TPL.format(clip=clip_b)
    # Two passes with -filter_complex; trims by frame-index then concats.
    a0, a1 = frames_a; b0, b1 = frames_b
    fc = (
        f"[0:v]select='between(n\\,{a0}\\,{a1})',setpts=PTS-STARTPTS,"
        f"scale=-2:{target_height},fps={fps}[a];"
        f"[1:v]select='between(n\\,{b0}\\,{b1})',setpts=PTS-STARTPTS,"
        f"scale=-2:{target_height},fps={fps}[b];"
        f"[a][b]concat=n=2:v=1:a=0[v]"
    )
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-i", pa, "-i", pb,
           "-filter_complex", fc,
           "-map", "[v]",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
           str(out_path)]
    subprocess.run(cmd, check=True)


def build_c2w_per_frame(cam_pos, mean_obj_pos):
    """Synthesize a 4x4 c2w for each frame from positions only:
       * translation = cam_pos[t] (already in anchor coords)
       * forward (camera +Z in OpenCV / vipe convention) = direction the camera
         is "looking"; we estimate it as the unit vector from cam_pos[t] toward
         the mean object centroid at that frame. This produces a frustum that
         points at the action — consistent with what the viewer expects from
         camera.c2w_per_frame[t][:3, 2] (the third column).
       Rotation is built so the frustum's image-plane Y matches world-Y as
       closely as possible; the actual basis used by the frustum renderer
       comes from world-up cross products so a fully correct rotation is not
       required for visualization purposes.
    """
    F = cam_pos.shape[0]
    out = np.zeros((F, 4, 4), dtype=np.float32)
    out[:, 3, 3] = 1.0
    for t in range(F):
        fwd = mean_obj_pos[t] - cam_pos[t]
        n = np.linalg.norm(fwd)
        if n < 1e-6:
            fwd = np.array([0, 0, 1], dtype=np.float32)
        else:
            fwd = (fwd / n).astype(np.float32)
        wup = np.array([0, -1, 0], dtype=np.float32)   # display-frame up ≈ -Y
        right = np.cross(fwd, wup)
        if np.linalg.norm(right) < 1e-3:
            right = np.array([1, 0, 0], dtype=np.float32)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        out[t, :3, 0] = right
        out[t, :3, 1] = up
        out[t, :3, 2] = fwd
        out[t, :3, 3] = cam_pos[t]
    return out


def project_anchor_frame(obj_anchor, vis_anchor, fx, fy, cx, cy, W, H):
    """Project anchor-frame 3D points to anchor-image pixels.
    Inputs: obj_anchor (K,3) in anchor coords (camera origin), vis_anchor (K,) bool.
    Returns: list of [u/W, v/H] in [0,1] or None per point.
    """
    out = []
    for k in range(obj_anchor.shape[0]):
        if not vis_anchor[k]:
            out.append(None); continue
        x, y, z = float(obj_anchor[k, 0]), float(obj_anchor[k, 1]), float(obj_anchor[k, 2])
        if z <= 0.05 or not np.isfinite(z):
            out.append(None); continue
        u = (fx * x / z + cx) / W
        v = (fy * y / z + cy) / H
        if not (0 <= u <= 1 and 0 <= v <= 1):
            out.append(None); continue
        out.append([u, v])
    return out


def build_chrono_from_video(mp4_path: Path, n_stamps: int, gt_2d_anchor) -> np.ndarray:
    """Bg = last frame; stamps = sampled earlier frames using anchor gt_2d as
    the cutout mask. Same convex-hull-with-dilate approach as prepare_hdepic.py."""
    cap = cv2.VideoCapture(str(mp4_path))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, T - 1)
    ok, bg = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read last frame of {mp4_path}")
    H, W = bg.shape[:2]
    return bg, T - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--name", default="hot3d_clip1995_clip1996")
    ap.add_argument("--caption", default="HOT3D cross-clip stitch · clip-001995 → clip-001996")
    ap.add_argument("--obj-keep", type=int, default=256,
                    help="Subsample obj K from 2000 to this many to keep JSON modest.")
    ap.add_argument("--target-height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    out_data = args.out_dir / "static" / "data"
    out_video = args.out_dir / "static" / "videos"
    out_data.mkdir(parents=True, exist_ok=True)
    out_video.mkdir(parents=True, exist_ok=True)

    meta, scene, obj, vis, color, cam = load_source()
    F, K, _ = obj.shape
    fx, fy = meta["intrinsics"]["fx"], meta["intrinsics"]["fy"]
    cx, cy = meta["intrinsics"]["cx"], meta["intrinsics"]["cy"]
    W_src, H_src = meta["intrinsics"]["W"], meta["intrinsics"]["H"]

    # ── 1. Sub-sample obj from K=2000 to keep_K, keeping anchor-visible ones ──
    keep_K = min(args.obj_keep, K)
    visible_at_anchor = np.where(vis[0])[0]
    rng = np.random.default_rng(7)
    if len(visible_at_anchor) >= keep_K:
        sub_idx = np.sort(rng.choice(visible_at_anchor, keep_K, replace=False))
    else:
        sub_idx = np.sort(rng.choice(K, keep_K, replace=False))
    obj_sub  = obj[:, sub_idx, :]    # (F, keep_K, 3)
    vis_sub  = vis[:, sub_idx]       # (F, keep_K)
    color_sub = color[sub_idx]       # (keep_K, 3) in [0,1]
    print(f"obj keep: {keep_K}/{K}; anchor-visible pool was {len(visible_at_anchor)}")

    # ── 2. Build gt_3d / vis / pred_3d / pt_colors_rgb / gt_2d ──
    gt_3d = []
    for t in range(F):
        frame_pts = []
        for k in range(keep_K):
            if not vis_sub[t, k]:
                frame_pts.append(None); continue
            x, y, z = obj_sub[t, k]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                frame_pts.append(None); continue
            frame_pts.append([float(x), float(y), float(z)])
        gt_3d.append(frame_pts)

    pt_colors_rgb = [[int(c[0]*255), int(c[1]*255), int(c[2]*255)] for c in color_sub]

    # gt_2d: only the anchor frame has a meaningful projection (camera is at
    # origin in anchor coords). Other frames are null — we ship without per-
    # frame full poses, so 2D overlays past frame 0 would be guesswork.
    gt_2d_anchor = project_anchor_frame(obj_sub[0], vis_sub[0], fx, fy, cx, cy, W_src, H_src)
    gt_2d = [None] * F
    gt_2d[0] = gt_2d_anchor

    vis_list = [[bool(vis_sub[t, k]) for k in range(keep_K)] for t in range(F)]

    # ── 3. Camera per-frame c2w (positions from cam.bin + synthesized forward) ──
    mean_obj_pos = np.zeros((F, 3), dtype=np.float32)
    for t in range(F):
        m = vis_sub[t]
        mean_obj_pos[t] = obj_sub[t][m].mean(axis=0) if m.any() else cam[t]
    c2w_per_frame = build_c2w_per_frame(cam, mean_obj_pos)

    # ── 4. Stitch + downsample mp4 ──
    A0, A1 = meta["video"]["frames_A"]
    B0, B1 = meta["video"]["frames_B"]
    mp4_dst = out_video / f"{args.name}.mp4"
    stitch_mp4(meta["video"]["clip_A"], (A0, A1),
               meta["video"]["clip_B"], (B0, B1),
               mp4_dst, target_height=args.target_height, fps=args.fps)

    # ── 5. Chronophoto: bg = last frame of stitched mp4, stamps = sampled earlier
    #     frames cut by convex hull of anchor 2D track set (only anchor frame has
    #     2D, so the chrono is a 1-stamp object cutout over the bg). Skip stamps
    #     entirely — we don't have 2D for non-anchor frames — and just use the
    #     last frame straight as the chrono. The viewer is happy with that.
    cap = cv2.VideoCapture(str(mp4_dst))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    ok, bg = cap.read()
    cap.release()
    chrono_path = out_video / f"{args.name}_chrono.jpg"
    cv2.imwrite(str(chrono_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # ── 6. Scene PC binary ──
    pc_path = out_data / f"{args.name}_pc.bin"
    write_pc_binary(pc_path, scene[:, :3], scene[:, 3:6])

    # ── 7. Bundle JSON ──
    config = {
        "gt_3d":   gt_3d,
        "pred_3d": gt_3d,                  # no separate prediction — mirror GT
        "vis":     vis_list,
        "obj_name": "hot3d_object_cloud",
        "t":       0,
        "hist_frames":   list(range(0,  F // 2)),
        "future_frames": list(range(F // 2, F)),
        "all_frames":    list(range(F)),
        "n_hist": F // 2,
        "l2": 0.0,
        "gt_2d":   gt_2d,
        "pred_2d": gt_2d,
        "pt_colors_rgb":     pt_colors_rgb,
        "color_sample_frame": 0,
    }
    bundle = {
        "configs":  [config],
        "mse": 0.0, "l2": 0.0,
        "n_configs": 1,
        "num_frames": F,
        "fps": float(args.fps),
        "video_fps_mult": 1,
        "caption": args.caption,
        "raw_meta": {
            "n_points": keep_K,
            "n_frames": F,
            "video_dim_HW": [args.target_height, args.target_height],
            "src_clip_start": 0,
            "src_clip_end": F - 1,
            "source_2d": "(anchor-frame projection only)",
            "source_3d": str(SRC),
            "note": "HOT3D cross-clip example: anchor-frame coords, no per-frame full pose.",
        },
        "chrono": {
            "image_url": f"static/videos/{args.name}_chrono.jpg",
            "frame_indices": [F - 1],
            "mode": "last_frame_only",
            "dilate_px": 0,
        },
        "pc_bin": {
            "url": f"static/data/{args.name}_pc.bin",
            "n_points": int(scene.shape[0]),
            "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
            "n_concat_frames": 1,
            "subsample": 1,
            "frame_indices_original": [meta["anchor"]["frame"]],
        },
        "camera": {
            "c2w_frame0": c2w_per_frame[0].tolist(),
            "intrinsics_frame0": [float(fx), float(fy), float(cx), float(cy)],
            "video_stem": args.name,
            "video_dim_HW": [args.target_height, args.target_height],
            "c2w_per_frame": c2w_per_frame.tolist(),
            "intrinsics_per_frame": [[float(fx), float(fy), float(cx), float(cy)]] * F,
        },
    }
    out_json = out_data / f"{args.name}.json"
    out_json.write_text(json.dumps(bundle))

    print(f"wrote {out_json} ({out_json.stat().st_size//1024} KB)")
    print(f"wrote {mp4_dst}  ({mp4_dst.stat().st_size//1024} KB, {F} frames)")
    print(f"wrote {chrono_path}")
    print(f"wrote {pc_path}  ({scene.shape[0]} pts)")


if __name__ == "__main__":
    main()
