#!/usr/bin/env python3
"""
Build a motion-teaser-viz bundle directly from canonical HD-EPIC artifacts.

For each clip (e.g. P05-20240427-145526-105) this writes:
  static/data/{clip}.json            — per-object configs (hist=first half, future=second half)
  static/data/{clip}_pc.bin          — dense PC from frame-0 depth backprojection
  static/videos/{clip}.mp4           — copied/transcoded RGB
  static/videos/{clip}_chrono.jpg    — object-stamps chronophoto

Inputs (paths fixed from /weka/prior-default/jianingz/home/dataset/README.md §2):
  base = /weka/prior-default/jianingz/home/project/_GenTraj/vipe
  base/final_tracks/{vid}_2d.npz       (dict of {obj: (T,N,2) px})
  base/final_tracks/{vid}_3d.npz       (dict of {obj: (N,T,3) world})
  base/vipe_results/pose/{vid}.npz     (T,4,4 c2w)
  base/vipe_results/intrinsics/{vid}.npz (T,4 fx fy cx cy)
  base/vipe_results/depth/{vid}.zip    (per-frame EXR Z)
  base/vipe_results/rgb/{vid}.mp4      (480p 15fps)
  base/final_tracks/{vid}_filter_meta.npz (P_original (P,T,3) — raw, mixed objects)
  base/track_output/{vid}/{vid}_merged.npz (raw 2D tracks — mixed)

The bundle JSON format mirrors `static/data/basic_pick_place_14851_t29.json`
so index.html needs zero plumbing changes for the per-object configs.
"""
import argparse, json, subprocess, tempfile, zipfile, os
from pathlib import Path

import cv2
import numpy as np

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

VIPE = Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe")


# ─────────── EXR depth → world PC ───────────
def load_exr_depth(raw_bytes: bytes) -> np.ndarray:
    import OpenEXR, Imath
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
        f.write(raw_bytes); fname = f.name
    try:
        exr = OpenEXR.InputFile(fname)
        dw  = exr.header()['dataWindow']
        W   = dw.max.x - dw.min.x + 1
        H   = dw.max.y - dw.min.y + 1
        buf = exr.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
        d   = np.frombuffer(buf, dtype=np.float32).reshape(H, W).copy()
        exr.close()
    finally:
        os.unlink(fname)
    return d


def backproject(depth, rgb, c2w, intr, subsample=3):
    H, W = depth.shape
    fx, fy, cx, cy = intr
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
    col = rgb[vv, uu].astype(np.uint8)
    return xyz, col


def write_pc_binary(path: Path, xyz, col):
    N = int(xyz.shape[0])
    with open(path, 'wb') as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(col.astype(np.uint8).tobytes())


# ─────────── frame helpers ───────────
def grab_frame(mp4_path, fi):
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, f = cap.read()
    cap.release()
    if not ok: raise RuntimeError(f"can't read frame {fi}")
    return f


def grab_frames(mp4_path, fis):
    cap = cv2.VideoCapture(str(mp4_path))
    out = []
    for fi in fis:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok: raise RuntimeError(f"can't read frame {fi}")
        out.append(f)
    cap.release()
    return out


def sample_color(img, u, v, half=3):
    H, W = img.shape[:2]
    cx, cy = int(round(u*(W-1))), int(round(v*(H-1)))
    x0=max(0,cx-half); x1=min(W,cx+half+1); y0=max(0,cy-half); y1=min(H,cy+half+1)
    patch = img[y0:y1, x0:x1].reshape(-1,3)
    m = patch.mean(axis=0)
    return [int(m[2]), int(m[1]), int(m[0])]  # BGR→RGB


def build_chrono(bg, stamp_frames, stamp_pts2d, dilate_px=2, edge_blur=3):
    H, W = bg.shape[:2]
    out = bg.astype(np.float32).copy()
    for frame, pts in zip(stamp_frames, stamp_pts2d):
        coords = []
        for pt in pts:
            if pt is None or pt[0] is None: continue
            u, v = pt[0], pt[1]
            if not (0 <= u <= 1 and 0 <= v <= 1): continue
            coords.append([int(round(u*(W-1))), int(round(v*(H-1)))])
        if len(coords) < 3: continue
        hull = cv2.convexHull(np.array(coords, dtype=np.int32))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1,)*2)
            mask = cv2.dilate(mask, k)
        soft = (cv2.GaussianBlur(mask, (edge_blur, edge_blur), 0).astype(np.float32)/255.0)[..., None] \
               if edge_blur > 1 else (mask.astype(np.float32)/255.0)[..., None]
        out = out * (1.0 - soft) + frame.astype(np.float32) * soft
    return out.clip(0, 255).astype(np.uint8)


def transcode_mp4(src: Path, dst: Path, fps: float):
    """Re-encode source mp4 to libx264 yuv420p so all browsers can play it."""
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-i", str(src),
           "-vf", f"fps={fps}",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
           str(dst)]
    subprocess.run(cmd, check=True)


# ─────────── main ───────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--caption", default="")
    ap.add_argument("--n-stamps", type=int, default=4)
    ap.add_argument("--pc-subsample", type=int, default=3)
    args = ap.parse_args()

    vid = args.vid
    out_data = args.out_dir / "static" / "data"
    out_video = args.out_dir / "static" / "videos"
    out_data.mkdir(parents=True, exist_ok=True)
    out_video.mkdir(parents=True, exist_ok=True)

    # Load raw artifacts
    d2 = np.load(VIPE/f"final_tracks/{vid}_2d.npz", allow_pickle=True)
    d3 = np.load(VIPE/f"final_tracks/{vid}_3d.npz", allow_pickle=True)
    poses = np.load(VIPE/f"vipe_results/pose/{vid}.npz")['data'].astype(np.float32)
    intrs = np.load(VIPE/f"vipe_results/intrinsics/{vid}.npz")['data'].astype(np.float32)
    rgb_path = VIPE/f"vipe_results/rgb/{vid}.mp4"
    depth_zip_path = VIPE/f"vipe_results/depth/{vid}.zip"

    tracks_dict = d2['tracks'][()] if d2['tracks'].dtype == object else {"object": d2['tracks']}
    vis_dict    = d2['visibility'][()] if d2['visibility'].dtype == object else {"object": d2['visibility']}
    pts3d_dict  = d3['points_3d'][()] if d3['points_3d'].dtype == object else {"object": d3['points_3d']}

    H_W = d2['dim'].astype(np.int64)
    H_raw, W_raw = int(H_W[0]), int(H_W[1])

    # Frame count from the smallest array (objects share T)
    T = next(iter(tracks_dict.values())).shape[0]
    fps = 15.0  # HD-EPIC vipe re-encodes to 15 fps

    # Build configs: one per object. hist = first half, future = second half.
    n_hist = T // 2
    if n_hist < 2: n_hist = 2
    if n_hist >= T: n_hist = T - 1
    hist_frames   = list(range(0, n_hist))
    future_frames = list(range(n_hist, T))
    all_frames    = list(range(T))

    # Sample point colors at frame 0 of clipped video
    f0 = grab_frame(rgb_path, 0)

    configs = []
    for obj, t2d in tracks_dict.items():
        # t2d (T, N, 2) px ; vis (T, N) ; p3d (N, T, 3)
        v2d_obj = vis_dict[obj]
        p3d_obj = pts3d_dict[obj]

        # gt_2d: list[T] of list[N] of [u,v] in [0,1] or None (when not visible)
        gt_2d = []
        for t in range(T):
            frame_pts = []
            for n in range(t2d.shape[1]):
                if not bool(v2d_obj[t, n]):
                    frame_pts.append(None); continue
                u = float(t2d[t, n, 0]) / max(W_raw, 1)
                v = float(t2d[t, n, 1]) / max(H_raw, 1)
                frame_pts.append([u, v])
            gt_2d.append(frame_pts)

        # gt_3d: list[T] of list[N] of [x,y,z] (world) or None when invalid.
        # p3d_obj is (N, T, 3) → transpose to (T, N, 3). Invalid values must
        # be encoded as `null` (None) so the viewer's `if (!pt) continue`
        # checks skip them; emitting [0,0,0] makes invalid points converge
        # to the world origin (typically the camera frame-0 position) and
        # produces the "rays from camera" artifact in the paper-figure view.
        p3d_t = np.transpose(p3d_obj, (1, 0, 2))
        gt_3d = []
        for t in range(T):
            frame_pts = []
            for n in range(p3d_t.shape[1]):
                x, y, z = float(p3d_t[t, n, 0]), float(p3d_t[t, n, 1]), float(p3d_t[t, n, 2])
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    frame_pts.append(None)
                else:
                    frame_pts.append([x, y, z])
            gt_3d.append(frame_pts)

        # vis (T, N) bool list
        vis_list = [[bool(v2d_obj[t, n]) for n in range(t2d.shape[1])] for t in range(T)]

        # Per-point colors at frame 0 (sampled from RGB at gt_2d[0])
        pt_colors_rgb = []
        for n in range(t2d.shape[1]):
            pt = gt_2d[0][n]
            if pt is None or pt[0] is None or not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
                pt_colors_rgb.append(None)
            else:
                pt_colors_rgb.append(sample_color(f0, pt[0], pt[1]))

        # No prediction → mirror gt as pred (viewer expects the field)
        configs.append({
            "gt_3d": gt_3d,
            "pred_3d": gt_3d,
            "vis": vis_list,
            "obj_name": obj,
            "t": 0,
            "hist_frames":   hist_frames,
            "future_frames": future_frames,
            "all_frames":    all_frames,
            "n_hist": n_hist,
            "l2": 0.0,
            "gt_2d": gt_2d,
            "pred_2d": gt_2d,
            "pt_colors_rgb": pt_colors_rgb,
            "color_sample_frame": 0,
        })

    # Chronophoto: bg = last frame; stamps = sampled earlier frames using FIRST object's gt_2d
    n_stamps = min(args.n_stamps, T)
    local_idxs = sorted(set(int(round(i)) for i in np.linspace(0, T-1, n_stamps)))
    if local_idxs and local_idxs[-1] == T-1:
        local_idxs = local_idxs[:-1]
    bg_frame = grab_frame(rgb_path, T-1)
    stamp_frames = grab_frames(rgb_path, local_idxs)
    stamp_pts2d = [configs[0]["gt_2d"][k] for k in local_idxs]
    chrono_img = build_chrono(bg_frame, stamp_frames, stamp_pts2d, dilate_px=2, edge_blur=3)
    chrono_path = out_video / f"{vid}_chrono.jpg"
    cv2.imwrite(str(chrono_path), chrono_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # PC: depth backprojection at frame 0
    with zipfile.ZipFile(depth_zip_path) as zf:
        names = sorted(zf.namelist())
        depth0 = load_exr_depth(zf.read(names[0]))
    f0_rgb = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
    xyz, col = backproject(depth0, f0_rgb, poses[0], intrs[0], subsample=args.pc_subsample)
    pc_path = out_data / f"{vid}_pc.bin"
    write_pc_binary(pc_path, xyz, col)

    # mp4: transcode to ensure libx264 yuv420p
    mp4_dst = out_video / f"{vid}.mp4"
    transcode_mp4(rgb_path, mp4_dst, fps=fps)

    # Bundle
    bundle = {
        "configs": configs,
        "mse": 0.0,
        "l2": 0.0,
        "n_configs": len(configs),
        "num_frames": T,
        "fps": fps,
        "video_fps_mult": 1,
        "caption": args.caption,
        "raw_meta": {
            "n_points": int(sum(c["gt_3d"][0].__len__() for c in configs)),
            "n_frames": T,
            "video_dim_HW": [H_raw, W_raw],
            "src_clip_start": 0,
            "src_clip_end": T-1,
            "source_2d": str(VIPE/f"final_tracks/{vid}_2d.npz"),
            "source_3d": str(VIPE/f"final_tracks/{vid}_3d.npz"),
            "note": "HD-EPIC: per-object dict tracks, configs[i] = i-th object",
        },
        "chrono": {
            "image_url": f"static/videos/{vid}_chrono.jpg",
            "frame_indices": local_idxs + [T-1],
            "mode": "object_stamps_on_last_frame_convex_hull",
            "dilate_px": 2,
        },
        "pc_bin": {
            "url": f"static/data/{vid}_pc.bin",
            "n_points": int(xyz.shape[0]),
            "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
            "n_concat_frames": 1,
            "subsample": args.pc_subsample,
            "frame_indices_original": [0],
        },
        "camera": {
            "c2w_frame0": poses[0].tolist(),
            "intrinsics_frame0": [float(intrs[0,0]), float(intrs[0,1]),
                                  float(intrs[0,2]), float(intrs[0,3])],
            "video_stem": vid,
            "video_dim_HW": [H_raw, W_raw],
            # FULL per-frame trajectory so the viewer can render a moving frustum
            "c2w_per_frame": poses.tolist(),
            "intrinsics_per_frame": intrs.tolist(),
        },
    }

    out_json = out_data / f"{vid}.json"
    out_json.write_text(json.dumps(bundle))

    print(f"wrote {out_json}    ({out_json.stat().st_size//1024} KB)")
    print(f"wrote {mp4_dst}     ({mp4_dst.stat().st_size//1024} KB, {T} frames @ {fps} fps)")
    print(f"wrote {chrono_path}")
    print(f"wrote {pc_path}      ({xyz.shape[0]} pts)")
    print(f"  configs: {[c['obj_name'] for c in configs]}")


if __name__ == "__main__":
    main()
