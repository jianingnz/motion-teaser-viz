#!/usr/bin/env python3
"""
Re-bake the DROID `_pc.bin` for a clip at full per-pixel density.

motion5-viz's preprocess uses `subsample=2` plus a trajectory-bbox crop and a
30k random cap. For DROID that lands at ~7k points — much sparser than
EgoDex's ~45k, which is what made the rendered DROID PC look unstructured.

This script:
  * Loads the camframe.npz to get the at-depth-resolution intrinsics (3×3).
  * Loads depth at the same reference frame motion5-viz used (track-fps
    `hist_frames[0]` of the source clip JSON, indexed straight into the
    video-fps depth h5 — preserving the original frame the JSON's PC was
    built at, so tracks and PC stay aligned).
  * Back-projects EVERY valid depth pixel (no stride, no bbox crop) into the
    DROID camera frame (c2w = identity).
  * Samples RGB from the source mp4 at the same video-fps frame.
  * Optionally caps at MAX_POINTS via uniform random subsampling.
  * Overwrites `static/data/<clip-id>_pc.bin` and patches the served JSON's
    `pc_bin.n_points` field accordingly.

The clip-id maps back to its DROID UUID by parsing the prefix of the source
clip filename: `AUTOLab_<hex>_<timestamp>_<cam>_..._t<N>` →
`AUTOLab+<hex>+<timestamp>` (file separator) and `<cam>` is the camera
serial. Pass --uuid / --cam explicitly if your clip-id doesn't follow that
pattern.
"""
import argparse, json
from pathlib import Path

import cv2
import h5py
import numpy as np


DROID_ROOT = Path("/weka/oe-training-default/chenhaoz/droid_pointworld/droid_all")


def parse_droid_id(clip_id_or_src_stem: str):
    """Try to recover (uuid, cam) from a clip-id like
    `AUTOLab_<hex>_<timestamp>_<cam>_..._t<N>`."""
    parts = clip_id_or_src_stem.split("_")
    # Expected: ["AUTOLab", hex, timestamp, cam, ...]
    if len(parts) >= 4 and parts[0] == "AUTOLab":
        uuid_pieces = [parts[0], parts[1], parts[2]]
        uuid = "+".join(uuid_pieces)  # filesystem separator is `+`
        cam = parts[3]
        return uuid, cam
    raise RuntimeError(f"can't parse uuid/cam from id: {clip_id_or_src_stem}")


def write_pc_bin(path: Path, xyz: np.ndarray, colors_u8: np.ndarray):
    N = int(xyz.shape[0])
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(colors_u8.astype(np.uint8).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-json", required=True, type=Path,
                    help="The motion5-viz source clip JSON (we read hist_frames[0] from here).")
    ap.add_argument("--src-mp4",  required=True, type=Path,
                    help="The DROID source mp4 (60fps native) — used to sample RGB.")
    ap.add_argument("--out-dir",  required=True, type=Path,
                    help="Output dir (motion-teaser-viz root).")
    ap.add_argument("--clip-id",  required=True,
                    help="Output clip id; matches the existing bin/JSON files in out-dir.")
    ap.add_argument("--uuid", default=None,
                    help="DROID file uuid (with `+` separators). If omitted we parse from --clip-id.")
    ap.add_argument("--cam",  default=None,
                    help="DROID camera serial, e.g. 24400334. If omitted we parse from --clip-id.")
    ap.add_argument("--subsample", type=int, default=1,
                    help="Pixel stride when back-projecting (1 = every pixel).")
    ap.add_argument("--max-points", type=int, default=0,
                    help="Random subsample cap. 0 (default) keeps the full strided grid intact "
                         "so the result is a STRUCTURED uniform sample.")
    ap.add_argument("--track-densify-radius", type=int, default=0,
                    help="Pixel radius around every GT 2D-track (across ALL frames) inside which a "
                         "stride-1 pass adds EXTRA points to emphasise the robotic arm / "
                         "manipulated-object region. 0 disables.")
    ap.add_argument("--track-densify-stride", type=int, default=1,
                    help="Stride used INSIDE the track-densify radius (default 1 = every pixel).")
    args = ap.parse_args()

    if args.uuid is None or args.cam is None:
        # Parse from the SOURCE filename, not the output id (output id may
        # have been shortened). Default to the source-json stem.
        base = args.src_json.stem  # AUTOLab_<hex>_<ts>_<cam>_object_t27
        u, c = parse_droid_id(base)
        uuid = args.uuid or u
        cam  = args.cam  or c
    else:
        uuid, cam = args.uuid, args.cam

    h5_path     = DROID_ROOT / uuid / "depth" / f"{uuid}_depth.h5"
    camframe    = DROID_ROOT / uuid / f"{cam}_smoothed_camframe.npz"
    if not h5_path.exists():   raise RuntimeError(f"missing depth h5: {h5_path}")
    if not camframe.exists():  raise RuntimeError(f"missing camframe: {camframe}")

    src = json.loads(args.src_json.read_text())
    cfg0 = src["configs"][0]
    ref_frame_track = int(cfg0["hist_frames"][0])  # track-fps index, same as motion5-viz used
    mult = int(src.get("video_fps_mult", 1))
    ref_frame_video = ref_frame_track * mult

    # Intrinsics at depth resolution
    cf = np.load(camframe)
    K = cf["intrinsics"].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Depth at the reference frame
    with h5py.File(h5_path, "r") as f:
        ds = f[f"{cam}+ext/depth"]
        # motion5-viz indexes the depth h5 directly with the track-fps idx
        # (DROID is largely stationary, so this is a deliberate simplification
        # of the canonical pipeline). We mirror that behaviour to keep tracks
        # and PC aligned with the existing JSON.
        depth_uint16 = ds[ref_frame_track]
    H, W = depth_uint16.shape
    depth_m = depth_uint16.astype(np.float32) / 1000.0  # mm → metres

    s = max(1, int(args.subsample))
    us = np.arange(0, W, s, dtype=np.int32)
    vs = np.arange(0, H, s, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    z = depth_m[vv, uu]
    valid = (z > 0.05) & np.isfinite(z) & (z < 20.0)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    xc = (uu.astype(np.float32) - cx) / fx * z
    yc = (vv.astype(np.float32) - cy) / fy * z
    xyz = np.stack([xc, yc, z], axis=1).astype(np.float32)  # camera frame

    # RGB at the same frame from the source mp4 (60fps native, multiply for video idx)
    cap = cv2.VideoCapture(str(args.src_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.src_mp4}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_video)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {ref_frame_video} from {args.src_mp4}")
    if frame_bgr.shape[:2] != (H, W):
        frame_bgr = cv2.resize(frame_bgr, (W, H))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    colors = rgb[vv, uu].astype(np.uint8)

    # ── Optional track-proximity densification (arm / manipulated-object emphasis) ──
    # When --track-densify-radius > 0, sweep every GT 2D-track in the source
    # clip JSON (across ALL frames + ALL configs), upscale to depth-resolution
    # pixels, and add a SECOND, finer-stride pass over depth pixels within
    # `radius` of any track pixel. Produces a denser PC blob exactly where
    # the manipulated object + arm tip live, without needing a segmentation
    # network.
    if args.track_densify_radius > 0:
        # Source JSON tracks are normalised to [0, 1] in (u, v); depth is
        # at (W, H). Collect every visible track pixel (any frame, any config).
        track_uv_norm = []
        for cfg in src.get("configs", []):
            for fr in cfg.get("gt_2d", []) or []:
                for pt in fr or []:
                    if pt is None: continue
                    u, v = pt[0], pt[1]
                    if u is None or v is None: continue
                    if not (0 <= u <= 1 and 0 <= v <= 1): continue
                    track_uv_norm.append((u, v))
        if track_uv_norm:
            tu = np.array([p[0] for p in track_uv_norm], dtype=np.float32) * (W - 1)
            tv = np.array([p[1] for p in track_uv_norm], dtype=np.float32) * (H - 1)
            # Track-proximity mask via 2D euclidean distance (broadcast).
            from scipy.spatial import cKDTree
            tree = cKDTree(np.stack([tu, tv], axis=-1))
            uu_full, vv_full = np.meshgrid(np.arange(0, W, max(1, args.track_densify_stride),
                                                     dtype=np.int32),
                                           np.arange(0, H, max(1, args.track_densify_stride),
                                                     dtype=np.int32))
            uu_full, vv_full = uu_full.ravel(), vv_full.ravel()
            d, _ = tree.query(np.stack([uu_full, vv_full], axis=-1).astype(np.float32), k=1)
            near = d < args.track_densify_radius
            uu2, vv2 = uu_full[near], vv_full[near]
            z2 = depth_m[vv2, uu2]
            valid2 = (z2 > 0.05) & np.isfinite(z2) & (z2 < 20.0)
            uu2, vv2, z2 = uu2[valid2], vv2[valid2], z2[valid2]
            if len(uu2) > 0:
                xc2 = (uu2.astype(np.float32) - cx) / fx * z2
                yc2 = (vv2.astype(np.float32) - cy) / fy * z2
                xyz_arm = np.stack([xc2, yc2, z2], axis=1).astype(np.float32)
                col_arm = rgb[vv2, uu2].astype(np.uint8)
                # Concatenate; downstream cap (if requested) trims the union.
                xyz    = np.concatenate([xyz, xyz_arm], axis=0)
                colors = np.concatenate([colors, col_arm], axis=0)
                print(f"  track-densify: +{len(xyz_arm)} pts within "
                      f"{args.track_densify_radius}px of {len(track_uv_norm)} GT tracks "
                      f"(stride={args.track_densify_stride})")

    # Optional uniform random cap (default 0 = keep full structured grid)
    if args.max_points > 0 and len(xyz) > args.max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(xyz), args.max_points, replace=False)
        xyz, colors = xyz[idx], colors[idx]

    # Write the bin (overwrites existing)
    bin_path = args.out_dir / "static" / "data" / f"{args.clip_id}_pc.bin"
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    write_pc_bin(bin_path, xyz, colors)

    # Patch the served JSON's pc_bin.n_points so the viewer header line is right
    json_path = args.out_dir / "static" / "data" / f"{args.clip_id}.json"
    if json_path.exists():
        served = json.loads(json_path.read_text())
        served.setdefault("pc_bin", {})["n_points"] = int(len(xyz))
        served["pc_bin"]["source"] = (f"densified by rebuild_pc_droid_dense.py "
                                      f"(stride={s}, max={args.max_points})")
        json_path.write_text(json.dumps(served))

    print(f"wrote {bin_path}  ({bin_path.stat().st_size//1024} KB, {len(xyz)} points)")
    print(f"  uuid={uuid} cam={cam} ref_frame_track={ref_frame_track} video={ref_frame_video}")
    print(f"  intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  depth {W}×{H}")


if __name__ == "__main__":
    main()
