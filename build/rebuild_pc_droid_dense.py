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
    ap.add_argument("--max-points", type=int, default=70000,
                    help="Random-cap output to at most this many points (0 = no cap).")
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

    # Optional uniform random cap
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
