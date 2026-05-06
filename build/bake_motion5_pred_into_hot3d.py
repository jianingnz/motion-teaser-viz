#!/usr/bin/env python3
"""Bake motion5-viz's hotworld prediction (and GT) trails into an existing
motion-teaser-viz HOT3D bundle.

Why: the user wants to compare motion5's model prediction against GT in our
HOT3D viewer, while keeping our existing visualization (MoGe-dense scene PC
+ 2000-pt object cloud from gt3d.bin). motion5's hotworld JSON stores 80
sampled points × (3 hist + 12 future) frames of GT and predicted 3D
positions, plus per-point RGB.

Coordinate-frame caveat: motion5's hot3d source coords come from
`dense_3d.npz` (un-rotated), while our `gt3d.bin` is in the anchor-display
frame produced by prepare_hot3d_singleclip.py (rotated by `R_ROT90`). They
differ by a rigid transform. We solve for R, t via Kabsch/Procrustes on
nearest-neighbor pairs (motion5 GT ↔ our gt3d.bin at the same frame), then
apply that transform to motion5's gt_3d and pred_3d before stuffing them
into the bundle.

Result: a new top-level field `m5_trails` on the bundle JSON containing the
transformed gt_3d/pred_3d arrays, vis, per-point RGB, and frame metadata.
The viewer reads this field and renders motion5's trails alongside the
existing object cloud.

Usage:
    python build/bake_motion5_pred_into_hot3d.py \
        --motion5-json tmp/motion5_hotworld/<id>_obj<N>_t2.json \
        --bundle-json  static/data/hot3d_<clip>_obj<N>_<window>.json
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def decode_gt3d_bin(path: Path):
    """Decode gt3d v2 binary → (xyz (F,N,3) float32 with NaN for unannotated,
    vis (F,N) bool, rgb (N,3) uint8). Mirrors the JS loader (loadGT3DBinary)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GT3D", f"bad magic: {magic!r}"
        version = struct.unpack("<I", f.read(4))[0]
        assert version == 2, f"unsupported version: {version}"
        N = struct.unpack("<I", f.read(4))[0]
        F = struct.unpack("<I", f.read(4))[0]
        nObj = struct.unpack("<I", f.read(4))[0]
        axisMin = np.frombuffer(f.read(12), dtype=np.float32).copy()
        axisMax = np.frombuffer(f.read(12), dtype=np.float32).copy()
        obj_ids = np.frombuffer(f.read(N), dtype=np.uint8).copy()
        rgb = np.frombuffer(f.read(N * 3), dtype=np.uint8).copy().reshape(N, 3)
        quant = np.frombuffer(
            f.read(F * N * 3 * 2), dtype=np.int16
        ).copy().reshape(F, N, 3)
        visBytes = np.frombuffer(f.read(F * N), dtype=np.uint8).copy().reshape(F, N)
    SENT = -32768
    K = 1.0 / 32767.0
    center = (axisMin + axisMax) / 2
    half = (axisMax - axisMin) / 2
    xyz = quant.astype(np.float32) * K * half + center
    valid = quant[..., 0] != SENT
    xyz[~valid] = np.nan
    return xyz, valid, rgb, obj_ids, axisMin, axisMax


def kabsch(A: np.ndarray, B: np.ndarray):
    """Solve B ≈ A @ R.T + t for rigid R (3,3), t (3,). Returns R, t,
    plus per-pair residual norms (m)."""
    ca = A.mean(0)
    cb = B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    A_x = A @ R.T + t
    err = np.linalg.norm(A_x - B, axis=1)
    return R, t, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion5-json", required=True, type=Path)
    ap.add_argument("--bundle-json", required=True, type=Path,
                    help="motion-teaser-viz HOT3D bundle JSON to update in place.")
    ap.add_argument("--max-nn-dist", type=float, default=0.5,
                    help="NN-pair filter (m). Drops pairs farther than this "
                         "before fitting the rigid transform.")
    args = ap.parse_args()

    m5 = json.loads(args.motion5_json.read_text())
    cfg5 = m5["configs"][0]
    m5_gt = np.asarray(cfg5["gt_3d"], dtype=np.float32)
    m5_pred = np.asarray(cfg5["pred_3d"], dtype=np.float32)
    m5_vis = np.asarray(cfg5["vis"], dtype=bool)
    # pt_colors_rgb is optional — motion5 hotworld JSONs don't ship it (the
    # site renders trails with a colormap gradient, not per-track RGB).
    m5_pt_colors = cfg5.get("pt_colors_rgb")
    m5_hist = cfg5["hist_frames"]
    m5_future = cfg5["future_frames"]
    m5_n_hist = int(cfg5["n_hist"])
    T_m5, N_m5 = m5_gt.shape[0], m5_gt.shape[1]
    print(f"motion5: T={T_m5}, N={N_m5}, hist={m5_hist}, future_count={len(m5_future)}")

    bundle = json.loads(args.bundle_json.read_text())
    bundle_dir = args.bundle_json.parent
    gt3d_url = bundle["gt3d_bin"]["url"]
    gt3d_path = bundle_dir.parent.parent / gt3d_url
    if not gt3d_path.exists():
        gt3d_path = Path(gt3d_url)
    if not gt3d_path.exists():
        gt3d_path = (args.bundle_json.parent / Path(gt3d_url).name).resolve()
    print(f"reading gt3d.bin from {gt3d_path}")
    mtv_xyz, mtv_vis, _, _, _, _ = decode_gt3d_bin(gt3d_path)
    F_mtv = mtv_xyz.shape[0]
    print(f"mtv gt3d: F={F_mtv}, N={mtv_xyz.shape[1]}")

    # Pair (motion5 GT) with (mtv gt3d at same frame, NN). Use only frames
    # both have (the first min(T_m5, F_mtv)).
    F_pair = min(T_m5, F_mtv)
    A_list = []  # motion5 GT positions
    B_list = []  # mtv gt3d positions (NN-matched)
    for fi in range(F_pair):
        mtv_f = mtv_xyz[fi]
        valid = mtv_vis[fi] & np.isfinite(mtv_f).all(axis=1)
        if not valid.any():
            continue
        tree = cKDTree(mtv_f[valid])
        m5_f = m5_gt[fi]
        m5_v = m5_vis[fi] if m5_vis.shape[0] > fi else np.ones(N_m5, dtype=bool)
        m5_keep = m5_v & np.isfinite(m5_f).all(axis=1)
        if not m5_keep.any():
            continue
        d, idx = tree.query(m5_f[m5_keep], k=1)
        keep = d < args.max_nn_dist
        A_list.append(m5_f[m5_keep][keep])
        B_list.append(mtv_f[valid][idx[keep]])
    A = np.concatenate(A_list, axis=0)
    B = np.concatenate(B_list, axis=0)
    print(f"NN pairs: {len(A)}")

    R, t, err = kabsch(A, B)
    print(f"rigid fit: median resid={np.median(err)*1000:.2f}mm  "
          f"mean={err.mean()*1000:.2f}mm  max={err.max()*1000:.2f}mm")

    def xform(arr):
        # arr shape (T, N, 3) → (T, N, 3) after applying R @ p + t.
        flat = arr.reshape(-1, 3)
        out = flat @ R.T + t
        return out.reshape(arr.shape).astype(np.float32)

    m5_gt_x = xform(m5_gt)
    m5_pred_x = xform(m5_pred)

    bundle["m5_trails"] = {
        "gt_3d":   np.round(m5_gt_x, 5).tolist(),
        "pred_3d": np.round(m5_pred_x, 5).tolist(),
        "vis":     m5_vis.tolist(),
        "pt_colors_rgb": m5_pt_colors,
        "n_hist":  m5_n_hist,
        "hist_frames":   m5_hist,
        "future_frames": m5_future,
        "obj_name": cfg5.get("obj_name", ""),
        "fit_residual_mm_median": round(float(np.median(err) * 1000), 2),
        "fit_residual_mm_max":    round(float(err.max() * 1000), 2),
        "n_pairs": int(len(A)),
        "source_motion5_json": str(args.motion5_json),
    }
    # Make pred trail visible by default — that's the comparison we want.
    bundle.setdefault("viewer_defaults", {})
    bundle["viewer_defaults"]["showPred"] = True

    args.bundle_json.write_text(json.dumps(bundle))
    print(f"wrote {args.bundle_json} ({args.bundle_json.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
