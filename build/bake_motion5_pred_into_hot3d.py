#!/usr/bin/env python3
"""Bake motion5-viz's hotworld prediction into an existing motion-teaser-viz
HOT3D bundle as a dense rigid-body overlay.

Approach:

  1. NN-match motion5's frame-0 sampled points → our gt3d.bin frame-0
     query points to recover the per-sample mtv index (one mtv index per
     motion5 sample).
  2. Fit a global motion5→mtv displacement rotation R_disp using Kabsch on
     the (motion5, mtv) displacement pairs over all hist+future frames.
     This rotates motion5's coord frame into mtv's.
  3. For each frame f:
       a. Express motion5's predicted positions at frame f in mtv's frame:
            pred_mtv_at_kept[f] = mtv_anchor[kept] + R_disp @
                                  (motion5_pred[f][kept_m5] - motion5_anchor[kept_m5])
       b. Fit a per-frame rigid (R_f, t_f) in mtv's frame that maps
          mtv_anchor[kept] → pred_mtv_at_kept[f]. (Kabsch.)
       c. Apply (R_f, t_f) to **all 2000 mtv query points'** anchor
          positions → dense pred trail covering the whole object.
  4. GT trail is just mtv_xyz across all frames (literal subset of the
     dense object cloud — tracks it exactly).
  5. Both trails coincide at frame 0 by construction (R_0 = I, t_0 = 0).

The bundle is updated with a dense `m5_trails` block: gt_3d (F, 2000, 3),
pred_3d (F, 2000, 3), plus visibility from gt3d.bin and frame metadata.
The viewer uses cfg._goodIdx (the picker's track selection) to choose
which subset of the 2000 trails to actually render.

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
    return xyz, valid


def kabsch(A: np.ndarray, B: np.ndarray):
    """Solve B ≈ A @ R.T + t via Kabsch. Returns R, t, residuals (m)."""
    ca = A.mean(0); cb = B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    A_x = A @ R.T + t
    return R, t, np.linalg.norm(A_x - B, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion5-json", required=True, type=Path)
    ap.add_argument("--bundle-json", required=True, type=Path)
    ap.add_argument("--max-nn-dist", type=float, default=0.5,
                    help="NN cutoff (m) when matching motion5 → mtv at frame 0.")
    args = ap.parse_args()

    m5 = json.loads(args.motion5_json.read_text())
    cfg5 = m5["configs"][0]
    m5_gt = np.asarray(cfg5["gt_3d"], dtype=np.float32)
    m5_pred = np.asarray(cfg5["pred_3d"], dtype=np.float32)
    m5_vis = np.asarray(cfg5["vis"], dtype=bool)
    m5_hist = cfg5["hist_frames"]
    m5_future = cfg5["future_frames"]
    m5_n_hist = int(cfg5["n_hist"])
    T_m5, N_m5, _ = m5_gt.shape
    print(f"motion5: T={T_m5}  N={N_m5}  hist={m5_hist}  fut={len(m5_future)}")

    bundle = json.loads(args.bundle_json.read_text())
    bundle_dir = args.bundle_json.parent
    gt3d_url = bundle["gt3d_bin"]["url"]
    gt3d_path = (bundle_dir.parent.parent / gt3d_url) if not Path(gt3d_url).is_absolute() else Path(gt3d_url)
    if not gt3d_path.exists():
        gt3d_path = (args.bundle_json.parent / Path(gt3d_url).name).resolve()
    mtv_xyz, mtv_vis = decode_gt3d_bin(gt3d_path)
    F_mtv, N_mtv, _ = mtv_xyz.shape
    print(f"mtv gt3d: F={F_mtv}  N={N_mtv}")

    # ── Step 1: motion5 → mtv NN at frame 0 ──
    mtv0 = mtv_xyz[0]
    mtv0_valid = mtv_vis[0] & np.isfinite(mtv0).all(axis=1)
    tree = cKDTree(mtv0[mtv0_valid])
    mtv0_orig = np.where(mtv0_valid)[0]
    m5_0 = m5_gt[0]
    m5_0_valid = (m5_vis[0] if m5_vis.shape[0] > 0 else np.ones(N_m5, dtype=bool)) \
                 & np.isfinite(m5_0).all(axis=1)
    dists = np.full(N_m5, np.inf, dtype=np.float32)
    mtv_idx = np.full(N_m5, -1, dtype=np.int64)
    if m5_0_valid.any():
        d, ii = tree.query(m5_0[m5_0_valid], k=1)
        dists[m5_0_valid] = d
        mtv_idx[m5_0_valid] = mtv0_orig[ii]
    keep = (dists < args.max_nn_dist) & (mtv_idx >= 0)
    n_keep = int(keep.sum())
    kept_m5 = np.where(keep)[0]
    kept_mtv = mtv_idx[keep]
    print(f"frame-0 NN match: {n_keep}/{N_m5}")

    # ── Step 2: motion5 → mtv displacement rotation R_disp ──
    A_disp = []
    B_disp = []
    F_use = min(T_m5, F_mtv)
    A0_m5 = m5_gt[0, kept_m5]
    B0_mtv = mtv_xyz[0, kept_mtv]
    for fi in range(1, F_use):
        a_disp = m5_gt[fi, kept_m5] - A0_m5
        b_disp = mtv_xyz[fi, kept_mtv] - B0_mtv
        valid = (np.isfinite(a_disp).all(axis=1)
                 & np.isfinite(b_disp).all(axis=1))
        if valid.any():
            A_disp.append(a_disp[valid])
            B_disp.append(b_disp[valid])
    A_disp = np.concatenate(A_disp, axis=0) if A_disp else np.zeros((0, 3))
    B_disp = np.concatenate(B_disp, axis=0) if B_disp else np.zeros((0, 3))
    if len(A_disp) >= 4:
        H = A_disp.T @ B_disp
        U, _, Vt = np.linalg.svd(H)
        R_disp = Vt.T @ U.T
        if np.linalg.det(R_disp) < 0:
            Vt[-1, :] *= -1
            R_disp = Vt.T @ U.T
        err = np.linalg.norm(A_disp @ R_disp.T - B_disp, axis=1)
        print(f"R_disp fit: {len(A_disp)} pairs  median resid={np.median(err)*1000:.2f}mm "
              f"max={err.max()*1000:.2f}mm")
    else:
        print("not enough displacement pairs — using identity R_disp.")
        R_disp = np.eye(3, dtype=np.float32)

    # ── Step 3: per-frame rigid (R_f, t_f) in mtv's frame, applied to ALL
    #            2000 mtv anchor points to get the dense pred trail. ──
    pred_trail = np.full((F_use, N_mtv, 3), np.nan, dtype=np.float32)
    pred_trail[0] = mtv_xyz[0]   # by construction, R_0=I, t_0=0
    mtv_anchor = mtv_xyz[0]      # (N_mtv, 3)
    mtv_anchor_kept = mtv_anchor[kept_mtv]
    for fi in range(1, F_use):
        m5_disp = m5_pred[fi, kept_m5] - A0_m5            # (n_keep, 3) motion5 frame
        mtv_disp = m5_disp @ R_disp.T                      # (n_keep, 3) mtv frame
        target_kept = mtv_anchor_kept + mtv_disp           # (n_keep, 3)
        # Need finite source + target
        finite = (np.isfinite(mtv_anchor_kept).all(axis=1)
                  & np.isfinite(target_kept).all(axis=1))
        if finite.sum() < 4:
            pred_trail[fi] = mtv_anchor       # not enough — freeze at anchor
            continue
        R_f, t_f, _ = kabsch(mtv_anchor_kept[finite], target_kept[finite])
        pred_trail[fi] = mtv_anchor @ R_f.T + t_f

    # ── Step 4: GT trail = mtv positions across frames (literal). ──
    gt_trail = mtv_xyz[:F_use].copy()
    vis_trail = mtv_vis[:F_use].copy()

    # ── Step 5: write back ──
    bundle["m5_trails"] = {
        "gt_3d":   np.round(gt_trail, 5).tolist(),
        "pred_3d": np.round(pred_trail, 5).tolist(),
        "vis":     vis_trail.tolist(),
        "n_hist":  m5_n_hist,
        "hist_frames":   m5_hist,
        "future_frames": m5_future,
        "obj_name": cfg5.get("obj_name", ""),
        "n_pts": int(N_mtv),
        "alignment": "per_frame_rigid_in_mtv_frame_from_motion5_displacements",
        "n_motion5_samples_used": int(n_keep),
        "source_motion5_json": str(args.motion5_json),
    }
    bundle.setdefault("viewer_defaults", {})
    bundle["viewer_defaults"]["showPred"] = True

    args.bundle_json.write_text(json.dumps(bundle))
    print(f"wrote {args.bundle_json} ({args.bundle_json.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
