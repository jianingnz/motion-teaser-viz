#!/usr/bin/env python3
"""Bake motion5-viz's hotworld prediction into an existing motion-teaser-viz
HOT3D bundle as an overlay trail.

Why displacement-based: motion5's hotworld JSON ships gt_3d / pred_3d for
80 sampled points in a slightly different coordinate frame from our
gt3d.bin (different upstream npz, no R_ROT90 applied). Trying to globally
rigid-align them leaves a residual that visibly offsets the trails from
the dense 2000-pt object cloud the viewer renders.

Approach instead:

  1. NN-match motion5's frame-0 points → our gt3d.bin frame-0 points.
     This gives 80 mtv indices, one per motion5 sample point.
  2. **GT trail** = our gt3d.bin's positions at those 80 mtv indices, across
     all bundle frames. By construction this trail IS a strict subset of
     the dense object cloud, so it tracks it exactly.
  3. **Pred trail** = anchor at the GT frame-0 position + R @ (motion5_pred[f]
     - motion5_pred[0]). The rotation R comes from a Kabsch fit between
     motion5 and mtv per-point displacements; t is implicit (anchor pinned
     to GT frame-0). This makes pred and GT coincide at frame 0 by
     construction and uses motion5's predicted relative motion thereafter.

Caveats:
  * Bundles produced this way no longer represent the literal motion5
    eval pred — the predicted positions are translated to our anchor.
    The shape and direction of the predicted trajectory is preserved.
  * If motion5's NN-match to mtv has poor coverage (low pair count or
    high pair distance), the pred shape may not perfectly mirror what
    motion5 displays. This is a one-shot demo overlay, not a metric eval.

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
    """Decode gt3d v2 binary → (xyz (F,N,3) with NaN for unannotated,
    vis (F,N) bool). Mirrors the JS loader."""
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


def kabsch_R_only(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve for R such that displacements (B - B0) ≈ R @ (A - A0). Returns
    just R (3x3); translation is irrelevant since we anchor explicitly."""
    ca = A.mean(0); cb = B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion5-json", required=True, type=Path)
    ap.add_argument("--bundle-json", required=True, type=Path,
                    help="motion-teaser-viz HOT3D bundle JSON to update in place.")
    ap.add_argument("--max-nn-dist", type=float, default=0.5,
                    help="NN-pair distance cutoff (m) when matching motion5 → mtv at frame 0.")
    args = ap.parse_args()

    m5 = json.loads(args.motion5_json.read_text())
    cfg5 = m5["configs"][0]
    m5_gt = np.asarray(cfg5["gt_3d"], dtype=np.float32)        # (T_m5, N_m5, 3)
    m5_pred = np.asarray(cfg5["pred_3d"], dtype=np.float32)    # (T_m5, N_m5, 3)
    m5_vis = np.asarray(cfg5["vis"], dtype=bool)
    m5_hist_frames = cfg5["hist_frames"]
    m5_future_frames = cfg5["future_frames"]
    m5_n_hist = int(cfg5["n_hist"])
    T_m5, N_m5, _ = m5_gt.shape
    print(f"motion5: T={T_m5}  N={N_m5}  hist={m5_hist_frames}  "
          f"future_count={len(m5_future_frames)}")

    bundle = json.loads(args.bundle_json.read_text())
    bundle_dir = args.bundle_json.parent
    gt3d_url = bundle["gt3d_bin"]["url"]
    gt3d_path = (bundle_dir.parent.parent / gt3d_url) if not Path(gt3d_url).is_absolute() else Path(gt3d_url)
    if not gt3d_path.exists():
        gt3d_path = (args.bundle_json.parent / Path(gt3d_url).name).resolve()
    print(f"reading gt3d.bin: {gt3d_path}")
    mtv_xyz, mtv_vis = decode_gt3d_bin(gt3d_path)
    F_mtv = mtv_xyz.shape[0]
    print(f"mtv gt3d: F={F_mtv}  N={mtv_xyz.shape[1]}")

    # ── Step 1: match motion5 frame-0 → mtv frame-0 NN. Drop any motion5
    #            point that doesn't have a close mtv neighbour. ──
    mtv0 = mtv_xyz[0]
    mtv0_valid = mtv_vis[0] & np.isfinite(mtv0).all(axis=1)
    tree = cKDTree(mtv0[mtv0_valid])
    mtv0_orig_idx = np.where(mtv0_valid)[0]   # map filtered-tree-idx → original mtv idx
    m5_0 = m5_gt[0]
    m5_0_valid = (m5_vis[0] if m5_vis.shape[0] > 0 else np.ones(N_m5, dtype=bool)) \
                 & np.isfinite(m5_0).all(axis=1)
    dists = np.full(N_m5, np.inf, dtype=np.float32)
    mtv_idx = np.full(N_m5, -1, dtype=np.int64)
    if m5_0_valid.any():
        d, ii = tree.query(m5_0[m5_0_valid], k=1)
        dists[m5_0_valid] = d
        mtv_idx[m5_0_valid] = mtv0_orig_idx[ii]
    keep = (dists < args.max_nn_dist) & (mtv_idx >= 0)
    n_keep = int(keep.sum())
    print(f"frame-0 NN match: {n_keep}/{N_m5} motion5 pts paired (median dist "
          f"{np.median(dists[keep])*1000:.2f}mm)")

    # ── Step 2: GT trail = mtv positions at the matched mtv indices,
    #            across all min(T_m5, F_mtv) frames. ──
    F_use = min(T_m5, F_mtv)
    kept_idx = np.where(keep)[0]              # motion5-point indices we keep
    kept_mtv = mtv_idx[keep]                  # corresponding mtv indices
    gt_trail = np.full((F_use, n_keep, 3), np.nan, dtype=np.float32)
    for fi in range(F_use):
        gt_trail[fi] = mtv_xyz[fi, kept_mtv, :]
    # Mark invisible frames (NaN) explicitly so the JS renderer skips.
    vis_trail = mtv_vis[:F_use, kept_mtv]

    # ── Step 3: pred trail. Solve R from frame-0 displacements between
    #            motion5 (relative to motion5_pred[0]) and mtv (relative to
    #            mtv[0]) at the kept points. Since pred[0]==gt[0], we can
    #            just use motion5_gt at later frames or motion5_pred — for
    #            R fitting we use motion5_gt frame 1's displacement, since
    #            it should match mtv frame 1's displacement. ──
    # Stack displacements from all hist+early-future frames where both have data
    A_disp = []
    B_disp = []
    A0 = m5_gt[0, kept_idx]                   # motion5 anchor
    B0 = mtv_xyz[0, kept_mtv]                 # mtv anchor
    for fi in range(1, F_use):
        a_disp = m5_gt[fi, kept_idx] - A0
        b_disp = mtv_xyz[fi, kept_mtv] - B0
        valid = (np.isfinite(a_disp).all(axis=1)
                 & np.isfinite(b_disp).all(axis=1))
        if valid.any():
            A_disp.append(a_disp[valid])
            B_disp.append(b_disp[valid])
    if A_disp:
        A_disp = np.concatenate(A_disp, axis=0)
        B_disp = np.concatenate(B_disp, axis=0)
        # Solve for R via Kabsch (no centering — these are displacement vectors)
        H = A_disp.T @ B_disp
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        A_x = A_disp @ R.T
        err = np.linalg.norm(A_x - B_disp, axis=1)
        print(f"motion5→mtv displacement R fit: {len(A_disp)} pairs  "
              f"median resid={np.median(err)*1000:.2f}mm  "
              f"max={err.max()*1000:.2f}mm")
    else:
        print("not enough displacement pairs — using identity R.")
        R = np.eye(3, dtype=np.float32)

    # Build pred trail: anchor at mtv frame 0, displacement from motion5_pred.
    pred_trail = np.full((F_use, n_keep, 3), np.nan, dtype=np.float32)
    A0_kept = m5_pred[0, kept_idx]            # = m5_gt[0] at hist frames
    for fi in range(F_use):
        # Motion5 predicted displacement from motion5's anchor at this frame.
        m5_disp = m5_pred[fi, kept_idx] - A0_kept     # (n_keep, 3)
        # Rotate into mtv's frame; add mtv anchor.
        pred_trail[fi] = (m5_disp @ R.T) + B0
        # Drop frames where any input was NaN.
        bad = ~np.isfinite(m5_pred[fi, kept_idx]).all(axis=1)
        pred_trail[fi, bad] = np.nan

    # ── Step 4: write back ──
    bundle["m5_trails"] = {
        "gt_3d":   np.round(gt_trail, 5).tolist(),
        "pred_3d": np.round(pred_trail, 5).tolist(),
        "vis":     vis_trail.tolist(),
        "n_hist":  m5_n_hist,
        "hist_frames":   m5_hist_frames,
        "future_frames": m5_future_frames,
        "obj_name": cfg5.get("obj_name", ""),
        "n_pts": int(n_keep),
        "alignment": "frame0_anchor_with_displacement_R",
        "source_motion5_json": str(args.motion5_json),
    }
    # Make pred trail visible by default — that's the comparison we want.
    bundle.setdefault("viewer_defaults", {})
    bundle["viewer_defaults"]["showPred"] = True

    args.bundle_json.write_text(json.dumps(bundle))
    print(f"wrote {args.bundle_json} ({args.bundle_json.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
