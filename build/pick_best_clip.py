#!/usr/bin/env python3
"""Score every egodex/test clip and pick the best teaser candidate."""

import json, math
from pathlib import Path
import numpy as np

MOTION5 = Path("/weka/prior-default/jianingz/home/visual/motion5-viz")
DATA_DIR = MOTION5 / "static" / "data" / "modeling_json" / "egodex" / "test"
MANIFEST = json.loads((MOTION5 / "static" / "data" / "manifest.json").read_text())

# Categories that tend to make clear teaser figures
GOOD_CATS = {
    "pour", "stack_unstack_bowls", "stack_unstack_cups", "stack_unstack_plates",
    "pick_place_food", "basic_pick_place", "vertical_pick_place",
    "scoop_dump_ice", "throw_and_catch_ball", "play_mancala",
    "open_close_insert_remove_tupperware", "add_remove_lid",
    "stack_unstack_tetra_board", "stack",
    "wipe_kitchen_surfaces", "wipe_screen",
}


def trajectory_motion(cfg):
    """Total per-point world-space displacement summed across all frames."""
    gt = cfg["gt_3d"]  # (F, P, 3)
    F = len(gt); P = len(gt[0]) if F else 0
    if F < 2 or P == 0: return 0.0
    arr = np.full((F, P, 3), np.nan, dtype=np.float32)
    for f in range(F):
        for p in range(P):
            pt = gt[f][p]
            if pt is not None and pt[0] is not None and math.isfinite(pt[0]):
                arr[f, p] = pt
    deltas = np.diff(arr, axis=0)            # (F-1, P, 3)
    mags = np.linalg.norm(deltas, axis=-1)   # (F-1, P)
    # mean motion per frame averaged over points (only valid)
    m = np.nanmean(mags) if np.any(np.isfinite(mags)) else 0.0
    return float(m if np.isfinite(m) else 0.0)


def hull_area_frac(cfg):
    """Fraction of unit-image area covered by the convex hull of frame-0 2D points."""
    g2 = cfg.get("gt_2d")
    if not g2 or not g2[0]: return 0.0
    pts = [p for p in g2[0] if p is not None and p[0] is not None and 0 <= p[0] <= 1 and 0 <= p[1] <= 1]
    if len(pts) < 3: return 0.0
    pts = np.array(pts, dtype=np.float32)
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)  # 'volume' for 2D hull = area
    except Exception:
        # shoelace fallback on raw point cloud min-area-rect
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        return float((x1 - x0) * (y1 - y0))


def main():
    rows = []
    for clip in MANIFEST["clips"]["egodex"]["test"]:
        if clip.get("mse_norm") is None: continue
        if not (0.012 < clip["mse_norm"] < 0.06): continue
        if clip["category"] not in GOOD_CATS: continue
        path = DATA_DIR / (clip["id"] + ".json")
        if not path.exists(): continue
        cd = json.loads(path.read_text())
        cfg = cd["configs"][0]
        n_pts = len(cfg["gt_3d"][0])
        if n_pts < 30: continue  # need enough pts for a clean hull
        motion = trajectory_motion(cfg)
        if motion < 0.005: continue  # too still
        area = hull_area_frac(cfg)
        if area < 0.005 or area > 0.35: continue  # not too tiny, not whole frame
        # Composite score: motion is most important; prefer mid mse_norm; prefer mid area
        score = (motion * 100) * (1.0 / max(0.03, clip["mse_norm"])) * min(1.0, area / 0.05)
        rows.append({
            "id": clip["id"],
            "cat": clip["category"],
            "mse_norm": clip["mse_norm"],
            "l2": clip["l2"],
            "n_pts": n_pts,
            "motion": motion,
            "area": area,
            "score": score,
            "caption": cd.get("caption", ""),
            "n_frames": len(cfg["all_frames"]),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    print(f"{'rank':>4}  {'cat':<28}  {'mse_n':>6}  {'mot':>6}  {'area':>5}  {'pts':>4}  {'F':>3}  caption / id")
    for i, r in enumerate(rows[:15]):
        print(f"  {i+1:>2}.  {r['cat'][:28]:<28}  {r['mse_norm']:.4f}  {r['motion']:.4f}  {r['area']:.3f}  {r['n_pts']:>4}  {r['n_frames']:>3}  {r['caption'][:35]:<35}  {r['id']}")


if __name__ == "__main__":
    main()
