#!/usr/bin/env python3
"""
Render paper-quality static 3D PNG panels using matplotlib (the same library
visual.py uses). Inputs come from the enriched JSON written by prepare_clip.py:
- the dense binary PC (clip['pc_bin'].url + frame-0 c2w / intrinsics)
- gt_3d / pred_3d trajectories
- pt_colors_rgb per query point

Output: 4 PNGs per clip — `_history.png`, `_gt.png`, `_pred.png`, `_all.png`.

Following visual.py's PC look (`s=1`, `linewidths=0`, `depthshade=True`) and
floor-grid style (`Line3DCollection`, `linewidths=0.5, alpha=0.6`).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# Matching the three.js viewer
TRAIL_COLORS = {
    "hist": (59 / 255, 130 / 255, 246 / 255),   # vivid blue
    "gt":   (34 / 255, 197 / 255, 94 / 255),    # bright green
    "pred": (249 / 255, 115 / 255, 22 / 255),   # bright orange
}
POINT_SUBSAMPLE = 4  # match the three.js viewer


def load_pc_binary(path: Path):
    raw = path.read_bytes()
    N = int(np.frombuffer(raw[:4], dtype=np.uint32)[0])
    xyz = np.frombuffer(raw[4 : 4 + N * 12], dtype=np.float32).reshape(N, 3)
    rgb = np.frombuffer(raw[4 + N * 12 : 4 + N * 12 + N * 3], dtype=np.uint8).reshape(N, 3)
    return xyz.copy(), rgb.copy()


def ego_view_angles(c2w: np.ndarray, elev_down: float = 25.0):
    """matplotlib (azim, elev) aligned to the camera's facing direction at frame 0."""
    fwd_w = c2w[:3, 2]                      # camera +Z (OpenCV optical axis)
    fwd_p = fwd_w[[0, 2, 1]]                # plot-axis swap [X,Z,Y]
    horiz = np.array([-fwd_p[0], -fwd_p[1], 0.0])
    nrm = float(np.linalg.norm(horiz))
    if nrm < 1e-6:
        horiz = np.array([1.0, 0.0, 0.0])
    else:
        horiz /= nrm
    azim_vi = float(np.degrees(np.arctan2(horiz[1], horiz[0])))
    return -azim_vi, elev_down


def collect_trail_segments(pts3d_per_frame, p_indices, fStart, fEnd):
    """Return list of [[A], [B]] segments, world-space."""
    segs = []
    nF = len(pts3d_per_frame)
    for p in p_indices:
        for fi in range(fStart, min(fEnd, nF - 1)):
            a = pts3d_per_frame[fi][p]
            b = pts3d_per_frame[fi + 1][p]
            if a is None or b is None:
                continue
            if not (math.isfinite(a[0]) and math.isfinite(b[0])):
                continue
            segs.append([a, b])
    return segs


def render_panel(
    pc_xyz: np.ndarray,
    pc_rgb: np.ndarray,
    cfg: dict,
    pt_colors_rgb: list,
    c2w: np.ndarray,
    out_path: Path,
    kind: str,                        # 'hist' | 'gt' | 'pred' | 'all'
    azim: float, elev: float,
    img_w: int = 900, img_h: int = 600,
    pc_point_size: float = 0.8,
    grid_scale: float = 2.5,
    line_w: float = 2.4,
    dot_size: float = 26,
):
    """One paper-quality 3D PNG."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # plot coord swap [0,2,1]: world (X,Y,Z) → plot (X,Z,Y) — same as visual.py
    pc_plot = pc_xyz[:, [0, 2, 1]].astype(np.float32)

    # Frame on the TRAJECTORY (not the whole room) so the action fills the canvas.
    # We want the trajectory to occupy ~50% of the view and still get scene context.
    traj_pts = []
    for fr in cfg["gt_3d"]:
        for p in fr:
            if p is None or p[0] is None: continue
            if not math.isfinite(p[0]): continue
            traj_pts.append([p[0], p[2], p[1]])  # plot-space swap
    for fr in cfg["pred_3d"]:
        for p in fr:
            if p is None or p[0] is None: continue
            if not math.isfinite(p[0]): continue
            traj_pts.append([p[0], p[2], p[1]])
    traj_arr = np.array(traj_pts, dtype=np.float32)
    cx = float((traj_arr[:, 0].min() + traj_arr[:, 0].max()) / 2)
    cy = float((traj_arr[:, 1].min() + traj_arr[:, 1].max()) / 2)
    cz = float((traj_arr[:, 2].min() + traj_arr[:, 2].max()) / 2)
    traj_interval = float(max(
        traj_arr[:, 0].max() - traj_arr[:, 0].min(),
        traj_arr[:, 1].max() - traj_arr[:, 1].min(),
        traj_arr[:, 2].max() - traj_arr[:, 2].min(),
        0.15,
    ))
    half = traj_interval * 1.5     # 3× the trajectory extent → ~33% fill of view
    interval = half * 2
    x_lo, x_hi = cx - half, cx + half
    y_lo, y_hi = cy - half, cy + half
    z_lo, z_hi = cz - half, cz + half

    # floor grid (in plot coords; world Y → plot Z after the swap)
    # use the PC's max plot-Z (= world Y_max = floor) so the grid sits on the actual floor
    z_floor_plot = float(pc_plot[:, 2].max())
    hw = half * grid_scale
    n_lines = 11
    grid_segs = []
    for v in np.linspace(cx - hw, cx + hw, n_lines):
        grid_segs.append([[v, cy - hw, z_floor_plot], [v, cy + hw, z_floor_plot]])
    for v in np.linspace(cy - hw, cy + hw, n_lines):
        grid_segs.append([[cx - hw, v, z_floor_plot], [cx + hw, v, z_floor_plot]])

    # convert trajectories to plot space too (each point [x, y, z] → [x, z, y])
    def swap_pt(pt):
        if pt is None or pt[0] is None: return None
        return [pt[0], pt[2], pt[1]]
    def swap_frames(frames):
        return [[swap_pt(p) for p in fr] for fr in frames]

    n_hist = cfg["n_hist"]
    nF = len(cfg["all_frames"])
    P = len(cfg["gt_3d"][0])
    p_indices = list(range(0, P, POINT_SUBSAMPLE))

    gt3d_plot   = swap_frames(cfg["gt_3d"])
    pred3d_plot = swap_frames(cfg["pred_3d"])

    # ── shared axis setup ──
    def _make_ax(transparent_bg=False):
        f = Figure(figsize=(img_w / 150, img_h / 150), dpi=150)
        c = FigureCanvasAgg(f)
        a = f.add_subplot(111, projection="3d", computed_zorder=False)
        a.set_xlim(x_lo, x_hi); a.set_ylim(y_lo, y_hi); a.set_zlim(z_lo, z_hi)
        a.invert_zaxis()
        a.view_init(elev=elev, azim=-azim)
        a.set_axis_off()
        f.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
        if transparent_bg:
            f.patch.set_alpha(0.0)
            a.set_facecolor((0, 0, 0, 0))
            a.xaxis.pane.fill = False
            a.yaxis.pane.fill = False
            a.zaxis.pane.fill = False
        return f, c, a

    # ── Pass 1: PC + floor grid (opaque) ──
    fig1, c1, ax1 = _make_ax()
    ax1.add_collection3d(Line3DCollection(grid_segs, colors=(0.72, 0.72, 0.72),
                                          linewidths=0.5, alpha=0.55))
    pc_col_f = pc_rgb.astype(np.float32) / 255.0
    ax1.scatter(pc_plot[:, 0], pc_plot[:, 1], pc_plot[:, 2],
                c=pc_col_f, s=pc_point_size, linewidths=0, depthshade=True)
    c1.draw()
    panel_pc = np.array(c1.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    import matplotlib.pyplot as plt; plt.close(fig1)

    # ── Pass 2: trajectories on transparent bg ──
    fig2, c2, ax2 = _make_ax(transparent_bg=True)
    show_hist = kind in ("hist", "gt", "pred", "all")
    show_gt   = kind in ("gt", "all")
    show_pred = kind in ("pred", "all")

    if show_hist and n_hist > 0:
        segs = collect_trail_segments(gt3d_plot, p_indices, 0, n_hist - 1)
        if segs:
            ax2.add_collection3d(Line3DCollection(
                segs, colors=TRAIL_COLORS["hist"], linewidths=line_w, alpha=0.95))
    if show_gt:
        segs = collect_trail_segments(gt3d_plot, p_indices, max(0, n_hist - 1), nF - 1)
        if segs:
            ax2.add_collection3d(Line3DCollection(
                segs, colors=TRAIL_COLORS["gt"], linewidths=line_w, alpha=0.97))
    if show_pred:
        segs = collect_trail_segments(pred3d_plot, p_indices, max(0, n_hist - 1), nF - 1)
        if segs:
            ax2.add_collection3d(Line3DCollection(
                segs, colors=TRAIL_COLORS["pred"], linewidths=line_w,
                alpha=0.97, linestyles="--"))

    # current-frame dots in object color
    def _scatter_endpoint(pts_per_frame, fi, fallback):
        pts, cols = [], []
        for p in p_indices:
            pt = pts_per_frame[fi][p] if fi < len(pts_per_frame) else None
            if pt is None or pt[0] is None: continue
            if not math.isfinite(pt[0]): continue
            pts.append(pt)
            c = pt_colors_rgb[p] if (pt_colors_rgb and pt_colors_rgb[p] is not None) else None
            cols.append((c[0]/255.0, c[1]/255.0, c[2]/255.0) if c else fallback)
        if pts:
            pts = np.array(pts, dtype=np.float32)
            ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=cols, s=dot_size, linewidths=0, zorder=10)

    last_fi = nF - 1
    if kind == "hist" and n_hist > 0:
        _scatter_endpoint(gt3d_plot, n_hist - 1, TRAIL_COLORS["hist"])
    elif kind == "gt":
        _scatter_endpoint(gt3d_plot, last_fi, TRAIL_COLORS["gt"])
    elif kind == "pred":
        _scatter_endpoint(pred3d_plot, last_fi, TRAIL_COLORS["pred"])
    elif kind == "all":
        _scatter_endpoint(gt3d_plot,   last_fi, TRAIL_COLORS["gt"])
        _scatter_endpoint(pred3d_plot, last_fi, TRAIL_COLORS["pred"])

    c2.draw()
    traj_rgba = np.array(c2.buffer_rgba(), dtype=np.uint8)
    plt.close(fig2)

    # ── Alpha-composite trails over PC ──
    alpha = traj_rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = (panel_pc.astype(np.float32) * (1.0 - alpha)
           + traj_rgba[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-json", required=True, type=Path)
    ap.add_argument("--site-root", required=True, type=Path,
                    help="Site root containing static/data and where panels/ will be created")
    ap.add_argument("--azim", type=float, default=None,
                    help="Override matplotlib azimuth (deg). Default: ego from c2w_frame0.")
    ap.add_argument("--elev", type=float, default=None,
                    help="Override matplotlib elevation (deg). Default: 25.")
    ap.add_argument("--img-w", type=int, default=900)
    ap.add_argument("--img-h", type=int, default=600)
    args = ap.parse_args()

    clip = json.loads(args.clip_json.read_text())
    cfg  = clip["configs"][0]
    pt_colors_rgb = cfg.get("pt_colors_rgb") or []

    if "pc_bin" not in clip:
        raise SystemExit("clip JSON has no pc_bin — run prepare_clip.py first")
    pc_bin = args.site_root / clip["pc_bin"]["url"]
    pc_xyz, pc_rgb = load_pc_binary(pc_bin)
    print(f"loaded PC: {pc_xyz.shape[0]} pts from {pc_bin}")

    cam = clip.get("camera", {})
    if cam.get("c2w_frame0") and args.azim is None:
        c2w = np.array(cam["c2w_frame0"], dtype=np.float32)
        azim, elev = ego_view_angles(c2w, elev_down=(args.elev or 25.0))
        print(f"ego camera: azim={azim:.1f}°  elev={elev:.1f}°")
    else:
        azim = args.azim if args.azim is not None else 45.0
        elev = args.elev if args.elev is not None else 25.0

    panels_dir = args.site_root / "static" / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    clip_id = args.clip_json.stem

    panels_meta = {}
    for kind in ["hist", "gt", "pred", "all"]:
        out_path = panels_dir / f"{clip_id}_{kind}.png"
        render_panel(pc_xyz, pc_rgb, cfg, pt_colors_rgb,
                     c2w=np.array(cam.get("c2w_frame0", np.eye(4))),
                     out_path=out_path, kind=kind, azim=azim, elev=elev,
                     img_w=args.img_w, img_h=args.img_h)
        kb = out_path.stat().st_size // 1024
        print(f"  wrote {out_path}  ({kb} KB)")
        panels_meta[kind] = f"static/panels/{out_path.name}"

    # Update JSON with panel URLs
    clip["static_panels"] = {
        "azim_deg": azim,
        "elev_deg": elev,
        "img_size": [args.img_w, args.img_h],
        "panels": panels_meta,
    }
    args.clip_json.write_text(json.dumps(clip))
    print("updated JSON with static_panels metadata")


if __name__ == "__main__":
    main()
