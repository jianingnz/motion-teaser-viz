#!/usr/bin/env python3
"""Regenerate the HOT3D scene point cloud + per-point pix-distance using
PIXEL-STRIDE sampling (structured) instead of random subsampling.

The original scene.bin shipped by visual_example/build_scene.py was a
random sub-sample from the full 1408×1408 MoGe backprojection — that's
why it reads as visually noisy. By stride-sampling the pixel grid before
backprojection we get a regular grid of points that lines up with the
camera-side image structure (visible surfaces show up as smooth surfaces
in the cloud, not as random splatter).

Inputs:
  * Anchor RGB (clip-001995 frame 84) from .../tmp/rgbs/clip-001995_rgb.mp4
  * Cached MoGe (a, b) depth-scale coefficients from
    /weka/prior-default/jianingz/home/dataset/hot3d_repo/visual_example/data/meta.json
  * Anchor visible 2D tracks (for the per-point min-pix-dist side-channel),
    loaded from the dense per-clip tracks at
    /weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/hot3d_dense_16k/

Outputs (overwriting in place — re-run prepare_hot3d.py is NOT needed):
  static/data/hot3d_clip1995_clip1996_pc.bin
  static/data/hot3d_clip1995_clip1996_pc_dist.bin

Run inside the `moge` conda env (`/root/.conda/envs/moge`).
"""
import argparse, json, struct
from pathlib import Path

import numpy as np
import cv2

CLIP_A = "clip-001995"
T_A_START = 84
RGB_MP4 = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"
META_JSON = Path("/weka/prior-default/jianingz/home/dataset/hot3d_repo/visual_example/data/meta.json")
DENSE_TRACKS = Path("/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/hot3d_dense_16k")

FX = 608.9346; FY = 608.9346; CX = 701.7970; CY = 705.7216
W = H = 1408


def write_pc_binary(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray):
    N = int(xyz.shape[0])
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.astype(np.uint8).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="motion-teaser-viz repo root (containing static/).")
    ap.add_argument("--name", default="hot3d_clip1995_clip1996")
    ap.add_argument("--stride", type=int, default=3,
                    help="Pixel stride. 3 → ~155k pts before mask; 4 → ~87k.")
    ap.add_argument("--moge-resolution", type=int, default=9)
    ap.add_argument("--depth-min", type=float, default=0.05)
    ap.add_argument("--depth-max", type=float, default=5.0)
    args = ap.parse_args()

    out_data = args.out_dir / "static" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    # ── Cached (a, b) ──
    meta = json.load(open(META_JSON))
    a, b = float(meta["moge"]["a"]), float(meta["moge"]["b"])
    print(f"using MoGe affine scale  a={a:.5f}  b={b:.5f}")

    # ── Read anchor RGB ──
    cap = cv2.VideoCapture(RGB_MP4.format(clip=CLIP_A))
    cap.set(cv2.CAP_PROP_POS_FRAMES, T_A_START)
    ok, anchor_bgr = cap.read(); cap.release()
    assert ok, "could not read anchor frame"
    anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)
    print(f"anchor RGB: {anchor_rgb.shape}")

    # ── Run MoGe-1 ──
    import torch
    from moge.model.v1 import MoGeModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading MoGe on {device}...")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    img = torch.from_numpy(anchor_rgb).permute(2, 0, 1).float() / 255.0
    fov_x_deg = 2 * np.degrees(np.arctan2(W / 2, FX))
    print(f"running MoGe at resolution_level={args.moge_resolution}, fov_x={fov_x_deg:.2f}°")
    with torch.inference_mode():
        out = moge.infer(img, fov_x=fov_x_deg, resolution_level=args.moge_resolution,
                         apply_mask=True, force_projection=True, use_fp16=True)
    depth = out["depth"].cpu().numpy()                # (H, W)
    mask  = out["mask"].cpu().numpy().astype(bool)    # (H, W)
    metric_d = a * depth + b
    print(f"MoGe done. valid mask frac: {mask.mean():.3f}")

    # ── Stride-sample the pixel grid ──
    s = int(args.stride)
    yy, xx = np.mgrid[0:H:s, 0:W:s]
    valid = mask[yy, xx] & np.isfinite(metric_d[yy, xx]) \
            & (metric_d[yy, xx] > args.depth_min) & (metric_d[yy, xx] < args.depth_max)
    print(f"stride-{s} grid: {yy.size} pts pre-mask, {valid.sum()} pts post-mask")

    u = xx[valid].astype(np.float32)
    v = yy[valid].astype(np.float32)
    z = metric_d[yy, xx][valid].astype(np.float32)
    x = (u - CX) / FX * z
    y = (v - CY) / FY * z
    xyz = np.stack([x, y, z], axis=-1)
    rgb_u8 = anchor_rgb[yy, xx][valid].astype(np.uint8)
    print(f"final scene PC: {xyz.shape[0]} points")

    pc_path = out_data / f"{args.name}_pc.bin"
    write_pc_binary(pc_path, xyz, rgb_u8)
    print(f"wrote {pc_path}  ({pc_path.stat().st_size//1024//1024} MB)")

    # ── Per-point pixel distance to nearest anchor 2D-track ──
    print("computing min_pix_dist via KD-tree against the dense anchor 2D tracks...")
    d2A = np.load(DENSE_TRACKS / f"{CLIP_A}_2d.npz")
    tracks_anchor = d2A["tracks"][T_A_START]                   # (N, 2)
    vis_anchor    = d2A["visibility"][T_A_START]               # (N,)
    valid_uv = (
        vis_anchor
        & np.isfinite(tracks_anchor).all(axis=1)
        & (tracks_anchor[:, 0] >= 0) & (tracks_anchor[:, 0] < W)
        & (tracks_anchor[:, 1] >= 0) & (tracks_anchor[:, 1] < H)
    )
    track_pix = tracks_anchor[valid_uv].astype(np.float32)
    print(f"  anchor visible 2D-track count: {len(track_pix)}")

    from scipy.spatial import cKDTree
    pix_uv_scene = np.stack([u, v], axis=-1)                   # (N_scene, 2)
    if len(track_pix) > 0:
        tree = cKDTree(track_pix)
        min_pix_dist, _ = tree.query(pix_uv_scene, k=1)
    else:
        min_pix_dist = np.full((xyz.shape[0],), 1e6, dtype=np.float32)
    min_pix_dist = min_pix_dist.astype(np.float32)
    print(f"  min_pix_dist median={np.median(min_pix_dist):.1f}px "
          f"frac<30px={(min_pix_dist<30).mean()*100:.1f}%")

    pc_dist_path = out_data / f"{args.name}_pc_dist.bin"
    pc_dist_path.write_bytes(min_pix_dist.tobytes())
    print(f"wrote {pc_dist_path}  ({pc_dist_path.stat().st_size//1024} KB)")

    # ── Patch the bundle JSON's pc_bin.n_points + pc_dist_bin.n_points ──
    bundle_json = out_data / f"{args.name}.json"
    if bundle_json.exists():
        bundle = json.loads(bundle_json.read_text())
        bundle["pc_bin"]["n_points"] = int(xyz.shape[0])
        bundle["pc_dist_bin"]["n_points"] = int(xyz.shape[0])
        bundle["pc_bin"]["subsample"] = s
        bundle["pc_bin"]["sampling"]  = "pixel-stride"
        bundle_json.write_text(json.dumps(bundle, indent=2))
        print(f"patched {bundle_json} (n_points = {xyz.shape[0]})")


if __name__ == "__main__":
    main()
