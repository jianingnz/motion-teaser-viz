#!/usr/bin/env python3
"""Regenerate HOT3D 3D/2D tracks at higher density for clip-001995 + clip-001996.

Wraps `vis_clips_backproject.py:process_clip_tracks_only` and re-uses the same
mesh-sampling logic, just bumping `--num_track_points` from the default 2000.
Output goes to a fresh dir so the canonical /tmp/all_tracks/ stays untouched.

Run inside the `hot3d` conda env (so trimesh + the BOP Toolkit imports work).
"""
import argparse, os, sys
from pathlib import Path

VIS_DIR = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp"
sys.path.insert(0, VIS_DIR)
import vis_clips_backproject as VCB     # noqa: E402

OBJECT_MODELS_DIR = "/weka/prior-default/jianingz/home/dataset/hot3d/object_models_eval"
ARIA_TARS = Path("/weka/prior-default/jianingz/home/dataset/hot3d/train_aria")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-track-points", type=int, default=16000,
                    help="Per-object surface samples. 16000 ≈ 8× the canonical 2000.")
    ap.add_argument("--out-dir", default="/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/hot3d_dense_16k")
    ap.add_argument("--clips", nargs="+", default=["clip-001995", "clip-001996"])
    ap.add_argument("--sample-mode", choices=["random", "even", "pixellift"], default="pixellift",
                    help="'pixellift' = anchor-frame pixel-grid back-projection: "
                         "stride-sample pixels INSIDE each object's modal mask at "
                         "frame 84 of clip-001995, ray-cast through the camera "
                         "into the mesh, and use those object-space hit points as "
                         "the tracked surface samples. Mimics 'video pixels lifted "
                         "to 3D' — same regular grid pattern as the MoGe scene PC. "
                         "'even' = mesh Poisson-disk surface sampling. "
                         "'random' = uniform random mesh sampling.")
    ap.add_argument("--pixel-stride", type=int, default=4,
                    help="Pixel stride for --sample-mode pixellift. Lower = denser "
                         "cloud per object. With 1408×1408 input frames and a "
                         "typical object mask of ~80k pixels, stride=4 yields "
                         "~5k tracks per object.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load every BOP mesh + sample N points per object surface ──────
    import trimesh
    import numpy as np
    object_meshes = {}
    model_files = sorted(f for f in os.listdir(OBJECT_MODELS_DIR) if f.endswith(".glb"))
    for f in model_files:
        bop_id = int(f.split(".glb")[0].split("obj_")[1])
        mesh = VCB.load_mesh_raw(os.path.join(OBJECT_MODELS_DIR, f))
        object_meshes[bop_id] = mesh
    print(f"loaded {len(object_meshes)} BOP meshes")

    # ── Build object_surface_points by sample-mode ────────────────────
    object_surface_points = {}
    if args.sample_mode == "pixellift":
        # Open the anchor clip's tar, walk to clip-001995 frame 84, get
        # per-object mask + pose + camera, then for each object:
        #   1. Stride-sample pixels inside its modal mask
        #   2. Ray-cast each pixel through the pinhole camera into the
        #      mesh (in OBJECT space — applied via T_obj_from_eye)
        #   3. Save first-hit object-space positions
        # The result is exactly "the pixels you can see of this object
        # at the anchor frame, lifted to 3D on its own mesh surface".
        import tarfile, sys as _sys
        _sys.path.insert(0, VIS_DIR)
        import clip_util  # noqa: E402
        anchor_clip = "clip-001995"
        anchor_frame = 84
        anchor_tar = ARIA_TARS / f"{anchor_clip}.tar"
        if not anchor_tar.exists():
            raise FileNotFoundError(anchor_tar)
        print(f"pixel-lift from {anchor_clip} frame {anchor_frame} (stride={args.pixel_stride})")
        tar = tarfile.open(anchor_tar, mode="r")
        cameras, _ = clip_util.load_cameras(tar, f"{anchor_frame:06d}")
        camera_pinhole = clip_util.convert_to_pinhole_camera(cameras["214-1"])
        objects = clip_util.load_object_annotations(tar, f"{anchor_frame:06d}")
        tar.close()
        # Camera intrinsics + world↔eye pose at anchor (pinhole)
        T_we = camera_pinhole.T_world_from_eye
        fx, fy = camera_pinhole.f
        cx, cy = camera_pinhole.c
        img_w, img_h = camera_pinhole.width, camera_pinhole.height
        # Pre-compute the fisheye→pinhole warp map (needed to warp the
        # fisheye-space modal masks into the pinhole space we ray-cast in).
        warp_map_x, warp_map_y = VCB.compute_warp_maps(cameras["214-1"], camera_pinhole)
        stream_id = "214-1"
        for bid_str, inst_list in objects.items():
            inst = inst_list[0]
            bid = int(inst["object_bop_id"])
            if bid not in object_meshes:
                continue
            mask_data = inst["masks_modal"][stream_id] if (
                "masks_modal" in inst and stream_id in inst["masks_modal"]
            ) else None
            if mask_data is None:
                print(f"  obj {bid}: no mask at anchor — skipping (mesh-sampling fallback)")
                pts, _ = trimesh.sample.sample_surface_even(object_meshes[bid],
                                                            args.num_track_points)
                object_surface_points[bid] = np.array(pts, dtype=np.float32)
                continue
            T_wo = clip_util.se3_from_dict(inst["T_world_from_object"])
            # Decode mask, warp to pinhole
            import cv2
            mask_fisheye = clip_util.decode_binary_mask_rle(mask_data).astype(np.uint8)
            mask_pin = VCB.fast_warp(
                mask_fisheye * 255, warp_map_x, warp_map_y,
                interpolation=cv2.INTER_NEAREST,
            ) > 128
            # Stride-sample pixels inside the mask. NOTE: vis_clips_backproject
            # rotates the 2D tracks by rot90(k=3) at the very end so that they
            # match the display-orientation RGB mp4 (which is itself produced by
            # extract_rgbs.py with the same rot90). The MoGe scene PC is built
            # directly in display coords, so to share its stride-N grid we must
            # pick pinhole pixels whose post-rotation display coords fall on
            # (u_disp % stride == 0, v_disp % stride == 0). With
            #   u_disp = (img_h - 1) - v_pin
            #   v_disp = u_pin
            # this requires xx % stride == 0  AND  yy % stride == (img_h-1) % stride.
            yy, xx = np.where(mask_pin)
            stride = max(1, int(args.pixel_stride))
            yy_off = (img_h - 1) % stride
            keep = ((xx % stride == 0) & (yy % stride == yy_off))
            xx = xx[keep]; yy = yy[keep]
            n_rays = len(xx)
            if n_rays == 0:
                print(f"  obj {bid}: empty mask — skipping (mesh-sampling fallback)")
                pts, _ = trimesh.sample.sample_surface_even(object_meshes[bid],
                                                            args.num_track_points)
                object_surface_points[bid] = np.array(pts, dtype=np.float32)
                continue
            # Pinhole camera-space ray dirs: dir_cam = ((u-cx)/fx, (v-cy)/fy, 1)
            rays_d_cam = np.column_stack([
                (xx.astype(np.float64) - cx) / fx,
                (yy.astype(np.float64) - cy) / fy,
                np.ones(n_rays, dtype=np.float64),
            ])
            # Transform rays from camera frame → object frame.
            # ray_obj = inv(T_world_from_object) · T_world_from_eye · ray_cam
            T_oe = np.linalg.inv(T_wo) @ T_we
            R_oe = T_oe[:3, :3]
            t_oe = T_oe[:3, 3]
            rays_d_obj = (R_oe @ rays_d_cam.T).T
            rays_o_obj = np.tile(t_oe[None, :], (n_rays, 1))
            # First-hit ray-mesh intersection in object space
            mesh = object_meshes[bid]
            try:
                hits, idx_ray, idx_tri = mesh.ray.intersects_location(
                    ray_origins=rays_o_obj, ray_directions=rays_d_obj,
                    multiple_hits=False)
            except Exception as e:
                print(f"  obj {bid}: ray-cast failed ({e}) — falling back to mesh sample")
                pts, _ = trimesh.sample.sample_surface_even(object_meshes[bid],
                                                            args.num_track_points)
                object_surface_points[bid] = np.array(pts, dtype=np.float32)
                continue
            if len(hits) == 0:
                print(f"  obj {bid}: no ray hits — falling back to mesh sample")
                pts, _ = trimesh.sample.sample_surface_even(object_meshes[bid],
                                                            args.num_track_points)
                object_surface_points[bid] = np.array(pts, dtype=np.float32)
                continue
            # Make every object's track count == args.num_track_points so
            # prep_hot3d's "N % N_OBJ == 0" assert holds. If we got more
            # hits than requested, stride sub-sample. If fewer, cycle-
            # replicate the lifts (visually duplicates overlay so cloud
            # density is unchanged).
            target = int(args.num_track_points)
            if len(hits) > target:
                step = len(hits) / target
                pick = np.array([int(i * step) for i in range(target)])
                pts = hits[pick]
            elif len(hits) < target:
                pad_idx = np.arange(target - len(hits)) % len(hits)
                pts = np.concatenate([hits, hits[pad_idx]], axis=0)
            else:
                pts = hits
            object_surface_points[bid] = pts.astype(np.float32)
            print(f"  obj {bid}: {n_rays} mask px → {len(hits)} pixel-lifts → "
                  f"{len(pts)} (padded/trimmed to target)")
    elif args.sample_mode == "even":
        for bid, mesh in object_meshes.items():
            pts, _ = trimesh.sample.sample_surface_even(mesh, args.num_track_points * 2)
            if len(pts) > args.num_track_points:
                step = len(pts) / args.num_track_points
                idx = np.array([int(i * step) for i in range(args.num_track_points)])
                pts = pts[idx]
            object_surface_points[bid] = np.array(pts, dtype=np.float32)
    else:  # random
        for bid, mesh in object_meshes.items():
            pts, _ = trimesh.sample.sample_surface(mesh, args.num_track_points)
            object_surface_points[bid] = np.array(pts, dtype=np.float32)

    # ── Process each clip ─────────────────────────────────────────────
    for clip in args.clips:
        clip_path = ARIA_TARS / f"{clip}.tar"
        if not clip_path.exists():
            raise FileNotFoundError(clip_path)
        # Force fresh output by removing any stale outputs first.
        for ext in ["_2d.npz", "_3d.npz"]:
            p = out_dir / f"{clip}{ext}"
            if p.exists(): p.unlink()
        print(f"\nprocessing {clip} → {out_dir}")
        ok = VCB.process_clip_tracks_only(
            clip_path=str(clip_path),
            object_meshes=object_meshes,
            object_surface_points=object_surface_points,
            output_dir=str(out_dir),
        )
        if not ok:
            print(f"  ! {clip} failed")
            continue
        # Quick verify
        import numpy as np
        d3 = np.load(out_dir / f"{clip}_3d.npz")
        d2 = np.load(out_dir / f"{clip}_2d.npz")
        N3 = d3["points_3d"].shape[0]
        T = d3["points_3d"].shape[1]
        N2, _, _ = d2["tracks"].shape if d2["tracks"].ndim == 3 else (None, None, None)
        print(f"  3d: points_3d {d3['points_3d'].shape}  vis {d3['visibility'].shape}")
        print(f"  2d: tracks {d2['tracks'].shape}  vis {d2['visibility'].shape}  dim {d2['dim']}")
        # Sanity: total points should be a multiple of num_track_points (=N_objects per clip)
        assert N3 % args.num_track_points == 0, f"N3={N3} not a multiple of {args.num_track_points}"
        print(f"  → {N3 // args.num_track_points} objects × {args.num_track_points} pts each")

    print("\ndone.")


if __name__ == "__main__":
    main()
