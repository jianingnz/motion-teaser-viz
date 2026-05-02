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
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load every BOP mesh + sample N points per object surface ──────
    import trimesh
    object_meshes = {}
    object_surface_points = {}
    model_files = sorted(f for f in os.listdir(OBJECT_MODELS_DIR) if f.endswith(".glb"))
    print(f"loading {len(model_files)} BOP meshes; sampling {args.num_track_points} per object...")
    for f in model_files:
        bop_id = int(f.split(".glb")[0].split("obj_")[1])
        mesh = VCB.load_mesh_raw(os.path.join(OBJECT_MODELS_DIR, f))
        object_meshes[bop_id] = mesh
        pts, _ = trimesh.sample.sample_surface(mesh, args.num_track_points)
        import numpy as np
        object_surface_points[bop_id] = np.array(pts, dtype=np.float32)
    print(f"  loaded {len(object_meshes)} meshes")

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
