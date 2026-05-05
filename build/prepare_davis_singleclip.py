#!/usr/bin/env python3
"""Build a motion-teaser-viz bundle for a DAVIS clip from raw motion5
inputs.

Pipeline mirrors what motion5-viz does for `camel_camal_t2`:

  1. Read a single prediction record from
     `eval_results/fulleval_rollout_3mix_p8_davis_<split>/predictions.jsonl`.
  2. Hand it to motion5-viz's `build_motion4_site.build_clip_json` (which we
     import directly) to produce a camel-style source JSON with
     gt_3d / pred_3d / gt_2d / pred_2d / vis / pc_xyz / pc_colors.
  3. Pipe that through `prepare_clip_simple.py` to land a self-contained
     bundle in `static/data/<clip_id>.{json,_pc.bin}` and
     `static/videos/<clip_id>.{mp4,_chrono.jpg}`.

Required source data (paths hard-coded for the current weka layout):
  * Predictions   → /weka/prior-default/chenhaoz/home/MotionPlanner/molmo2/eval_results/
  * Interp 3D GT  → /weka/prior-default/chenhaoz/data/interpolated_tracks/davis/
  * Final 2D GT   → /weka/prior-default/jianingz/home/project/_GenTraj/vipe/davis_final_tracks/
  * Vipe RGB+pose+intrinsics+depth
                  → /weka/prior-default/jianingz/home/project/_GenTraj/vipe/vipe_results/
  * DAVIS split   → /weka/prior-default/chenhaoz/home/motion_filtering_splits/davis_split.json

Usage:
    python build/prepare_davis_singleclip.py --video flamingo --out-dir .
    python build/prepare_davis_singleclip.py --video car-turn --out-dir . --t0 3
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

MOTION5_BUILD = Path("/weka/prior-default/jianingz/home/visual/motion5-viz/build")
sys.path.insert(0, str(MOTION5_BUILD))

import build_motion4_site as B4  # noqa: E402

EVAL_ROOT = Path("/weka/prior-default/chenhaoz/home/MotionPlanner/molmo2/eval_results")
VIPE_RGB_ROOT = Path(
    "/weka/prior-default/jianingz/home/project/_GenTraj/vipe/vipe_results/rgb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                    help="DAVIS video id (e.g. flamingo, car-turn).")
    ap.add_argument("--split", choices=["test", "traintest"], default="traintest",
                    help="Which prediction split to read from.")
    ap.add_argument("--t0", type=int, default=None,
                    help="History-end frame; selects the matching prediction. "
                         "Defaults to the prediction with smallest l2.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="motion-teaser-viz repo root (containing static/).")
    ap.add_argument("--clip-id", default=None,
                    help="Bundle id; defaults to <video>_<obj>_t<t0>.")
    ap.add_argument("--src-mp4", type=Path, default=None,
                    help="Override path to the source mp4. Defaults to the "
                         "vipe RGB clip.")
    ap.add_argument("--n-stamps", type=int, default=4)
    ap.add_argument("--dilate-px", type=int, default=2)
    args = ap.parse_args()

    pred_jsonl = EVAL_ROOT / f"fulleval_rollout_3mix_p8_davis_{args.split}" / "predictions.jsonl"
    if not pred_jsonl.exists():
        raise SystemExit(f"predictions file not found: {pred_jsonl}")

    preds = []
    with open(pred_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["video"] == args.video:
                preds.append(r)
    if not preds:
        raise SystemExit(
            f"no prediction lines for video={args.video!r} in {pred_jsonl}")

    if args.t0 is not None:
        cands = [p for p in preds if int(p["t0"]) == args.t0]
        if not cands:
            raise SystemExit(
                f"no prediction for video={args.video!r} t0={args.t0} "
                f"(available t0={[p['t0'] for p in preds]})")
        rec = cands[0]
    else:
        rec = min(preds, key=lambda p: float(p.get("l2") or float("inf")))

    print(f"  selected prediction: video={args.video} obj={rec['obj']} "
          f"t0={rec['t0']} l2={rec.get('l2')}")

    meta_index = B4.load_metadata_index("davis")
    if args.video not in meta_index:
        raise SystemExit(f"video {args.video!r} not in davis metadata index")

    tmp_root = args.out_dir / "tmp" / "davis_singleclip"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_video_dir = tmp_root / "videos"
    tmp_video_dir.mkdir(exist_ok=True)

    result = B4.build_clip_json(rec, "davis", meta_index, tmp_video_dir)
    if result is None:
        raise SystemExit(
            f"build_clip_json returned None for {args.video} — likely the 2D "
            f"projection failed (PROJ_LOADERS) or no GT found.")
    safe_id, safe_key, src_json, mse, l2 = result
    print(f"  built source JSON: id={safe_id} mse={mse} l2={l2}")
    if "pc_xyz" not in src_json:
        raise SystemExit(
            "source JSON has no pc_xyz — prepare_clip_simple.py won't work. "
            "Check that vipe depth+intrinsics+pose exist for this clip.")
    if src_json["configs"][0].get("gt_2d") is None:
        raise SystemExit(
            "source JSON has gt_2d=null — prepare_clip_simple.py samples colors "
            "from gt_2d so the bundle would be unusable.")

    clip_id = args.clip_id or safe_id
    src_json_path = tmp_root / f"{safe_id}_motion5src.json"
    src_json_path.write_text(json.dumps(src_json))
    print(f"  wrote source JSON → {src_json_path}")

    src_mp4 = args.src_mp4 or (VIPE_RGB_ROOT / f"{args.video}.mp4")
    if not src_mp4.exists():
        raise SystemExit(f"source mp4 not found: {src_mp4}")

    script = args.out_dir / "build" / "prepare_clip_simple.py"
    cmd = ["python3", str(script),
           "--src-json", str(src_json_path),
           "--src-mp4",  str(src_mp4),
           "--out-dir",  str(args.out_dir),
           "--clip-id",  clip_id,
           "--n-stamps", str(args.n_stamps),
           "--dilate-px", str(args.dilate_px)]
    print("  running:", " ".join(cmd))
    subprocess.check_call(cmd)

    print(f"\nbundle ready: {clip_id}")


if __name__ == "__main__":
    main()
