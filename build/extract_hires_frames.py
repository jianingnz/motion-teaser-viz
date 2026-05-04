#!/usr/bin/env python3
"""Extract high-resolution image-frame strips for the motion-teaser-viz
"image frames" panels.

Why this exists. The viewer renders two strips of sampled frames:
  ⓪b "Frames from full video"    — N_FULL evenly spaced across the full source mp4
  ⑥  "Raw RGB frames"            — N_CLIP evenly spaced across the cut clip

Today both are drawn by seeking the in-page low-res `<video>` element
(EgoDex cut = 640×360, EgoDex full = 854×480, HOT3D = 480×480), so paper
figures end up grainy. This script pre-extracts the strip frames from the
ORIGINAL high-resolution source — and ONLY for the strips, not playback —
then writes them as standalone jpgs that the viewer loads directly.

Sources:
  EgoDex   /weka/prior-default/jianingz/home/dataset/egodex/{part}/{task}/{idx}.mp4
           native 1920×1080 @ 30 fps (the bundle's frame indices live in a
           15-fps vipe-derived domain, so we map vipe_fi → src_30fps_fi = 2·vipe_fi)
  HOT3D    /weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/clip-NNNNNN_rgb.mp4
           native 1408×1408 @ 30 fps. The viewer's stitched mp4 (clip-001995[84..149]
           ∪ clip-001996[0..39]) is at the source 30 fps too, so viewer frame
           N → (source_clip, source_frame) is read from the stitch table below.

Other datasets (HD-EPIC, DROID, DAVIS) are skipped — pass --allow-skip and
they exit with code 0 instead of failing.

Usage
-----
    $PYTHON build/extract_hires_frames.py \
        --clip-id clean_surface_1713_t10 \
        --out-dir /weka/prior-default/jianingz/home/visual/motion-teaser-viz \
        [--n-clip 10] [--n-full 14]

In-place edits the JSON to add `clip_frames_hires` and (when applicable)
`full_video_frames_hires` arrays. Run again to refresh — old jpgs are
overwritten.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


# ────────────────────── source-path resolution ──────────────────────

EGODEX_ROOT = Path("/weka/prior-default/jianingz/home/dataset/egodex")
HOT3D_RGB_TPL = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"

# The HOT3D bundle is a stitch — viewer frames map back to two source clips.
# Mirrors the constants in build/prepare_hot3d.py (CLIP_A/B + T_*_START/END).
HOT3D_STITCH = {
    "hot3d_clip1995_clip1996": [
        ("clip-001995", 84, 149),  # viewer frames 0..65   (66 frames)
        ("clip-001996",  0,  39),  # viewer frames 66..105 (40 frames)
    ],
    "hot3d_clip1995_clip1996_lastframe": [
        ("clip-001995", 84, 149),
        ("clip-001996",  0,  39),
    ],
}


def is_egodex(video_stem: str) -> bool:
    """EgoDex video stems look like `partN_<task_with_underscores>_<int>`."""
    return bool(re.match(r"^part\d+_.+_\d+$", video_stem))


def split_egodex_stem(video_stem: str):
    """Split `part1_clean_surface_1713` → ('part1', 'clean_surface', '1713')."""
    m = re.match(r"^(part\d+)_(.+)_(\d+)$", video_stem)
    if not m:
        raise ValueError(f"not an EgoDex stem: {video_stem}")
    return m.group(1), m.group(2), m.group(3)


def egodex_source_mp4(video_stem: str) -> Path:
    """Resolve the 1920×1080 @ 30 fps original EgoDex mp4 for a video_stem.
    Searches the canonical part-folder first; if the task name isn't there
    (cross-task index reuse), falls back to a glob."""
    part, task, idx = split_egodex_stem(video_stem)
    canonical = EGODEX_ROOT / part / task / f"{idx}.mp4"
    if canonical.exists():
        return canonical
    matches = list(EGODEX_ROOT.glob(f"{part}/*/{idx}.mp4"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"no EgoDex source mp4 found for stem={video_stem} "
            f"(tried {canonical} and {EGODEX_ROOT}/{part}/*/{idx}.mp4)")
    raise RuntimeError(
        f"ambiguous EgoDex source for stem={video_stem}: {matches}")


def hot3d_stitch_for_clip(clip_id: str):
    if clip_id in HOT3D_STITCH:
        return HOT3D_STITCH[clip_id]
    return None


# ────────────────────── frame extraction primitives ──────────────────────

def grab_frames_from_mp4(mp4: Path, indices: list[int]) -> list[np.ndarray]:
    """Read BGR frames at the given source-frame indices. Indices are in the
    mp4's NATIVE frame domain (30 fps for EgoDex source / HOT3D source)."""
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4}")
    out = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to read frame {fi} of {mp4}")
        out.append(frame)
    cap.release()
    return out


def write_jpgs(frames: list[np.ndarray], out_dir: Path, repo_root: Path,
               quality: int = 92) -> list[str]:
    """Write frames to out_dir/NN.jpg. Returns repo-root-relative URLs
    (e.g. `static/videos/<clip_id>_clipframes/00.jpg`) suitable for the
    JSON bundle's `url` field."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = []
    for i, frame in enumerate(frames):
        path = out_dir / f"{i:02d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        rels.append(str(path.relative_to(repo_root)).replace("\\", "/"))
    return rels


# ────────────────────── per-dataset strip builders ──────────────────────

def even_indices(n_total: int, n_pick: int) -> list[int]:
    """N evenly-spaced integer indices in [0, n_total-1] inclusive."""
    if n_total <= 0 or n_pick <= 0:
        return []
    if n_pick == 1:
        return [0]
    return [int(round(i * (n_total - 1) / (n_pick - 1))) for i in range(n_pick)]


def build_egodex_clip_strip(clip: dict, video_stem: str, n_clip: int):
    """N_CLIP frames evenly across the cut clip, pulled from the 1920×1080
    @30 fps source. The bundle's src_clip_start/end are in the 15-fps vipe
    domain — convert via ×2."""
    raw_meta = clip.get("raw_meta") or {}
    s = raw_meta.get("src_clip_start")
    e = raw_meta.get("src_clip_end")
    if s is None or e is None:
        raise RuntimeError(
            "clip JSON has no raw_meta.src_clip_start/src_clip_end — required.")
    n_clip_frames = e - s + 1                                   # 15-fps frames
    vipe_picks = even_indices(n_clip_frames, n_clip)            # 0..n_clip_frames-1
    src_picks_30fps = [(s + v) * 2 for v in vipe_picks]         # 30-fps src indices
    src_mp4 = egodex_source_mp4(video_stem)
    print(f"  EgoDex clip strip: {len(src_picks_30fps)} frames at 30-fps "
          f"src indices {src_picks_30fps} from {src_mp4.name}")
    return src_mp4, vipe_picks, src_picks_30fps


def build_egodex_full_strip(clip: dict, video_stem: str, n_full: int):
    """N_FULL frames evenly across the FULL EgoDex source mp4.

    The viewer's full mp4 is the 15-fps vipe re-encode; the strip thumbnails
    are pulled from the original 30-fps EgoDex source. We pick N evenly
    across the SOURCE mp4 directly (not the vipe domain) — that's robust
    against ±1 frame rounding between the two encodings, and the strip
    label reads in seconds anyway, so vipe-frame alignment isn't required.
    """
    fv = clip.get("full_video") or {}
    if not fv:
        return None
    src_mp4 = egodex_source_mp4(video_stem)
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src_mp4}")
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if n_src <= 0:
        raise RuntimeError(f"unable to read frame count from {src_mp4}")
    src_picks = even_indices(n_src, n_full)
    print(f"  EgoDex full strip: {len(src_picks)} frames @ {src_fps:.2f}fps "
          f"from {src_mp4.name} (n_src={n_src})  src_indices={src_picks}")
    return src_mp4, src_picks, src_fps


def build_hot3d_clip_strip(clip: dict, clip_id: str, n_clip: int):
    """For HOT3D the viewer mp4 is a stitch of two source clips at the
    source 30 fps (1 source frame → 1 viewer frame). Map each viewer-frame
    index to the (source_clip, source_frame) pair via HOT3D_STITCH."""
    stitch = hot3d_stitch_for_clip(clip_id)
    if stitch is None:
        raise RuntimeError(
            f"HOT3D clip_id={clip_id} not in HOT3D_STITCH table — "
            f"add the stitch boundaries before re-running.")
    n_total = clip.get("num_frames")
    if n_total is None:
        raise RuntimeError("HOT3D clip JSON missing num_frames.")
    viewer_picks = even_indices(int(n_total), n_clip)

    # Resolve each viewer-frame to its (source_clip, source_frame).
    pairs = []
    cumulative = 0
    seg_table = []
    for clp, a, b in stitch:
        seg_len = b - a + 1
        seg_table.append((clp, a, b, cumulative, cumulative + seg_len - 1))
        cumulative += seg_len
    if cumulative != n_total:
        print(f"  WARN: stitch sums to {cumulative} but JSON num_frames={n_total} "
              f"— assuming the table is authoritative.")
    for vf in viewer_picks:
        for clp, a, b, vstart, vend in seg_table:
            if vstart <= vf <= vend:
                src_fi = a + (vf - vstart)
                pairs.append((clp, src_fi, vf))
                break
        else:
            raise RuntimeError(f"viewer frame {vf} fell outside HOT3D stitch table")
    print(f"  HOT3D clip strip: {len(pairs)} frames "
          f"({[(c, f) for c, f, _ in pairs]})")
    return pairs


# ────────────────────── pipeline ──────────────────────

def extract_for_clip(out_dir: Path, clip_id: str, n_clip: int, n_full: int,
                     allow_skip: bool):
    json_path = out_dir / "static" / "data" / f"{clip_id}.json"
    if not json_path.exists():
        raise RuntimeError(f"clip JSON not found: {json_path}")
    clip = json.loads(json_path.read_text())
    cam = clip.get("camera") or {}
    video_stem = cam.get("video_stem", "")

    is_hot3d = clip_id in HOT3D_STITCH
    if not is_hot3d and not is_egodex(video_stem):
        msg = (f"clip {clip_id}: not EgoDex (stem={video_stem!r}) and not in "
               f"HOT3D stitch table — only EgoDex+HOT3D are supported.")
        if allow_skip:
            print(f"SKIP: {msg}"); return
        raise RuntimeError(msg)

    print(f"=== {clip_id} ===")

    clip_dir = out_dir / "static" / "videos" / f"{clip_id}_clipframes"
    full_dir = out_dir / "static" / "videos" / f"{clip_id}_fullframes"

    # ── Cut-clip strip (panel ⑥) ──
    if is_hot3d:
        pairs = build_hot3d_clip_strip(clip, clip_id, n_clip)
        # Group by source clip so we open each rgb mp4 once.
        per_src = {}
        for clp, src_fi, vf in pairs:
            per_src.setdefault(clp, []).append((src_fi, vf))
        # Read frames preserving the original (vf-ordered) sequence.
        idx_to_frame = {}
        for clp, items in per_src.items():
            mp4 = Path(HOT3D_RGB_TPL.format(clip=clp))
            if not mp4.exists():
                raise FileNotFoundError(f"HOT3D source mp4 missing: {mp4}")
            src_indices = [src_fi for src_fi, _ in items]
            frames = grab_frames_from_mp4(mp4, src_indices)
            for (_, vf), frame in zip(items, frames):
                idx_to_frame[vf] = frame
        ordered_vfs = [vf for _, _, vf in pairs]
        frames_out = [idx_to_frame[vf] for vf in ordered_vfs]
        rels = write_jpgs(frames_out, clip_dir, out_dir)
        H, W = frames_out[0].shape[:2]
        clip["clip_frames_hires"] = [
            {"url": rel,
             "frame_idx_clip": int(vf),
             "src_clip": clp,
             "frame_idx_src": int(src_fi)}
            for rel, (clp, src_fi, vf) in zip(rels, pairs)
        ]
    else:
        src_mp4, viewer_picks, src_picks_30fps = build_egodex_clip_strip(
            clip, video_stem, n_clip)
        frames_out = grab_frames_from_mp4(src_mp4, src_picks_30fps)
        rels = write_jpgs(frames_out, clip_dir, out_dir)
        H, W = frames_out[0].shape[:2]
        clip["clip_frames_hires"] = [
            {"url": rel,
             "frame_idx_clip": int(vf),
             "frame_idx_src_30fps": int(src_fi)}
            for rel, vf, src_fi in zip(rels, viewer_picks, src_picks_30fps)
        ]
    print(f"  wrote {len(rels)} clip-strip jpgs ({W}x{H}) → {clip_dir}")

    # ── Full-source strip (panel ⓪b) ──
    # HOT3D: there's no separate "full source" panel today — the viewer mp4
    # already IS the source for every frame, so the clip strip alone is
    # sufficient. Skip full strip for HOT3D.
    if not is_hot3d:
        result = build_egodex_full_strip(clip, video_stem, n_full)
        if result is not None:
            src_mp4, src_picks, src_fps = result
            frames_out = grab_frames_from_mp4(src_mp4, src_picks)
            rels = write_jpgs(frames_out, full_dir, out_dir)
            H, W = frames_out[0].shape[:2]
            clip["full_video_frames_hires"] = [
                {"url": rel,
                 "frame_idx_src": int(src_fi),
                 "t_sec": float(src_fi) / max(src_fps, 1e-6)}
                for rel, src_fi in zip(rels, src_picks)
            ]
            print(f"  wrote {len(rels)} full-strip jpgs ({W}x{H}) → {full_dir}")
        else:
            print(f"  no full_video on this bundle — skipping full strip")
            clip.pop("full_video_frames_hires", None)

    json_path.write_text(json.dumps(clip))
    print(f"  updated {json_path} ({json_path.stat().st_size//1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", action="append", required=True,
                    help="Clip id (the JSON stem in static/data/). "
                         "Repeat to process multiple.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="motion-teaser-viz repo root.")
    ap.add_argument("--n-clip", type=int, default=10,
                    help="Number of frames in the cut-clip strip (panel ⑥).")
    ap.add_argument("--n-full", type=int, default=14,
                    help="Number of frames in the full-source strip (panel ⓪b).")
    ap.add_argument("--allow-skip", action="store_true",
                    help="Skip clips that aren't EgoDex/HOT3D instead of erroring.")
    args = ap.parse_args()

    rc = 0
    for cid in args.clip_id:
        try:
            extract_for_clip(args.out_dir, cid, args.n_clip, args.n_full,
                             args.allow_skip)
        except Exception as e:
            print(f"FAIL: {cid}: {e}", file=sys.stderr)
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
