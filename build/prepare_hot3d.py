#!/usr/bin/env python3
"""Build the HOT3D cross-clip motion-teaser-viz bundle in single-file form.

Reads dense per-object surface tracks (default 16000 pts/object × 6 objects =
96000 tracks/clip), stitches clip-001995[84..149] + clip-001996[0..39] into
the anchor frame, and emits FOUR consolidated artefacts (plus the existing
scene PC files):

  static/data/{name}.json           metadata + URLs only (small, < 1 MB)
  static/data/{name}_gt3d.bin       per-frame 3D positions, INT16-quantized
                                    (sentinel −32768 = invisible) +
                                    per-track RGB (anchor-frame, uint8) +
                                    per-track object id (uint8). Header
                                    carries the per-axis dequantization
                                    bounds so the viewer can recover
                                    metric coords.
  static/data/{name}_cam.bin        per-frame 4×4 c2w (float32) +
                                    intrinsics (fx, fy, cx, cy) (float32)
  static/videos/{name}.mp4          stitched 30-fps source-rate video
                                    (one source frame → one mp4 frame)
  static/data/{name}_pc.bin         MoGe scene cloud (existing)
  static/data/{name}_pc_dist.bin    per-scene-pt min-pix-dist (existing)
  static/videos/{name}_chrono.jpg   chronophoto background (existing)

Compared with the previous JSON-inline version this trims the bundle from
32 MB to ~50 MB total spread across 4 binaries; the JSON itself is now a
~10 KB metadata pointer file.

Why Int16 quantization for positions? Anchor-frame XYZ all fit comfortably
in [-1.5 m, +1.5 m]; an int16 axis encoding gives ~0.05 mm precision per
component which is well below visualization-meaningful resolution. At
96000 pts × 106 frames × 6 bytes the gt3d.bin is ~58 MB, comfortably
under GitHub's 100 MB single-file limit (Float32 would have been ~115 MB).
"""
import argparse, json, struct, subprocess
from pathlib import Path

import cv2
import numpy as np

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Source paths
ALL_TRACKS_DEFAULT = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/hot3d_dense_16k"
RGB_TPL    = "/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/{clip}_rgb.mp4"
SCENE_DIR  = Path("/weka/prior-default/jianingz/home/dataset/hot3d_repo/visual_example/data")
CAM_DIR    = "/tmp/{tag}_cams"

CLIP_A = "clip-001995"
CLIP_B = "clip-001996"
T_A_START, T_A_END = 84, 149
T_B_START, T_B_END = 0,  39

FX = 608.9346; FY = 608.9346; CX = 701.7970; CY = 705.7216
W_SRC = H_SRC = 1408
N_OBJ = 6

R_ROT90 = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]], dtype=np.float64)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]],
        dtype=np.float64)


def load_T_we(path, stream="214-1"):
    d = json.load(open(path))
    e = d[stream]["T_world_from_camera"]
    R = quat_to_R(e["quaternion_wxyz"])
    t = np.array(e["translation_xyz"], dtype=np.float64)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def get_T_we(tag, frame):
    return load_T_we(f"{CAM_DIR.format(tag=tag)}/{frame:06d}.cameras.json")


def diag4(R):
    T = np.eye(4); T[:3, :3] = R; return T


def transform(M, P):
    Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    return (M @ Ph.T).T[:, :3]


def write_pc_binary(path: Path, xyz, rgb_norm):
    N = int(xyz.shape[0])
    rgb_u8 = np.clip(rgb_norm * 255.0, 0, 255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.tobytes())


def stitch_mp4(out_path, target_height=480, fps=30):
    """1 source frame → 1 output frame, native 30 fps preserved."""
    pa, pb = RGB_TPL.format(clip=CLIP_A), RGB_TPL.format(clip=CLIP_B)
    fc = (
        f"[0:v]trim=start_frame={T_A_START}:end_frame={T_A_END + 1},"
        f"setpts=PTS-STARTPTS,scale=-2:{target_height}[a];"
        f"[1:v]trim=start_frame={T_B_START}:end_frame={T_B_END + 1},"
        f"setpts=PTS-STARTPTS,scale=-2:{target_height}[b];"
        f"[a][b]concat=n=2:v=1:a=0,fps={fps}[v]"
    )
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-i", pa, "-i", pb,
        "-filter_complex", fc, "-map", "[v]",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
        "-vsync", "cfr",
        str(out_path)], check=True)


def build_c2w_per_frame(cam_pos, mean_obj_pos):
    """Synthesise a 4×4 c2w per frame from cam centre + a forward direction
    pointing at the per-frame mean object centroid. Rotation is built so the
    frustum's "up" matches world-up after the y/z scene flip."""
    F = cam_pos.shape[0]
    out = np.zeros((F, 4, 4), dtype=np.float32); out[:, 3, 3] = 1.0
    for t in range(F):
        fwd = mean_obj_pos[t] - cam_pos[t]
        n = np.linalg.norm(fwd)
        fwd = (fwd / n).astype(np.float32) if n > 1e-6 else np.array([0,0,1], dtype=np.float32)
        wup = np.array([0, -1, 0], dtype=np.float32)
        right = np.cross(fwd, wup)
        if np.linalg.norm(right) < 1e-3: right = np.array([1,0,0], dtype=np.float32)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        out[t, :3, 0] = right; out[t, :3, 1] = up; out[t, :3, 2] = fwd
        out[t, :3, 3] = cam_pos[t]
    return out


def write_gt3d_binary(path: Path, positions: np.ndarray, vis: np.ndarray,
                      annotated: np.ndarray, obj_ids: np.ndarray,
                      rgb_u8: np.ndarray, n_obj: int):
    """positions: (N, F, 3) float32; vis: (N, F) bool; annotated: (N, F) bool;
       obj_ids: (N,) uint8; rgb_u8: (N, 3) uint8.

    Layout v2 (little-endian):
      MAGIC                           4 bytes 'GT3D'
      version                         uint32 = 2
      N (#tracks)                     uint32
      F (#frames)                     uint32
      n_objects                       uint32
      x_min, x_max, y_min, y_max, z_min, z_max     float32 ×6  (per-axis bounds)
      obj_ids                         uint8 × N
      rgb                             uint8 × (N*3)
      positions                       int16 × (F*N*3)   (sentinel -32768 = NOT annotated)
      vis                             uint8 × (F*N)     (1 = visible/cloud-eligible, 0 = occluded)

    Compared with v1 (where SENT = invisible conflated 'object not in
    annotations' with 'occluded'), v2 separates the two:
      • position SENT  ⇔ object not annotated this frame (no pose available)
      • vis = 0        ⇔ position IS available, but point fails the
                         camera-facing / modal-mask / depth-buffer test
    Trails default to using positions through occluded frames (smooth);
    the cloud renderer respects the vis byte to suppress non-camera-facing
    points. A viewer toggle ('trail honors visibility') flips the trail
    to break at vis=0 frames as well — the v1 'cut' look.
    """
    N, F, _ = positions.shape
    assert vis.shape == (N, F)
    assert annotated.shape == (N, F)
    assert obj_ids.shape == (N,)
    assert rgb_u8.shape == (N, 3)
    valid_mask = annotated & np.isfinite(positions).all(axis=-1)   # (N, F)
    if not valid_mask.any():
        raise RuntimeError("no annotated positions to encode")
    valid_pts = positions[valid_mask]                              # (M, 3)
    axis_min = valid_pts.min(axis=0).astype(np.float32)
    axis_max = valid_pts.max(axis=0).astype(np.float32)
    center = (axis_max + axis_min) / 2.0
    half = (axis_max - axis_min) / 2.0
    half = np.where(half < 1e-9, 1e-9, half)
    quant = np.full((N, F, 3), -32768, dtype=np.int16)
    normed = (positions[valid_mask] - center) / half
    normed = np.clip(np.round(normed * 32767.0), -32767, 32767).astype(np.int16)
    quant[valid_mask] = normed
    quant_fnt = np.ascontiguousarray(quant.transpose(1, 0, 2))     # (F, N, 3)
    vis_fn    = np.ascontiguousarray(vis.T.astype(np.uint8))       # (F, N)
    with open(path, 'wb') as f:
        f.write(b"GT3D")
        f.write(struct.pack("<I", 2))      # version
        f.write(struct.pack("<I", N))
        f.write(struct.pack("<I", F))
        f.write(struct.pack("<I", n_obj))
        for v in axis_min: f.write(struct.pack("<f", float(v)))
        for v in axis_max: f.write(struct.pack("<f", float(v)))
        f.write(obj_ids.astype(np.uint8).tobytes())
        f.write(rgb_u8.astype(np.uint8).tobytes())
        f.write(quant_fnt.tobytes())
        f.write(vis_fn.tobytes())
    return {
        "format":   "GT3D v2",
        "n_tracks": int(N),
        "n_frames": int(F),
        "n_objects": int(n_obj),
        "axis_min":  [float(v) for v in axis_min],
        "axis_max":  [float(v) for v in axis_max],
    }


def write_cam_binary(path: Path, c2w: np.ndarray, intrinsics):
    """c2w: (F, 4, 4) float32. intrinsics: [fx, fy, cx, cy].

    Layout:
      MAGIC 'CAM4' (4 bytes)
      version uint32 = 1
      F uint32
      intrinsics float32 × 4 (fx, fy, cx, cy)
      c2w        float32 × (F * 16)   row-major flatten of each 4×4
    """
    F = c2w.shape[0]
    assert c2w.shape[1:] == (4, 4)
    with open(path, 'wb') as f:
        f.write(b"CAM4")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", F))
        for v in intrinsics: f.write(struct.pack("<f", float(v)))
        f.write(c2w.astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-tracks-dir", default=ALL_TRACKS_DEFAULT)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--name", default="hot3d_clip1995_clip1996")
    ap.add_argument("--caption", default="HOT3D cross-clip stitch · clip-001995 → clip-001996 (16k pts/object)")
    ap.add_argument("--target-height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--force-random-scene-pc", action="store_true",
                    help="Overwrite the structured stride-sampled scene PC "
                         "(built by regen_hot3d_dense_scene_pc.py) with the "
                         "canonical random sub-sample. Off by default — re-runs "
                         "of this script preserve the structured PC.")
    ap.add_argument("--apply-half-cut", action="store_true",
                    help="Drop per-frame per-object far-side points whose "
                         "projection onto the camera→centroid axis lies "
                         "behind the object's midline. Originally added to "
                         "compensate for vis_clips_backproject's 10%% depth-"
                         "tolerance leaking back-of-mesh points through the "
                         "self-occlusion test in random-sample mode. "
                         "Pixellift takes ray first-hit so every point is "
                         "already on the camera-facing surface — the half-"
                         "cut then drops legitimate visible points and the "
                         "object reads as 'sliced'. Off by default.")
    args = ap.parse_args()

    out_data  = args.out_dir / "static" / "data";  out_data.mkdir(parents=True, exist_ok=True)
    out_video = args.out_dir / "static" / "videos"; out_video.mkdir(parents=True, exist_ok=True)

    # ── Load dense per-clip tracks ──────────────────────────────────
    tracks_root = Path(args.all_tracks_dir)
    d3A = np.load(tracks_root / f"{CLIP_A}_3d.npz")
    d2A = np.load(tracks_root / f"{CLIP_A}_2d.npz")
    d3B = np.load(tracks_root / f"{CLIP_B}_3d.npz")
    d2B = np.load(tracks_root / f"{CLIP_B}_2d.npz")
    pts3_A = d3A["points_3d"]; vis3_A = d3A["visibility"].squeeze()
    tracks_A = d2A["tracks"];   vis2_A = d2A["visibility"]
    pts3_B = d3B["points_3d"]; vis3_B = d3B["visibility"].squeeze()
    tracks_B = d2B["tracks"];   vis2_B = d2B["visibility"]
    print("source: A", pts3_A.shape, "B", pts3_B.shape)
    assert pts3_A.shape[0] % N_OBJ == 0, pts3_A.shape
    pts_per_obj = pts3_A.shape[0] // N_OBJ
    print(f"  → {N_OBJ} objects × {pts_per_obj} pts/object")

    # All tracks; deterministic source order (same per-object slice for A and B).
    K_total = pts_per_obj * N_OBJ
    sub_idx = np.arange(K_total, dtype=np.int64)
    obj_ids = np.repeat(np.arange(N_OBJ, dtype=np.uint8), pts_per_obj)

    # ── Camera poses + anchor-frame transforms ──────────────────────
    T_we_A0       = get_T_we("c1995", 0)
    T_we_A_anchor = get_T_we("c1995", T_A_START)
    T_we_A = [get_T_we("c1995", t) for t in range(T_A_START, T_A_END + 1)]
    T_we_B0 = get_T_we("c1996", 0)
    T_we_B  = [get_T_we("c1996", t) for t in range(T_B_START, T_B_END + 1)]
    M_anchor_from_world = diag4(R_ROT90) @ np.linalg.inv(T_we_A_anchor)
    M_anchor_from_A0    = M_anchor_from_world @ T_we_A0 @ diag4(R_ROT90.T)
    M_anchor_from_B0    = M_anchor_from_world @ T_we_B0 @ diag4(R_ROT90.T)

    # ── Stitched per-frame data ─────────────────────────────────────
    # `annotated_arr` is True at frames where the object's pose was in the
    # source clip's `*.objects.json` — i.e., a real 3D position is available
    # (whether or not the point is camera-facing / inside the modal mask).
    # vis_clips_backproject leaves `points_3d` at zero for non-annotated
    # frames, so all-zero rows distinguish missing-from-annotations from
    # 'annotated but invisible'. v2 gt3d.bin stores positions for all
    # annotated frames so the viewer can render a smooth trail through
    # occlusion gaps; the cloud renderer separately respects vis_arr.
    F = (T_A_END - T_A_START + 1) + (T_B_END - T_B_START + 1)
    obj_arr  = np.zeros((K_total, F, 3), dtype=np.float32)
    vis_arr  = np.zeros((K_total, F), dtype=bool)
    annot_arr = np.zeros((K_total, F), dtype=bool)
    cam_pos = np.zeros((F, 3), dtype=np.float32)
    for fi, t in enumerate(range(T_A_START, T_A_END + 1)):
        P = pts3_A[sub_idx, t, :]
        obj_arr[:, fi] = transform(M_anchor_from_A0, P).astype(np.float32)
        vis_arr[:, fi] = vis3_A[sub_idx, t]
        annot_arr[:, fi] = (P != 0).any(axis=1)
        cam_h = np.append(T_we_A[fi][:3, 3], 1.0)
        cam_pos[fi] = (M_anchor_from_world @ cam_h)[:3]
    for j, t in enumerate(range(T_B_START, T_B_END + 1)):
        fi = (T_A_END - T_A_START + 1) + j
        P = pts3_B[sub_idx, t, :]
        obj_arr[:, fi] = transform(M_anchor_from_B0, P).astype(np.float32)
        vis_arr[:, fi] = vis3_B[sub_idx, t]
        annot_arr[:, fi] = (P != 0).any(axis=1)
        cam_h = np.append(T_we_B[j][:3, 3], 1.0)
        cam_pos[fi] = (M_anchor_from_world @ cam_h)[:3]

    # ── Per-track RGB sampled at the FIRST frame each track becomes
    #    visible. Held constant for that track across all frames so the
    #    cloud reads as one consistent surface even when a particular
    #    track is invisible at the anchor (we no longer leave those grey).
    #    This requires per-frame RGB lookups — load both clips' video
    #    frames into memory once (clip-A frames 84..149 + clip-B frames
    #    0..39, ≈ 106 frames at 1408×1408 = ~630 MB raw, manageable). ──
    print("loading per-frame RGB for per-track colour sampling...")
    def load_clip_frames(clip, fr_start, fr_end):
        cap = cv2.VideoCapture(RGB_TPL.format(clip=clip))
        out = []
        for f in range(fr_start, fr_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            if not ok: raise RuntimeError(f"failed to read {clip} frame {f}")
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return out
    rgb_A = load_clip_frames(CLIP_A, T_A_START, T_A_END)
    rgb_B = load_clip_frames(CLIP_B, T_B_START, T_B_END)
    print(f"  loaded {len(rgb_A)} A-frames + {len(rgb_B)} B-frames")

    # For each track, find the FIRST stitched frame where it's visible
    # AND has finite uv, then sample THAT frame's RGB at that uv.
    rgb_u8 = np.full((K_total, 3), 128, dtype=np.uint8)   # neutral fallback
    F_A = T_A_END - T_A_START + 1
    for k in range(K_total):
        # Walk stitched frames 0..F-1 in order:
        #   stitched f in [0, F_A) → clip A frame T_A_START + f
        #   stitched f in [F_A, F) → clip B frame T_B_START + (f - F_A)
        found = False
        for f in range(F):
            if not vis_arr[k, f]: continue
            if f < F_A:
                src_f = T_A_START + f
                uv = tracks_A[src_f, sub_idx[k]]
                vis2 = vis2_A[src_f, sub_idx[k]]
                img = rgb_A[f]
            else:
                src_f = T_B_START + (f - F_A)
                uv = tracks_B[src_f, sub_idx[k]]
                vis2 = vis2_B[src_f, sub_idx[k]]
                img = rgb_B[f - F_A]
            if not vis2: continue
            u_raw, v_raw = float(uv[0]), float(uv[1])
            if not (np.isfinite(u_raw) and np.isfinite(v_raw)): continue
            if not (0 <= u_raw < W_SRC - 1 and 0 <= v_raw < H_SRC - 1): continue
            u0 = int(np.floor(u_raw)); v0 = int(np.floor(v_raw))
            du = u_raw - u0; dv = v_raw - v0
            rgbf = img.astype(np.float32)
            c = (rgbf[v0, u0]   * (1 - du) * (1 - dv)
               + rgbf[v0, u0+1] *      du  * (1 - dv)
               + rgbf[v0+1, u0] * (1 - du) *      dv
               + rgbf[v0+1, u0+1] *    du  *      dv)
            rgb_u8[k] = np.clip(c, 0, 255).astype(np.uint8)
            found = True
            break
        # if `found` is False, the track was never visible AND in-bounds
        # in this 106-frame window — leave at neutral grey.
    print(f"  per-track colour sampled (fallback grey for "
          f"{(rgb_u8 == 128).all(axis=1).sum()} never-visible tracks)")

    # ── Per-frame mean object centre (drives the synthesised camera fwd) ──
    mean_obj = np.zeros((F, 3), dtype=np.float32)
    for fi in range(F):
        m = vis_arr[:, fi]
        mean_obj[fi] = obj_arr[m, fi].mean(axis=0) if m.any() else cam_pos[fi]
    c2w_per_frame = build_c2w_per_frame(cam_pos, mean_obj)

    # ── Camera-side half cut (opt-in via --apply-half-cut) ──────────
    if args.apply_half_cut:
        print("applying camera-side half-cut to vis_arr...")
        cs_dropped = 0
        for fi in range(F):
            for obj in range(N_OBJ):
                obj_visible = np.where(vis_arr[:, fi] & (obj_ids == obj))[0]
                if len(obj_visible) < 3: continue
                pts = obj_arr[obj_visible, fi]
                centroid = pts.mean(axis=0)
                view = (centroid - cam_pos[fi]).astype(np.float32)
                n = np.linalg.norm(view)
                if n < 1e-6: continue
                view /= n
                proj = (pts - centroid) @ view
                far_mask = proj > 0.0
                if far_mask.any():
                    vis_arr[obj_visible[far_mask], fi] = False
                    cs_dropped += int(far_mask.sum())
        print(f"  dropped {cs_dropped} far-side points across all frames")
    else:
        print("camera-side half-cut: SKIPPED (use --apply-half-cut to enable). "
              "Pixellift first-hit rays already give camera-facing surface "
              "samples, so showing every vis-flagged point.")

    # ── Stitched mp4 ───────────────────────────────────────────────
    mp4_dst = out_video / f"{args.name}.mp4"
    stitch_mp4(mp4_dst, target_height=args.target_height, fps=args.fps)

    # Chrono = last frame of stitched mp4
    cap = cv2.VideoCapture(str(mp4_dst))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    ok, bg = cap.read(); cap.release()
    chrono_path = out_video / f"{args.name}_chrono.jpg"
    cv2.imwrite(str(chrono_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # ── Scene PC + per-pt distance side-channel ────────────────────
    # If the structured stride-sampled scene PC is already in place
    # (built by build/regen_hot3d_dense_scene_pc.py), keep it — that
    # file lives in the viewer repo and reflects MoGe-stride sampling
    # which is way better than the visual_example random sub-sample.
    # Otherwise fall back to the canonical (random) scene.bin.
    pc_path      = out_data / f"{args.name}_pc.bin"
    pc_dist_path = out_data / f"{args.name}_pc_dist.bin"
    structured_present = pc_path.exists() and pc_dist_path.exists()
    if structured_present and not args.force_random_scene_pc:
        scene_n = struct.unpack("<I", pc_path.open("rb").read(4))[0]
        print(f"keeping existing structured scene PC at {pc_path} (N={scene_n})")
    else:
        scene = np.fromfile(SCENE_DIR / "scene.bin", dtype=np.float32).reshape(-1, 6)
        write_pc_binary(pc_path, scene[:, :3], scene[:, 3:6])
        min_pix_dist = np.fromfile(SCENE_DIR / "min_pix_dist.bin", dtype=np.float32)
        pc_dist_path.write_bytes(min_pix_dist.astype(np.float32).tobytes())
        scene_n = scene.shape[0]
        print(f"wrote canonical (random) scene PC at {pc_path} (N={scene_n})")

    # ── GT3D bin (positions int16-quantized, RGB uint8, obj_ids uint8) ──
    # GitHub Pages has a 50 MB soft per-file limit (the warning we hit on
    # earlier pushes). Single-file at 16k pts/obj × 6 obj × 106 frames
    # comes out to ~58 MB — over the limit and known to "Failed to fetch"
    # on some browsers. We write the bin then SPLIT byte-wise into two
    # ~30 MB parts that the viewer concatenates back together. Splitting
    # at an arbitrary byte boundary is safe since the loader reassembles
    # the full buffer before parsing the GT3D layout.
    gt3d_path = out_data / f"{args.name}_gt3d.bin"
    gt3d_meta = write_gt3d_binary(gt3d_path, obj_arr, vis_arr, annot_arr,
                                   obj_ids, rgb_u8, N_OBJ)
    gt3d_urls = [f"static/data/{args.name}_gt3d.bin"]
    if gt3d_path.stat().st_size > 49_000_000:
        data = gt3d_path.read_bytes()
        mid  = len(data) // 2
        a_path = out_data / f"{args.name}_gt3d_a.bin"
        b_path = out_data / f"{args.name}_gt3d_b.bin"
        a_path.write_bytes(data[:mid])
        b_path.write_bytes(data[mid:])
        gt3d_path.unlink()
        gt3d_urls = [f"static/data/{args.name}_gt3d_a.bin",
                     f"static/data/{args.name}_gt3d_b.bin"]
        print(f"split gt3d into {a_path.name} ({a_path.stat().st_size//1024//1024} MB) "
              f"+ {b_path.name} ({b_path.stat().st_size//1024//1024} MB)")

    # ── Camera bin (per-frame 4×4 + intrinsics) ────────────────────
    cam_path = out_data / f"{args.name}_cam.bin"
    write_cam_binary(cam_path, c2w_per_frame, [FX, FY, CX, CY])

    # ── Bundle JSON (metadata + URLs only) ─────────────────────────
    config = {
        "obj_name": "hot3d_objects",
        "t": 0,
        "hist_frames":   list(range(0, F // 2)),
        "future_frames": list(range(F // 2, F)),
        "all_frames":    list(range(F)),
        "n_hist": F // 2,
        "l2": 0.0,
        # gt_3d / pred_3d / pt_colors_rgb / vis come from gt3d.bin at load time.
        "gt_3d": None, "pred_3d": None, "vis": None,
        "gt_2d": None, "pred_2d": None,
        "pt_colors_rgb": None,
        "color_sample_frame": 0,
    }
    bundle = {
        "configs": [config],
        "mse": 0.0, "l2": 0.0, "n_configs": 1,
        "num_frames": F, "fps": float(args.fps),
        "video_fps_mult": 1,
        "caption": args.caption,
        "raw_meta": {
            "n_points": K_total, "n_frames": F,
            "video_dim_HW": [args.target_height, args.target_height],
            "src_clip_start": 0, "src_clip_end": F - 1,
            "source_3d": str(tracks_root),
            "note": f"HOT3D cross-clip; {N_OBJ} obj × {pts_per_obj} pts (dense).",
        },
        "chrono": {
            "image_url": f"static/videos/{args.name}_chrono.jpg",
            "frame_indices": [F - 1], "mode": "last_frame_only", "dilate_px": 0,
        },
        "pc_bin": {
            "url": f"static/data/{args.name}_pc.bin",
            "n_points": int(scene_n),
            "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
            "n_concat_frames": 1, "subsample": 1,
            "frame_indices_original": [T_A_START],
        },
        "pc_dist_bin": {
            "url": f"static/data/{args.name}_pc_dist.bin",
            "n_points": int(scene_n),
            "format": "float32 N",
        },
        "gt3d_bin": {
            # If split, the viewer fetches each url and concatenates the
            # ArrayBuffers in order before parsing. Single-file bundles
            # still set `url` for backward compatibility.
            "url": gt3d_urls[0] if len(gt3d_urls) == 1 else None,
            "urls": gt3d_urls,
            **gt3d_meta,
        },
        "cam_bin": {
            "url": f"static/data/{args.name}_cam.bin",
            "format": "CAM4 v1: magic | uint32 v | uint32 F | float32×4 intrinsics | float32 F*16 c2w",
            "n_frames": F,
        },
        "camera": {
            "intrinsics_frame0": [FX, FY, CX, CY],
            "video_stem": args.name,
            "video_dim_HW": [args.target_height, args.target_height],
        },
        "viewer_defaults": {
            "objMaskRadiusPx": 30,
            "objectCloud": True,
            # Cloud-dominant default (~12 px on a 480-tall panel) so the
            # surface cloud reads as the primary visualization, with the
            # trails behind acting as supporting motion-direction cues.
            "objCloudPointPx": 12,
            "showHist": False, "showPred": False,
            # Static-object surface point clusters were showing up as
            # 'dots on the trail' once the cloud got dense; the trail-
            # ball + endpoint markers are not used on the HOT3D
            # representation either, so suppress them entirely.
            "showBalls": False, "showEndpoints": False,
            "trackSmoothWindow": 1,
            "maxSceneDepth": 0.85,
        },
    }
    out_json = out_data / f"{args.name}.json"
    out_json.write_text(json.dumps(bundle, indent=2))

    print(f"\nwrote {out_json}     ({out_json.stat().st_size//1024} KB)")
    if gt3d_path.exists():
        print(f"wrote {gt3d_path}     ({gt3d_path.stat().st_size//1024//1024} MB, "
              f"{K_total} pts × {F} frames Int16-quantized)")
    else:
        total_mb = sum((out_data / Path(u).name).stat().st_size for u in gt3d_urls) // 1024 // 1024
        print(f"wrote split gt3d ({len(gt3d_urls)} chunks, {total_mb} MB total, "
              f"{K_total} pts × {F} frames Int16-quantized)")
    print(f"wrote {cam_path}      ({cam_path.stat().st_size} B)")
    print(f"wrote {mp4_dst}       ({mp4_dst.stat().st_size//1024} KB, {F} frames @ {args.fps} fps)")
    print(f"wrote {pc_path}       ({pc_path.stat().st_size//1024//1024} MB, {scene_n} pts)")
    print(f"wrote {pc_dist_path}  ({pc_dist_path.stat().st_size//1024} KB)")
    print(f"wrote {chrono_path}   ({chrono_path.stat().st_size//1024} KB)")

    # Quick visibility debug
    print(f"\nVISIBILITY: total {K_total} tracks across {N_OBJ} objects")
    for fi in [0, 1, 5, 30, 53, 65, 66, 100, F - 1]:
        per_obj = np.zeros(N_OBJ, dtype=int)
        for k in range(K_total):
            if vis_arr[k, fi]: per_obj[obj_ids[k]] += 1
        print(f"  frame {fi:3d}: total={vis_arr[:, fi].sum():5d}  per-obj={per_obj.tolist()}")


if __name__ == "__main__":
    main()
