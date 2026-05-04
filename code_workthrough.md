# Code walkthrough — motion-teaser-viz

The whole viewer is **one self-contained `index.html`** (≈6500 lines:
CSS + HTML + ES-module JS). Data prep is a sibling Python `build/`
directory. There is **no bundler / no build step** — three.js is loaded
from a CDN via an import map.

## File layout
```
motion-teaser-viz/
├─ index.html                  # the entire viewer (self-contained)
├─ README.md                   # paper-style + run instructions
├─ high_level_idea.md          # this project's "what / why"
├─ code_workthrough.md         # this file
├─ problem.md                  # mistakes & solutions log
├─ build/                      # data-prep Python scripts (CLI)
│  ├─ prepare_clip.py
│  ├─ prepare_clip_simple.py
│  ├─ prepare_full_video.py
│  ├─ prepare_hdepic.py
│  ├─ prepare_hot3d.py
│  ├─ build_scene_moge_lastframe.py
│  ├─ build_scene_monst3r.py
│  ├─ rebuild_pc_droid_dense.py
│  ├─ regen_hot3d_dense_scene_pc.py
│  ├─ regen_hot3d_dense_tracks.py
│  ├─ extract_hires_frames.py  # paper-quality strip frames (EgoDex+HOT3D)
│  └─ pick_best_clip.py
├─ static/
│  ├─ data/                    # JSON + .bin per clip
│  └─ videos/                  # mp4 + chrono jpg + (hi-res) frame-strip jpgs per clip
└─ tmp/                        # scratch tests, screenshots, picker tests
```

## Bundle contract (per-clip JSON + binaries)
Every prep script emits the same shape (some fields optional):
```jsonc
{
  "configs": [{                  // 1+ tracked-object configs per clip
    "obj_name": "soy sauce",
    "t":   29,                   // start frame (in source video)
    "n_hist": 3,
    "hist_frames":   [int, ...], // length n_hist
    "future_frames": [int, ...], // length 30 typical
    "all_frames":    [int, ...], // history + future, after remap to 0..T-1
    "gt_3d":   [F][P][3],        // ground-truth 3D positions
    "pred_3d": [F][P][3],        // model prediction
    "vis":     [F][P]bool,       // visibility mask
    "gt_2d":   [F][P][2],        // image-space coords (px)
    "pred_2d": [F][P][2],
    "pt_colors_rgb": [P][3],     // sampled RGB at color_sample_frame
    "color_sample_frame": int,
    "l2": float
  }, ...],
  "num_frames": 33,    "fps": 15,    "video_fps_mult": 1,
  "caption": "...",
  "mse": float,        "l2": float,
  "raw_2d": [F][R][2], "raw_3d": [F][R][3],
  "raw_meta":  { n_points, n_frames, video_dim_HW, src_clip_start, src_clip_end, source_2d, source_3d, note },
  "chrono":    { image_url, frame_indices, mode, dilate_px },
  "pc_bin":    { url, n_points, format, n_concat_frames, subsample, frame_indices_original },
  "camera":    { c2w_frame0, intrinsics_frame0, video_stem, /* opt */ c2w_per_frame, intrinsics_per_frame },

  // Optional extras
  "full_video":          { url, n_frames, fps, src_clip_start, src_clip_end, video_dim_HW },
  "full_pc_bin":         { url, n_points, format, frame_index_original, subsample },
  "full_video_2d / _3d / _raw_2d / _raw_3d": { tracks, n_points, n_frames, ... },

  // Hi-res image-frame strips (EgoDex + HOT3D only — built by extract_hires_frames.py).
  // Viewer prefers these over seeking the low-res mp4 when present.
  "clip_frames_hires":       [{ url, frame_idx_clip, frame_idx_src_30fps?, src_clip?, frame_idx_src? }, …],
  "full_video_frames_hires": [{ url, frame_idx_full, frame_idx_src_30fps, t_sec }, …],

  // HOT3D-only
  "gt3d_bin":   { url(s) },              // per-track [F,3] int16 + RGB + obj_id
  "cam_bin":    { url },                 // per-frame 4×4 c2w + intrinsics
  "pc_dist_bin":{ url },                 // per-PC-point 2D-track-distance (px)

  // Viewer behaviour overrides
  "viewer_defaults": { /* mutates Settings + DOM at load time */ }
}
```

### `.bin` formats
- **`*_pc.bin`** (scene PC):
  `uint32 N · float32 positions[N*3] · uint8 colors[N*3]`
  → `loadPCBinary` (line 3890) inflates to `{ N, positions: Float32, colors: Float32 in [0,1] }`.
- **`*_gt3d_a/b.bin`** (HOT3D dense object tracks): see `loadGT3DBinary`
  (line 3915). Int16-quantized per-frame XYZ + per-track RGB + obj-id +
  per-frame visibility mask. Multi-chunk via `urls: [...]`.
- **`*_cam.bin`** (HOT3D per-frame camera): see `loadCamBinary` (line
  4012). Per-frame 4×4 c2w + intrinsics.

## Boot flow (`init`, line 4236)
1. `fetch(CLIP_URL)` → `clipData`. Resolve `cfg = configs[?obj=…] || configs[0]`.
2. `populateObjSelector` (multi-config bundles).
3. `applyBundleViewerDefaults(clipData)` (1st pass: mutates `Settings`+inputs).
4. `precomputeSmoothCamera` — gaussian-smooths c2w positions/forwards for
   the frustum trail.
5. `snapshotOriginalTracks` → `applyTrackSmoothing` (idempotent rebuild
   from snapshot per slider change).
6. Hydrate binaries: `_altPC`, `_pc`, `_pcExtras`, HOT3D `_gt3d` + `cam`,
   `_pcDist`. Fallback to inline `pc_xyz`/`pc_colors` if no `.bin`.
7. `filterObjectPC` — k-th-NN-within-R rule (defaults R=0.10 m, K=4)
   → `_pcObject` for panel ⑤.
8. Compute curated indices `cfg._goodIdx` (live), `cfg._goodIdxPaper`
   (paper-mode tail-clean). HOT3D auto-routes to `selectSpatialIdx`
   (3D K-means / FPS).
9. Build header caption + frame-count meta.
10. Hook `<video>` src + chrono img.
11. Instantiate **panels** (line 4442): `canvas-all`, `canvas-gt`,
    `canvas-pred`, `canvas-cmp`, `canvas-pcrgb` (frame-0-only object
    cloud), `canvas-paper` (static), and conditionally `canvas-altpc`
    (full-video PC composite).
12. 2nd `applyBundleViewerDefaults` pass — propagates layer toggles into
    `panel.opts`.
13. `panels.forEach(p => p.buildStatic(clipData, cfg))` — one-shot
    static-layer build (PC, ground grid, frustum, picker spheres).
14. Paper-mode panels rendered once at the last frame; live panels are
    updated inside `tick()` whenever the integer frame index changes.
15. `bindControls()` — wires up the entire sidebar (sliders, colour
    pickers, palette/preset buttons, layer toggles, eraser, …).

## Per-frame tick (line 4520)
```
processKeyboardCamera(now)        // WASD + E/C; mutates camState
fi = round(video.currentTime * fps).clamp(start, end)
if fi !== App._lastFi:
    update labels + scrub bar
    panels (non-paperMode).updateDynamic(cfg, fi).render()
    drawVideoOverlay(fi)
requestAnimationFrame(tick)
```
Only an **integer frame change** triggers re-render — playhead /
trail-ball positions never drift between sub-frame `tick`s.

## `class Panel` (line 2046)
```
constructor(canvasId, opts)              // 2047
_setupInput()                            // 2112  drag/scroll/click/eraser
_eraserBuildCache() / _eraseAt(e)        // 2161+ pre-projected screen cache
applyCamera()                            // 2256  syncs camState → camera
_resizeRenderer()                        // 2266
render()                                 // 2279  applies sceneRotX/Y/Z then renders
buildStatic(clipData, cfg)               // 2368  PC, frustum, ground grid, init dots
_buildTrailLayer(pts, fStart, fEnd, ...) // 2550  vertex-coloured Line2 + balls
updateDynamic(cfg, fiFrac)               // 2867  per-frame trails + spheres
                                         //       + HOT3D object cloud + ghost
                                         //       frames + per-frame camera frustum
```
Each panel owns `staticGroup` (renderOrder 0 — PC) and `dynGroup`
(renderOrder 2 — everything that should overlay). PC materials use a
custom shader (`makeDiscMaterial`, line 1512) for round, anti-aliased
discs that respect renderOrder via `forceTransparent`.

## Index-selection algorithms (line 1577–1955)
- `pointMaxJumpScores(pts, fStart?, fEnd?)` — per-track max inter-frame
  Δ-norm over a window. Forms the basis for every smoothness / jumpiness
  ranking.
- `selectSmoothestIdx` / `selectJumpiestIdx` / `selectMixedRawIdx`
  (mostly cross-section + N forced-jumpy) / `selectRandomIdx`
  (`_seededRand`/`_mulberry32`) / `selectUniformIdx`.
- `selectIdxByStrategy(strategy, scores, n_keep, n_jumpy)` — single
  dispatch the rest of the code calls.
- **`selectSpatialIdx` / `kmeansSelectIdx`** (HOT3D-only) — 3D K-means /
  FPS over the moving object's frame-0 cloud, capped at
  `SPATIAL_MAX_POOL=4000`. Auto-elevates the dropdown when `_gt3d`
  is present (line 4366).

## Trail rendering (`_buildTrailLayer`, line 2550)
- Catmull-Rom samples (`smoothCurve3`, `SMOOTH_SAMPLES`) between consecutive
  GT frames so polylines read as smooth curves, not jagged segments.
- Vertex colours come from one of two modes (per role, branched on the
  shared `_useColormapFor(role)` helper):
  - **two-colour** (legacy): lerp `oldEnd → newEnd` (the live colour pickers
    in the Layers section) along arc length (`trailGradientByArcLen=true`)
    or frame index, with `mixLo` (pale floor) and `tExp` (curve) shaping `t`.
  - **colormap LUT**: sample `COLORMAP_LUTS[name]` at the same `t` (after
    `mixLo` / `tExp` shaping). 8 palettes shipped: matplotlib's
    *magma · inferno · plasma · viridis · cividis · turbo* + seaborn's
    *rocket · mako*. Per-role `Settings.colorPresetReverse[role]` flips the
    lookup direction. Replaces the old single-toggle rainbow mode entirely.
- **Trail balls**: `InstancedMesh` of `SphereGeometry` placed every
  `TRAIL_BALL_FRAME_STRIDE` frames. Radius scales pale-min → vivid-max
  (`TRAIL_BALL_R_MIN/MAX_*`) so the size encodes time alongside colour.
  Colour selection follows the same per-role colormap branch as the lines.
- Pred uses a dashed `LineMaterial` so it still reads against GT under
  full overlap.

## Two-pass viewer defaults
`applyBundleViewerDefaults` runs twice: once before panel construction
(to mutate `Settings` + DOM input values so the first computation is
correct) and once after (to propagate per-panel layer toggles). HOT3D
sets `objectCloud=true`, hides pred, etc. via this hook; the *_lastframe
HOT3D bundle adds `staticObjFrameMode='last'` and
`skipStaticObjCloud=true`.

## Sidebar accordion → settings map
Almost every section of the sidebar maps 1-to-1 onto a key on the
`Settings` object. `bindControls()` (line 5417) installs `change/input`
listeners that set the field then call one of:
- `rerenderAll()`              — cheap re-draw with current dynamic state
- `rebuildAllStatic()`         — rebuild PC/frustum geometry on each panel
- `recomputeGtIdx(n)` / `recomputeRawIdx()` — refresh the curated picks
- `applyTrackSmoothing(clipData)` — rebuild gt_3d/pred_3d/raw_3d from snapshot
- `applyPalette(name)` / `_applyPreset(p)` — bulk-set settings from a preset

The `__MTV` debug hook on `window` exposes live `Settings`, `cfg`,
`clipData`, `panels`, `recomputeGtIdx`, `selectSpatialIdx` so a browser
console / smoke-test can introspect without reaching inside the module.

## Build scripts (`build/`)
All scripts share the same *bundle contract* output. Common steps:
1. Read source motion5 JSON (or HOT3D / HD-EPIC native format).
2. Trim mp4 to the clip window via ffmpeg.
3. Re-index `gt_2d / pred_2d / *_frames` so frame 0 is the first cut frame.
4. Sample `pt_colors_rgb` from frame 0 of the trimmed mp4.
5. Build a depth-backprojected scene PC (subsample=3) — or load a dense
   alternative (MoGe / MOnST3R for HOT3D & EgoDex variants).
6. Render the chronophotography composite (`min` blend keeps moving
   foreground sharp; rest of the pipeline lives in
   `prepare_clip.py:build_chrono`).
7. Write `static/data/<clip_id>.json` + `static/data/<clip_id>_pc.bin` +
   `static/videos/<clip_id>.mp4` + `_chrono.jpg` (and any extras).

Per-dataset specifics:
- **`prepare_hot3d.py`** quantises per-track per-frame XYZ to int16 and
  emits `_gt3d_a.bin / _gt3d_b.bin` (chunked) + `_cam.bin` + an optional
  `_pc_dist.bin` (per-PC-point distance to nearest 2D track at frame 0,
  used by the object-mask-radius slider).
- **`prepare_hdepic.py`** ships multiple `configs` (one per tracked
  object) and a `pc_extras` list (frame-N depth backprojections).
- **`build_scene_moge_lastframe.py`** swaps in a MoGe lifting of the
  *last* frame as the scene PC; viewer pins static-object pose to the
  last frame to match.
- **`build_scene_monst3r.py`** uses MOnST3R for the scene reconstruction
  variant.
- **`extract_hires_frames.py`** runs *after* the dataset-specific prep
  scripts. EgoDex: maps each strip frame's vipe-domain index back to its
  30-fps source mp4 (`src_30fps_fi = 2·vipe_fi`) and pulls from
  `/weka/.../egodex/<part>/<task>/<idx>.mp4` (1920×1080). HOT3D: walks the
  cross-clip stitch table (`HOT3D_STITCH`) so each viewer frame resolves to
  the right `clip-NNNNNN_rgb.mp4` source frame at 1408×1408 (no re-encode).
  In-place patches the JSON to add `clip_frames_hires` /
  `full_video_frames_hires` arrays.

## Hi-res screenshot capture
- `Panel.capture(scale)` (line ~2376) renders the panel into a backing
  buffer of `cssSize × scale`, returning a PNG `Blob`. Steps:
  1. Save current pixel-ratio + canvas CSS box.
  2. `setPixelRatio(1)` + `setSize(W, H, false)` so only the drawing buffer
     grows; keep the on-screen `<canvas>` CSS box untouched.
  3. Update each `lineMats[k].resolution` and the PC shader's `uViewport`
     uniform with the new pixel space — Line2 widths and disc point sizes
     scale 1:1 so paper figures look identical (just sharper) to the live view.
  4. Re-apply scene rotation, run `renderer.render(...)`, snapshot via
     `canvas.toBlob('image/png')`.
  5. Restore via `_resizeRenderer()` + `render()`.
- `WebGLRenderer` is now constructed with `preserveDrawingBuffer:true` so
  step 4's `toBlob` reads the just-rendered pixels reliably across browsers.
- Sidebar "Capture" section (in `bindControls`): panel selector + scale
  dropdown (1×, 2×, 4×, 8×) + "All panels" mode that walks every visible
  3D panel and downloads `<clip_id>_<canvas-id>_<scale>x.png`.
