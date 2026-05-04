# motion-teaser-viz — High-level idea

## Goal
Single-clip 4D motion-prediction **teaser visualizer** — paper-quality figures
and an interactive web demo for the MotionPlanner / 4D point-tracking work.
A clean rewrite of `motion5-viz` focused on showing **one clip really well**,
not browsing a dataset. Live demo:
[github.com/jianingnz/motion-teaser-viz](https://github.com/jianingnz/motion-teaser-viz)
(GitHub Pages → `https://jianingnz.github.io/motion-teaser-viz/`).

## What "one clip" means
A clip bundle is a triplet `{ JSON config, MP4 video, binary scene PC (.bin) }`,
sometimes with extras (chrono jpg, full-video mp4, full-video PC, HOT3D
gt3d/cam binaries, lastframe variants). Bundles live under `static/data/` +
`static/videos/`. The viewer fetches one JSON at boot (driven by
`?clip=<json>` or the dropdown), pulls the binaries it references, and lays
out 6+ synced panels.

## Panels (top → bottom)
| # | what | source |
|---|------|--------|
| ⓪ / ⓪b | full source mp4 + sampled-frame strip (with the cut clip bracketed on a timeline) | `clipData.full_video` |
| ① | RGB video + 2D track overlay (cut clip) | `cfg.gt_2d`, `cfg.pred_2d`, `clipData.raw_2d` |
| ② | chronophotography (motion ghost) image + full 2D tracks | `clipData.chrono.image_url` |
| ③ | big interactive 3D scene — scene PC + GT + pred + raw + camera frustum | scene `_pc`, `cfg.gt_3d`, `cfg.pred_3d`, `clipData.raw_3d` |
| ③b | **paper-figure** static composite — every frame at once, temporal gradient | same |
| ③c | alt-PC big panel — same combined view but scene PC = full-video frame 0 | `_altPC` from `full_pc_bin` |
| ④a/b/c | small synced 3D panels: GT only · Pred only · GT+Pred+error vectors | same as ③ |
| ⑤ | initial **object-only** PC (frame 0, k-NN-strict object filter) + tracked-point markers | `_pcObject` derived from `_pc` ∩ `cfg.gt_3d[0]` |
| ⑥ | raw RGB frame strip from the cut clip | mp4 sampled, **or pre-extracted hi-res jpgs** when `clipData.clip_frames_hires` is present (EgoDex 1920×1080 / HOT3D 1408×1408) |
| time legend | pale→vivid + small→large encoding key | — |

All 3D panels share **one camera state** (`camState`) — drag in any panel,
they all orbit. WASD pans, E/C raises/lowers, scroll zooms, `.` toggles the
sidebar.

## Visual encoding rules
- **Per-track identity** = colour sampled from the object RGB at frame 0
  (`pt_colors_rgb`) for the object-cloud / dot layers.
- **Time** = pale→vivid + small→large gradient along each trajectory's
  trail. Each role (GT / Pred / Raw / object-cloud) has an independent
  colour mode: **`twoColor`** (legacy oldEnd → newEnd lerp using the colour
  pickers) or one of 8 perceptual **colormap palettes** (matplotlib's
  *magma · inferno · plasma · viridis · cividis · turbo* + seaborn's
  *rocket · mako*). A per-role `reverse` flag flips the colormap direction.
  Stored as `Settings.colorPreset[role]` / `Settings.colorPresetReverse[role]`;
  LUT data lives in `COLORMAP_LUTS` (256-step `Uint8Array`s, ~22 KB inline).
  Gradient knobs `mixLo` (pale floor), `tExp` (curve), and
  `trailGradientByArcLen` apply identically to both modes.
- **Role = colour family**: GT (pink default `#f0529c`), Pred (mint
  `#34d399`, dashed), Raw (red, no smoothing), Endpoint (yellow `#fde047`).
  All user-pickable; presets in `PALETTES` (each preset can specify
  per-role colormap names too).
- **Pred is dashed** so it stays visually distinct from GT even when they
  overlap.

## Track curation
For each clip the viewer scores points by per-frame jumpiness
(`pointMaxJumpScores`) and selects subsets via swappable strategies:
`smoothest` (default GT), `jumpiest`, `random`, `uniform`, `mixed` (raw —
mostly cross-section + a forced-jumpy minority). HOT3D bundles auto-elevate
to a **`spatial`** strategy: 3D K-means / FPS over the moving object's
frame-0 surface for even visual coverage. A separate **paper-mode** GT
selection drops noisy tail frames (`paperGtTailDrop=9`) and re-ranks via
tail-cleanliness in a wider pool (`PAPER_GT_POOL_MULT=1.7`).
Manual override: the `<select multiple>` pickers (`gt-pick`, `raw-pick`).

## Bundle types observed
- **EgoDex** (clean_surface, basic_pick_place, part4_pour, …) — 33-frame
  clips (3 hist + 30 future @ 15 fps), depth-backprojected sparse PC
  (~45k pts), full_video + full PC available.
- **HD-EPIC** (P05/P06) — multi-object configs (`obj_name` selector
  visible), pour/scrub tasks.
- **DROID** (AUTOLab, GuptaLab, PennPAL) — wider clip range, JSON-fallback
  PC (no .bin).
- **DAVIS** (camel) — single object benchmark.
- **HOT3D** (clip1995→1996) — cross-clip stitched scene with **dense gt3d
  binaries** (`hot3d_*_gt3d_a/b.bin`, ~thousands of tracks per object) +
  per-frame camera bin. Renders as a per-frame coloured object cloud
  (`objectCloud`) with prediction-mode (frame 0 SOLID = "input", playhead
  + ghost frames at low opacity = "predicted future"). Two scene PC
  variants — first-frame MoGe and last-frame MoGe (`_lastframe.json`).

## Data prep (`build/`)
Each dataset has its own prep script that emits the same bundle contract:
- `prepare_clip.py` — generic EgoDex/DROID/DAVIS path
- `prepare_clip_simple.py` — minimal version
- `prepare_full_video.py` — adds full-video tracks + alt PC
- `prepare_hdepic.py` — HD-EPIC multi-object
- `prepare_hot3d.py` — HOT3D dense gt3d + cam.bin emission
- `build_scene_moge_lastframe.py`, `build_scene_monst3r.py` — alternate
  scene-PC sources (MoGe last-frame, MOnST3R)
- `rebuild_pc_droid_dense.py` — DROID dense PC rebuild
- `regen_hot3d_dense_*.py` — HOT3D track / scene PC regeneration

All produce: depth-backprojected scene PC (subsample=3 typical), per-frame
2D & 3D tracks, frame-0 object-RGB samples, chronophotography composite.

## Anatomy of `index.html` (single self-contained app)
- **lines 1–364**: CSS (light theme, sidebar, card grid, time legend).
- **lines 366–373**: import map → three.js + addons CDN.
- **lines 375–1152**: HTML body — header w/ clip & object selectors,
  cards for every panel, fixed-position `<aside class="sidebar">` controls.
- **lines 1154–6517**: ES module script.
  - 1162–1485: constants + `Settings` (live UI-mutated state).
  - 1493–1576: shared `camState` + custom round-disc PC shader.
  - 1577–1955: index-selection algorithms (jump scores, smoothest,
    jumpiest, mixed, uniform, random, **3D K-means / FPS spatial**).
  - 1957–2044: rainbow palette + Catmull-Rom smoothing.
  - 2046–3812: `class Panel` — one 3D viewer (constructor / input /
    eraser / `applyCamera` / `_resizeRenderer` / `render` / `buildStatic`
    / `_buildTrailLayer` / `updateDynamic`).
  - 3813–4234: app-level state + binary loaders (`loadPCBinary`,
    `loadGT3DBinary`, `loadCamBinary`) + `precomputeSmoothCamera` +
    `applyBundleViewerDefaults` + `applyTrackSmoothing`.
  - 4236–4555: `init()` — fetches JSON, hydrates binaries, instantiates
    panels, kicks the per-frame `tick()` loop driven by the cut-clip
    `<video>` element.
  - 4557–4960: full-video panel (separate timeline) + clip↔full-video
    point-index mapping for ③c.
  - 4961–6228: control bindings (`bindControls`, `bindPalettes`,
    `bindPresets`, `bindKeyboardCamera`, `bindTrajectoryPickers`,
    `bindFullVideoControls`, `bindPcSourceSelector`).
  - 6228–6517: sidebar open/close + palette/preset utilities + WASD pan +
    layer toggles.
- **`window.__MTV`** (line 3823): debug hook exposing live `Settings`,
  `cfg`, `clipData`, `panels`, `recomputeGtIdx`, `selectSpatialIdx`.

## Things that are easy to miss
- The `<video>` element drives the clock; `tick()` is `requestAnimationFrame`
  but the trajectory only re-renders on **integer** frame change, so paper
  mode (which is static) is updated exactly once at boot.
- Two render groups: `staticGroup.renderOrder=0` (PC) and
  `dynGroup.renderOrder=2` (trails / spheres). PC materials are pushed into
  the **transparent queue at α=1** via `forceTransparent` so renderOrder
  actually wins over a transparent line layer.
- HOT3D's `cfg.gt_3d` is a **sparse stub** — every frame is a shared empty
  array. Real data lives in `clipData._gt3d` (typed arrays). Code uses
  `if (!pt) continue` everywhere to handle this gracefully.
- `clipData._pcHidden : Set<number>` collects click-to-hide / eraser
  removals; every panel that draws the main PC reads it.
- Track-smoothing slider rebuilds `cfg.gt_3d / pred_3d` and `clipData.raw_3d`
  from a snapshot on every change so going back to "Original" is exact.

## Recent additions (2026-05-04)
- **Per-role colormap palette mode** (replaced the old single rainbow
  toggle). 8 LUTs (magma/inferno/plasma/viridis/cividis/turbo/rocket/mako)
  inlined as 256-step `Uint8Array`s; `colormapRgb01(t, name, reverse)` is the
  one helper called by every gradient site (3D trails, trail balls, HOT3D
  object-cloud trails, 2D video overlay, 2D raw overlay, time legend).
  Per-role dropdowns + reverse toggles in the sidebar's "Time gradient"
  section. Settings keys: `colorPreset.{gt,pred,raw,obj}` and
  `colorPresetReverse.{...}`.
- **Hi-res image-frame strips** for EgoDex + HOT3D paper figures. Build-time
  pre-extract via `build/extract_hires_frames.py` writes
  `static/videos/<clip_id>_clipframes/NN.jpg` (cut strip) and
  `_fullframes/NN.jpg` (full strip) at native source resolution
  (EgoDex 1920×1080 from `dataset/egodex/<part>/<task>/<idx>.mp4`; HOT3D
  1408×1408 from `dataset/hot3d_repo/tmp/rgbs/clip-NNNNNN_rgb.mp4`). The JSON
  bundle gains `clip_frames_hires` and `full_video_frames_hires` arrays;
  `buildRawThumbs` / `buildVideoFrameStrip` consume them when present and
  fall back to the low-res `<video>`-seek path otherwise. Defaults: 10 cut
  frames + 14 full frames (tunable via CLI flags).
- **Hi-res screenshot capture** for the three.js panels. `Panel.capture(scale)`
  resizes the WebGL drawing buffer to the on-screen size × scale, updates
  line/PC viewport uniforms so widths and disc sizes scale proportionally,
  renders, and returns a PNG blob. WebGL renderer now constructed with
  `preserveDrawingBuffer:true` so `canvas.toBlob` works after the render.
  Sidebar "Capture" section: panel selector + 1×/2×/4× (4K) / 8× (8K) scale +
  "All panels" mode that downloads one PNG per visible 3D panel.

## Status
Bundles deployed. Live demo on GitHub Pages. Ongoing tuning: paper-figure
knobs, HOT3D spatial picks, ghost-frame composition.
