# motion-teaser-viz

Single-clip 4D motion-prediction teaser visualizer for the MotionPlanner work.
Built as a clean rewrite of `motion5-viz` focused on **paper-quality teaser
figures** rather than batch dataset browsing.

Live demo: https://jianingnz.github.io/motion-teaser-viz/

---

## What's in the viewer

For one selected clip the page shows, all in sync:

1. **RGB video + 2D track overlay** — original mp4 with per-point dots colored
   by the object's actual RGB (sampled from frame 0). History trails are cyan,
   GT future is green, prediction is dashed orange. Pred is drawn as an open
   ring so it's visually distinct from GT even when they overlap.

2. **Chronophotography (ghost-trail) image + 2D tracks** — all sampled frames
   composited into one image (`min` blend keeps moving foreground sharper). On
   top, the *full* 2D track of every point is drawn from start of history to
   end of prediction. This single image is intended to work as a method-figure
   thumbnail: one frame that summarizes the whole clip.

3. **3D combined viewer** — large interactive scene with the sparse RGB point
   cloud + history + GT future + prediction trails + current-frame spheres in
   per-point object color. Drag to orbit, scroll to zoom.

4. **Four small panels** showing the same scene with progressive overlay:
   `history only` → `+ GT future` → `+ prediction` → `+ GT/Pred error vectors`.
   All five 3D panels share one camera, so dragging in any panel orbits all.

Toggle buttons at the bottom let you turn each layer on/off in the big panel,
or reset the camera.

## Color conventions

| element | color | meaning |
|---|---|---|
| Per-point sphere / 2D dot | sampled object RGB from frame 0 | object identity |
| History trail | cyan `#7dd3fc` | given context |
| GT future trail | green `#22e36b` | ground truth |
| Pred future trail | orange `#ff7e3c` (dashed) | model prediction |
| Error vector | thin red | GT↔Pred residual at current frame |

## Rendering choices vs. the original

- Round, anti-aliased point sprites via a custom `gl_PointCoord` shader
  (no more square `gl.POINTS`).
- Trail lines via `Line2` / `LineMaterial` (3 px) instead of `LineBasicMaterial`
  — WebGL ignores `linewidth>1` on `LineBasicMaterial`.
- Current-frame trajectory dots are real `InstancedMesh` `SphereGeometry` so
  they have proper depth + lighting and read as 3D from any viewing angle.
- Soft radial ground disc instead of a flat grid helper.
- Background is a vertical gradient.

## Build

```bash
# from a checkout of motion5-viz containing the source clip
PYTHON=/path/to/python  # needs cv2 + numpy
$PYTHON build/prepare_clip.py \
  --src-json <motion5-viz>/static/data/modeling_json/egodex/test/<clip_id>.json \
  --src-mp4  <motion5-viz>/static/videos/modeling/egodex/<video_stem>.mp4 \
  --out-dir  . \
  --clip-id  bowls_nest_t91
```

This produces:
```
static/data/<clip_id>.json    # source JSON enriched with pt_colors_rgb + chrono meta
static/videos/<clip_id>.mp4
static/videos/<clip_id>_chrono.jpg
```

The viewer loads `static/data/clean_surface_1713_t10.json` by default; the
header has a clip-selector dropdown for switching, or pass
`?clip=static/data/<other>.json` directly.

## Source clips used for the demo

| clip id | source | description |
|---|---|---|
| `clean_surface_1713_t10` | `part1_clean_surface_1713_obj0_t10` (EgoDex test) | default — wiping a surface |
| `basic_pick_place_14851_t29` | `part2_basic_pick_place_14851_obj2_t29` (EgoDex test) | basic pick-and-place |

Both bundles are 33-frame clips (3 history + 30 future) at 15 fps, with a
single-frame depth-backprojected sparse PC (subsample=3, ~45k points).

## Per-role colormap palette (paper figures)

Every gradient (3D trails, trail balls, HOT3D dense object-cloud trails, 2D
video/chrono overlays) supports two colour modes per role:

- **`twoColor`** — legacy `oldEnd → newEnd` lerp using the colour pickers in
  the Layers section. The pale floor (`mixLo`) and curve (`tExp`) sliders
  shape the gradient.
- **colormap LUT** — sample one of 8 perceptual colormaps at the same `t`.
  256-step LUTs sourced from
  [matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
  (*magma · inferno · plasma · viridis · cividis · turbo*) and
  [seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html#perceptually-uniform-palettes)
  (*rocket · mako*). LUT generated at build time via
  `cmap(np.linspace(0, 1, 256))`; viewer linearly interpolates.

Sidebar **Time gradient** section: dropdown + `⇄` reverse toggle for each of
GT, Pred, Raw, object-cloud trails. `mixLo` / `tExp` apply identically to
both modes — colormap mode honours the existing pale-floor and curve knobs.

## Hi-res image-frame strips (EgoDex + HOT3D)

Panel ⓪b (full-source) and panel ⑥ (cut-clip) render thumbnail strips. By
default they sample the in-page low-res mp4 (`<video>`-seek + `drawImage`),
which caps at 640×360 (EgoDex cut), 854×480 (EgoDex full), or 480×480 (HOT3D
viewer mp4). For paper figures we ship a build-time pre-extract:

```bash
$PYTHON build/extract_hires_frames.py \
  --clip-id clean_surface_1713_t10 \
  --clip-id hot3d_clip1995_clip1996 \
  --out-dir . \
  --n-clip 10 --n-full 14
```

Sources:
- **EgoDex** — original 1920×1080 @ 30 fps mp4 at
  `/weka/prior-default/jianingz/home/dataset/egodex/<part>/<task>/<idx>.mp4`.
  Bundles' `src_clip_start/end` are 15-fps vipe indices; we map back via
  `src_30fps_fi = 2 · vipe_fi` and clamp to the source mp4's frame count.
- **HOT3D** — 1408×1408 @ 30 fps `clip-NNNNNN_rgb.mp4` at
  `/weka/prior-default/jianingz/home/dataset/hot3d_repo/tmp/rgbs/`. Cross-clip
  bundles (e.g. `hot3d_clip1995_clip1996`) walk a `HOT3D_STITCH` table to
  resolve which source-clip each viewer-frame index belongs to.

The script writes JPEGs (q=92) into
`static/videos/<clip_id>_clipframes/NN.jpg` and `<clip_id>_fullframes/NN.jpg`,
then in-place patches the JSON with `clip_frames_hires` and (EgoDex only)
`full_video_frames_hires` arrays. Viewer auto-uses these when present; falls
back to `<video>`-seek when absent (HD-EPIC / DROID / DAVIS keep the
existing low-res path).

## Hi-res screenshot capture (paper figures from the 3D panels)

Sidebar **Capture** section. Picks any 3D canvas (`canvas-all`, `canvas-paper`,
`canvas-altpc`, `canvas-gt`, `canvas-pred`, `canvas-cmp`, `canvas-pcrgb`) or
"All panels"; scale dropdown 1×, 2×, 4× (≈4K from a HD-sized panel), 8×
(≈8K). Click 📷 Capture PNG and the browser downloads
`<clip_id>_<canvas-id>_<scale>x.png`.

Implementation in `Panel.capture(scale)`:

1. `setSize(W*scale, H*scale, /*updateStyle=*/false)` resizes the WebGL
   drawing buffer only. CSS box stays at `cssW×cssH px` so the on-screen
   layout is undisturbed.
2. `LineMaterial.resolution` and the PC shader's `uViewport` uniform are
   updated to the new pixel space — Line2 widths and disc point sizes scale
   1:1 with the buffer.
3. `renderer.render(scene, camera)` then `await canvas.toBlob('image/png')`.
   Works because `WebGLRenderer` is constructed with
   `preserveDrawingBuffer:true`.
4. Restore via `setPixelRatio(prevPR)` + `_resizeRenderer()` + `render()`.

WebGL `MAX_TEXTURE_SIZE` is the hard upper bound (typically 16384 on modern
GPUs). 4K and 8K are safe; 16K is borderline.
