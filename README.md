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
