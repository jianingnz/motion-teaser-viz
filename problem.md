# Mistakes / bugs / solutions log

> Append-only. Each entry: short title, what was wrong, why, the fix.
> Read this before debugging anything in this repo so we don't re-run
> the same wall-bumps.

## 2026-05-04 — extract_hires_frames.py emitted wrong JSON URL prefix
**Symptom**: After the first run, the bundle JSON's `clip_frames_hires[].url`
came out as `videos/<clip_id>_clipframes/00.jpg`. The viewer expects URLs
relative to the repo root (it serves `static/...` paths everywhere else),
so loading the page would 404 the strip frames.
**Root cause**: `write_jpgs` computed `path.relative_to(out_dir.parent.parent)`,
which evaluated to `out_dir / static / videos / ..` → only `videos/...`
because `out_dir` was already the repo root. The intent was repo-root
relative.
**Fix**: pass `repo_root` explicitly into `write_jpgs` and compute
`path.relative_to(repo_root)` (`build/extract_hires_frames.py:125-136`).
**Prevent**: when generating URLs for the JSON bundle, always derive them
from the explicit repo root, never from the output directory's `parent`s.

## 2026-05-04 — first EgoDex full-strip run failed on out-of-bounds frame
**Symptom**: `failed to read frame 302 of /weka/.../egodex/.../1713.mp4`
(the source mp4 has 300 frames; vipe-domain → 30-fps index calculation
generated 302).
**Root cause**: The vipe-domain `n_frames` from the bundle (152) maps to a
30-fps source range of `[0, 2·151] = [0, 302]`, but the source mp4 actually
has frame indices `[0, 299]`. We ignored that the source mp4 may run a
half-frame short of `2 × n_vipe`.
**Fix**: clamp the chosen 30-fps src indices to `[0, n_src - 1]` after
opening the source mp4 and reading `CV_CAP_PROP_FRAME_COUNT`
(`build/extract_hires_frames.py` — see `build_egodex_full_strip`).
**Prevent**: any time we map between fps domains for a finite-length mp4,
read the source's `CAP_PROP_FRAME_COUNT` and clamp before seeking.

## 2026-05-13 — 2D-track overlay drifted from the clip video
**Symptom**: On `insert_remove_utensils_534_t185` (and several other
EgoDex / DROID / DAVIS bundles) the 2D-track overlay in panel ① lagged
the video by a noticeable amount and never quite caught up before the
loop reset. Affected mp4s were ~1.5–4× too long: `num_frames`=24 but the
mp4 had 40 frames at 15 fps; `clean_surface_3603_t12` had 33 vs 142, etc.
**Root cause**: `build/prepare_clip.py`'s ffmpeg pipeline encoded with
`-vf select=...,setpts=PTS-STARTPTS,fps=15` but without `-fps_mode cfr`
or an explicit output `-r`. libx264 padded/duplicated frames during
encoding, so the mp4 had more frames than the trajectory, and the
viewer's `fi = round(currentTime * fps)` formula then drove the overlay
to a frame past the displayed mp4 frame.
**Fix**: Stopped depending on a "correctly encoded" mp4 in the viewer.
`index.html` now treats the mp4's full duration as the [0, T-1]
trajectory timeline (linear map), and sets `video.playbackRate` so each
trajectory frame plays for exactly 1/15 s wall-clock regardless of how
the mp4 was encoded — see the `loadedmetadata` handler around line 4877
and the `tick()` loop. The 2D overlay also now redraws every animation
frame using a fractional `fiFrac` so the current-frame dots glide
smoothly with the video (trails still snap on integer-frame boundaries).
**Prevent**: For new bundles, fix the encoder side too — `prepare_clip.py`
should pass `-fps_mode cfr -r {fps}` (or `-frames:v {n_clip}`) so the
mp4 always has exactly `num_frames` frames at the declared fps. The
viewer's linear-map sync is still the more robust contract.

## 2026-05-13 — Camera frustum didn't move on `insert_remove_utensils`
**Symptom**: On `insert_remove_utensils_534_t185` the camera frustum in
the 3D panels was static across the whole clip, while neighbouring EgoDex
bundles (`part4_pour_1027`) animated correctly.
**Root cause**: `build/prepare_clip.py` only stored `camera.c2w_frame0`
(a single 4×4) — not `camera.c2w_per_frame` — so
`precomputeSmoothCamera()` fell back to a 1-frame trajectory.
**Fix**: One-off `tmp/add_per_frame_camera.py` reads
`{vipe_root}/pose/{video_stem}.npz` + `intrinsics/...npz`, slices
[`src_clip_start`, `src_clip_end`] inclusive, and writes
`c2w_per_frame` + `intrinsics_per_frame` into the existing bundle JSON.
Applied to `insert_remove_utensils_534_t185.json`; other affected
bundles (`basic_pick_place_14851_t29`, `clean_surface_*`, `pick_food_*`)
can be repaired the same way.
**Prevent**: `prepare_clip.py` should also save the full per-frame slice
(it has `poses_full` in scope) — mirror the HD-EPIC builder
(`prepare_hdepic.py:347-349`).

## 2026-05-13 (later) — linear-map sync was WORSE than `round(t*fps)`
**Symptom**: After the first 2026-05-13 patch landed, the 2D overlay
was visibly *more* off than before — the trail head sat several frames
ahead of the visible utensils on `insert_remove_utensils_534_t185`.
**Root cause**: The linear map `fi = floor(currentTime / (duration/T))`
assumed the T trajectory frames are uniformly spread over `duration`
mp4-seconds. For `insert_remove_utensils_534_t185.mp4` that's false —
the mp4 has 24 unique frames at native 15 fps (mediaTime 0..1.533 s)
followed by 16 *duplicate* frames of trajectory frame 23 at the end
(libx264 pad). So a uniform map mid-way-through-mp4 said `fi=12` while
the displayed frame was actually source frame 9.
**Fix**: Reverted `tick()` / `tickFullVideo()` to the per-frame formula
`fi = round(mediaTime * App.fps)`. To get the *exactly-displayed*
mediaTime instead of polling `video.currentTime` (which can lead by up
to half a frame in Chrome), we set up `video.requestVideoFrameCallback`
in the `loadedmetadata` handler and stash `meta.mediaTime` on
`App._rvfcMediaTime`. The early-loop guard `currentTime > clipEndTime +
0.5/fps` skips the trailing duplicate frames so the trajectory never
"freezes" on the last frame. `playbackRate` is now `15 / App.fps` so a
30-fps HOT3D mp4 plays at 0.5×, a 24-fps DAVIS mp4 at 0.625×, and
every clip looks like 15 trajectory-frames/sec wall-clock.
**Prevent**: When mp4 PTS layout matters for overlay sync, verify by
MSE-matching mp4 frames against source frames (see
`tmp/...` one-shot from this debug) before changing the sync math.

## 2026-05-13 — GIF worker blocked by CORS on GitHub Pages
**Symptom**: `GIF capture failed: Failed to construct 'Worker': Script
at 'https://cdn.jsdelivr.net/.../gif.worker.js' cannot be accessed
from origin 'https://jianingnz.github.io'`.
**Root cause**: `new GIF({ workerScript: <cross-origin URL> })` makes
the browser spawn a Worker directly from the foreign URL; github.io's
strict-origin Worker constructor refuses unless the response carries
`Access-Control-Allow-Origin: *`, which jsdelivr doesn't reliably set
for arbitrary npm files.
**Fix**: Added `_ensureGifWorkerUrl()` which fetches the worker script
text (falls through a list of CDN candidates: jsdelivr → unpkg →
classic gif.js), wraps it in a same-origin `Blob`, and uses the
resulting `blob:` URL as `workerScript`. Blob URLs are always
same-origin, so the Worker constructor accepts them. Cached for the
page's lifetime.
**Prevent**: When using libraries that spawn Workers via a URL
argument, always wrap third-party worker scripts in a same-origin
blob — same trick applies to mediabunny, ffmpeg.wasm, etc.

### Template
```
## <YYYY-MM-DD> — short title
**Symptom**: what looked wrong from the outside.
**Root cause**: the actual bug.
**Fix**: code change made (file:line).
**Prevent**: what to look for next time so it isn't re-introduced.
```
