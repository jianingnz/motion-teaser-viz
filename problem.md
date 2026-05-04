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

### Template
```
## <YYYY-MM-DD> — short title
**Symptom**: what looked wrong from the outside.
**Root cause**: the actual bug.
**Fix**: code change made (file:line).
**Prevent**: what to look for next time so it isn't re-introduced.
```
