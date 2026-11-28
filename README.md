# ID Photo: webcam ID photos with live relighting

For the noncommercial FBA and clothing experiments, see [the Korean report](README_kr.md) and [the English experiment report](CLOTHING_EXPERIMENTS.md).

A Python desktop app that turns a webcam session into résumé (3 × 4 cm), student or employee ID (3.5 × 4.5 cm) and square profile photos, plus 6 × 4 in print sheets. Every frame is relit to an even light from the face mesh, the background is replaced, and seven checks run before the shutter. The plan, formulas and thresholds are in [docs/id-photo-survey-plan-v2.md](docs/id-photo-survey-plan-v2.md).

The photos are edited (relit, background replaced), so the app is **not for passports or other government IDs**. It says so on screen and in every saved file.

## Module B: traditional alpha and comparison

The app starts with **FBA + Clothing** when its optional dependencies and verified local research assets are available. Clothing uses our traditional wall-colour, face-prior and guided-filter approach to produce an automatic trimap for the pretrained FBA model. Uncheck FBA for traditional Clothing alone, or also uncheck Clothing for Main. Pretrained MediaPipe face landmarks support these approaches. The panel also offers pretrained MODNet as an opt-in comparison; without `--presentation-modnet` and its local dependencies and model, that choice is disabled. BiRefNet retains its separate capture-only requirements below.

If the mode or capture settings change while a traditional photo is rendering, the old result is discarded rather than displayed under the new selection.

Traditional mode keeps its original single-face detector and does **not** guarantee selection of the person nearest the centre or removal of other people. MODNet uses the guarded centre-face selection described below, but can also miss faces or fail to separate overlapping people.

In a local 12-photo experiment, the traditional alpha and output pixels using an experimental centre detector matched the original single-face path. That detector and its multi-face abstention are not enabled in traditional mode. The photographs have no ground-truth masks and do not establish accuracy.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync --group dev
```

```bash
uv run python tools/fetch_model.py
```

The second command downloads MediaPipe's face landmark model (3.6 MB) to `models/` if it isn't there yet. FBA is not installed with the basic dependencies. Its public weights are **noncommercial only** per the [authors' terms](https://github.com/MarcoForte/FBA_Matting). After reviewing those terms yourself, prepare the optional dependencies and local research assets from this source checkout:

```bash
uv sync --extra fba --group dev
uv run --extra fba python tools/fba_reference.py --accept-noncommercial
```

The app never downloads FBA assets or accepts the terms for you. Then start it without an extra flag:

```bash
uv run idphoto
```

With the prepared assets, FBA + Clothing is selected automatically. If dependencies, the source adapter or verified assets are missing, the app displays the reason and preparation commands and clearly selects traditional Clothing instead. The FBA adapter and preparation script are available in the source checkout, not the installed wheel. Ordinary `uv run` retains an already installed extra; a later plain `uv sync` removes extras unless requested again. `--presentation-fba` remains accepted for compatibility. To use Apple GPU for FBA, add `--fba-device mps`; an unavailable MPS device is reported rather than silently replaced with CPU.

FBA is not a real-time video model here: the main view shows the latest completed FBA snapshot, while Clothing continues live and is shown while the first snapshot is pending or no face is found. The checker and auto shutter use Clothing; capture runs FBA on the frozen photo at the live width (640 px), upsamples its alpha and color correction, and keeps the full-resolution pixels; at camera resolution the trimap's unknown band is too wide in pixels and FBA left the wall in it. FBA errors are shown explicitly. See the [Korean report](README_kr.md) for details.

Other ways to run it:

```bash
uv run idphoto --source demo
```

```bash
uv run idphoto --source photo.jpg --lang en
```

- `--source`: camera index (default `0`), `demo` (a synthetic sitter, no camera needed), or an image or video file.
- `--lang ko|en`: interface language (default: the system language). The panel's language button switches it live.
- `--camera-size 1920x1080`: request a larger camera frame for sharper saved photos.

In the app, hold **B** to see the frame before processing, and press **Space** to take a photo without waiting for the auto shutter.

On macOS, the first start asks for camera access on behalf of the app you launched it from (Terminal, iTerm and so on): click Allow and the live view starts. If no dialog appears, which can happen in an editor's built-in terminal, run it from Terminal, or turn that app on in System Settings → Privacy & Security → Camera and restart it.

## What happens to each frame

![pipeline](docs/images/fig2-system-pipeline.png)

| Step | Module (owner) | File | Plan |
|---|---|---|---|
| Camera, exposure and white-balance lock, lock check | A | [capture.py](idphoto/capture.py) | B.2 |
| sRGB → linear, luminance | A | [color.py](idphoto/color.py) | B.2 |
| 478 pretrained face landmarks | C | [landmarks.py](idphoto/landmarks.py) | A.3 |
| Mesh normals, 9-term SH light fit, 5-frame average, left-right evening from mirror points, ratio relighting | A | [relight.py](idphoto/relight.py), [mesh.py](idphoto/mesh.py) | B.2 |
| Background and person color models, score, Otsu, opening/closing, guided filter, spill removal, compositing | B | [matting.py](idphoto/matting.py) | B.3 |
| Crown, pose, crop and scale, print sheet, 300 DPI files | C | [geometry.py](idphoto/geometry.py) | B.5 |
| Seven checks with coaching messages, auto shutter | D | [checker.py](idphoto/checker.py) | B.4 |
| Glue, settings, shutter render | all | [pipeline.py](idphoto/pipeline.py) | 2.2 |
| GUI (Korean/English) | C | [gui.py](idphoto/gui.py), [i18n.py](idphoto/i18n.py) | Fig. 4 |

All modules exchange one [`FrameContext`](idphoto/context.py) (B.6). The traditional image processing uses NumPy ([npimg.py](idphoto/npimg.py): box filters from cumulative sums, guided filter, Laplacian, Otsu, morphology, separable bicubic resampling, polygon fill, triangle rasterizer). FBA additionally uses its pretrained model and Torch. OpenCV is used to open the camera and to read and write image files.

Every starting value from B.2–B.5 is in [config.py](idphoto/config.py), so tuning in weeks 2–5 happens in one file.

The live view runs on frames downscaled to 640 px wide. When the shutter fires, the same camera frame is processed again at full resolution with the live light average, and that is what gets saved.

## Saved files

**Save…** in the result window writes a folder `idphoto_YYYYmmdd_HHMMSS/` with:

- `resume_3x4cm.jpg` (354 × 472), `id_3.5x4.5cm.jpg` (413 × 531), `profile_800px.jpg` (800 × 800), all marked 300 DPI;
- `print_sheet_resume_3x4cm_6x4in.jpg` (10 photos) and `print_sheet_id_3.5x4.5cm_6x4in.jpg` (8 photos), 1800 × 1200 at 300 DPI;
- `report.json`: the seven check values, R_LR before and after, the fitted light, crown and crop boxes. This is data for the evaluation in C.1.

The app does not intentionally upload photos or video. However, MediaPipe 1.2.0 emitted a failed request to Google's logging endpoint during a local run; its payload was not verified. An offline guarantee therefore requires network isolation.

## Optional local matting experiments

The experimental MODNet path is opt-in; on Apple Silicon, install the parsing extra and provide a local MODNet ONNX model to try the fixed-grid CoreML provider. BiRefNet is a separate, optional **shutter-only CPU** path: MODNet remains the live preview, and selecting BiRefNet never replaces or silently falls back to another capture model. The expected portrait model SHA-256 is `1ba1c8ff5a7bbfadc8d8d13fb11d7be793f91f23d9d466549e37a854f6668f99`; weights are not included or downloaded automatically.

```bash
uv sync --extra parsing --group dev
uv run python tools/fetch_model.py
env -u OMP_NUM_THREADS uv run idphoto \
  --presentation-modnet \
  --modnet-model models/modnet.onnx \
  --modnet-provider coreml-static \
  --birefnet-model models/birefnet-portrait.onnx \
  --face-model models/face_landmarker.task \
  --lang en
```

For an installed wheel, supply `--face-model /absolute/path/to/face_landmarker.task`; the wheel does not contain the MediaPipe model or fetch script. All model paths must point to local files. Missing dependencies, a missing or hash-mismatched BiRefNet model, or a provider mismatch is an explicit error. See the [Korean report](README_kr.md) for UI behavior, benchmark commands, and local 12-photo comparisons, including FBA combinations and shape priors. Those stills have no ground-truth masks, and several thresholds were selected while viewing the same photos; they are diagnostic examples, not an accuracy claim or evidence of generalization.

In MODNet mode, faces are collected from the full frame and overlapping crops, deduplicated, and selected by Euclidean distance from the face bounding-box centre to the image centre; an exact tie favours the leftmost face, then the uppermost. The selected face supplies pose and geometry. If full and crop detection find no face, the previous-position hint and two rotating search tiles are also checked in the current frame. The all-person matte is retained for wall estimation; selected-person ownership uses foreground components above alpha 0.02. A shared component, an unexplained component of at least max(64 pixels, 0.05% of the frame), missing face, or missing background samples causes abstention. Pending or ambiguous frames show a solid background and disable capture; the Before shortcut cannot reveal the raw frame in this mode. When several faces are detected, “original wall” uses a solid estimated wall colour, and a failed cutout check does not restore the raw wall with other people in it. The selected subject and ownership are rechecked on the frozen MODNet or BiRefNet capture frame.

This is a conservative heuristic, not person-instance identification: faces can be missed, especially when small, turned away or occluded, and undetected people connected to the selected foreground may remain. Detection, separation and occluded-body reconstruction are not guaranteed. On one public synthetic pair with separated bodies it removed the other person; touching and overlapping pairs abstained. A real public NASA five-person photograph produced no detected face and abstained. The 12 local single-person photos had unchanged live alpha, rendered pixels and checker decisions compared with the prior face path, using the same MODNet CPU predictions; they do not establish accuracy.

An 80-second public-video replay after restoring frozen manual captures, with window exposure observed and outbound networking denied for the process, reached 25.29 fully processed paints/s and 77.10 ms p95, but one 10-second window was below 24/s (23.9/s), so the sustained gate **failed**. A separate one-shutter run also failed the rate gate (24.26/s overall, minimum window 20.9/s, 78.84 ms p95). Its manual BiRefNet capture produced and displayed the requested frame after a live track change, but the capture interval reached only 19.80 fully processed paints/s. A real local BiRefNet capture of a separated synthetic pair also produced the selected subject. These results do not establish physical-camera performance or identity stability through display.

An isolated detector profile on 60 sampled public-video frames and four public multi-person images measured 12.0 ms median total face detection; median full, left, right and centre calls were 3.68, 0.82, 0.84 and 3.48 ms respectively. The left and right crops each found faces missed by earlier views in the synthetic group examples. The centre crop added none in this small sample, but can independently find a more central face; skipping it is not supported as a safe optimization. These isolated timings are not additive whole-app replay timings.

An installed-wheel CPU/CoreML alpha comparison on the 12 local evaluation photos used ONNX Runtime 1.32.0. `CoreMLExecutionProvider` was active on all photos across the tested fixed grids `512x288`, `512x352`, and `512x672`. The float alpha arrays were not bitwise identical; maximum and mean absolute differences were `1.76e-5` and `3.45e-8`, and the lowest binary-mask IoU at alpha 0.5 was `0.99999983`. These are implementation-parity measurements, not ground-truth accuracy.

The first strict 80-second replay passed the sustained preview gate but failed its capture gate: the capture checker raised before a visible result dialog. The checker runs after the BiRefNet matte call, so that failure does not establish that inference never ran. After fixing the missing native-frame input to the sharpness check, a fresh 80-second replay used the same five public clips in the order `0004 → 0000 → 0001 → 0002 → 0003`, 30 Hz pacing, 3-second warmup, and one manual capture at +13.000 seconds. It produced one CPU BiRefNet result and a visible dialog, with verified frame/crop lineage. The press-to-dialog interval was 8.027 seconds; during that interval fully processed preview paints were 26.54/s, p95 source-read-to-paint was 90.81 ms, and the endpoint-inclusive longest paint gap was 104.28 ms.

That replay **failed the overall sustained preview gate**: 23.81 fully processed paints/s across 80.001 seconds, full-run p95 105.35 ms, and the final four 10-second windows were 17.7, 16.3, 20.9, and 19.5 paints/s (below 24/s). The first four windows were 27.6–29.6/s. A matched-frame comparison shows later increases in source-read-to-prepare and job-finish-to-finalize latency, while model queue wait stays low; the available traces do not establish the cause. The capture-only gate passing is not evidence that the full preview requirement passes. Both strict replay reports preserve their individual outcomes. These were local public-video replays, not camera tests.

#### Latest live-wall and shutter retest

The MODNet **live preview only** now uses a packed OpenCV wall-estimator implementation; the reference estimator remains in use for full-resolution BiRefNet shutter rendering. Before integration, an isolated 20-pair/case synthetic benchmark at 640 × 360 verified bit-exact global/local wall arrays, composites, encoded pixels, and rule-7 decisions on edge cases. Its median estimator time changed from 13.87 ms to 7.53 ms (p95 18.22 ms to 12.40 ms), with both AB and BA execution orders improving. The same candidate was then checked against cached real RGB/alpha pairs from all 12 local photos: wall fields, rendered floats/bytes, and rule-7 verdicts were exact. These are implementation equivalence and timing checks, not mask accuracy.

With that code, a fresh 80-second no-shutter run averaged 26.94 fully processed paints/s and had full-run p95 below 100 ms, but its 40–50 s and 50–60 s windows reached only 21.3 and 22.5/s; the sustained gate failed. A separate fresh run with one +13 s manual shutter passed the full replay gate: 29.44 fully processed paints/s on average, all eight 10-second windows at or above 24/s, and full-run p95 source-read-to-paint 71.99 ms. Exactly one full-resolution CPU BiRefNet inference and visible `ResultDialog` were linked to source frame 485 and its recorded crop; press-to-result was 9.030 s. The press-to-result preview interval itself was 26.91 paints/s with p95 89.68 ms; including both endpoints, the longest paint gap was 309.12 ms (reported, not separately gated). A single successful shutter run does not erase the no-shutter sustained failure or establish a universal performance guarantee, and the observed rate change cannot be attributed solely to the estimator from separate runs.

An already-frozen manual BiRefNet photo remains bound to its original frame and selected subject if the live track later changes. A change before freezing rejects the request; automatic captures and stale live previews are still invalidated, as are captures after settings or mode changes.

The new JSON/CSV files `modnet-live-wall-fast-no-shutter-80s` and `modnet-live-wall-fast-one-shutter-80s` preserve the matched replay outcomes. The CSVs now also include per-frame landmark, relight, matting, geometry, and checker timings. The tests use public video clips and local model files; no camera was opened.

A follow-up source-only diagnostic found that 30 Hz decoding and resizing of public 4K clip `0002` used about 2.8 process-CPU seconds per wall second, versus about 0.97–1.02 for the other clips, without the app or model running. This can add contention in replay, but does not explain the entire slowdown: the no-shutter run also slowed during `0001`. An isolated CoreML ORT spinning probe showed negligible idle CPU and no meaningful paced CPU reduction from disabling spinning. No runtime or replay settings were changed, and the sustained failure remains unresolved.

On the same product code, a later **exposure-verified no-shutter series** passed all three pre-specified 80-second attempts: 29.83, 29.81, and 29.81 fully processed paints/s, with full-run p95 57.76, 54.15, and 51.91 ms. All eight 10-second windows in each attempt were at least 29.3 paints/s and had p95 at most 75.55 ms. No fresh result lacked a paint. The harness kept the window on top, raised and activated it once, and used `caffeinate -d -i`; exposure/visibility/minimization were sampled and Qt exposure changes recorded. All three stayed exposed and visible. These extra harness conditions differ from the earlier failed run, whose exposure was not recorded; the failure and its cause remain unresolved. The series is not a camera test or a general performance guarantee.

After the earlier wall timing runs, a fresh wheel was built and installed into an environment outside the checkout. Its 12-photo 640 px CPU/CoreML alpha comparison used ONNX Runtime 1.32.0: CoreML was active on all photos for grids `512x288`, `512x352`, and `512x672`; max/mean absolute alpha differences were `1.42e-5` and `3.28e-8`, with minimum 0.5-mask IoU `1.0`. The installed wheel's CLI help worked, and its direct, crop-reacquisition, and reset landmark paths each detected 478 points using an explicitly supplied local face model. This was an installed-artifact check on cached local evaluation inputs, not a ground-truth accuracy result or camera test.

## Tools

| Command | What it does |
|---|---|
| `uv run python tools/relight_demo.py` | Reproduces Figure 3 with the real code: prints R_LR and shadow share before and after, saves `out/relight_demo.png` |
| `uv run python tools/eval_synthetic.py` | Synthetic version of C.1: R_LR over a sweep of side lights, mask IoU against the true mask, staged faults and per-rule recall |
| `uv run python tools/eval_checker.py testset/` | Precision and recall of each checker rule on real photos sorted into one folder per staged fault (C.1); saves every measured value to `out/checker_eval.csv` |
| `uv run python tools/make_faults.py docs/evaluation/test/*.jpg` | Makes blurred, tilted and side-shadowed copies of good photos in `testset/`, for `eval_checker.py` |
| `uv run python tools/make_rating_pairs.py docs/evaluation/test/*.jpg` | Relit vs unrelit résumé photos side by side as A/B in random order for the blind rating, with the answer key (`out/rating/`) |
| `uv run python tools/score_rating.py responses.csv` | Scores the blind rating from the form's CSV export: raters preferring relit, sign test |
| `uv run python tools/score_usability.py usability.csv` | Scores the usability test: retakes and time, our app vs the phone workflow (Wilcoxon, bootstrap) |
| `uv run --with onnxruntime python tools/identity_check.py` | ArcFace similarity of each relit photo to its unrelit original (evaluation only; downloads a 13 MB model to `models/`) |
| `uv run python tools/bench_fps.py` | Frames per second of the full live pipeline on this machine (target ≥ 10) |
| `uv run python tools/process_image.py photo.jpg` | Runs the pipeline on still photos and saves the same files as the app |
| `uv run python tools/crown_inspector.py photo.jpg` | Crown and crop checking window: both crown cues, the crop box and the saved photo, with sliders for k, the mask band and the head share. Click the top of the hair to mark the true crown; it shows the head-length error in mm and the best-fit k. `--demo` uses synthetic faces |
| `uv run python tools/check_lock.py` | Tests the webcam's exposure / white-balance lock on this machine: lock state, then brightness drift while you hold still. On macOS run it from a terminal that has camera access |
| `uv run python tools/capture_frames.py` | Saves webcam frames, their landmarks and each matting step to `out/debug/` (debugging, test sets). On macOS run it from a terminal that has camera access |
| `uv run python tools/replay_frames.py out/debug/<folder>` | Re-runs the pipeline on saved frames, to compare a code change on real rooms |
| `uv run python tools/extract_canonical_mesh.py` | Rebuilds `idphoto/data/canonical_face_mesh.npz` from the model bundle |

Tests: `uv run pytest` (331 passed, 2 skipped in the latest local run).

## Results so far

Measured on an Apple M4 Pro, mostly with the synthetic sitter (`idphoto/synthetic.py`), which renders MediaPipe's canonical face with a chosen light, so each module can be checked against known truth. These are not the plan's results; those need real people (C.1).

- **Relighting:** a side window that leaves R_LR = 0.41 is relit to 0.96, and the shadowed share of the face drops from 15% to 0% (`tools/relight_demo.py`). R_LR lands in 0.90–1.10 for side lights down to R_LR ≈ 0.28, relit to 0.96 (`tools/eval_synthetic.py`); a harsher light that stays out of the band makes rule 1 ask the user to turn toward the light, as C.4 plans.
- **Matting:** IoU 1.000 against the true mask on four wall and top combinations. A top the color of the wall drops to 0.82, and rule 7 catches it (contrast below 0.12).
- **A real, busy room** (clothes rack, white doors, ceiling light, black T-shirt; six webcam frames): hair, face and the black top are kept and the room is replaced. The edge on the clothes-rack side is not exact, and rule 7 asks for a plain wall there, since only 65% of that background is one color.
- **Relighting on real photos** (the 11 photos in `docs/evaluation/test`, R_LR 1.01–1.37 before): R_LR after relighting is 0.98–1.03. On three side-lit frames taken from screen recordings of the app at strength 0 (R_LR 1.93–2.64 before), it is 1.04–1.08. The ratio's fine detail (standard deviation over skin of the log ratio minus its 13 × 13 mean, on the 128 × 128 grid) is 0.015–0.059 on these 14 images, against 0.030–0.145 with the plan's frontal target and hard clip, where that detail showed as a gray patch beside the nose and an orange cheek. The nine overhead-light webcam frames measured earlier (0.97–1.05) were not kept, so they are not re-measured.
- **Checker:** each staged fault (side shadow, glare, blur, tilt, turn, closed eyes, open mouth, top like the wall) fails its rule; the good frame passes all seven.
- **Speed:** 16–22 fps for the whole live pipeline including MediaPipe (45–62 ms per frame from a 1280 × 720 camera frame, varying between runs).
- **Landmarks:** MediaPipe finds the synthetic face with a mean landmark error of 1.6% of the face width, which confirms that the renderer and the landmark numbering agree.

## Where the code goes beyond or departs from the plan

These are choices made while implementing the drafts in Appendix B. Each is commented in the code.

1. **Mesh normals** are stored on the 128 × 128 face crop the light is fitted on (`FrameContext.normals` plus `face_box`), not as a full (H, W, 3) array.
2. **Guided filter (B.3 step 3):** the guide is the frame's color instead of its luminance (He et al.'s color-guide variant), since a lit neck and a wall of the same brightness differ only in color. The radius is 6 px at 640 px wide instead of the starting 15 px: in a real room, a 15-px window spread a gray halo wherever the mask edge was a few pixels off.
3. **Background and person color models (B.3 step 1):** the single wall color from the upper corners failed in a real room (corners holding a window, a patterned ceiling and a door frame), and dark hair was classified as a shadowed wall. The background is now k-means clusters (8) of the pixels the landmarks place well outside the person, and the person gets clusters (5) of the face, the head under the crown and a column straight below the chin. Each pixel weighs its distance to the nearest background color (allowing it 0.5–1.25 times as bright, so a wall shadow stays background) against its distance to the nearest person color; distances past 2.5 spreads count the same. In front of a plain wall this reduces to the plan's wall model.
4. **Head-and-shoulders prior:** besides the region where the person can be, a prior probability that is high in the skull and neck (0.88) and inside the shoulder line (0.7), so a black top in front of dark clutter is kept; after Otsu and morphology, only the region connected to the face is kept and holes are filled.
5. **Live speed:** in the live view, steps 1–3 and the guided-filter coefficients run at half resolution and the coefficients are upsampled (fast guided filter); connectivity runs on a grid 4 times coarser. The saved photo uses the exact full-resolution filter.
6. **Spill removal:** μ_B in F = (I − (1 − α) μ_B) / max(α, 0.1) is the background color around each edge pixel (the mean of nearby background pixels), not one wall color.
7. **Target light follows the head's yaw (B.2):** "frontal" means in front of the face, not the camera (this now applies to the shape part, see 19). The yaw comes from the mesh's mean normal, averaged over the same 5 frames as the light. With a camera-fixed target, a head turned a few degrees left one cheek brighter after relighting.
8. **Ratio outside the face** fades to the face mean weighted by brightness, so hair, neck and shoulders keep the exposure the face keeps.
9. **Cheek polygons** sit below the lower orbital rim so glasses frames stay outside them.
10. **Pitch ratio:** d(10, 1) / d(1, 152) is divided by its value on MediaPipe's canonical face (1.134), so 1.0 means a level head and the 0.85–1.15 band applies.
11. **Sharpness (rule 3)** is measured on the camera's full-resolution frame, so the 256 × 256 face crop is not an upsampled copy of the 640-px live frame. The bound of 120 is calibrated per camera as B.4 asks: after 15 frames it becomes 70% of the camera's sharp frames (the 80th percentile of the last 90), never above 120 and never below 15. A MacBook webcam scores 40–50 and could never pass 120.
12. **Rule 7:** the hair band is everything above the mouth within 1.1 face widths of the midline. Contrast compares the top with the background in the same rows, on each side of the neck, and the weaker side counts. When the cutout fails, the message asks for a top unlike the wall (low contrast), for a plain wall (less than 70% of the background is one color), or to move away from the wall.
13. **Crown:** measured, not averaged. The crown is where the person mask ends on the face midline; the eyes-halfway proportion (k = 2) only checks it, and stands in when the mask top lies outside 0.85–1.25 of the proportion's head length (hair cut off, wall or hat included, or the mask leaves the frame). On hand-marked photos the hair top fitted k = 1.99–2.25 depending on the hairstyle, which is why k alone is not used.
14. **Exposure lock:** on macOS the camera is read through AVFoundation (pyobjc), because OpenCV's macOS backend has no exposure control. `AVCaptureDevice` offers what the hardware supports: the built-in MacBook camera supports neither lock, and an external Brio 300 supports the exposure lock only (state `partial`). Where nothing can be locked (`unsupported`), the lock did not hold (`failed`) or face brightness drifts more than 3% while the head holds still (`drifting`), the light is averaged over 10 frames instead of 5 (the C.4 fallback). The lock is set after at least 1.5 s and once the mean brightness is steady (at most 5 s); `tools/check_lock.py` tests it on a machine.
15. **Rule 7 fallback (C.4):** if the cutout check fails when a photo is taken, the saved photo keeps the original wall and the result window says so.
16. **Print sheets** are made for the résumé and ID formats; the profile photo is for screens.
17. **Rule 1 shadow share** is measured below the eyes and 4 px inside the face edge (on the 128 × 128 grid), not over the whole face. On real photos, bangs and hair at the temples fall inside the face outline and counted as shadow (9–33% on our 11 test photos, limit 5%); now they measure 0–2%. A side shadow still shows: an unrelit side-lit face gives 13–38%.
18. **Left-right evening is measured, not modeled (B.2):** on real side light, nine SH terms put the largest correction next to the nose and the smallest on the darker outer cheek, which showed as a gray patch beside the nose. The left-right part of the ratio now compares the skin brightness at each grid pixel with the same point on the other side of the face (the canonical mesh is exactly symmetric, so each triangle has a mirror triangle) and moves both to the same level (see item 23 for where they meet). Brightness is averaged over a wide window (two box passes of radius 6) over skin only (pixels darker than 0.3 × the median, such as bangs, are left out). Two passes each measure what is still uneven after the whole ratio (shape, bounds and smoothing included) is applied.
19. **Shape part of the ratio:** the SH fit is still used, made even across the midline, to turn the room light into the target (frontal, 20° from above, turned with the head). The target's fill is 2.0 times the key light instead of 0.8, closer to a studio's wide softbox; the plan's value shaded the jaw line darker. This part is softly bounded to 0.8–1.25.
20. **Skin color is evened like brightness:** shadowed skin is redder and a highlight paler, so a brightened shadow turned orange and a darkened highlight gray. Per-channel gains move each point's averaged skin color toward its mirror point's at equal luminance, by the same shares as the brightness (item 23) (bounded to 0.8–1.25). They are stored in `FrameContext.color_gain` and applied with `ratio_map` to the relit frame and the FBA foreground.
21. **Bounds and smoothing of the ratio:** the plan's hard clip to [0.5, 2] left flat patches where it was reached. The bounds are applied softly (unchanged up to half the bound in log, then bending toward it), and the ratio is smoothed over the face only (two box passes of radius 6 on the 128 × 128 grid), since averaging in the value outside the face weakened the correction at the face edge. Relighting takes about 22 ms per live frame on an Apple M5, against about 12 ms with the plan's ratio.
22. **Left-right passes see the guided upsampling:** the ratio was evened on the grid, then the guided filter that brings it to the frame flattened it, so on six recorded side-lit webcam frames (R_LR 0.35–0.43) the grid reached 0.93 but the frame, which rule 1 measures, only 0.88, and the status line showed the grid's value. The passes now measure the grid with the guided upsampling applied, `r_lr_after` is measured the same way, and the left-right bound is widened from 0.5–2.0 (a dark side needing about 1.6x sat past the old bound's knee). Those frames now pass rule 1, and the reported value matches it.
23. **The dark side rises more than the lit side falls:** both sides of the face used to meet at their geometric mean, so the lit cheek darkened as much as the shadowed one brightened (sRGB 168 → 133 on the recorded frames) and the face looked darker. They now meet at B^0.75 D^0.25 of the brighter (B) and darker (D) level (`MIRROR_LIFT`), with the left-right bound at 0.4–3.5. On the six recorded frames the face mean goes from 121 to 132 (was 121) and the lit cheek keeps 149, rule 1 reads 0.93–0.99, and noise on the brightened cheek rises from 2.5 to 3.2 (high-pass std, ×1000). Raising only the dark side (`MIRROR_LIFT` 1) stopped at R_LR 0.84 on the harshest synthetic light, with more noise; 0.75 keeps 0.96 there.

## Not done yet

- Part D (checker and evaluation) steps, how to take test photos, and what D needs from the team: [docs/part-d-steps.md](docs/part-d-steps.md).
- Not yet tested on the team's webcams and rooms. The Nov 8 gate (10 real frames, side and overhead light) is the next check.
- The C.1 baselines (ArcFace identity, rembg masks, blind rating, usability test) are separate studies and are not in this repository.
- Thresholds are the drafts from B.4 and are calibrated in week 5. EAR ≥ 0.23 is close to the canonical face's 0.234, so it may need a lower or per-user value.
- Stretch goals (lighting presets, more background colors, job-portal pixel sizes) are not started.

## Credits

Face landmarks: [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) (Apache 2.0). `idphoto/data/canonical_face_mesh.npz` is MediaPipe's canonical face mesh, extracted from the model bundle.
