# ID Photo: webcam ID photos with live relighting

Team 11 · Relight

A Python desktop app that turns a webcam session into résumé (3 × 4 cm), student or employee ID (3.5 × 4.5 cm) and square profile photos, plus 6 × 4 in print sheets. Every frame is relit to an even light from the face mesh, the background is replaced, and seven checks run before the shutter.

The photos are edited (relit, background replaced), so the app is **not for passports or other government IDs**.

## Requirements

- [uv](https://docs.astral.sh/uv/), the Python package manager. It also installs Python 3.12 (pinned in `.python-version`) if it is missing.
- git (the FBA setup script clones the FBA source)
- A webcam (optional: `--source demo` runs without one)

To install uv on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then open a new terminal so the `uv` command is found.

## Install

The app mats every photo with our traditional **Clothing** method plus the pretrained **FBA** matting model, so FBA is required. FBA's public weights are **noncommercial only** per the [authors' terms](https://github.com/MarcoForte/FBA_Matting). After reviewing those terms, run:

```bash
uv sync --extra fba
```

```bash
uv run --extra fba python tools/fba_reference.py --accept-noncommercial
```

The second command downloads the FBA source and weights (about 140 MB) to `models/fba/`. The MediaPipe face landmark model is already included in `models/face_landmarker.task`; if it is missing, run `uv run python tools/fetch_model.py`.

## Run

```bash
uv run --extra fba idphoto
```

If FBA is not prepared, the app prints the reason and the commands above, and does not start.

Other ways to run it:

```bash
uv run --extra fba idphoto --source demo
```

```bash
uv run --extra fba idphoto --source photo.jpg --lang en
```

- `--source`: camera index (default `0`), `demo` (a synthetic sitter, no camera needed), or an image or video file.
- `--lang ko|en`: interface language (default: the system language). The panel's language button switches it live.
- `--camera-size 1920x1080`: request a larger camera frame for sharper saved photos.
- `--fba-device mps`: run FBA on the Apple GPU (default `cpu`).

FBA is too slow for live video, so the main view shows the latest finished FBA snapshot (its age is shown at the top left) and the checks run on the live Clothing result. The photo itself is matted with FBA. In the app, hold **B** to see the frame before processing, and press **Space** to take a photo without waiting for the auto shutter.

On macOS, the first start asks for camera access for the app you launched it from (Terminal, iTerm and so on): click Allow. If no dialog appears, run it from Terminal, or allow that app in System Settings → Privacy & Security → Camera and restart it.

### Visual Studio (Windows)

Python has no compile step, so Visual Studio opens and runs the source as it is; **Build** succeeds without producing an executable.

1. Install Visual Studio 2022 with the **Python development** workload.
2. In this folder, run the two commands under [Install](#install). `uv sync` creates the `.venv` environment that the project uses.
3. Open `Team11_Relight.sln`. The project's environment is `.venv (Python 3.12)` and its startup file is `main.py`.
4. Press **F5** (or Ctrl+F5) to start the app. To run without a camera, set **Project → Properties → Debug → Script Arguments** to `--source demo`.

## Saved files

**Save…** in the result window writes a folder `idphoto_YYYYmmdd_HHMMSS/` with:

- `resume_3x4cm.jpg`, `id_3.5x4.5cm.jpg`, `profile_800px.jpg`, all at 300 DPI;
- `print_sheet_resume_3x4cm_6x4in.jpg` and `print_sheet_id_3.5x4.5cm_6x4in.jpg` (1800 × 1200 at 300 DPI);
- `report.json`: the seven check values and the measured light.

## Code layout

| File | What it does |
|---|---|
| `main.py` | Starts the app (Visual Studio startup file) |
| `Team11_Relight.sln`, `Team11_Relight.pyproj` | Visual Studio solution and Python project |
| `idphoto/capture.py` | Camera, exposure and white-balance lock |
| `idphoto/color.py` | sRGB ↔ linear, luminance |
| `idphoto/landmarks.py` | 478 pretrained face landmarks (MediaPipe) |
| `idphoto/relight.py`, `idphoto/mesh.py` | Mesh normals, SH light fit, left-right evening, ratio relighting |
| `idphoto/matting.py` | Background/person color models, Otsu, morphology, guided filter, compositing |
| `idphoto/geometry.py` | Crown, pose, crop and scale, print sheets |
| `idphoto/checker.py` | Seven checks with coaching messages, auto shutter |
| `idphoto/npimg.py` | Our NumPy image operations (box filter, guided filter, Otsu, morphology, resampling) |
| `idphoto/pipeline.py`, `idphoto/config.py` | Glue code and all tunable values |
| `idphoto/gui.py`, `idphoto/i18n.py` | GUI (Korean/English) |
| `tools/presentation_fba.py`, `tools/fba_reference.py`, `tools/eval_clothing.py` | FBA model setup and FBA + Clothing matting |

## Credits

Face landmarks: [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) (Apache 2.0). `idphoto/data/canonical_face_mesh.npz` is MediaPipe's canonical face mesh, extracted from the model bundle. Matting: [FBA Matting](https://github.com/MarcoForte/FBA_Matting) (noncommercial weights, not included).
