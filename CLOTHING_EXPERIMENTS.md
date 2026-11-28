[App overview](README.md) | [한국어](README_kr.md)

# Module B Clothing-Recovery Experiments

Comparison images and `test9.jpg` are not distributed here; obtain the input locally with appropriate consent to reproduce the 12-photo evaluation.



If the existing background-separation pipeline is unfamiliar, first read the [Module B matting overview](https://github.com/hanbin5/relight/blob/module-b/matting/README.md). This report does not repeat that pipeline; it describes the attempts to recover missing clothing and their results. The selected change is in the app's `idphoto/matting.py`, not just an offline experiment.

The **person prior** uses facial landmarks to assume where a person might be, and the **gate** limits that region. **Color-sample candidates** are pixels eligible to be sampled for a color model. **Alpha** is 0 for background, 1 for foreground, and intermediate where the two colors are mixed. The app has no semantic clothing-segmentation model. SCHP, Sapiens and FBA are offline experiments, not app features; Sapiens and FBA have noncommercial restrictions.

## Latest Experiment: All 12 Photos

`Previous clothing` is app code `4276adf6a4cba13d0b98f4a8e333498e8a5c40db`; `Clothing (NumPy)` includes the current close-up fallback. `FBA` uses a trimap from Main alpha, and `NumPy + FBA` uses one made **by the same rules from the current clothing alpha**. Both receive the original photo. SCHP and Sapiens instead combine their predictions with **Main alpha**, not FBA: `clothing` uses garment classes, and `person` includes all predicted person parts. Here Sapiens receives aspect-preserving black padding and a fixed gamma of 0.55 on its input only.

All 12 photos use the same 640 px-wide inputs and landmarks in live mode (`s=2`) and on the full grid (`s=1`) (local image). The [live CSV](docs/evaluation/clothing/research-gamma/live/measurements.csv) and [full-grid CSV](docs/evaluation/clothing/research-gamma/full-grid/measurements.csv) retain all 12. The figure shows `shot2` as the representative of the similar `shot2`–`shot4` scene. Left and right are as viewed in the photo.

| Photo | NumPy clothing and background | SCHP person | Sapiens pad + gamma person | Sapiens vs SCHP pixels: live; full grid |
|---|---|---|---|---|
| shot2 | Jacket remains, but its edge is rough and fragments on the full grid. | More jacket and hair remain. | Similar jacket and hair; edges are not perfect. | +343/-690; +382/-703 |
| shot3 | Left jacket is partly missing. | More jacket remains; the left edge can be translucent. | Jacket and hair remain, with an imperfect edge. | +444/-644; +453/-641 |
| shot4 | The right-side background object remains. | More jacket remains and the object is reduced. | Jacket remains and the object is reduced, with a rough edge. | +446/-629; +447/-630 |
| shot5 | Both sides of the close-up jacket return; the right shoulder is rough. | Jacket and clipped hair return. | Jacket and hair return, but their edges remain imperfect. | +1,259/-750; +1,259/-750 |
| test1 | Dark room objects and holes in the top remain. | Side background is reduced and the top is more connected. | Gamma restores much of the top versus padding alone, but conspicuous holes and clipped hair remain. | +1,382/-4,094; +1,458/-3,964 |
| test2 | Dark objects remain on both sides. | Hair and background improve, but the right shoulder and sleeve are cut. | Almost the entire top is removed despite gamma. | +0/-54,200; +0/-54,202 |
| test3 | A neck notch and background at the right remain. | Background is reduced; top and hair are more complete. | The gray top remains, but much hair is still clipped. | +626/-10,381; +627/-8,780 |
| test4 | A blue object and dark background remain. | Objects near the top are reduced and more hair remains. | Almost the entire top is removed. | +0/-56,237; +0/-56,057 |
| test5 | The right white shoulder and a left wall strip remain. | The wall strip is reduced; clothing and neck edges are imperfect. | The wall is reduced, but a conspicuous hole cuts through the printed shirt. | +6,993/-22,509; +6,792/-22,533 |
| test6 | Wall leakage behind the shoulder remains. | The wall is reduced, but a bright foreground hole remains. | The wall is reduced, but large parts of the shirt and lower foreground are cut. | +10,591/-41,028; +10,227/-41,082 |
| test8 | Collar notches and blocky edges remain. | The top is more connected, with residue at the right sleeve. | Gamma restores part of the top versus padding alone, but leaves speckled holes. | +4,898/-4,278; +4,871/-4,184 |
| test9 | Hoodie and hair are clipped, with attached background. | More hoodie and hair remain, but the clothing edge is rough. | More hoodie and hair remain, with gray shadow-side residue and a rough edge. | +13,681/-915; +13,378/-915 |

The last column counts whole-image pixels crossing alpha 0.5: `+` is above 0.5 only in Sapiens person, and `-` only in SCHP person. It includes hair, clothing and background, not just clothing; counts are not accuracy or recovery and image sizes differ. All parser-versus-Main and clothing-only counts are in the CSVs.

**There are no ground-truth masks for these photos.** The 12 photos are not independent scenes, and both fusion thresholds and gamma were explored on the same set. Relative to previous clothing at alpha 0.5, shot5 adds 31,525 pixels and removes none in live mode and adds 31,292 and removes none on the full grid; the other 11 binary masks are unchanged. Added pixels may include background. SCHP person is the more useful **offline reference for these examples**, but it also cuts test2 clothing; Sapiens and the new HSV proposal are not reliable overall. No pretrained parser or new HSV logic was added to the app.

### Methods and Decisions

**App close-up fallback.** When the existing chest candidates number fewer than 50 on the processing grid and more than 0.3 eye-to-chin distances (`ec`) remain below the chin, narrow strips at the visible lower sides become clothing candidates. Their upper boundary is `max(chin + 0.25 ec, chin + 0.75 × visible height below chin)` and their distance from the center is `0.4–0.85` face widths (`fw`). A wider side-shoulder gate starts at `chin - 0.45 ec`, and its surroundings are protected from clothing-background sampling. Candidates at normalized distance 0.9 or less from the background model are discarded; fewer than 50 surviving candidates or background candidates skips the pass and clears clothing warm state on updating frames. This fallback was triggered only for shot5 in both modes. Of 608 candidates, 590 were used in live mode; on the full grid, 2,618 were candidates, at most 2,000 were sampled, and 1,998 survived.

This uses landmark-based geometry and the existing face-connectivity processing, without a separate pose model or per-photo exceptions. We also tried cutting along an image-edge shoulder path and retaining only face-connected regions after erosion, but these cut the white shirts in test5/test6 or the test9 hoodie and were rejected. A close-up sample can still confuse similarly colored walls and clothing, and occlusion or unusual poses can invalidate the sampling assumptions.

**Offline SCHP parsing.** We run a LIP 20-class ONNX conversion of [Self-Correction for Human Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing), using [this checkpoint](https://huggingface.co/pirocheto/schp-lip-20/tree/7c0028fe34dcd8bd534ffa291a9086cdc856e4a9) on CPU. We use the original inference's BGR channel order and normalization, pad the image to a black square, and resize to 473 × 473 with Pillow bilinear interpolation. This is not pixel-identical to the original affine transform or the converted model card's RGB square stretch. We softmax the 20 logits and use clothing classes 5, 6, 7, 10, and 11 and background class 0.

Both combinations attenuate Main alpha below the chin as SCHP background probability rises from 0.65 to 0.95, while protecting the central neck. Clothing or whole-person probabilities are remapped from 0.15–0.85 to 0–1 and added only in face-associated proposals; the face is preserved. The clothing-only result also uses a landmark-based clothing gate. Connectivity is approximated on a reduced grid and does not perfectly identify the subject. This is our experimental fusion, tuned on these same 12 photos, not a standard method from the paper. SCHP misclassifications, including lost shoulders and holes, propagate into the result.

The original source and converted model card are each labeled MIT, but redistribution and commercial rights for the original training data and weights have not been independently established. The two weight files total about 255 MiB and are pinned to revision `7c0028fe34dcd8bd534ffa291a9086cdc856e4a9`, with SHA-256 `41c9d6c753a387f794bf3cbd9aed5e20c81a934fdc91731475bd46138937f84c` and `be931b91c76ed7707b5426abbccee98620a122b777c7cbd1f771a122803666e7`. The project plan does not permit an additional pretrained segmentation model in the app, so ONNX Runtime and Pillow are only a `parsing` optional dependency and SCHP is not wired into the app. A single live run had a median SCHP inference plus pre/postprocessing time of about 190 ms; this excludes model loading, fusion, and the rest of the app.

**Offline Sapiens parsing.** We ran the official [Sapiens 0.3B segmentation TorchScript checkpoint](https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/tree/18e07d7474a00d30728f22de33325741ce27046f). Its 1,358,871,599-byte file is pinned to revision `18e07d7474a00d30728f22de33325741ce27046f` and SHA-256 `735a9a8d63fe8f3f6a4ca3d787de07e69b1f9708ad550e09bb33c9854b7eafbc`. The [code](https://github.com/facebookresearch/sapiens/blob/main/LICENSE) and model card specify CC BY-NC 4.0: noncommercial use only.

The model takes RGB at 1024 × 768, normalized with mean `[123.5, 116.5, 103.5]` and standard deviation `[58.5, 57, 57.5]`. We sum its apparel, lower-clothing and upper-clothing probabilities (classes 1, 12, 22) and use class 0 for background. The same Main-alpha fusion described above is used for both parsers, although its thresholds were tuned on SCHP and their probabilities and class definitions need not be calibrated alike; this is not a controlled comparison of models alone.

We tested three whole-frame inputs on all 12 photos and both grids: official-demo stretch (local image) (full grid (local image)), aspect-preserving black padding (local image) (full grid (local image)), and padding plus fixed gamma 0.55 (the top result). Each directory includes its 12-row CSV. Stretch loses most of shot5 and test1/2/3/4/8 clothing. Padding restores shot5 and test3 but still largely removes test1/2/4, makes holes in test5/6 and leaves speckles in test8. Gamma was explored after viewing these same photos: it restores parts of test1/8 versus padding, yet test2/4 still lose almost the entire top and test5/6 have large holes. Gamma changes only the parser input; compositing still uses the original RGB. No variant is suitable to add to the app.

On an Apple M4 Pro with 24 GiB RAM, Python 3.12.13, PyTorch 2.15.0 and ONNX Runtime 1.32.0, one padding run had median Sapiens MPS inference plus pre/postprocessing of about 1.6 s per live image versus about 0.18 s for SCHP on CPU. The devices differ; model loading, fusion and the rest of the app are excluded. On the same shot5 input, actual CPU and MPS Sapiens clothing/background probabilities differed by at most 0.000023 and had no threshold-0.5 disagreements. These are exploratory timings, not an app benchmark.

**HSV, geometry and connectivity probe.** A separate offline NumPy hypothesis models HSV background colors, protects a face-derived central torso, and removes background-like regions connected to background seeds before face-connectivity cleanup. It is not a reproduction of a full pose-estimation paper. The live (local image) and full-grid (local image) comparisons and CSVs cover all 12 photos. At alpha 0.5 the probe removed 12,889 and 26,669 pixels respectively versus current NumPy; without labels, those totals do not distinguish clothing from background.

The labeled synthetic counterexample in `tests/test_hsv_probe.py` exposes the tradeoff: of 2,150 known sleeve pixels resembling the wall, current NumPy retains 2,020 live and 1,997 full-grid, while the probe retains only 692 and 679; it removes **zero** background pixels in that scene. We rejected this candidate rather than treating a smaller mask as an improvement. The existing app close-up fallback remains; the new probe changes no app code.

### Other Research Candidates

| Candidate | Availability, constraints and decision |
|---|---|
| [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) | The pinned converted model ran locally on CPU; source and model card are labeled MIT, but original weight and training-data rights require separate verification. Offline comparison only. |
| [Sapiens](https://github.com/facebookresearch/sapiens) | The official 0.3B model ran locally, but its code and weights are CC BY-NC 4.0 and the observed failures preclude app adoption. |
| [Sapiens2](https://github.com/facebookresearch/sapiens2) | Official 0.4B segmentation and 1B matting weights are public but add about 1.63 GB and 6.16 GB. Its [license](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md) forbids biometric processing, among other uses, and use accepts the agreement. Whether an ID-photo use falls within that restriction requires legal review. We did not download or run it. |
| [Spectrum](https://arxiv.org/abs/2508.06032) | The paper describes worn-clothing parsing, but its paper and project page did not expose a reproducible code/weight download when checked; model-use terms could not be established. Not run. |
| [D2FP](https://github.com/cvlab-yongin/D2FP) | Code is labeled MIT and [official release weights](https://github.com/cvlab-yongin/D2FP/releases/tag/weights) are about 958 MB each. The setup needs Detectron2 and a CUDA extension; the Python debug fallback still imports that extension. This Mac has neither Detectron2 nor CUDA, and weight/data rights need separate review. Not run. |
| [SAM 3](https://github.com/facebookresearch/sam3) | Weights are [manually gated](https://huggingface.co/facebook/sam3) and under the custom [SAM license](https://github.com/facebookresearch/sam3/blob/main/LICENSE); the official path expects CUDA and the checkpoint is about 3.45 GB. Not run or sent to an external demo; a clothing prompt also requires subject association. |
| HSV/morphology and pose/connection ideas | Tested as the NumPy probe and the existing face-landmark close-up fallback, not as the cited papers' complete systems. The new probe was rejected on the labeled counterexample above. |

For traceability, the unrun Sapiens2 [segmentation `model.safetensors`](https://huggingface.co/facebook/sapiens2-seg-0.4b/tree/449b3c5335e6722bb94990abdd1aa6e612432f22) has revision `449b3c5335e6722bb94990abdd1aa6e612432f22`, 1,626,451,404 bytes and LFS SHA-256 `b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9`. The unrun [matting `model.safetensors`](https://huggingface.co/facebook/sapiens2-matting-1b/tree/88b79074a58a0e2bb5befc00a237488c43cd94da) has revision `88b79074a58a0e2bb5befc00a237488c43cd94da`, 6,157,412,312 bytes and LFS SHA-256 `2a0ea4ea2234e8dbd51e952950d8cedfe3ee2eca9099003784fd2d9cec6bc6e1`. These are repository metadata, not checksums verified from local downloads.

### Reproduce the New Comparison

After reviewing the FBA and Sapiens noncommercial restrictions, run locally. Photos are not transmitted, and models are stored under Git-ignored `models/`. The commands below use Apple MPS for Sapiens; use `--sapiens-device cpu` where MPS is unavailable.

```bash
uv sync --group dev --extra fba --extra parsing
uv run python tools/fetch_model.py
uv run --extra fba python tools/fba_reference.py --accept-noncommercial
uv run --extra parsing python -m tools.schp_reference --download
uv run --extra fba python -m tools.sapiens_reference --accept-noncommercial
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --out out/clothing-research/stretch/live
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --full-grid --out out/clothing-research/stretch/full-grid
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --out out/clothing-research/pad/live
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --full-grid --out out/clothing-research/pad/full-grid
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --sapiens-gamma 0.55 --out out/clothing-research/gamma/live
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --sapiens-gamma 0.55 --full-grid --out out/clothing-research/gamma/full-grid
uv run python -m tools.eval_hsv_probe --out out/clothing-research/hsv/live
uv run python -m tools.eval_hsv_probe --full-grid --out out/clothing-research/hsv/full-grid
uv run --extra fba --extra parsing pytest -q
```

Each evaluation writes one comparison figure and a 12-row CSV. Run a parser comparison before the HSV probe so its landmark cache exists. Main and the parser combinations use Main's same local-background estimate and linear-light composite; FBA composites its estimated foreground and alpha in sRGB, so colors are not strictly like-for-like. FBA's automatic trimap and measurement details are described below. The fixed Main and previous-clothing Git commits must be available locally; results can change if landmarks are redetected rather than reused from cache.

## Earlier Clothing and FBA Experiment: 12 Photo Files

The tables and figures below describe the **earlier run before the close-up fallback**, using code `4276adf6a4cba13d0b98f4a8e333498e8a5c40db`. For example, the zero shot5 samples and zero FBA-combination change below are historical, not the current result.

We compared `shot2`–`shot5`, `test1`–`test6`, `test8`, and `test9` in `docs/evaluation/test`. The four photos in `docs/evaluation/crown-crop/originals/` are byte-identical copies of `shot2`–`shot5`, so they are not counted again.

The 12 files are not 12 independent scenes: `shot2`–`shot4` are very similar views of the same scene. The comparison and trimap sheets show `shot2` as their representative and retain the differently framed `shot5`; the table and CSVs still include all 12 photos.

The existing baseline is main commit `e96bb08cb678634207ea3aa45f086152c4c96f11`; the clothing implementation is `06ec619158d68d7c3417fd47ce8fd3703ae269ca`. The FBA reference uses a trimap derived from main alpha and landmarks; the existing combination uses a trimap derived from clothing alpha and the same landmarks. The more conservative Guarded FBA combination is described below. All process the **whole 640 px-wide frame**, including clothing, not the earlier head-only FBA crop. Each input and set of landmarks is shared; relighting strength is 0 and the replacement background is white. Each app run uses a new `Matter` with `update=False`, so color models do not carry over from another photograph.

**There are no ground-truth masks for the real photos.** The first table's numbers count pixels whose alpha was at most 0.5 in main and above 0.5 in ours; they are not comparisons against FBA. They are not counts of actual clothing pixels, accuracy, or recall; added background is included. Images have different sizes, so counts across photos are not comparable as quality scores. Observations refer to the live result; left and right are as viewed in the image.

| Photo | Ours added: live | Ours added: full grid | Ours: live result and remaining problem | FBA: live observation |
|---|---:|---:|---|---|
| shot2 | 15,927 | 5,355 | Much more khaki jacket remains, but outer sleeves and a small bright hole are still missing. | The jacket and hair remain more broadly, although the jacket is partly translucent. |
| shot3 | 7,916 | 14,472 | More jacket remains on both sides, but much of the left sleeve is missing. | More jacket is retained; the left side is partly transparent and some background remains. |
| shot4 | 6,699 | 6,157 | More outer jacket remains, but so does room clutter on the right. | Most of the jacket remains, with some background at the right. |
| shot5 | 0 | 0 | The close-up has no clothing samples, so the extra recovery is skipped. | Hair is retained more broadly, but much of the jacket still fades away. |
| test1 | 26,880 | 10,719 | More black patterned clothing remains, with shoulder holes and dark-room leakage. | More hair and clothing remain, with conspicuous room objects and haze. |
| test2 | 15,067 | 16,128 | More patterned clothing remains, but dark objects at the sides stay connected. | Hair and clothing remain, but the dark room and haze remain at both sides. |
| test3 | 7,138 | 7,011 | More gray clothing remains near the neck and shoulders; a neck notch remains. | The hair and shoulders remain more fully, with background haze on the right. |
| test4 | 7,185 | 7,109 | More clothing remains near the shoulder, along with a blue object and dark side clutter. | The hair and shoulders remain more fully, but dark objects remain at both sides. |
| test5 | 12,618 | 12,714 | The right white shoulder returns, but a left wall strip remains. | The shirt remains, together with wall texture and a dark strip on the left. |
| test6 | 11,753 | 12,569 | More outer white clothing remains, but wall behind the shoulders is also newly included. | The shirt remains, with faint wall texture behind the head. |
| test8 | 15,664 | 14,794 | Outer black sleeves return, with collar notches and blocky edges remaining. | The top remains more fully, but the fabric backdrop leaves haze. |
| test9 | 34,428 | 15,183 | More dark hoodie remains, but shadow-side gaps, attached background, and a clipped hair edge remain. | Hair and hoodie extend farther, but broad background haze remains. |

### FBA and Clothing Improvement + FBA: 12 Photos

The counts below are pixels added / removed by the **combined live result relative to FBA**. Each counts whole-frame pixels crossing the alpha 0.5 threshold in one direction. These are not actual clothing pixels or ground-truth errors; they include background leakage.

| Photo | Added / removed vs FBA | Combined result |
|---|---:|---|
| shot2 | 2,079 / 0 | The translucent jacket becomes more opaque, but some background remains on the right. |
| shot3 | 1,111 / 999 | The left jacket is less translucent, but some background remains. |
| shot4 | 22 / 433 | Most of the jacket remains, much like FBA, along with a background object on the right. |
| shot5 | 0 / 0 | No clothing samples are available; both trimaps and FBA alphas match, and the jacket still fades. |
| test1 | 18,831 / 439 | The top remains, but dark room objects at the sides are more prominent than in FBA. |
| test2 | 8,747 / 1 | The top remains connected to dark background objects on both sides. |
| test3 | 2,044 / 10 | The shoulders remain, but so does background haze on the right. |
| test4 | 2,979 / 91 | Hair and shoulders remain along with dark objects and haze on both sides. |
| test5 | 1,090 / 43 | The white top remains along with the left wall strip and wall texture. |
| test6 | 10,256 / 36 | The wall behind the right shoulder is more prominent than in FBA. |
| test8 | 38 / 2,640 | The black top and faint fabric-backdrop haze remain, with some alpha changes from FBA. |
| test9 | 65,273 / 0 | The hoodie is more opaque, but background around the head and upper left is much more prominent. |

### Guarded FBA: 12 Photos

Converting the existing clothing result to a trimap promotes added regions to sure foreground and also relaxes surrounding sure background to unknown. In live test6 and test9, respectively 16,263 and 31,940 pixels became sure foreground in the existing combined trimap, and 2,885 and 6,788 pixels changed from sure background to unknown. No sure-background pixels changed directly to sure foreground in these two photos. Many of the new sure-foreground pixels are clothing, but FBA can retain wall or shadow in the enlarged unknown area. These are class transitions, not counts of mislabeled background.

Guarded FBA **keeps the main trimap**, promoting an unknown pixel to sure foreground only when the clothing trimap marks it sure foreground and the first, main-based FBA alpha exceeds 0.75. Main's sure background is not relaxed; FBA then runs again on the original photo. The 0.75 cutoff is an experimental choice made after inspecting these photos, not validated on an independent set.

The live counts below are added / removed pixels relative to the existing clothing + FBA result at alpha 0.5. They include both clothing and background and are neither accuracy nor clothing-recovery measurements.

| Photo | Added / removed vs existing combination | Guarded FBA observation |
|---|---:|---|
| shot2 | 572 / 875 | The jacket is more opaque than in FBA, but partly less opaque than in the existing combination. |
| shot3 | 397 / 4 | Most of the jacket remains, although the left side is partly translucent. |
| shot4 | 51 / 8 | Similar to the existing combination, including a background object at the right. |
| shot5 | 0 / 0 | No clothing samples are available; the result matches FBA and the jacket still fades. |
| test1 | 134 / 11,701 | The top remains, but room objects and haze remain at both sides. |
| test2 | 24 / 1,728 | The top remains connected to dark room objects. |
| test3 | 6 / 1,895 | The shoulders remain along with background haze on the right. |
| test4 | 5 / 4,016 | Hair and clothing remain, with objects at both sides. |
| test5 | 0 / 1,072 | The white top remains along with the left wall strip and wall texture. |
| test6 | 0 / 9,456 | The top remains and wall behind the shoulder is lighter than in the existing combination, but not gone. |
| test8 | 803 / 3 | The top remains with background haze; its boundary differs from the existing combination. |
| test9 | 0 / 48,623 | The hoodie remains and background around the head is lighter than in the existing combination, but haze remains. |

In live experiments we also required `clothing_evidence` to exceed 0, 0.5, or 1.0: this reduced some background but from 0.5 removed the shot2 jacket again. Broadening the clothing shade range to `(0.6, 1.5)`, `(0.4, 2.5)`, or `(0.25, 4.0)` kept more clothing in some cases, but also retained more wall in test6 and test8, so neither variant was adopted. Guarded FBA is a limited offline experiment that reduces leakage in test6 and test9; it does not fix the standalone clothing result or change the app algorithm. The first FBA can itself mistakenly confirm background or miss clothing, and unknown regions can change on the second pass. It is not uniformly better on all photos.

### Jackets: shot2–shot5

Much of shot2's previously missing khaki jacket returns, but not every sleeve or background error is fixed. In shot5 the face fills the frame and there are zero clothing-sample candidates, so the baseline result is left unchanged.

### Dark room: test1–test4

In test1 and test2, a wider area of black clothing remains, but dark objects in the room remain with it. A larger mask alone does not establish that the clothing has been separated correctly.

### Light walls and other poses: test5, test6, test8, test9

The right white shoulder in test5 is wider than before, but in test6 the bright wall behind the shoulders is newly included as well. Separating similarly colored clothing and wall remains unsolved.

### FBA Reference, Combination, and Trimaps

FBA estimates foreground, background, and alpha from an image and a **trimap**, not from the image alone. The reference receives a trimap derived from main alpha; the combination receives a trimap derived from **clothing-improved alpha using the same rules**. Both use the original photo as FBA input. Thus this experiment runs clothing improvement first to change FBA's input, rather than applying the clothing model as a postprocess to an FBA result.

We threshold each alpha at 0.5, erode by 0.06 face widths to mark sure foreground, and mark the face mask as foreground. Pixels beyond a 0.5-face-width dilation are sure background, except inside a generous landmark-based body envelope below the chin; those pixels remain unknown unless already sure foreground. The envelope's half-width grows from 1 to 3.5 face widths over one eye-to-chin distance, using the square root of the normalized distance. Erosion and dilation use square windows, with radii rounded to the nearest pixel and a minimum of one pixel.



These are **FBA conditioned on the main or improved mask and landmarks**, not an independent automatic segmentation baseline or a manually corrected trimap. Background erroneously marked sure foreground cannot be removed, and clothing marked sure background cannot be recovered. If clothing improvement wrongly includes background that becomes sure foreground in the combined trimap, FBA cannot remove it. The wide body region allows FBA to decide on sleeves outside the old mask, but also allows wall leakage. The unknown share of the main-derived live trimaps ranges from 31.1% to 59.7% (median 42.5%); this describes the input, not quality. The combination makes shot2's jacket more opaque, but also makes background more prominent in test6 and test9, so it is not uniformly better.

FBA runs on the same resized 640 px input as the app comparisons, on CPU with four threads and a 1024 px tile size. All 12 frames fit in one tile; padding to a multiple of eight preserves the input pixel grid. The reference enforces the trimap's known labels and uses FBA's estimated foreground and alpha for an sRGB white composite, without additional refinement. The app instead composites in linear light with its own spill removal, so composite colors are not strictly like-for-like. In the previous single live run, the FBA call took a median 291 ms (234–1,491 ms range) and the existing combined call 292 ms (236–683 ms range). The existing combination and Guarded FBA are later calls on the same model, and Guarded FBA also needs the first FBA result. The numbers exclude model loading, trimap generation, landmark detection, and input-mask generation; they are not fair speed comparisons, end-to-end times, or real-time measurements.

The live path calculates color decisions and filter coefficients on a **half-size grid, `s=2`**, from the 640 px input. The full-grid comparison (local image) applies the shutter algorithm's **`s=1` to the same 640 px input** and derives both trimaps from that mode's main and improved alphas. The clothing-only shoulders in shot2 and shot4 remain fragmented on this path, while shot3 gains more than in live mode. **This is not an original-resolution export evaluation**, nor evidence that the two paths produce identical boundaries.

The full-grid FBA trimaps and alphas (local image) and main-versus-ours live alpha comparison (local image) are also available.

The [live CSV](docs/evaluation/clothing/measurements.csv) and [full-grid CSV](docs/evaluation/clothing/full-grid/measurements.csv) record thresholds, sample-candidate counts, application status, and single-run times for all 12 photos. Across the set, live mode adds 161,275 pixels and removes 9; full grid adds 122,211 and removes 6. Removed pixels had alpha above 0.5 before and at most 0.5 after. Recovery only adds to the binary mask **before** cleanup; subsequent cleanup and guided-filter recomputation can also change existing edges.

The [FBA live CSV](docs/evaluation/clothing/fba/measurements.csv) and [FBA full-grid CSV](docs/evaluation/clothing/full-grid-fba/measurements.csv) also record trimap class sizes, pixels added/removed for the existing results and Guarded FBA at alpha 0.5, tile counts, and processing times. These differences are not ground-truth errors or clothing accuracy.

The figures show white-background composites regardless of rule 7. The app can retain the original wall when rule 7 fails at the shutter, so the figures do not necessarily represent what would be saved.

## 1. Why Clothing Was Missing

The old gate serves two roles: in the scoring stage it excludes the person outside the assumed shape, and pixels outside it are candidates for the background color model. A shoulder wider than assumed can therefore be cut off and also teach the model that its fabric is background.

In the earlier [matting failure analysis](https://github.com/hanbin5/relight/blob/module-b/matting/README.md), about 20 background samples came from the end of test5's right white shoulder and formed a white cluster. In shot2, the khaki jacket resembled a dark shade of the gray blind cluster, and 59% of a landmark-defined outer-shoulder region was included in the mask. These are **observations from the earlier original-resolution analysis**, not recall against labeled clothing. They are not a before/after measurement paired with the 640 px comparison above.

The old person-color samples represent the face, hair, neck, and a narrow chest together. If a black inner top occupies the middle of the chest, an outer jacket's color may be poorly represented. In a dark room, black clothing and room objects can have similar colors. The torso prior of 0.7 is not certain foreground, so background color evidence and the threshold can still exclude clothing.

## 2. Separate Clothing Models and Guarded Background Samples

### Why and What Was Selected

The first attempt widened the global shoulder gate higher up and expanded chest samples. More clothing remained, but the global background/person models and Otsu threshold also changed, with visible changes to the hair boundary in shot3 and test9. Rather than redefine the head color model, we selected a separate clothing score. The intermediate code and figures for that first attempt were not retained, so this is not an independently reproducible quantitative ablation.

We also narrowed the allowed shading of clothing colors and required a close match to the clothing samples. This still did not prevent wall leakage in test6. Instead of replacing existing body labels, the selected method **adds only supported pixels to the existing mask** to reduce new holes; it can also retain background that was already incorrectly included.

### Process

1. Face width `fw`, eye-to-chin distance `ec`, and the midline from the landmarks define a clothing search region that widens higher up than the old shoulder gate. Pixels in a wide guard below the chin are excluded **only from clothing-background candidates**; the whole guard is not forced to foreground.
2. Clothing-color candidates come from the chest below the chin; clothing-background candidates are old background candidates outside the guard. For example, test5 has 20,243 clothing and 44,745 background candidates on the live processing grid; at most 2,000 from each are used by the models. Candidate regions can contain pixels that are not actually clothing or background.
3. Separate clothing foreground/background color models score the search region using color distances and a torso prior. A pixel is added only if its clothing score exceeds the original Otsu threshold and its distance to the clothing model is below 2.5.
4. The added decisions are combined with the old binary mask before the existing opening/closing, face connectivity, hole filling, and guided filter. The original head color models and scoring definitions are unchanged, but later connectivity and edge processing are not guaranteed to be pixel-identical.



For test5, the guard excludes 4,607 of the 49,352 old background candidates, leaving 44,745 for the clothing-background model. The blue region includes wall as well: it is not all contaminated samples or all clothing. The green region is not guaranteed to be clothing either.

The app's image processing uses NumPy. There is no additional segmentation or depth model, calibration frame, manual per-photo region, or photo-name-specific threshold. Facial landmarks still use the existing MediaPipe model.

### Settings and Insufficient Samples

| Item | Setting |
|---|---|
| Clothing search region | Starts at chin + 0.2 `ec`; half-width increases from 0.6 `fw` to 2.8 `fw` with a square-root progression over 0.8 `ec`, then fades across 0.25 `fw` |
| Background guard | Below the chin, excludes old background candidates inside that half-width plus 0.6 `fw` from clothing-background candidates |
| Clothing candidates | Chin + 0.65 `ec` through + 2.5 `ec`; half-width grows from 0.35 `fw` to 1.0 `fw` with a square-root progression over 1 `ec` |
| Neck protection | Additional decisions only below chin + 0.65 `ec` or at least 0.45 `fw` from the midline |
| Clothing prior | 0.5 at the margin, up to the existing `PRIOR_TORSO=0.7` inside the torso |
| Color models | 8 background and 5 clothing clusters, at most 2,000 samples each; sampling RNG seeds 3 and 4 respectively |
| k-means | 8 iterations cold, 3 on an updating frame with previous models |
| Shading and color support | Clothing linear-intensity scale 0.8–1.2; its square root is used for color distance. Distance to the clothing model must be below the existing cap of 2.5 |
| Threshold | Original score's Otsu value, clipped to 0.35–0.65 |

If the original clothing candidates number fewer than 50, the close-up fallback described above is attempted first. If the usable clothing samples or guarded background candidates still number fewer than 50 on the processing grid, the additional pass is skipped. On an updating frame, the old clothing models are also cleared; `Matter.reset()` clears both original and clothing models. Application status and candidate counts are recorded in `ctx.stats["matting"]`.

**The frame-border fallback still matters.** If the old background or old foreground candidates number fewer than 50, the original code replaces background candidates with the frame border. The clothing-background candidates then also come from that border minus the guard; if at least 50 remain, they can train the clothing-background model. They are not guaranteed to be clean wall pixels.

The guarded background model is used only for clothing scoring. The original head/background models, wall statistics, plain-wall check, and local-spill estimation keep their original definitions. This does not purify every background estimate in the pipeline.

## 3. Synthetic Regression Scenes With Known Masks

Unlike the real photos, the person region in each 320 × 280 synthetic scene is known when it is rendered. We constructed wider, higher shoulders and compared the same baseline and improved code. Retention below is the fraction of the **specified synthetic target region** whose alpha exceeds 0.5, not full-image IoU.

| Synthetic target | Target pixels | Baseline retention: live / full grid | Improved: live / full grid |
|---|---:|---:|---:|
| Khaki jacket outside a black inner top | 13,932 | 0% / 0% | 100% / 100% |
| White sleeve where the old gate is below 0.05 | 658 | 0% / 0% | 100% / 100% |
| Dark top with separate dark room clutter | 26,304 | 95.38% / 95.15% | 100% / 100% |

After recovery, false foreground in these scenes occupied about 0.013–0.016% of the **whole image**, not of the background alone. The tests require target retention above 98% and false foreground below 0.5% of the whole image. They also check the sampling guard and clearing warm state when the chest is cropped.

These simple scenes check the intended clothing-only regressions; they do not evaluate the FBA combination or establish accuracy on real photos, separation of identical clothing and wall colors, coverage of varied poses, or hair-boundary quality.

## 4. Reproduction and Limitations

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. These commands write under `out/` rather than overwriting the tracked figures and CSVs.

```bash
uv sync --group dev
uv run python tools/fetch_model.py

uv run python tools/eval_clothing.py --sample-figure --out out/clothing-report/live
uv run python tools/eval_clothing.py --full-grid --out out/clothing-report/full-grid

uv run pytest tests/test_matting_checker.py -q
uv run pytest -q
```

Each evaluation writes one comparison figure and `measurements.csv`; the figure shows 10 photos and the CSV retains all 12. `--sample-figure` also produces the `sample-regions.jpg` shown above. The default baseline is the fixed main commit above; `--baseline SHA` overrides it. The tool loads that revision's `matting.py` from Git using the current shared configuration, so the baseline commit must be present locally. The Main baseline does not use the clothing-shade and close-up sample-distance constants added on this branch. Rerunning these commands with current code can change shot5 and the historical FBA-combination values above.

To reproduce the FBA comparison, first review the [authors' license and model notice](https://github.com/MarcoForte/FBA_Matting). The source is MIT-licensed, but the released weights are **noncommercial only**; this comparison does not authorize commercial or product use. After acknowledging that restriction:

```bash
uv sync --extra fba --group dev
uv run --extra fba python tools/fba_reference.py --accept-noncommercial
uv run --extra fba python tools/eval_clothing.py --fba --out out/clothing-report/live-fba
uv run --extra fba python tools/eval_clothing.py --fba --full-grid --out out/clothing-report/full-grid-fba
uv run --extra fba pytest tests/test_matting_checker.py -q
uv run --extra fba pytest -q
```

The commands download the authors' source and approximately 139 MB of weights into ignored `models/fba/`; photos remain local. The adapter verifies source revision `8dd100d760186150710510ee8d78ed916cdb17a4` and weight SHA-256 `f3f8a2b6cefd7200368b6008a712468ed1debc6658693db31b0f719f658a2967` and loads weights with `weights_only=True`. Each FBA run writes one comparison figure including Guarded FBA, one trimap and alpha figure, and a CSV. Missing or invalid assets and trimaps raise errors rather than silently substituting another method.

Landmarks are detected in image mode and cached by default in `out/clothing-landmarks/`; the same values are used within each before/after pair. The cache is not tracked by Git. Redetection in a fresh environment or with a fresh `--cache` directory can change landmarks and measurements. Missing photos or faces raise errors instead of being silently omitted.

The earlier FBA experiment passed 215 tests overall, including 45 matting/checker tests. Main and improved CSV times are single cold-run measurements of the matting stage only; they exclude loading, landmark detection, and relighting. They are diagnostics, not a latency guarantee or a 10 fps claim.

This is a **bounded clothing-recovery improvement, not a general portrait-segmentation solution**. Recovery is skipped when even the close-up fallback cannot obtain enough samples; similar-colored objects in test1/test2/test9 and the wall in test6 can remain. Crossed arms or an occluded chest can invalidate the assumed sample and shoulder locations. Original-resolution export, warm starts on real sequences, landmark jitter, and sustained webcam performance were not evaluated. Accuracy or production-readiness claims need hand-drawn masks and temporal evaluation.

For general app usage, see the [main README](https://github.com/hanbin5/relight/blob/main/README.md). The app produces edited photos and is not for passports or other government IDs.
