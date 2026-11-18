# Part D: checker and evaluation — steps

Owner: D. Branch: `module-d/checker-eval`. Plan sections: B.4 (rules), C.1 (evaluation), Appendix D (blind rating, usability test).

The checker itself (`idphoto/checker.py`) and the auto shutter already work. What's left is testing them on real people and running the two studies.

## Steps

| # | Step | When | Status |
|---|---|---|---|
| 1 | `tools/eval_checker.py`: precision and recall per rule on a folder of labeled photos | Nov 17–18 | Done |
| 2 | Collect the test set: ~20 people × 1 good photo + 7 staged faults | Nov 18–20 | To do |
| 3 | Tune the thresholds in `config.py` from the CSV the script saves; target ≥ 90% recall per rule | Nov 20–21 | To do |
| 4 | Blind rating: 50 raters, relit vs unrelit (pass: ≥ 32 of 50 prefer relit) | Nov 16–22 | To do |
| 5 | Usability test: 10 users, our app vs phone camera + an app (pass: fewer retakes for ≥ 6 of 10) | Nov 19–22 | To do |
| 6 | Identity check: ArcFace similarity, relit vs unrelit and AI profile vs unrelit | after Nov 22 if needed | To do |
| 7 | Write up results for the report, including anything that missed its target | Nov 22–28 | To do |

Nov 22 is the feature freeze (C.3).

## How to take test photos (step 2)

For each person, take one photo of each of these and save it in the matching folder under `testset/`. Use the same file name in every folder (for example `alex.jpg`):

| Folder | What to do |
|---|---|
| `good/` | Face the camera, even light, eyes open, mouth closed |
| `side_shadow/` | Light from one side only (a window or a desk lamp) |
| `glare/` | Glasses with a light reflecting in the lenses |
| `blur/` | Move during the shot, or smudge the lens |
| `tilt/` | Tilt the head to one side |
| `eyes_closed/` | Close your eyes |
| `mouth_open/` | Open your mouth / talk |
| `top_like_wall/` | Wear a top close to the wall color |

`uv run python tools/capture_frames.py` works for taking them, or any webcam photo. Then:

```bash
uv run python tools/eval_checker.py testset/
```

Get consent before taking anyone's photo. `testset/` is in `.gitignore`, so **don't commit photos**; share them only inside the team, and delete them after the semester (C.1).

## What D needs from everyone

- **Everyone:** sit for the test photos (step 2), and if you can, bring a friend or two.
- **Everyone:** help find raters for the blind rating (step 4); a link in a group chat is fine.
- **A:** if you already ran ArcFace for the Nov 8 relighting gate, tell D so step 6 isn't done twice.
- **A and C:** changes to relighting or the crop box can change rules 1 and 7, so mention it when you merge.

## Notes so far

- First run on the 11 photos in `docs/evaluation/test/` (all good photos, 5 people), Nov 18: 0 of 11 passed every rule with the draft bounds. Rules 2, 5 and 6 passed all 11 (the rule 6 worry was only the synthetic face). Rule 4 failed 10 (mostly pitch ratio 1.16–1.68), rule 3 failed 7 (low-res webcams), rule 7 failed 6.
- Rule 1 counted bangs and hair at the temples as shadow (9–33% of the face; glasses frames were barely involved). Fixed: the shadow share is now measured below the eyes, 4 px inside the face edge. On the same photos it is 0–2%, and rule 1 failed 1 of 11 instead of 11 (test1, R_LR 0.89). Without relighting, a synthetic side light still gives 13–38% shadow, so real shadows are still caught.
- `eval_checker.py` runs each photo on its own, so rule 3 uses the starting bound (120), not the per-camera one. Pick each camera's bound from the sharpness values in the CSV.
