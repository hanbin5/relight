# Relighting before and after

The [comparison image](before-after.png) takes one frame from a screen recording of the
app under a side light, with relighting off, and relights it at 100% with `main`
(`e0f193a`) and with `relight-improvement`. Only the face is cropped.

- **Before** (`main`): the correction peaks beside the nose, which leaves a gray patch
  there and an orange cheek on the brightened side; R_LR overshoots to 1.22.
- **After**: left-right evening is measured between mirror-symmetric mesh points and skin
  color is evened too; R_LR is 0.92 (target 0.90–1.10). A faint specular highlight beside
  the nose remains.

R_LR is shown as the app shows it (the recording is mirrored). The numbers at the bottom
of the image come from 3 side-lit frames and the 13 test images; they are measurements on
these images, not a general accuracy result.
