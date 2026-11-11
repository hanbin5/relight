"""Starting values from the plan, Appendix B.2–B.5.

All values here are drafts to tune in weeks 2–4 (relighting, matting) and week 5
(checker thresholds), not measured results. Pixel sizes are given at REF_WIDTH and
scaled with the frame width, so a 1280-px shutter frame behaves like the 640-px live view.
"""

# Frame sizes ---------------------------------------------------------------------------
CAMERA_SIZE = (1280, 720)  # requested camera resolution (w, h); the shutter frame
LIVE_WIDTH = 640  # live pipeline runs on frames downscaled to this width
REF_WIDTH = 640  # pixel radii below are given at this width

# B.2 Camera lock and real-time relighting ----------------------------------------------
WARMUP_S = 1.5  # let auto exposure / white balance settle, then lock
LOCK_CHECK_S = 1.0  # hold-still window for the lock check
LOCK_MAX_DRIFT = 0.03  # face brightness may drift at most 3% while holding still
FIT_SIZE = 128  # light is fitted on a FIT_SIZE x FIT_SIZE face crop
SMOOTH_FRAMES = 5  # fitted light is averaged over the last 5 frames
SMOOTH_FRAMES_FALLBACK = 10  # C.4 fallback when the exposure lock is not available
KEEP_FIT_FRAMES = 15  # keep the last good fit this long when the face is lost
RATIO_CLIP = (0.5, 2.0)  # clip of target / fitted shading
TARGET_ELEVATION_DEG = 20.0  # target light: frontal, slightly from above
TARGET_AMBIENT = 0.8  # soft fill added to the target's key light (relative to the key)
FIT_RIDGE = 1e-3  # small ridge on the l = 2 terms keeps the 9-term fit stable
FIT_Y_RANGE = (0.004, 0.92)  # skin pixels outside this linear range count as clipped
RATIO_GF_RADIUS = 4  # guided upsampling of the ratio, radius at FIT_SIZE
RATIO_GF_EPS = 1e-2
DEFAULT_STRENGTH = 1.0  # s in [0, 1] blends the ratio toward 1

# B.3 Single-frame matting and background replacement -----------------------------------
BG_CLUSTERS, FG_CLUSTERS = 8, 5  # colors in the background and foreground models
CLUSTER_SAMPLES = 2000  # pixels sampled per model each frame
PLAIN_WALL_SHARE = 0.7  # rule 7 asks for a plain wall when less of the background is its main color
CLUSTER_SPREAD_FLOOR = 0.03  # smallest cluster spread, in sqrt-linear color (sensor noise)
BG_SHADOW_SCALE = (0.5, 1.25)  # a background color this much darker/brighter is still background
FG_SHADE_SCALE = (0.6, 1.5)  # shading varies across the person too
SCORE_TAU = 1.0  # spreads of distance difference per unit of log-odds
OUTLIER_DISTANCE = 2.5  # distances beyond this many spreads count the same (color seen by neither model)
PRIOR_CORE = 0.88  # prior probability of the person in the skull and the neck
PRIOR_TORSO = 0.7  # ... and inside the shoulder line, where shoulder shapes vary more
OPEN_RADIUS, CLOSE_RADIUS = 2, 5  # morphology at REF_WIDTH
# Guided filter for the edge. B.3 starts at r = 15 px, eps = 1e-3 (luminance guide); with the
# color guide, a 15-px window spread a gray halo along edges the mask placed a few px off.
GF_RADIUS = 6
GF_EPS = 2e-3
SPILL_ALPHA_FLOOR = 0.1  # F = (I - (1 - a) mu_B) / max(a, 0.1)

# B.4 Checker rules ----------------------------------------------------------------------
EVEN_LIGHT_RANGE = (0.90, 1.10)  # rule 1: R_LR
SHADOW_FACTOR = 0.45  # rule 1: a skin pixel darker than 0.45 x face mean is shadow
SHADOW_MAX = 0.05  # rule 1: shadow share < 5%
GLARE_Y, GLARE_SAT = 0.94, 0.12  # rule 2: glare pixel
GLARE_FACE_MAX = 0.01  # rule 2: < 1% of the face
GLARE_LENS_MAX = 0.005  # rule 2: "no glare on lenses", with a tiny tolerance for noise
SHARPNESS_MIN = 120.0  # rule 3: variance of the Laplacian on a 256 x 256 face crop
SHARPNESS_CROP = 256
SHARPNESS_WINDOW = 90  # recent frames that calibrate the camera (about 5 s)
SHARPNESS_WARMUP = 15  # frames before the calibration applies
SHARPNESS_REL = 0.7  # a frame must reach 70% of the camera's sharp frames ...
SHARPNESS_FLOOR = 15.0  # ... and never less than this
ROLL_MAX_DEG = 2.5  # rule 4
YAW_RATIO_MAX = 0.06  # rule 4
PITCH_RATIO_RANGE = (0.85, 1.15)  # rule 4, relative to the canonical face (1.0 = level)
EAR_MIN = 0.23  # rule 5
LIP_GAP_MAX = 0.03  # rule 6
UNCERTAIN_MAX = 0.02  # rule 7: 0.1 < alpha < 0.9 outside the hair band
CONTRAST_MIN = 0.12  # rule 7: person-wall contrast at the shoulder line
SHUTTER_FRAMES = 5  # auto shutter: all rules pass for 5 frames in a row

# B.5 Crop geometry -----------------------------------------------------------------------
CROWN_K = 2.0  # crown ~ chin - k (chin - eye line)
CROWN_MASK_AGREE = (0.85, 1.25)  # mask-top cue is used only if it agrees with the proportion
CROWN_MASK_WEIGHT = 0.5  # share of the mask-top cue when it is used (0 = proportion only, 1 = mask only)
# Crop mode. "head": the head with its hair fills head_share of the photo (plan B.5).
# "face": chin to skull top (the proportion rule, hair left out) fills face_share, so faces come
# out the same size whatever the hairstyle; the hair top (the crown above) only moves the crop up
# or zooms out so the hair is not cut. Per-format shares are in geometry.FORMATS.
CROP_MODE = "head"
HAIR_MARGIN_MIN = 0.04  # face mode: least room above the hair, as a share of the photo height
CHIN_MAX = 0.92  # face mode: lowest the chin may sit, as a share of the photo height
PRINT_DPI = 300
SHEET_PX = (1800, 1200)  # 6 x 4 in landscape at 300 DPI (w, h)
