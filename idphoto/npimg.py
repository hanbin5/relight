"""NumPy image-processing primitives shared by all modules.

The plan uses OpenCV only for camera and image I/O; everything here is NumPy: box
filters from cumulative sums (O(N) for any radius), the guided filter (He et al., ECCV
2010), the 3 x 3 Laplacian, Otsu's threshold, binary morphology, separable bicubic and
bilinear resampling, polygon fill and z-buffered triangle rasterization.

Coordinates are continuous pixel units: pixel (i, j) covers [j, j + 1) x [i, i + 1), so its
center is (j + 0.5, i + 0.5). Landmarks use the same convention.
"""

import numpy as np

# ---------------------------------------------------------------------------------------
# Box filters


def _box_sum_rows(a: np.ndarray, r: int) -> np.ndarray:
    """Windowed sum along axis 0 from a running sum (row-by-row adds are the fastest in NumPy)."""
    n = a.shape[0]
    k = min(r, n - 1)
    c = np.array(a, np.float32)  # becomes c[i] = a[0] + ... + a[i]
    for i in range(1, n):
        c[i] += c[i - 1]
    # out[i] = c[min(i + k, n - 1)] - c[i - k - 1] (the second term is 0 for i <= k)
    out = np.empty_like(c)
    out[: n - k] = c[k:]
    if k > 0:
        out[n - k :] = c[n - 1]
    if k + 1 < n:
        out[k + 1 :] -= c[: n - k - 1]
    return out


def window_counts(n: int, r: int) -> np.ndarray:
    """Number of in-frame samples in a radius-r window at each of n positions."""
    idx = np.arange(n)
    return (np.minimum(idx + r + 1, n) - np.maximum(idx - r, 0)).astype(np.float32)


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r + 1) x (2r + 1) window, clipped at the frame border. a is (H, W[, C])."""
    rows = _box_sum_rows(a, r)
    t = np.ascontiguousarray(np.swapaxes(rows, 0, 1))
    return np.ascontiguousarray(np.swapaxes(_box_sum_rows(t, r), 0, 1))


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over the in-frame part of a (2r + 1) x (2r + 1) window."""
    cnt = np.outer(window_counts(a.shape[0], r), window_counts(a.shape[1], r))
    if a.ndim == 3:
        cnt = cnt[..., None]
    return (box_sum(a, r) / cnt).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Guided filter (B.3)


def guided_filter_coeffs(I: np.ndarray, p: np.ndarray, r: int, eps: float):
    """Window-averaged coefficients (a_bar, b_bar) of the guided filter q = a_bar I + b_bar.

    a_k = (mean(I p) - mu_k p_bar_k) / (sigma_k^2 + eps), b_k = p_bar_k - a_k mu_k.
    Returning the averaged coefficients lets callers upsample them (fast guided upsampling).
    """
    m = box_mean(np.stack([I, p, I * I, I * p], axis=-1), r)
    mean_i, mean_p, corr_ii, corr_ip = m[..., 0], m[..., 1], m[..., 2], m[..., 3]
    a = (corr_ip - mean_i * mean_p) / (corr_ii - mean_i * mean_i + eps)
    b = mean_p - a * mean_i
    ab = box_mean(np.stack([a, b], axis=-1), r)
    return ab[..., 0], ab[..., 1]


def guided_filter_color_coeffs(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Window-averaged coefficients (a_r, a_g, a_b, b) of the color-guided filter.

    He et al.'s color-guide variant: a_k = (Sigma_k + eps U)^-1 cov_k(I, p), b_k = p_bar_k - a_k . mu_k,
    with the 3 x 3 inverse written out per pixel (adjugate over determinant).
    """
    r_, g_, b_ = I[..., 0], I[..., 1], I[..., 2]
    m = box_mean(np.stack([r_, g_, b_, p, r_ * p, g_ * p, b_ * p,
                           r_ * r_, r_ * g_, r_ * b_, g_ * g_, g_ * b_, b_ * b_], axis=-1), r)
    mr, mg, mb, mp = m[..., 0], m[..., 1], m[..., 2], m[..., 3]
    cr, cg, cb = m[..., 4] - mr * mp, m[..., 5] - mg * mp, m[..., 6] - mb * mp
    vrr, vrg, vrb = m[..., 7] - mr * mr + eps, m[..., 8] - mr * mg, m[..., 9] - mr * mb
    vgg, vgb, vbb = m[..., 10] - mg * mg + eps, m[..., 11] - mg * mb, m[..., 12] - mb * mb + eps
    irr, irg, irb = vgg * vbb - vgb * vgb, vgb * vrb - vrg * vbb, vrg * vgb - vgg * vrb
    igg, igb, ibb = vrr * vbb - vrb * vrb, vrb * vrg - vrr * vgb, vrr * vgg - vrg * vrg
    det = vrr * irr + vrg * irg + vrb * irb
    ar = (irr * cr + irg * cg + irb * cb) / det
    ag = (irg * cr + igg * cg + igb * cb) / det
    ab = (irb * cr + igb * cg + ibb * cb) / det
    b = mp - ar * mr - ag * mg - ab * mb
    return box_mean(np.stack([ar, ag, ab, b], axis=-1), r)


def guided_filter_color(I: np.ndarray, p: np.ndarray, r: int, eps: float, subsample: int = 1) -> np.ndarray:
    """Edge-preserving smoothing of p guided by a color image I (H, W, 3); see guided_filter."""
    if subsample <= 1:
        c = guided_filter_color_coeffs(I, p, r, eps)
    else:
        h, w = p.shape
        hs, ws = max(h // subsample, 1), max(w // subsample, 1)
        small = resize(np.concatenate([I, p[..., None]], axis=-1), hs, ws, kernel="linear")
        c = guided_filter_color_coeffs(small[..., :3], small[..., 3], max(r // subsample, 1), eps)
        c = resize(c, h, w, kernel="linear")
    return c[..., 0] * I[..., 0] + c[..., 1] * I[..., 1] + c[..., 2] * I[..., 2] + c[..., 3]


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float, subsample: int = 1) -> np.ndarray:
    """Edge-preserving smoothing of p guided by the gray image I.

    subsample > 1 computes the coefficients on a coarser grid and upsamples them (He & Sun's
    fast guided filter); the live view uses 2, the saved photo the exact filter.
    """
    if subsample <= 1:
        a, b = guided_filter_coeffs(I, p, r, eps)
        return a * I + b
    h, w = I.shape
    hs, ws = max(h // subsample, 1), max(w // subsample, 1)
    small = resize(np.stack([I, p], axis=-1), hs, ws, kernel="linear")
    a, b = guided_filter_coeffs(small[..., 0], small[..., 1], max(r // subsample, 1), eps)
    ab = resize(np.stack([a, b], axis=-1), h, w, kernel="linear")
    return ab[..., 0] * I + ab[..., 1]


# ---------------------------------------------------------------------------------------
# Filters, thresholds and morphology


def laplacian3(g: np.ndarray) -> np.ndarray:
    """3 x 3 Laplacian [[0, 1, 0], [1, -4, 1], [0, 1, 0]] with edge replication."""
    p = np.pad(np.asarray(g, np.float32), 1, mode="edge")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]


def otsu_threshold(values: np.ndarray, bins: int = 256, lo: float = 0.0, hi: float = 1.0) -> float:
    """Otsu's threshold: maximizes the between-class variance of a 1-D histogram."""
    v = np.asarray(values, np.float32).ravel()
    if v.size == 0:
        return 0.5 * (lo + hi)
    q = np.clip(((v - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
    p = np.bincount(q, minlength=bins).astype(np.float64)
    p /= p.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(bins))
    between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    between = between[:-1]
    # Any cut in an empty gap between the classes scores the same; take the gap's middle.
    best = np.flatnonzero(between >= between.max() * (1.0 - 1e-6))
    k = int(round(float(best.mean())))
    return lo + (k + 1) / bins * (hi - lo)


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with a (2r + 1) square; the frame border does not erode."""
    if r <= 0:
        return mask.astype(bool)
    cnt = np.outer(window_counts(mask.shape[0], r), window_counts(mask.shape[1], r))
    return box_sum(mask.astype(np.float32), r) >= cnt - 0.5


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with a (2r + 1) square."""
    if r <= 0:
        return mask.astype(bool)
    return box_sum(mask.astype(np.float32), r) > 0.5


def opening(mask: np.ndarray, r: int) -> np.ndarray:
    return dilate(erode(mask, r), r)


def closing(mask: np.ndarray, r: int) -> np.ndarray:
    return erode(dilate(mask, r), r)


# ---------------------------------------------------------------------------------------
# Resampling (B.5: separable bicubic; scaling is always uniform)


def _cubic(x: np.ndarray, a: float = -0.5) -> np.ndarray:
    x = np.abs(x)
    x2, x3 = x * x, x * x * x
    near = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
    far = a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
    return np.where(x <= 1.0, near, np.where(x < 2.0, far, 0.0))


def _linear(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.abs(x))


_KERNELS = {"cubic": (_cubic, 2.0), "linear": (_linear, 1.0)}


def resample_taps(n_in: int, n_out: int, start: float, length: float, kernel: str = "cubic"):
    """Source indices and weights, both (n_out, taps), that resample [start, start + length)
    of a 1-D signal to n_out samples.

    The kernel is widened when downscaling (antialiasing); taps outside the signal are
    clamped to its edge. Each row of weights sums to one.
    """
    f, support = _KERNELS[kernel]
    scale = length / n_out
    s = max(scale, 1.0)
    centers = start + (np.arange(n_out) + 0.5) * scale - 0.5
    ntaps = int(np.ceil(2.0 * support * s)) + 1
    k = np.floor(centers - support * s).astype(np.int64)[:, None] + 1 + np.arange(ntaps)[None, :]
    w = f((k - centers[:, None]) / s)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.clip(k, 0, n_in - 1), w.astype(np.float32)


def _apply_taps(img: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Resample along axis 0 with per-output taps (gathers rows, no dense matrices)."""
    shape = (-1,) + (1,) * (img.ndim - 1)
    out = img[idx[:, 0]] * w[:, 0].reshape(shape)
    for t in range(1, idx.shape[1]):
        out += img[idx[:, t]] * w[:, t].reshape(shape)
    return out


def resize(img: np.ndarray, out_h: int, out_w: int, kernel: str = "cubic", box=None) -> np.ndarray:
    """Separable resampling of img (H, W[, C]) to (out_h, out_w), float32.

    box = (x0, y0, x1, y1) in continuous pixel coordinates selects the source window
    (default: the whole image); parts outside the image take the nearest edge value.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box if box is not None else (0.0, 0.0, float(w), float(h))
    iy, wy = resample_taps(h, out_h, y0, y1 - y0, kernel)
    ix, wx = resample_taps(w, out_w, x0, x1 - x0, kernel)
    rows = _apply_taps(img, iy, wy)  # gathers only the needed rows; float32 from here on
    cols = _apply_taps(np.ascontiguousarray(np.swapaxes(rows, 0, 1)), ix, wx)
    return np.ascontiguousarray(np.swapaxes(cols, 0, 1))


def downscale_u8(img: np.ndarray, factor: int) -> np.ndarray:
    """Average factor x factor blocks of a uint8 image (fast integer-factor downscale)."""
    if factor <= 1:
        return img
    h, w = img.shape[0] // factor * factor, img.shape[1] // factor * factor
    total = np.zeros((h // factor, w // factor) + img.shape[2:], np.uint32)
    for dy in range(factor):
        for dx in range(factor):
            total += img[dy:h:factor, dx:w:factor]
    return ((total + factor * factor // 2) // (factor * factor)).astype(np.uint8)


def downscale_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Average factor x factor blocks of a float image (rows and columns past a whole block are dropped)."""
    if factor <= 1:
        return img
    h, w = img.shape[0] // factor * factor, img.shape[1] // factor * factor
    total = np.zeros((h // factor, w // factor) + img.shape[2:], np.float32)
    for dy in range(factor):
        for dx in range(factor):
            total += img[dy:h:factor, dx:w:factor]
    return total / (factor * factor)


def upsample_nearest(img: np.ndarray, factor: int, shape) -> np.ndarray:
    """Repeat each pixel factor x factor times and fit the result to shape (h, w) by edge padding."""
    big = np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)[: shape[0], : shape[1]]
    pad = ((0, shape[0] - big.shape[0]), (0, shape[1] - big.shape[1])) + ((0, 0),) * (img.ndim - 2)
    return np.pad(big, pad, mode="edge") if any(p[1] for p in pad) else big


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples of a 2-D image at continuous coordinates (edge-clamped)."""
    h, w = img.shape[:2]
    fx = np.clip(np.asarray(xs, np.float64) - 0.5, 0, w - 1)
    fy = np.clip(np.asarray(ys, np.float64) - 0.5, 0, h - 1)
    x0 = np.minimum(fx.astype(np.int64), w - 2 if w > 1 else 0)
    y0 = np.minimum(fy.astype(np.int64), h - 2 if h > 1 else 0)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    tx, ty = fx - x0, fy - y0
    top = img[y0, x0] * (1 - tx) + img[y0, x1] * tx
    bot = img[y1, x0] * (1 - tx) + img[y1, x1] * tx
    return top * (1 - ty) + bot * ty


# ---------------------------------------------------------------------------------------
# Rasterization


def fill_polygon(shape, pts) -> np.ndarray:
    """Boolean mask of the pixels whose centers lie inside a polygon (even-odd rule)."""
    h, w = shape
    pts = np.asarray(pts, np.float64)
    mask = np.zeros((h, w), bool)
    x0 = max(int(np.floor(pts[:, 0].min())), 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + 1, w)
    y0 = max(int(np.floor(pts[:, 1].min())), 0)
    y1 = min(int(np.ceil(pts[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return mask
    ys = (np.arange(y0, y1) + 0.5)[:, None]
    xs = (np.arange(x0, x1) + 0.5)[None, :]
    inside = np.zeros((y1 - y0, x1 - x0), bool)
    xa, ya = pts[:, 0], pts[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    for k in range(len(pts)):
        if ya[k] == yb[k]:
            continue
        crosses = (ya[k] > ys) != (yb[k] > ys)
        x_hit = xa[k] + (ys - ya[k]) * (xb[k] - xa[k]) / (yb[k] - ya[k])
        inside ^= crosses & (xs < x_hit)
    mask[y0:y1, x0:x1] = inside
    return mask


def rasterize_triangles(v2d: np.ndarray, depth: np.ndarray, tris: np.ndarray, shape):
    """Z-buffered rasterization of a triangle mesh, fully vectorized.

    v2d: (V, 2) vertex positions in pixels; depth: (V,) with smaller = closer;
    tris: (T, 3) vertex indices. Returns tri_id (H, W) int32 (-1 = no triangle) and the
    barycentric weights (H, W, 3) of each covered pixel center.
    """
    h, w = shape
    a, b, c = v2d[tris[:, 0]], v2d[tris[:, 1]], v2d[tris[:, 2]]
    xs = np.stack([a[:, 0], b[:, 0], c[:, 0]], axis=1)
    ys = np.stack([a[:, 1], b[:, 1], c[:, 1]], axis=1)
    xmin = np.clip(np.ceil(xs.min(axis=1) - 0.5), 0, w).astype(np.int64)
    xmax = np.clip(np.floor(xs.max(axis=1) - 0.5), -1, w - 1).astype(np.int64)
    ymin = np.clip(np.ceil(ys.min(axis=1) - 0.5), 0, h).astype(np.int64)
    ymax = np.clip(np.floor(ys.max(axis=1) - 0.5), -1, h - 1).astype(np.int64)
    bw, bh = xmax - xmin + 1, ymax - ymin + 1
    den = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    cnt = np.where((bw > 0) & (bh > 0) & (np.abs(den) > 1e-12), bw * bh, 0)

    tri_id = np.full(h * w, -1, np.int32)
    bary = np.zeros((h * w, 3), np.float32)
    total = int(cnt.sum())
    if total == 0:
        return tri_id.reshape(h, w), bary.reshape(h, w, 3)
    tid = np.repeat(np.arange(len(tris)), cnt)
    k = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    px = xmin[tid] + k % bw[tid]
    py = ymin[tid] + k // bw[tid]
    qx, qy = px + 0.5 - a[tid, 0], py + 0.5 - a[tid, 1]
    e1x, e1y = b[tid, 0] - a[tid, 0], b[tid, 1] - a[tid, 1]
    e2x, e2y = c[tid, 0] - a[tid, 0], c[tid, 1] - a[tid, 1]
    d = den[tid]
    l1 = (qx * e2y - e2x * qy) / d
    l2 = (e1x * qy - qx * e1y) / d
    l0 = 1.0 - l1 - l2
    keep = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
    tid, px, py, l0, l1, l2 = tid[keep], px[keep], py[keep], l0[keep], l1[keep], l2[keep]
    t = tris[tid]
    z = l0 * depth[t[:, 0]] + l1 * depth[t[:, 1]] + l2 * depth[t[:, 2]]
    pix = py * w + px
    order = np.lexsort((z, pix))  # by pixel, then nearest first
    first = np.ones(order.size, bool)
    first[1:] = pix[order][1:] != pix[order][:-1]
    sel = order[first]
    tri_id[pix[sel]] = tid[sel]
    bary[pix[sel]] = np.stack([l0[sel], l1[sel], l2[sel]], axis=1)
    return tri_id.reshape(h, w), bary.reshape(h, w, 3)
