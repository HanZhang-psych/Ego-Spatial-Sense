"""Reconstruct the displays and precompute the model's SENSED evidence.

One pass: read dataset/saccades.csv (from pool_data.py), normalize
colors, assign a context id to every unique (setsize, targLoc,
singLoc, targCol, singCol, fixation) combination, render each unique
display once (feature search: target circle among heterogeneous
nontarget shapes, singleton in the opposite color), compute the
pre-window evidence maps (rectified template/distractor color
channels + the pixel-derived shape-match map), and SENSE each map at
the item centers (display_senses) - the point-sensing readout
F_i = P(x_i) adopted 2026-09-11 (see RESULTS). The cache is exact by
construction: the model only ever looks at the priority map at those
pixels.

The history kernel (PAINT_SIG = 0.03, peak height 1 - the
point-readout unit convention: a fully primed own location senses as
exactly beta) is precomputed as the matrix of each item's bump
sensed at every center (kernel_matrix; ~identity at this sigma).

Reconstruction assumptions: per-trial set size, colors, and item
distances come from the data files; shapes, item size, and background
are paper-sourced (the OSF trial files carry no display parameters).
All displays use the canonical green-target / red-singleton scheme
(see normalize_colors).

Channel units are GREYSCALE: the color channel (signed
template-axis contrast D_T) is divided by GREY_C - the strongest
|D_T| pixel of the canonical green/red display - so its pixels lie
in [-1, 1]; the shape channel is divided by its max, so it peaks at
1. With the history kernel's peak-1 convention, every weight then
reads the same way: priority delivered by a full-strength unit of
its channel.

Output: dataset/senses.npz -- A [ctx, 6, 2] (sensed greyscale
channels), GREY [1] (the color constant, for the record),
BH6/BH4 [6, 6] (sensed history kernel per set size). Plus
dataset/saccades_ctx.csv (saccades with ctx ids).

Usage: python build_contexts.py
"""

import numpy as np
import pandas as pd

from front_end import (COLORS, IMG, _gauss_blur, item_positions,
                       render, shape_for)




def normalize_colors(sacc):
    """Canonical color scheme: each subject's colors were fixed for
    their whole session, so the model only ever sees match-vs-mismatch
    structure - every display is reconstructed with a GREEN target
    color and a RED singleton. (Recorded cost: Stilwell 2023's
    within-study singleton-salience color manipulation is invisible to
    this reconstruction.)"""
    sacc = sacc.copy()
    sacc["targCol"] = "green"
    sacc["singCol"] = np.where(sacc.singLoc.values > 0, "red", "none")
    return sacc


def opponency_contrast(img):
    """Measure local visual contrast in color-opponent coordinates.

    An Itti-Koch-like center-surround front end that does not collapse
    into one unsigned salience map: it returns signed red-vs-green
    (RG) and blue-vs-yellow (BY) contrast maps, so the model can tell
    which color direction differs from the surround, plus an unsigned
    intensity/presence map (P). The goal later rotates the signed maps
    into target-relative axes; fitted gains build the goal-modified
    priority map.
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    out = {}
    for name, m in [("RG", r - g), ("BY", b - (r + g) / 2)]:
        c = np.zeros_like(m)
        for s_c, s_s in [(2, 8), (4, 16), (4, 48)]:
            c += _gauss_blur(m, s_c) - _gauss_blur(m, s_s)   # signed
        c[:10, :] = c[-10:, :] = c[:, :10] = c[:, -10:] = 0
        out[name] = c
    # presence: contrast of the color DEVIATION from the background -
    # fires on any visible object, whatever its color (intensity alone
    # is blind to items equiluminant with the background, and its
    # nonzero background level created image-border artifacts)
    from front_end import BG
    dev = np.sqrt(((img - np.array(BG)) ** 2).sum(-1))
    p = np.zeros_like(dev)
    for s_c, s_s in [(2, 8), (4, 16)]:
        p += np.abs(_gauss_blur(dev, s_c) - _gauss_blur(dev, s_s))
    p[:10, :] = p[-10:, :] = p[:, :10] = p[:, -10:] = 0
    out["P"] = p
    return out


def shape_kernels(r_px):
    """Binary footprint kernels at item scale for each shape."""
    n = int(3.2 * r_px) | 1
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    ks = {}
    ks["circle"] = ((xx - c) ** 2 + (yy - c) ** 2 < r_px ** 2)
    ks["square"] = (np.abs(xx - c) < 0.95 * r_px) & (np.abs(yy - c) < 0.95 * r_px)
    ks["diamond"] = (np.abs(xx - c) + np.abs(yy - c)) < 1.3 * r_px
    ks["triangle"] = ((yy - c > -0.9 * r_px)
                      & (np.abs(xx - c) < 0.95 * r_px
                         * (1 - (yy - c + 0.9 * r_px) / (2.0 * r_px))))
    ks["cross"] = (((np.abs(xx - c) < 0.45 * r_px) & (np.abs(yy - c) < 1.1 * r_px))
                   | ((np.abs(yy - c) < 0.45 * r_px) & (np.abs(xx - c) < 1.1 * r_px)))
    return {k: v.astype(float) for k, v in ks.items()}


def shape_match_map(img, template_shape="circle"):
    """PIXEL-DERIVED shape evidence: correlate the display's
    background-deviation map with the template-shape kernel, with the
    average all-shape kernel subtracted (so plain "an object is here"
    energy cancels and only shape-DISTINCTIVE structure remains).
    Graded and confusable by construction - a square partially matches
    a circle. Replaces the earlier analytic label channel. Note the
    remaining scope limit: the trial data never record item shapes, so
    displays are reconstructed with the template at the target's
    location; this channel makes the EVIDENCE pathway realistic, not
    the display's provenance."""
    from front_end import BG, ITEM_R
    scale = img.shape[0] // IMG        # supports hi-res renders
    r_px = ITEM_R / 1.5 * IMG * scale

    ks = shape_kernels(r_px)
    K = ks[template_shape]
    K = K / np.sqrt((K ** 2).sum())
    ones = np.ones_like(K)
    dev = np.sqrt(((img - np.array(BG)) ** 2).sum(-1))
    dev = (dev > 0.15).astype(float)   # binarize: shape, not color amplitude
    n = K.shape[0]

    def corr(kern):
        F = np.fft.rfft2(dev) * np.conj(np.fft.rfft2(kern, dev.shape))
        out = np.fft.irfft2(F, dev.shape)
        return np.roll(out, (n // 2, n // 2), axis=(0, 1))

    den = np.sqrt(np.maximum(corr(ones), 1e-9))

    def ncc(shape):
        Ks = ks[shape] / np.sqrt((ks[shape] ** 2).sum())
        return corr(Ks) / den          # normalized cross-correlation

    # discriminative match: template NCC minus the best competing shape's
    m = ncc(template_shape) - np.max(
        [ncc(sh) for sh in ks if sh != template_shape], axis=0)
    return _gauss_blur(np.maximum(m, 0), 2 * scale)


def template_axis(targCol):
    rgb = np.array(COLORS.get(targCol, COLORS["green"]))
    rg = rgb[0] - rgb[1]
    by = rgb[2] - (rgb[0] + rgb[1]) / 2
    n = np.hypot(rg, by)
    return (rg / n, by / n) if n > 1e-6 else (1.0, 0.0)


PAINT_SIG = 0.03      # fixed history-smoothing kernel (model assumption)

_GREY_C = None


def grey_color_constant():
    """Fixed full-scale unit for the color channel: the strongest
    |D_T| pixel of the canonical green/red set-size-6 display (the
    notebook's Sec. 3 demo display - target slot 2, singleton slot 5
    - so both pipelines share the constant exactly). Dividing by it
    makes the color map greyscale (pixels in [-1, 1]), so g_C reads
    as the priority delivered by a full-strength color pixel - the
    same convention as the shape channel (max-normalized, peak 1)
    and the history kernel (peak 1: beta = a fully primed
    location)."""
    global _GREY_C
    if _GREY_C is None:
        pos, _ = item_positions(6)
        items = [dict(x=pos[k][0], y=pos[k][1],
                      color=("red" if k + 1 == 5 else "green"),
                      shape=shape_for(k + 1, 2)) for k in range(6)]
        maps = opponency_contrast(render(items))
        u = template_axis("green")
        _GREY_C = float(np.abs(u[0] * maps["RG"] + u[1] * maps["BY"]).max())
    return _GREY_C


def item_centers(setsize):
    """The sensed pixels: each item's center, as (row, col)."""
    pos, _ = item_positions(setsize)
    return [(int(round((p[1] + 0.75) / 1.5 * IMG)),
             int(round((p[0] + 0.75) / 1.5 * IMG))) for p in pos]


def display_senses(setsize, targLoc, singLoc, targCol, singCol):
    """One display -> the model's sensed evidence A [6, 2]: the two
    channel maps (SIGNED template-axis color contrast D_T, shape
    match / its max) evaluated at each item's center, in GREYSCALE
    units (color divided by grey_color_constant, shape by its max).
    singCol only affects the rendering - the single goal gain g_C
    weighs the signed projection of whatever colors are on screen.
    Mirrors the notebook's channel_maps exactly."""
    pos, _ = item_positions(setsize)
    items = [dict(x=pos[k][0], y=pos[k][1],
                  color=(singCol if (k + 1) == singLoc and singCol != "none"
                         else targCol),
                  shape=shape_for(k + 1, targLoc)) for k in range(setsize)]
    img = render(items)
    maps = opponency_contrast(img)
    u = template_axis(targCol)
    gmap = u[0] * maps["RG"] + u[1] * maps["BY"]     # signed D_T
    gmap = gmap / grey_color_constant()              # greyscale: [-1, 1]
    sm = shape_match_map(img)
    sm = sm / max(sm.max(), 1e-9)                    # greyscale: peak 1
    chans = np.stack([gmap, sm])
    A = np.zeros((6, 2), dtype=np.float32)
    for j, (py, px) in enumerate(item_centers(setsize)):
        A[j] = chans[:, py, px]
    return A


def kernel_matrix(setsize):
    """BH[i, j]: item j's history-kernel bump (peak height 1 at its own
    center - a fully primed own location senses as exactly beta)
    sensed at item i's center. ~Identity at PAINT_SIG = 0.03."""
    pos, _ = item_positions(setsize)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    px_per_unit = IMG / 1.5
    cs = item_centers(setsize)
    BH = np.zeros((6, 6), dtype=np.float32)
    for j in range(setsize):
        px = (pos[j][0] + 0.75) / 1.5 * IMG
        py = (pos[j][1] + 0.75) / 1.5 * IMG
        bump = np.exp(-((xx - px) ** 2 + (yy - py) ** 2)
                      / (2 * (PAINT_SIG * px_per_unit) ** 2))
        for i in range(setsize):
            BH[i, j] = bump[cs[i][0], cs[i][1]]
    return BH


def main():
    sacc = pd.read_csv("dataset/saccades.csv", low_memory=False)
    sacc = normalize_colors(sacc)
    keys = (sacc[["setsize", "targLoc", "singLoc", "targCol", "singCol",
                  "fixloc"]].drop_duplicates().reset_index(drop=True))
    keys["ctx"] = np.arange(len(keys))
    n = len(keys)
    print(f"{n} unique contexts")
    A = np.zeros((n, 6, 2), dtype=np.float32)
    cache = {}
    for _, k in keys.iterrows():
        dk = (int(k.setsize), int(k.targLoc), int(k.singLoc),
              k.targCol, k.singCol)
        if dk not in cache:
            cache[dk] = display_senses(*dk)
        A[int(k.ctx)] = cache[dk]
    np.savez_compressed("dataset/senses.npz", A=A,
                        GREY=np.array([grey_color_constant()]),
                        BH6=kernel_matrix(6), BH4=kernel_matrix(4))
    merged = sacc.merge(keys, on=["setsize", "targLoc", "singLoc",
                                  "targCol", "singCol", "fixloc"], how="left")
    assert merged.ctx.notna().all()
    merged.to_csv("dataset/saccades_ctx.csv", index=False)
    print(f"saved senses.npz + saccades_ctx.csv "
          f"({n} contexts, {len(cache)} unique displays)")


if __name__ == "__main__":
    main()
