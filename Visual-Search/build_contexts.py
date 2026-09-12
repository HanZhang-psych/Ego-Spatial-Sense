"""Reconstruct the displays and precompute the model's sensory evidence.

One pass: read dataset/saccades.csv (from pool_data.py), normalize
colors, assign a context id to every unique (setsize, targLoc,
singLoc, targCol, singCol, fixation) combination, render each unique
display once (feature search: target circle among heterogeneous
nontarget shapes, singleton in the opposite color), compute the
pre-window evidence maps (template-rotated color channels + the
pixel-derived shape-match map), and regroup each map EXACTLY into
sector x distance-bin area averages (sector_geometry / _bin_map):
summing a map's binned contributions against the attention window's
per-bin values reproduces the sector average of window * map, so
training on these tables is the pixel construction, not an
approximation of it.

The painted history field is precomputed the same way as pure
geometry: history_matrix() bins each item's unit-history kernel bump
(fixed smoothing PAINT_SIG = 0.03, normalized so each bump carries
unit own-sector mass - sigma sets spread, not weight).

Reconstruction assumptions: per-trial set size, colors, and item
distances come from the data files; shapes, item size, and background
are paper-sourced (the OSF trial files carry no display parameters).
All displays use the canonical green-target / red-singleton scheme
(see normalize_colors).

Output: dataset/contexts.npz -- P [ctx, 6, NBINS, 3] (binned D_T,
D_O, presence), FORMP [ctx, 6, NBINS] (binned shape match), HM6/HM4
[6, 6, NBINS] (binned history-kernel geometry per setsize); each
array std-normalized as in main(). Plus dataset/saccades_ctx.csv
(saccades with ctx ids).

Usage: python build_contexts.py
"""

import numpy as np
import pandas as pd

from front_end import (COLORS, IMG, _gauss_blur, item_positions,
                       render, shape_for)

OPPONENT = {"green": "red", "red": "green", "blue": "orange",
            "orange": "blue", "pink": "teal", "teal": "pink"}
COLORS.setdefault("orange", (0.95, 0.55, 0.05))
NBINS = 24
MAXR = 1.1


def valid_color(c):
    return isinstance(c, str) and c in COLORS


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


def sector_geometry(setsize):
    """Pixel-level geometry for the sector x distance-bin readout:
    for each item sector (pie slice of pi/setsize half-angle... width
    2*pi/setsize), assign every pixel within MAXR of the center to a
    (sector, distance bin) cell. Returns (sector_idx, bin_idx, denom):
    -1 outside; denom = pixels per sector within MAXR."""
    pos, _ = item_positions(setsize)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    to_unit = lambda p: (p / IMG) * 1.5 - 0.75
    ux, uy = to_unit(xx + 0.5), to_unit(yy + 0.5)
    dist = np.hypot(ux, uy)
    ang = np.arctan2(uy, ux)
    radii = np.linspace(0.09, MAXR, NBINS)
    half = (radii[1] - radii[0]) / 2
    edges = np.concatenate([[0.0], (radii[:-1] + radii[1:]) / 2,
                            [radii[-1] + half]])
    bin_idx = np.digitize(dist, edges) - 1
    bin_idx[(dist >= edges[-1])] = -1
    sector_idx = np.full(ang.shape, -1)
    halfw = np.pi / setsize
    for k in range(setsize):
        ak = np.arctan2(pos[k][1], pos[k][0])
        dd = np.abs((ang - ak + np.pi) % (2 * np.pi) - np.pi)
        sector_idx[(dd < halfw) & (bin_idx >= 0)] = k
    denom = float((sector_idx >= 0).sum()) / setsize
    return sector_idx, bin_idx, denom


def _bin_map(m, sector_idx, bin_idx, denom, setsize):
    """Sector x distance-bin CONTRIBUTIONS of map m: out[i, d] = (sum of
    m over the cell) / (pixels per sector), so that (window(d) *
    out).sum(d) equals the sector average of window * m."""
    out = np.zeros((6, NBINS))
    ok = sector_idx >= 0
    flat = sector_idx[ok] * NBINS + bin_idx[ok]
    sums = np.bincount(flat, weights=m[ok], minlength=setsize * NBINS)
    out[:setsize] = sums.reshape(setsize, NBINS) / denom
    return out


def display_evidence(setsize, targLoc, singLoc, targCol, singCol):
    """One display -> the pre-window map, binned: P [6, NBINS, 3]
    (target-axis, orthogonal, presence contributions) and FP [6, NBINS]
    (shape-map contributions). Unnormalized."""
    pos, _ = item_positions(setsize)
    items = [dict(x=pos[k][0], y=pos[k][1],
                  color=(singCol if (k + 1) == singLoc and singCol != "none"
                         else targCol),
                  shape=shape_for(k + 1, targLoc)) for k in range(setsize)]
    img = render(items)
    maps = opponency_contrast(img)
    u = template_axis(targCol)
    perp = (-u[1], u[0])
    gmap = u[0] * maps["RG"] + u[1] * maps["BY"]
    omap = perp[0] * maps["RG"] + perp[1] * maps["BY"]
    smap = shape_match_map(render(items, scale=2))[::2, ::2]
    si, bi, denom = sector_geometry(setsize)
    P = np.stack([_bin_map(gmap, si, bi, denom, setsize),
                  _bin_map(omap, si, bi, denom, setsize),
                  _bin_map(maps["P"], si, bi, denom, setsize)], axis=-1)
    FP = _bin_map(smap, si, bi, denom, setsize)
    return P.astype(np.float32), FP.astype(np.float32)


def history_matrix(setsize):
    """HM[i, j, d]: sector i's binned contribution when item j's history
    equals 1, using the fixed smoothing kernel PAINT_SIG. The model's
    painted history field, precomputed as geometry."""
    pos, _ = item_positions(setsize)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    px_per_unit = IMG / 1.5
    si, bi, denom = sector_geometry(setsize)
    HM = np.zeros((6, 6, NBINS), dtype=np.float32)
    for jslot in range(setsize):
        px = (pos[jslot][0] + 0.75) / 1.5 * IMG
        py = (pos[jslot][1] + 0.75) / 1.5 * IMG
        bump = np.exp(-((xx - px) ** 2 + (yy - py) ** 2)
                      / (2 * (PAINT_SIG * px_per_unit) ** 2))
        HM[:, jslot] = _bin_map(bump, si, bi, denom, setsize)
    # normalize so a unit history contributes unit binned mass in its
    # own sector (pure reparameterization; keeps the betas O(1) and
    # makes PAINT_SIG control spread only, not weight)
    HM /= max(float(np.einsum("iid->i",
                              HM[:setsize, :setsize]).mean()), 1e-9)
    return HM


def main():
    sacc = pd.read_csv("dataset/saccades.csv", low_memory=False)
    sacc = normalize_colors(sacc)
    keys = (sacc[["setsize", "targLoc", "singLoc", "targCol", "singCol",
                  "fixloc"]].drop_duplicates().reset_index(drop=True))
    keys["ctx"] = np.arange(len(keys))
    n = len(keys)
    print(f"{n} unique contexts")
    P = np.zeros((n, 6, NBINS, 3), dtype=np.float32)
    FP = np.zeros((n, 6, NBINS), dtype=np.float32)
    cache = {}
    for _, k in keys.iterrows():
        dk = (int(k.setsize), int(k.targLoc), int(k.singLoc),
              k.targCol, k.singCol)
        if dk not in cache:
            cache[dk] = display_evidence(*dk)
        ci = int(k.ctx)
        P[ci], FP[ci] = cache[dk]
        if ci % 300 == 0:
            print(f"  {ci}/{n}", flush=True)
    P[..., :2] /= max(np.abs(P[..., :2]).std(), 1e-9)
    P[..., 2] /= max(P[..., 2].std(), 1e-9)
    FP /= max(FP.std(), 1e-9)
    np.savez_compressed("dataset/contexts.npz", P=P, FORMP=FP,
                        HM6=history_matrix(6), HM4=history_matrix(4))
    merged = sacc.merge(keys, on=["setsize", "targLoc", "singLoc",
                                  "targCol", "singCol", "fixloc"], how="left")
    assert merged.ctx.notna().all()
    merged.to_csv("dataset/saccades_ctx.csv", index=False)
    print(f"saved contexts.npz + saccades_ctx.csv "
          f"({n} contexts, {len(cache)} unique displays)")


if __name__ == "__main__":
    main()
