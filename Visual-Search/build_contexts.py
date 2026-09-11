"""Reconstruct the displays and precompute the model's sensory evidence.

One pass: read dataset/saccades.csv (from pool_data.py), normalize
colors, assign a context id to every unique (setsize, targLoc,
singLoc, targCol, singCol, fixation) combination, render each unique
display once (feature search: target diamond among heterogeneous
nontarget shapes, singleton in the study's opposite color), compute
opponency-contrast maps, and sample template-rotated radial profiles
along rays from each fixation.

Reconstruction assumptions: per-trial set size, colors, and item
distances come from the data files; shapes, item size, and background
are paper-sourced (the OSF trial files carry no display parameters).
Color fallbacks: junk/NaN targCol -> the subject's modal valid value,
else green; missing singCol on singleton-present trials -> the
opponent of the target color.

Output: dataset/contexts_v21.npz (profiles P [ctx, 6, NBINS, 3]:
template axis D_T, orthogonal D_O, presence D_P; FORM [ctx, 6]; range
R [ctx, 6]) + dataset/saccades_ctx.csv (saccades with ctx ids).

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
NRAYS = 90
WEDGE_DEG = 30.0
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


def wedge_profiles(maps, u, fix_xy, setsize):
    """Per item: signed radial profiles D_T, D_O and unsigned D_P,
    averaged over the item's wedge rays; plus range R (first energy)."""
    to_px = lambda v: (v + 0.75) / 1.5 * IMG
    fx, fy = to_px(fix_xy[0]), to_px(fix_xy[1])
    radii = np.linspace(0.09, MAXR, NBINS)
    pos, _ = item_positions(setsize)
    prof = np.zeros((6, NBINS, 3))
    R = np.ones(6)
    th_all = np.arange(NRAYS) * 2 * np.pi / NRAYS
    for j in range(setsize):
        dx, dy = pos[j][0] - fix_xy[0], pos[j][1] - fix_xy[1]
        tj = np.arctan2(dy, dx) % (2 * np.pi)
        dd = np.abs((th_all - tj + np.pi) % (2 * np.pi) - np.pi)
        rays = np.where(dd < np.deg2rad(WEDGE_DEG))[0]
        if len(rays) == 0:
            rays = [int(np.argmin(dd))]
        acc = np.zeros((NBINS, 3))
        rr = []
        for i in rays:
            th = th_all[i]
            px = np.clip(fx + to_px(radii * np.cos(th)) - to_px(0),
                         0, IMG - 1).astype(int)
            py = np.clip(fy + to_px(radii * np.sin(th)) - to_px(0),
                         0, IMG - 1).astype(int)
            crg = maps["RG"][py, px]
            cby = maps["BY"][py, px]
            acc[:, 0] += u[0] * crg + u[1] * cby          # template axis
            acc[:, 1] += -u[1] * crg + u[0] * cby         # orthogonal
            pprof = maps["P"][py, px]
            acc[:, 2] += pprof
            hit = np.where(pprof > 0.15 * maps["P"].max())[0]
            rr.append(radii[hit[0]] if len(hit) else MAXR)
        prof[j] = acc / len(rays)
        R[j] = float(np.mean(rr))
    return prof, R


def main():
    sacc = pd.read_csv("dataset/saccades.csv", low_memory=False)
    sacc = normalize_colors(sacc)
    keys = (sacc[["setsize", "targLoc", "singLoc", "targCol", "singCol",
                  "fixloc"]].drop_duplicates().reset_index(drop=True))
    keys["ctx"] = np.arange(len(keys))
    n = len(keys)
    print(f"{n} unique contexts")
    P = np.zeros((n, 6, NBINS, 3), dtype=np.float32)
    F = np.zeros((n, 6), dtype=np.float32)
    R = np.ones((n, 6), dtype=np.float32)
    cache = {}
    for _, k in keys.iterrows():
        dk = (int(k.setsize), int(k.targLoc), int(k.singLoc),
              k.targCol, k.singCol)
        if dk not in cache:
            setsize, targLoc, singLoc, targCol, singCol = dk
            pos, _ = item_positions(setsize)
            items = [dict(x=pos[j][0], y=pos[j][1],
                          color=(singCol if (j + 1) == singLoc
                                 and singCol != "none" else targCol),
                          shape=shape_for(j + 1, targLoc))
                     for j in range(setsize)]
            img = render(items)
            img_hi = render(items, scale=2)
            cache[dk] = (opponency_contrast(img), shape_match_map(img_hi),
                         template_axis(targCol))
        maps, fmap, u = cache[dk]
        setsize = int(k.setsize)
        pos, _ = item_positions(setsize)
        fix = (0.0, 0.0) if k.fixloc == 0 else pos[int(k.fixloc) - 1]
        ci = int(k.ctx)
        P[ci], R[ci] = wedge_profiles(maps, u, fix, setsize)
        sc = fmap.shape[0] // IMG
        to_pxf = lambda v: (v + 0.75) / 1.5 * IMG * sc
        for j in range(setsize):
            F[ci, j] = fmap[int(np.clip(to_pxf(pos[j][1]), 0, IMG * sc - 1)),
                            int(np.clip(to_pxf(pos[j][0]), 0, IMG * sc - 1))]
        if ci % 300 == 0:
            print(f"  {ci}/{n}", flush=True)
    P[..., :2] /= max(np.abs(P[..., :2]).std(), 1e-9)
    P[..., 2] /= max(P[..., 2].std(), 1e-9)
    F /= max(F.std(), 1e-9)
    np.savez_compressed("dataset/contexts_v21.npz", P=P, FORM=F, R=R)
    merged = sacc.merge(keys, on=["setsize", "targLoc", "singLoc",
                                  "targCol", "singCol", "fixloc"], how="left")
    assert merged.ctx.notna().all()
    merged.to_csv("dataset/saccades_ctx.csv", index=False)
    print(f"saved contexts_v21.npz + saccades_ctx.csv "
          f"({n} contexts, {len(cache)} unique displays)")


if __name__ == "__main__":
    main()
