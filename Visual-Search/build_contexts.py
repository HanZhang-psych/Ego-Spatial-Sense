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

from front_end import (COLORS, IMG, _gauss_blur, form_map, item_positions,
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
    sacc = sacc.copy()
    sacc["targCol"] = sacc.targCol.str.lower()
    sacc["singCol"] = sacc.singCol.str.lower()
    fixed = []
    for (_, _), sub in sacc.groupby(["study", "subj"], sort=False):
        tc = sub.targCol.where(sub.targCol.map(valid_color))
        mode = tc.mode()
        t_fallback = mode.iloc[0] if len(mode) else "green"
        t = tc.fillna(t_fallback)
        s = sub.singCol.where(sub.singCol.map(valid_color))
        s = s.fillna(t.map(OPPONENT))
        s = s.where(sub.singLoc > 0, "none")
        out = sub.copy()
        out["targCol"] = t
        out["singCol"] = s
        fixed.append(out)
    return pd.concat(fixed)


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
    i = (r + g + b) / 3
    p = np.zeros_like(i)
    for s_c, s_s in [(2, 8), (4, 16)]:
        p += np.abs(_gauss_blur(i, s_c) - _gauss_blur(i, s_s))
    p[:10, :] = p[-10:, :] = p[:, :10] = p[:, -10:] = 0
    out["P"] = p
    return out


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
            cache[dk] = (opponency_contrast(img), form_map(items),
                         template_axis(targCol))
        maps, fmap, u = cache[dk]
        setsize = int(k.setsize)
        pos, _ = item_positions(setsize)
        fix = (0.0, 0.0) if k.fixloc == 0 else pos[int(k.fixloc) - 1]
        ci = int(k.ctx)
        P[ci], R[ci] = wedge_profiles(maps, u, fix, setsize)
        to_px = lambda v: (v + 0.75) / 1.5 * IMG
        for j in range(setsize):
            F[ci, j] = fmap[int(np.clip(to_px(pos[j][1]), 0, IMG - 1)),
                            int(np.clip(to_px(pos[j][0]), 0, IMG - 1))]
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
