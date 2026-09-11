"""Reconstruct stimuli and precompute pixel-derived evidence (v2).

Reconstruction assumptions (licensed by the paradigm's inclusion
criteria and per-trial targCol/singCol columns): items on an
iso-eccentric ring; all items in the target color except the singleton,
which is in the opposite color; the target is the template shape
(diamond) among HETEROGENEOUS nontarget shapes (feature search - the
target is never a shape singleton, per the source studies' design). Displays repeat massively, so the front-end
runs once per unique context (setsize, targLoc, singLoc, targCol,
singCol, fixation) and every saccade looks its context up.

Per context, per item: wedge-integrated channel evidence from the ray
scan - SIM_T (target-color similarity), SIM_S (singleton-color
similarity), SAL (Itti&Koch salience incl. a wide-surround scale so a
color singleton pops against its neighbors), FORM (analytic template
shape; the stated shortcut) - plus the sensor range reading (distance
of first energy toward that item; feeds the envelope).

Color fallbacks: junk/NaN targCol -> the subject's modal valid value,
else green; missing singCol on singleton-present trials -> the
opponent of the target color.

Output: dataset/contexts.npz + a ctx index column appended to
dataset/saccades_ctx.csv.
"""

import numpy as np
import pandas as pd

from front_end import (COLORS, ECC, IMG, _gauss_blur, form_map,
                       render, ray_scan, shape_for)

OPPONENT = {"green": "red", "red": "green", "blue": "orange",
            "orange": "blue", "pink": "teal", "teal": "pink"}
COLORS.setdefault("orange", (0.95, 0.55, 0.05))
WEDGE_DEG = 30.0


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


def item_positions(setsize):
    ang = [2 * np.pi * j / setsize - np.pi / 2 for j in range(setsize)]
    return [(ECC * np.cos(a), ECC * np.sin(a)) for a in ang], ang


def color_similarity_maps(img, rgb_t, rgb_s):
    sim = {}
    for name, rgb in [("SIM_T", rgb_t), ("SIM_S", rgb_s)]:
        if rgb is None:
            sim[name] = np.zeros(img.shape[:2])
            continue
        d2 = ((img - np.array(rgb)) ** 2).sum(-1)
        sim[name] = _gauss_blur(np.exp(-d2 / (2 * 0.12 ** 2)), 2)
    return sim


def salience_map(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    sal = np.zeros(img.shape[:2])
    for m in (r - g, b - (r + g) / 2, (r + g + b) / 3):
        for s_c, s_s in [(2, 8), (4, 16), (4, 48)]:  # incl. wide surround
            sal += np.abs(_gauss_blur(m, s_c) - _gauss_blur(m, s_s))
    sal[:10, :] = sal[-10:, :] = sal[:, :10] = sal[:, -10:] = 0
    return sal / sal.max() if sal.max() > 0 else sal


def build_context(setsize, targLoc, singLoc, targCol, singCol):
    pos, _ = item_positions(setsize)
    items = []
    for j in range(setsize):
        color = singCol if (j + 1) == singLoc and singCol != "none" else targCol
        items.append(dict(x=pos[j][0], y=pos[j][1], color=color,
                          shape=shape_for(j + 1, targLoc)))
    img = render(items)
    maps = color_similarity_maps(
        img, COLORS.get(targCol),
        COLORS.get(singCol) if singCol != "none" else None)
    maps["SAL"] = salience_map(img)
    maps["FORM"] = form_map(items)
    return items, maps


def wedge_evidence(maps, fix_xy, setsize):
    scan, rng, names = ray_scan(
        {k: maps[k] for k in ("SIM_T", "SIM_S", "SAL", "FORM")}, fix_xy)
    order = ["SIM_T", "SIM_S", "SAL", "FORM"]
    idx = [names.index(n) for n in order]
    pos, _ = item_positions(setsize)
    n_rays = scan.shape[0]
    th = np.arange(n_rays) * 2 * np.pi / n_rays
    E = np.zeros((6, len(order)))
    R = np.ones(6)
    for j in range(setsize):
        dx, dy = pos[j][0] - fix_xy[0], pos[j][1] - fix_xy[1]
        tj = np.arctan2(dy, dx) % (2 * np.pi)
        dd = np.abs((th - tj + np.pi) % (2 * np.pi) - np.pi)
        w = dd < np.deg2rad(WEDGE_DEG)
        if w.sum() == 0:
            w = dd <= dd.min() + 1e-9
        E[j] = scan[w][:, idx].mean(0)
        R[j] = rng[w].mean()
    return E, R


def main():
    sacc = pd.read_csv("dataset/saccades.csv", low_memory=False)
    sacc = normalize_colors(sacc)
    keys = sacc[["setsize", "targLoc", "singLoc", "targCol", "singCol",
                 "fixloc"]].drop_duplicates().reset_index(drop=True)
    print(f"{len(keys)} unique contexts")
    Es = np.zeros((len(keys), 6, 4), dtype=np.float32)
    Rs = np.ones((len(keys), 6), dtype=np.float32)
    cache = {}
    for i, k in keys.iterrows():
        disp_key = (k.setsize, k.targLoc, k.singLoc, k.targCol, k.singCol)
        if disp_key not in cache:
            cache[disp_key] = build_context(*disp_key)
        _, maps = cache[disp_key]
        pos, _ = item_positions(int(k.setsize))
        fix = (0.0, 0.0) if k.fixloc == 0 else pos[int(k.fixloc) - 1]
        Es[i], Rs[i] = wedge_evidence(maps, fix, int(k.setsize))
        if i % 200 == 0:
            print(f"  {i}/{len(keys)}", flush=True)
    # z-scale each channel over real item slots so gains are comparable
    for c in range(4):
        slot = np.concatenate([Es[i, :int(keys.setsize[i]), c]
                               for i in range(len(keys))])
        Es[:, :, c] /= max(slot.std(), 1e-9)
    keys["ctx"] = np.arange(len(keys))
    np.savez_compressed("dataset/contexts.npz", E=Es, R=Rs)
    merged = sacc.merge(keys, on=["setsize", "targLoc", "singLoc",
                                  "targCol", "singCol", "fixloc"], how="left")
    assert merged.ctx.notna().all()
    merged.to_csv("dataset/saccades_ctx.csv", index=False)
    print(f"saved contexts.npz ({len(keys)} contexts, "
          f"{len(cache)} unique displays) and saccades_ctx.csv")


if __name__ == "__main__":
    main()
