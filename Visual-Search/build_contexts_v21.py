"""v2.1 contexts: goal-modified salience map (goal INSIDE the contrast).

Architecture change from v2 (user decision): no separate task-blind
salience channel. The front-end delivers signed multiscale
center-surround contrast on FIXED opponency axes (RG, BY, intensity);
the task set rotates these into template-referenced axes (the goal
supplies the direction of the target color in opponency space; the
fitted gain supplies only strength); the assembled map is the
RECTIFIED gain-weighted sum - so feature-level suppression can only
attenuate/relegate (drive toward zero), never write negatively.
Negative writing remains exclusive to the spatial sources (location
traces, IoR), applied downstream.

Because DoG is linear, gains commute with it: we precompute, per
context and per item wedge, the signed radial contrast PROFILES along
the template axis (D_T) and its orthogonal (D_O), plus an unsigned
intensity-contrast presence profile (D_P). At fit time the model
computes relu(g_T*D_T + g_O*D_O + w_p*D_P) per radial bin and
integrates - differentiable in the gains with no image ops.

Output: dataset/contexts_v21.npz (profiles [ctx, 6, NBINS, 3], FORM
[ctx, 6], R [ctx, 6]) aligned with dataset/saccades_ctx.csv ctx ids.
"""

import numpy as np
import pandas as pd

from build_contexts import (COLORS, item_positions, normalize_colors)
from front_end import ECC, IMG, _gauss_blur, form_map, render

NBINS = 24
NRAYS = 90
WEDGE_DEG = 30.0
MAXR = 1.1


def opponency_contrast(img):
    """Measure local visual contrast in color-opponent coordinates.

    This is an Itti-Koch-like center-surround front end, but it does
    not collapse everything into one unsigned salience map. It returns
    signed red-vs-green (RG) and blue-vs-yellow (BY) contrast maps, so
    the model can tell which color direction differs from the
    surround, plus an unsigned intensity/presence map (P) that says
    where any visible object contrast exists. Later code rotates the
    signed color maps into target-relative axes and combines them with
    fitted gains to build the goal-modified priority map.
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
            px = np.clip(fx + to_px(radii * np.cos(th)) - to_px(0), 0, IMG - 1).astype(int)
            py = np.clip(fy + to_px(radii * np.sin(th)) - to_px(0), 0, IMG - 1).astype(int)
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
    sacc = pd.read_csv("dataset/saccades_ctx.csv", low_memory=False)
    keys = (sacc[["setsize", "targLoc", "singLoc", "targCol", "singCol",
                  "fixloc", "ctx"]].drop_duplicates()
            .sort_values("ctx").reset_index(drop=True))
    n = int(keys.ctx.max()) + 1
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
                          shape=("diamond" if (j + 1) == targLoc
                                 else "circle"))
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
        # FORM: wedge-mean of the analytic template-shape map at item dirs
        to_px = lambda v: (v + 0.75) / 1.5 * IMG
        for j in range(setsize):
            F[ci, j] = fmap[int(np.clip(to_px(pos[j][1]), 0, IMG - 1)),
                            int(np.clip(to_px(pos[j][0]), 0, IMG - 1))]
        if ci % 300 == 0:
            print(f"  {ci}/{n}", flush=True)
    # scale for comparable gains
    P[..., :2] /= max(np.abs(P[..., :2]).std(), 1e-9)
    P[..., 2] /= max(P[..., 2].std(), 1e-9)
    F /= max(F.std(), 1e-9)
    np.savez_compressed("dataset/contexts_v21.npz", P=P, FORM=F, R=R)
    print(f"saved contexts_v21.npz ({n} contexts, {len(cache)} displays)")


if __name__ == "__main__":
    main()
