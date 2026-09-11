"""Pixel front-end for the search model: the visual LiDAR.

Renders a search display to an RGB image, computes Itti & Koch-style
feature maps (color opponency R-G and B-Y, intensity; center-surround
contrast via difference-of-Gaussians at two scales; combined salience),
and samples every map along rays from the current fixation - the exact
analog of the agent's LiDAR scan. Nothing here is fitted; this is the
perception module (sensor), fixed by construction like the agent's
LiDAR.

Outputs per fixation: a [n_rays, n_channels] scan, where each ray
carries (feature channels..., salience, form-match) integrated over
radial bins weighted toward the nearest energy - plus the radial
distance at which each ray first meets item energy (the "range
reading", used only for diagnostics; the model itself consumes the
scan, and eccentricity is implicit in the fixation-centered sampling).

The form-match channel (target shape) is the one stated shortcut: shape
identity is painted analytically rather than extracted from pixels (see
docs/priority_field_visual_search_model.md, front-end section).

Demo: python front_end.py  -> figures/front_end_demo.png
"""

import numpy as np

# display geometry (normalized units: ring radius 0.5 as in the fits)
IMG = 256               # rendered resolution (square)
ITEM_R = 0.055          # item radius in ring-diameter units
ECC = 0.5               # ring radius

COLORS = {
    "green": (0.10, 0.75, 0.20),
    "red": (0.85, 0.10, 0.10),
    "pink": (0.95, 0.45, 0.70),
    "teal": (0.05, 0.65, 0.65),
    "blue": (0.15, 0.30, 0.90),
    "gray": (0.55, 0.55, 0.55),
}
BG = (0.35, 0.35, 0.35)


def render(items):
    """items: list of dicts {x, y, color, shape ('circle'|'diamond')}
    in centered coords (units of ring diameter). Returns [IMG, IMG, 3]."""
    img = np.ones((IMG, IMG, 3)) * np.array(BG)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    to_px = lambda v: (v + 0.75) / 1.5 * IMG
    r_px = ITEM_R / 1.5 * IMG
    for it in items:
        cx, cy = to_px(it["x"]), to_px(it["y"])
        shape = it.get("shape", "circle")
        if shape == "diamond":
            mask = (np.abs(xx - cx) + np.abs(yy - cy)) < 1.3 * r_px
        elif shape == "square":
            mask = (np.abs(xx - cx) < 0.95 * r_px) & (np.abs(yy - cy) < 0.95 * r_px)
        elif shape == "triangle":
            mask = ((yy - cy > -0.9 * r_px)
                    & (np.abs(xx - cx) < 0.95 * r_px * (1 - (yy - cy + 0.9 * r_px)
                                                        / (2.0 * r_px))))
        elif shape == "cross":
            mask = (((np.abs(xx - cx) < 0.45 * r_px) & (np.abs(yy - cy) < 1.1 * r_px))
                    | ((np.abs(yy - cy) < 0.45 * r_px) & (np.abs(xx - cx) < 1.1 * r_px)))
        else:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r_px ** 2
        img[mask] = COLORS[it["color"]]
    return img


NONTARGET_SHAPES = ["circle", "square", "triangle", "cross", "hexagon"]


def shape_for(slot, targ_slot, template_shape="diamond"):
    """Feature-search displays: the target is the template shape among
    HETEROGENEOUS nontarget shapes (the target is never a shape
    singleton - inclusion criterion of the source studies)."""
    if slot == targ_slot:
        return template_shape
    return NONTARGET_SHAPES[slot % 4]


def _gauss_blur(m, sigma):
    k = int(3 * sigma) * 2 + 1
    k = min(k, (min(m.shape) // 2) * 2 - 1)   # kernel must fit the image
    ax = np.arange(k) - k // 2
    g = np.exp(-ax ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    m = np.apply_along_axis(lambda r: np.convolve(r, g, "same"), 0, m)
    return np.apply_along_axis(lambda r: np.convolve(r, g, "same"), 1, m)


def feature_maps(img):
    """Itti & Koch-style maps from pixels. Returns dict of [IMG, IMG]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    inten = (r + g + b) / 3
    rg = r - g                      # color opponency
    by = b - (r + g) / 2
    maps = {}
    for name, m in [("RG", rg), ("BY", by), ("I", inten)]:
        cs = np.zeros_like(m)
        for s_c, s_s in [(2, 8), (4, 16)]:   # center-surround (DoG) scales
            cs += np.abs(_gauss_blur(m, s_c) - _gauss_blur(m, s_s))
        cs[:8, :] = cs[-8:, :] = cs[:, :8] = cs[:, -8:] = 0   # border artifact
        maps[name] = cs / cs.max() if cs.max() > 0 else cs
    sal = sum(maps.values())
    maps["SAL"] = sal / sal.max() if sal.max() > 0 else sal
    # signed color channels for top-down gains (template/distractor color)
    maps["RGs"] = rg
    maps["BYs"] = by
    return maps


def form_map(items, template_shape="diamond"):
    """Analytic form-match channel (stated shortcut)."""
    m = np.zeros((IMG, IMG))
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    to_px = lambda v: (v + 0.75) / 1.5 * IMG
    r_px = ITEM_R / 1.5 * IMG
    for it in items:
        if it.get("shape", "circle") == template_shape:
            cx, cy = to_px(it["x"]), to_px(it["y"])
            m[(xx - cx) ** 2 + (yy - cy) ** 2 < (1.4 * r_px) ** 2] = 1.0
    return _gauss_blur(m, 2)


def ray_scan(maps, fix_xy, n_rays=90, n_bins=48, max_r=1.1):
    """Sample every map along rays from the fixation: the visual LiDAR.

    Returns scan [n_rays, n_channels] (nearest-weighted radial sum) and
    range [n_rays] (distance of first above-threshold salience energy).
    Channel order: RG, BY, I, SAL, RGs, BYs (+ FORM if present).
    """
    names = list(maps.keys())   # any channel set; dict order preserved
    to_px = lambda v: (v + 0.75) / 1.5 * IMG
    fx, fy = to_px(fix_xy[0]), to_px(fix_xy[1])
    # start beyond the fovea: the currently fixated item is not re-sensed
    radii = np.linspace(0.09, max_r, n_bins)
    w_near = np.exp(-2.0 * radii)          # nearest-energy weighting
    scan = np.zeros((n_rays, len(names)))
    rng = np.full(n_rays, max_r)
    for i in range(n_rays):
        th = 2 * np.pi * i / n_rays
        px = np.clip(fx + to_px(radii * np.cos(th)) - to_px(0), 0, IMG - 1)
        py = np.clip(fy + to_px(radii * np.sin(th)) - to_px(0), 0, IMG - 1)
        px, py = px.astype(int), py.astype(int)
        for c, nm in enumerate(names):
            prof = maps[nm][py, px]
            scan[i, c] = (np.abs(prof) * w_near).sum() * np.sign(
                prof[np.argmax(np.abs(prof))] if nm in ("RGs", "BYs") else 1.0)
        salprof = maps["SAL"][py, px]
        hit = np.where(salprof > 0.25)[0]
        if len(hit):
            rng[i] = radii[hit[0]]
    return scan, rng, names


def demo_display():
    """Gaspelin-style feature-search display: green diamond target among
    HETEROGENEOUS green nontarget shapes, one red singleton; 6 items."""
    items = []
    for j in range(6):
        a = 2 * np.pi * j / 6 - np.pi / 2
        color = "red" if j == 4 else "green"
        items.append(dict(x=ECC * np.cos(a), y=ECC * np.sin(a),
                          color=color, shape=shape_for(j + 1, 2)))
    return items


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = demo_display()
    img = render(items)
    maps = feature_maps(img)
    maps["FORM"] = form_map(items)
    scan_c, rng_c, names = ray_scan(maps, (0.0, 0.0))
    scan_f, rng_f, _ = ray_scan(maps, (items[1]["x"], items[1]["y"]))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes[0, 0].imshow(img, origin="lower")
    axes[0, 0].set_title("display (pixels in)")
    for ax, nm, ttl in [(axes[0, 1], "RG", "R-G opponency contrast"),
                        (axes[0, 2], "SAL", "salience map"),
                        (axes[0, 3], "FORM", "form-match (shortcut)")]:
        ax.imshow(maps[nm], origin="lower", cmap="magma")
        ax.set_title(ttl)
    th = np.linspace(0, 360, len(scan_c), endpoint=False)
    for ax, sc, ttl in [(axes[1, 0], scan_c, "ray scan from CENTER"),
                        (axes[1, 1], scan_f, "ray scan from TARGET (post-saccade)")]:
        for c, nm in enumerate(names):
            if nm in ("SAL", "RGs", "FORM"):
                ax.plot(th, sc[:, c], label=nm)
        ax.set_xlabel("ray angle (deg)")
        ax.legend(fontsize=7)
        ax.set_title(ttl)
    axes[1, 2].plot(th, rng_c, label="from center")
    axes[1, 2].plot(th, rng_f, label="from target")
    axes[1, 2].set_title("range reading (first salience energy)")
    axes[1, 2].set_xlabel("ray angle (deg)")
    axes[1, 2].legend(fontsize=8)
    axes[1, 3].axis("off")
    axes[1, 3].text(0.02, 0.5,
                    "visual LiDAR:\npixels -> I&K maps ->\nray scan from fixation\n"
                    "(re-centers each saccade,\nlike the agent's scan)",
                    fontsize=11, va="center")
    for ax in axes[0]:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/front_end_demo.png", dpi=120)
    print("saved figures/front_end_demo.png")
