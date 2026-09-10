"""Pre-onset gaze bias: does the spatial prior leak into where the
eyes sit BEFORE the display appears?

Model prediction (the pre-onset prior F = window x history): gaze at
display onset should be displaced toward recent TARGET locations
(the fast positive trace) and away from recent SINGLETON locations
(the slow negative trace). The source paper analyzed 0-400 ms after
onset; this analysis looks at the trial's initial fixation position -
determined before any stimulus-driven saccade.

Measure: each trial's initial fixation (first row of the fixation
report), expressed as an offset from the subject's own median initial
position (removes calibration bias). That offset is projected (in
pixels) onto unit directions from screen center toward:

  - the previous trial's target location (lag-1; contaminated by
    return-saccade undershoot, see control)
  - the previous trial's singleton location (undershoot-free: trials
    rarely end on the singleton)
  - the model's trace directions (history summed with the fitted
    rates eta_T = 0.61, eta_D = 0.23 over all past trials)
  - CONTROL: the previous trial's final fixated location. Return
    undershoot predicts bias here; the lag-1 target projection is
    also reported after residualizing against this control within
    subject.

Same-block lag-1 only (block breaks involve recalibration). Practice
excluded; Hamblin-Frohman excluded (no gaze coordinates). Stats:
subject-level mean projections, one-sample t across subjects.

Usage: python analyze_preonset.py --data_dir ".../Data Files"
"""

import argparse
import os

import numpy as np
import pandas as pd

from pool_data import CENTER, EXCLUDE_STUDYNAMES, item_coordinates, load_study

ETA_T, ETA_D = 0.61, 0.23      # fitted memory rates (final model)


def process(df, coords):
    df = df.copy()
    df["block"] = df["block"].fillna(0)
    df = df.sort_values(["subjNum", "block", "trial", "saccindex"])
    first = df.groupby(["subjNum", "block", "trial"], sort=False).first()
    last = df.groupby(["subjNum", "block", "trial"], sort=False).last()
    t = first.reset_index()
    t["fx"], t["fy"] = t.normX, t.normY
    t["end_x"], t["end_y"] = last.normX.values, last.normY.values
    t["practice"] = (t.practice == "Y").astype(int)
    t = t[(t.setsize >= 4) & (t.setsize <= 6)]
    return t


def unit_to(coords, setsize, loc):
    if loc < 1 or (setsize, loc) not in coords:
        return None
    x, y = coords[(setsize, loc)]
    v = np.array([x - CENTER[0], y - CENTER[1]])
    n = np.linalg.norm(v)
    return v / n if n > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(os.listdir(args.data_dir)):
        if not f.endswith(".txt"):
            continue
        df = load_study(os.path.join(args.data_dir, f))
        name = df.studyName.iloc[0]
        if name in EXCLUDE_STUDYNAMES:
            continue
        coords = item_coordinates(df)
        if not coords:
            print(f"skip {f} (no gaze coordinates)")
            continue
        t = process(df, coords)
        for subj, sub in t.groupby("subjNum", sort=False):
            sub = sub.sort_values(["block", "trial"])
            med_x, med_y = sub.fx.median(), sub.fy.median()
            hT = np.zeros(7)
            hD = np.zeros(7)
            prev = None
            for r in sub.itertuples():
                s = int(r.setsize)
                if (not r.practice and prev is not None
                        and prev["block"] == r.block
                        and np.isfinite(r.fx)):
                    off = np.array([r.fx - med_x, r.fy - med_y])
                    row = dict(study=name, subj=str(subj),
                               off_x=off[0], off_y=off[1])
                    uT = unit_to(coords, s, prev["targ"])
                    uS = unit_to(coords, s, prev["sing"])
                    uE = None
                    if np.isfinite(prev["end_x"]):
                        v = np.array([prev["end_x"] - CENTER[0],
                                      prev["end_y"] - CENTER[1]])
                        n = np.linalg.norm(v)
                        uE = v / n if n > 1 else None
                    vT = sum((hT[j] * u for j in range(1, s + 1)
                              if (u := unit_to(coords, s, j)) is not None),
                             np.zeros(2))
                    vD = sum((hD[j] * u for j in range(1, s + 1)
                              if (u := unit_to(coords, s, j)) is not None),
                             np.zeros(2))
                    row["p_prevT"] = off @ uT if uT is not None else np.nan
                    row["p_prevS"] = off @ uS if uS is not None else np.nan
                    row["p_end"] = off @ uE if uE is not None else np.nan
                    row["p_traceT"] = (off @ (vT / np.linalg.norm(vT))
                                       if np.linalg.norm(vT) > .05 else np.nan)
                    row["p_traceD"] = (off @ (vD / np.linalg.norm(vD))
                                       if np.linalg.norm(vD) > .05 else np.nan)
                    rows.append(row)
                # trace update happens for every analyzed trial
                hT *= (1 - ETA_T)
                if r.targLoc >= 1:
                    hT[int(r.targLoc)] += ETA_T
                hD *= (1 - ETA_D)
                if r.color_sing and r.singLoc >= 1:
                    hD[int(r.singLoc)] += ETA_D
                prev = dict(block=r.block, targ=int(r.targLoc),
                            sing=int(r.singLoc) if r.color_sing else 0,
                            end_x=r.end_x, end_y=r.end_y)
        print(f"{f}: done")

    d = pd.DataFrame(rows)
    print(f"\n{len(d)} trials, {d.groupby(['study','subj']).ngroups} subjects")
    print(f"median |offset| = "
          f"{np.hypot(d.off_x, d.off_y).median():.1f} px\n")

    def ttest(col, label):
        m = d.dropna(subset=[col]).groupby(["study", "subj"])[col].mean()
        n = len(m)
        t = m.mean() / (m.std() / np.sqrt(n))
        print(f"{label:52} {m.mean():+6.2f} px   t({n-1})={t:+6.2f}")

    ttest("p_prevT", "toward PREVIOUS TARGET location (lag-1)")
    ttest("p_prevS", "toward PREVIOUS SINGLETON location (lag-1)")
    ttest("p_traceT", "toward the target TRACE direction (fitted eta)")
    ttest("p_traceD", "toward the singleton TRACE direction")
    ttest("p_end", "CONTROL: toward previous trial's final gaze")

    # residualize the lag-1 target projection against the control,
    # within subject (removes return-saccade undershoot)
    dd = d.dropna(subset=["p_prevT", "p_end"]).copy()
    res = []
    for (st, sj), g in dd.groupby(["study", "subj"]):
        if len(g) < 20 or g.p_end.std() < 1e-9:
            continue
        b = np.cov(g.p_prevT, g.p_end)[0, 1] / g.p_end.var()
        res.append((g.p_prevT - b * g.p_end).mean())
    res = np.array(res)
    t = res.mean() / (res.std() / np.sqrt(len(res)))
    print(f"{'toward PREV TARGET, undershoot removed':52} "
          f"{res.mean():+6.2f} px   t({len(res)-1})={t:+6.2f}")
    d.to_csv("dataset/preonset_trials.csv", index=False)
    print("\nsaved dataset/preonset_trials.csv")


if __name__ == "__main__":
    main()
