# Visual-Search: the priority field fitted to human saccades

The search instantiation of the signed priority field
(`docs/priority_field_visual_search_model.md`) — the same structured
network as the reach-avoid agent (`2D-Escaping-Ball/model/goal_es2.py`),
behavior-cloned from human saccades instead of an expert controller.

Data: the pooled oculomotor-suppression studies from Drennan & Gaspelin
(2026, *Cognition*), "What can a half-million saccades tell us about
distractor suppression?" — 12 eye-tracking studies, N = 354. The
per-study fixation reports are NOT part of this repo (OSF:
https://osf.io/q27ph/); point `pool_data.py` at their folder.

## Model (7 fitted weights)

```
F_i = env(d_i) * (1 + g_T*isT_i + g_S*isS_i) + beta_T*hT_i + beta_D*hD_i
P(saccade -> i) = softmax over the current choice set
```

| Weight | Meaning | Agent counterpart |
| --- | --- | --- |
| k | attention-window falloff (sigmoid steepness, tied across directions) | per-ray k profile (untied) |
| g_T | template gain — attend the target's features (task set, static) | goal gain block |
| g_S | salience gain — willfully ignore the singleton (task set, static) | obstacle gain |
| beta_T, beta_D | history-trace expression weights (signed) | hand-set beta = 0.15 |
| eta_T, eta_D | trace accrual rates | hand-set eta = 0.05 |

Scope (decisions in the model doc, §7): saccades 1–5, each a
conditional choice from the current fixation; positions only (no
latency anywhere); one population-level fit (no individual
differences); bare field → softmax readout (no motor-repetition, no
lapse); no IoR term — the refixation rate is a reported diagnostic; no
transient map. Traces are conditioned on all analyzed trials in order
(practice included); the likelihood scores experimental, kept trials
only. Onset/motion-distractor trials are excluded entirely — the
analysis is color-singleton present vs. absent, as in the source paper.
Gaspelin & Luck 2018 E4 is excluded (block-alternating singleton colors
break the fixed-task-set assumption), as in the source paper's
suppression/priming analyses.

## Files

| File | Purpose |
| --- | --- |
| `pool_data.py` | 12 fixation reports → `dataset/saccades.csv` (one row per scoreable saccade, with per-item distances from the current fixation) + `dataset/events.csv` (per-trial target/singleton locations driving the traces) |
| `model.py` | `SearchEs2Model`: tied-k sigmoid envelope x channel gains + trace injection → field → softmax |
| `fit_pooled.py` | Pooled MLE (Adam), null model without traces, diagnostics, `results_fit.json` |

## Reproduce

```bash
python pool_data.py --data_dir "<...>/search_data/Data Files" --out_dir dataset
python fit_pooled.py --epochs 300
```

## Diagnostics reported by the fit

- Observed vs model-implied first-saccade rates (target / singleton /
  per-item nonsingleton baseline) — the oculomotor suppression effect.
- Trace-model vs no-trace null (total NLL difference for 4 extra weights).
- Refixation rate on saccades 2+, observed vs the no-IoR model's
  prediction — the pre-registered check that decides whether a single
  visited-item penalty gets added (model comparison, not assumption).
