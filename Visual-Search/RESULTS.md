# Pooled fit results (v1, run 2026-09-10)

Data: 217,595 saccades (indices 1–5; 124,834 first saccades), 333
subjects, 11 studies; color-singleton present vs. absent trials only;
practice conditions the traces but is not scored. `results_fit.json`.

## Estimates (7 pooled weights)

| Weight | Estimate | Reading |
| --- | --- | --- |
| g_T | **+3.20** | template gain: attend the target's features |
| g_S | **−1.85** | salience gain: the singleton is written *below* a plain nontarget — suppression as negative writing |
| beta_T | **+1.76** | prior target locations attract (strongly) |
| beta_D | **−0.24** | prior singleton locations repel (weaker, reliably signed) |
| eta_T | **0.62** | target-location history is heavily recency-weighted (~lag 1–2) |
| eta_D | **0.17** | singleton-location history accrues ~4x slower — a slower suppressive trace |
| k | 0.24 | envelope falloff ≈ flat (see caveat) |

Trace model vs. no-trace null: **ΔNLL = 18,788** for 4 extra weights
(NLL/saccade 1.286 vs. 1.372) — selection history is a first-order
component of saccade choice, not a correction term.

## Model vs. observed (first saccades, singleton present)

| Destination | Observed | Model |
| --- | --- | --- |
| target | 40.7% | 52.9% |
| singleton | 7.4% | 4.3% |
| nonsingleton (per item) | 13.4% | 11.1% |

Direction and ordering reproduced (suppression below baseline; compare
the source paper's 42.0 / 7.9 / 14.2 under slightly different
exclusions). Miscalibration: the model *over*-guides (too much target,
too much suppression) — a single g_T shared across saccade indices
compromises between first saccades (42% target) and later saccades
(higher target rates as search closes in).

## No-IoR refixation diagnostic (pre-registered in the model doc)

Saccades 2+ returning to an already-visited item: **observed 1.2%,
no-IoR model predicts 6.3%.** Humans avoid revisits ~5x more than the
bare field explains. The diagnostic fails in the direction that
warrants the single visited-item penalty, to be added as a model
comparison (one extra weight), per §7 of the model doc.

## Caveats on record

- **k ≈ 0.24 (near-flat envelope).** Under the v1 separability
  assumption (one multiplicative envelope shared by all channels), the
  likelihood prefers almost no distance falloff. Either proximity
  genuinely matters little for between-item choice in these dense
  6-item rings, or the shared-envelope form is strained (a far target
  is chosen anyway, dragging k down against the nonsingleton proximity
  structure). Distinguishing these needs the per-channel falloff
  extension.
- No uncertainty intervals yet (bootstrap over subjects planned).
- Parameter recovery on synthetic data not yet run.
