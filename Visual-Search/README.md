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
M(x) = alpha_P*P(x) + g_C*C_T(x) + g_F*S_T(x)
       + sum_j (beta_T*h_Tj + beta_D*h_Dj) * G(x - x_j)
F_i = M(x_i)
P(saccade -> i) = softmax over the current choice set
```

| Weight | Meaning | Agent counterpart |
| --- | --- | --- |
| alpha_P | goal-independent sensory color-salience gain | sensory/obstacle gain |
| g_C | target-color evidence gain | goal gain block |
| g_F | target-shape evidence gain | goal gain block |
| beta_T, beta_D | target/distractor history-field expression weights | history gain |
| eta_T, eta_D | target/distractor trace accrual rates | memory update rate |

Scope: first saccades only, launched from the display center; positions
only (no latency); one population-level fit (no individual differences);
bare field -> softmax readout (no motor-repetition, no lapse); no
attention window because all first-saccade items are equidistant from
fixation. Practice trials are excluded altogether (the memory traces
start cold at each subject's first experimental trial); the likelihood
scores kept trials. Trials with other distractor types (abrupt onsets,
singleton-onsets, motion singletons) are excluded entirely — the
analysis is color-singleton present vs. absent, as in the source paper
— with dropped counts printed at pooling time.
Gaspelin & Luck 2018 E4 is excluded (block-alternating singleton colors
break the fixed-task-set assumption), as in the source paper's
suppression/priming analyses.

## Files

| File | Purpose |
| --- | --- |
| `pool_data.py` | Fixation reports → `dataset/saccades.csv` + `dataset/events.csv` |
| `front_end.py` | The sensor: display rendering, Itti & Koch-style maps, ray scanning, item geometry |
| `build_contexts.py` | Reconstructs every unique display and precomputes sensed fields (`dataset/senses.npz` + `dataset/saccades_ctx.csv`) |
| `data.py` | Shared tensors (choice sets, distances, visited, trial-ordered events) + the subject split |
| `model.py` | `SearchModel` — the final model — and `load_final()` (`weights_final.json`) |
| `fit.py` | Pooled MLE -> `weights_final.json`, `results_final.json` |
| `reproduce.py` | Reproduction batteries on held-out people |
| `tutorial_visual_search.ipynb` | Teaching notebook: builds, trains, evaluates the final model on the real data |

Earlier model generations (v1 role-flag fits, the goal-late v2, the
window/history/sigma comparison scripts) were removed in a
consolidation; every number they produced is preserved in RESULTS.md,
and the git history holds the scripts.

## Reproduce

```bash
python pool_data.py --data_dir "<...>/search_data/Data Files" --out_dir dataset
python build_contexts.py
python fit.py                 # -> weights_final.json, results_final.json
python reproduce.py
```

## Diagnostics reported by the fit

- Observed vs model-implied first-saccade rates (target / singleton /
  per-item nonsingleton baseline) — the oculomotor suppression effect.
- Trace-model vs no-trace null (total NLL difference for 4 extra weights).
- Refixation rate on saccades 2+, observed vs the no-IoR model's
  prediction — the pre-registered check that decides whether a single
  visited-item penalty gets added (model comparison, not assumption).
