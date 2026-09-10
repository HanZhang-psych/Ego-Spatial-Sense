# A priority-field model of visual search

**Derived from the ego spatial sense (ES2) model; adaptation target: first-saccade
selection in additional-singleton-type search tasks.**

This document specifies the proposed model, states exactly what is preserved,
generalized, replaced, or added relative to the original ES2 model (as
implemented in `2D-Escaping-Ball/model/es2.py` and extended in
`2D-Escaping-Ball/model/goal_es2.py`), and records the design decisions made
so far, with the empirical debates each parameter engages.

Status: specification (v1). No implementation yet. The supporting simulation
evidence cited in §7 lives in `2D-Escaping-Ball/` (see `README_reach_avoid.md`
and `RESULTS_reach_avoid.md`).

---

## 1. Scope

- **Models:** the destination of the *first saccade* in displays of discrete
  items (target, distractors, optional salient singleton), trial by trial,
  including experience-dependent changes across trials.
- **v1 deliberately excludes:** saccade latency as a generated output,
  multi-fixation search, inhibition of return, foveal verification dynamics.
  Latency enters only as a *conditioning covariate* (§5). A v2 accumulator
  readout that generates latencies is sketched in §9.

## 2. One-sentence statement

Parallel feature maps and transient-weighted contrast (salience) maps,
combined under goal-driven feature-channel gains, enter a **single signed
priority field** weighted on entry by an **a priori gain map** — oculomotor
envelope, cued spatial knowledge, and presence-driven leaky history traces —
and the field is resolved into a saccade by a readout that is blind to which
source contributed what.

## 3. Model specification

### 3.1 Front-end (bottom-up)

For each item/location ℓ:

- **Feature channels** φ_c(ℓ): raw local feature values per dimension
  (color, orientation, size, ...). Kept un-collapsed so the template can
  weight them.
- **Contrast maps**: per-dimension local feature contrast (center–surround
  oddity), summed into a salience signal s(ℓ). This is a different
  computation from φ (relative, context-dependent), not a redundant copy;
  it is what task-independent capture rides on.
- **Transient weighting**: bottom-up signals are weighted by temporal change
  (onsets, motion). In static displays the display-onset transient is
  spatially uniform (no differential priority); differential transients
  exist only when the paradigm creates them (new objects vs. placeholders).

### 3.2 Top-down and a priori sources

- **Goal template (feature space, phasic):** a signed gain vector g over
  feature channels, set before display onset by instruction/task set.
  Template evidence at ℓ: T(ℓ) = Σ_c g_c φ_c(ℓ). Negative components are
  templates for rejection. "Search mode" (Bacon & Egeth) = where the gain
  sits: on specific feature channels (feature-search mode) vs. on the
  salience channel itself (singleton-detection mode).
- **A priori gain map (location space):** gain(ℓ) = envelope(ℓ) + cue(ℓ) +
  h(ℓ), fixed within a trial:
  - *Oculomotor envelope*: graded eccentricity-dependent sensitivity around
    current fixation (functional viewing field) plus saccade-cost biases.
    Observer-specific width is a fitted parameter.
  - *Cued spatial knowledge*: explicit advance "where" information (spatial
    precue, instruction, scene priors). Structurally absent in tasks without
    cues.
  - *History traces*: see §4.

### 3.3 The field

F(ℓ) = gain(ℓ) ⊗ [ s(ℓ) + T(ℓ) ]

- **Single and signed**: all sources express themselves only by writing into
  F; suppression is negative writing, not a separate pathway.
- **Source-blind readout**: downstream processing sees only the sum. This is
  the commitment that makes source-ablation diagnostics meaningful (§7).
- **⊗ is an estimable combination rule** (additive vs. multiplicative entry
  of the gain map), not an assumption. The two rules predict top-down
  effects that are constant (additive) vs. salience-scaled (multiplicative).

### 3.4 Readout (v1: endpoint only)

P(first saccade → item i) = softmax_i ( F(i) / τ )

- Conditional-logit likelihood; τ absorbs overall field scale.
- **Motor repetition** (position priming of the response) is a lagged
  covariate on the previous saccade vector *in the readout*, quarantined
  from the history traces so response repetition cannot masquerade as
  trace learning.
- **Latency as covariate**: source weights may interact with a fast/slow
  median split, recovering coarse time-course sensitivity (salience weight
  predicted higher for fast saccades) without generating latencies.
- Averaging (between-item) landings have no generative account in v1;
  assignment/exclusion rules must be pre-registered.

## 4. Selection history: presence-driven leaky traces

Two location-indexed traces (target, distractor), each updated every trial:

h_{t+1}(ℓ) = (1 − η) · h_t(ℓ) + η · e_t(ℓ)

- **Presence-driven (adopted design decision):** e_t(ℓ) indicates that a
  target (+) or distractor (−) *appeared* at ℓ — registration by the
  front-end, independent of where the saccade went. The update path is
  front-end → traces; there is no saccade → trace feedback.
- Two traces with separate (η, β) by default; the single signed trace
  (η_tgt = η_dist, β_tgt = β_dist) is the nested restriction, testable by
  model comparison. β (weight of the trace in gain(ℓ)) is separate from η
  (accrual rate): "stronger selection history" can mean faster buildup or
  larger asymptote, and the model distinguishes them.
- Spatial spread: updates convolved with a small kernel (fitted width),
  matching suppression gradients around high-probability locations.
- Presence sourced from the *contrast maps* (item as individuated oddity),
  a weak commitment predicting reduced trace learning for very low-salience
  items even under presence-driven updating.

**What presence-driven commits to (falsifiers):** learning rate is a
property of display statistics, not the observer's behavior — identical
sequences give identical traces regardless of individual capture rates;
suppression develops at the same η for distractors that never capture;
acquisition is simple-exponential (no self-limiting kink); and the
trial-conditional kernel shows **no** difference following captured vs.
clean trials at matched history. Violations favor selection-gated updating
(e updated only on selection) or the hybrid — both remain in the model
family as alternative update rules.

**Fitting:** η, β (and the kernel width) are identified from trial-order
dynamics of first-saccade choices alone — acquisition curves, lagged-
regression kernels (influence of a distractor k trials back decays as
(1−η)^k), and reversal transients. Hierarchical (subjects in tasks) fits
make "task A induces faster selection-history buildup than task B" a
posterior contrast on η. Public trial-level datasets (OSF: Gaspelin,
Theeuwes / van Moorselaar labs) suffice; parameter recovery on synthetic
data is required before fitting human data.

## 5. Task-silencing principle

The full model is the union of sources writing into the field. A given task
silences sources through its **input structure**, not through fitted zeros:

- No cue → cue(ℓ) receives no input (drops out structurally).
- Placeholder displays → no differential transients (transient term
  spatially uniform).
- Unbiased location statistics → traces converge to flat.

Parameters receiving no variance from the design are **fixed, not fitted**
(a fitted zero on an unidentifiable parameter is noise). Cross-paradigm
constancy of shared parameters (envelope width, τ, η) is itself a testable
claim.

## 6. Modifications from the original ES2 model

| Component | Original ES2 (this repo) | Search model | Status |
| --- | --- | --- | --- |
| Input | Two consecutive 360-ray LiDAR scans | Feature channels + per-dimension contrast maps over display items | **Replaced** (front-end swap) |
| Temporal structure | Scan delta × proximity (looming detector) | Transient/onset weighting of bottom-up signals | **Preserved** (same commitment, new stimulus) |
| A priori envelope | Learned per-ray k (sensing envelope from ego dynamics) | Graded oculomotor envelope (functional viewing field) + cue + history traces | **Generalized** (one parameter vector → three-source gain map) |
| Goal channel | Spatial bump at known goal bearing (goal_es2.py) | Feature-channel gain vector g ("what" knowledge); spatial bump remains the degenerate known-location case | **Generalized** (writes into the space the ego has advance knowledge of) |
| Field | Single signed 360-dim field; sources superimposed; readout source-blind | Same, over display locations | **Preserved** (core identity) |
| Combination rule | Envelope multiplicative (inside sigmoid), goal additive | ⊗ estimable (additive vs. multiplicative) | **Promoted** from implementation accident to fitted contrast |
| Readout | Instantaneous MLP → continuous (fx, fy) each step | v1: softmax over items → discrete first saccade (+ motor-repetition covariate); v2: leaky competing accumulators → endpoint and latency | **Replaced** |
| Learning | Offline behavior cloning of a potential-field expert; no test-time learning | Parameters fit to human choices; history traces update online (presence-driven, rate η) | **Replaced / New** — see §7 for why this is forced |
| Coordinates | Egocentric movement directions (action-referenced) | Retinotopic/display locations = saccade goal space (action-referenced) | **Preserved** (effector-referenced field) |

**Inherited commitments (the model's identity):** (1) one signed field,
source-blind readout; (2) a priori parametric weighting of evidence entry,
set by the observer's state before the stimulus; (3) transient-weighted
bottom-up input; (4) effector-referenced coordinates. Dropping any of these
makes the model a notational variant of Guided Search; keeping them makes it
a falsifiable member of the priority-map family with a cross-domain
instantiation.

## 7. Supporting evidence from this repo's simulations

(2D-Escaping-Ball reach-avoid extension; see RESULTS_reach_avoid.md.)

- **Competence:** goal-conditioned ES2 matches the expert's Pareto point
  (67.6 goals/min @ 0.4 collisions/min); goal-blind baselines reach ~0
  goals/min.
- **Source superposition is architectural, not free:** under the goal-swap
  probe, ES2 (and the goal Transformer) collapse goal-reaching (−97…−100%)
  with collision rates unchanged, while a capacity-comparable MLP trained on
  the same data entangles them (collisions double under a wrong goal). The
  clean dissociation follows from the field structure, not from the task or
  the data.
- **The learning rule, not the architecture, is the locus of selection
  history:** an ES2 agent behavior-cloned in a hazard-biased world (80% of
  threats on one side) transfers **no** directional asymmetry in k, field,
  or behavior — the theoretically expected null, because the imitated expert
  is memoryless. This forces the §4 design: history requires deployment-time
  updating from experienced statistics (under the presence-driven rule, an
  online occupancy trace; no reward machinery required).

## 8. Debates engaged as parameters

| Debate | Model expression |
| --- | --- |
| Stimulus-driven capture (Theeuwes) vs. contingent capture (Folk) | Default weight on the salience channel: nonzero fixed vs. fully task-set-controlled |
| Proactive suppression vs. capture-then-disengage vs. passive transient decay | Sign/baseline of suppression sources at t=0 vs. their onset latency vs. no suppression term (mostly a v2/time-course question; v1 touches it only via the fast/slow covariate split) |
| Search modes (singleton-detection vs. feature-search) | Which channel g targets (salience vs. specific features) |
| One selection-history mechanism or two | Single signed trace as nested restriction of the two-trace model |
| Rate vs. strength of selection history | η vs. β, separately fitted |
| What teaches attention (presence / selection / hybrid) | Update-rule variants for e_t; presence-driven adopted, falsifiers stated in §4 |
| Additive vs. multiplicative top-down modulation | The ⊗ combination rule |

## 9. v2 roadmap (documented, not committed)

- **Accumulator readout:** leaky competing accumulators over a motor map,
  driven by a time-varying field (fast salience ramp, slow template ramp,
  gain map constant); threshold crossing yields latency, activity-weighted
  centroid yields endpoint. Recovers the global effect and the
  latency–capture trade-off as emergent properties; enables the
  proactive-vs-reactive arbitration on time-resolved data. Readout is a
  swappable module so all v1 field-side fits carry over.
- **Simulation counterpart:** presence-driven online occupancy trace on the
  frozen hazard agent's per-ray gain (the predicted positive condition for
  the §7 null).
- Feature-history traces (priming of pop-out) as the feature-space analog of
  the location traces.

## 10. Relation to existing models

Kin to Guided Search (guidance = weighted feature channels + bottom-up
contrast + history + scene priors) and salience/priority-map theory
(Itti–Koch; Fecteau & Munoz); the field-as-valence reading descends from
Lewin's field theory, and the locomotor instantiation is an additive
goal+obstacle behavioral dynamics model in the Fajen–Warren lineage. The
distinctive claims are the four inherited commitments (§6) plus the
cross-domain evidence (§7) that the field structure — not capacity, data, or
task — is what yields modular, causally testable goal conditioning.
