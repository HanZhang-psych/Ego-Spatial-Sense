# A signed priority field for search and action

**One model, two instantiations: the eyes navigating a search space, and an
agent navigating an action space.** Derived from the ego spatial sense (ES2)
model (`2D-Escaping-Ball/model/es2.py`, extended in
`2D-Escaping-Ball/model/goal_es2.py`).

Status: specification (v1). The search instantiation is not yet implemented;
the action instantiation and its diagnostics live in `2D-Escaping-Ball/`
(see `README_reach_avoid.md`, `RESULTS_reach_avoid.md`).

Paper structure this document serves: (1) a computational model of visual
search — the signed priority field fitted to human first-saccade data; (2)
the same model, re-instantiated, driving an autonomous agent in the
reach-avoid task. The formulation below is written so that the transition
between the two takes three sentences (§5).

---

## 1. Master equation

Over a generic **effector-referenced space** q — the places the effector can
go: display locations for the eyes, movement directions for the body —

**F(q) = gain(q) ⊗ Σ_c g_c · φ_c(q, t)   →   action = soft readout of F**

- **Channels** φ_c(q,t): transient-weighted evidence maps. *The task supplies
  the channel list.* Signed channel gains g_c subsume attraction, repulsion,
  template guidance, and rejection templates in one notation.
- **A priori gain map** gain(q) = envelope(q) + h(q): a standing
  physiological component and an experience-driven component, both set
  before the stimulus. (Cued spatial knowledge is deliberately excluded —
  see §7 Design decisions.)
- **⊗**: additive vs. multiplicative entry of the gain map — an estimable
  combination rule, not an assumption.
- **Soft readout**: softmax over F. A *sample* from it is a saccade
  (ballistic, discrete); the *expectation* under it (activity-weighted
  vector average — a population vector) is a continuous movement command.
  Same readout family; the effector determines sample vs. expectation.
- **Single signed field, source-blind readout**: every source expresses
  itself only by writing into F; downstream processing sees only the sum.
  Suppression is negative writing, not a separate pathway.

## 2. Instantiation table

| Component | Visual search (eyes) | Reach-avoid (agent) |
| --- | --- | --- |
| Space q | Display item locations | Movement directions (360 rays) |
| Channels φ_c | Color, orientation, size, ... + per-dimension contrast (salience) channel; transient/onset-weighted | Obstacle proximity (transient-weighted = looming) + goal-presence indicator |
| Channel gains g_c | Template: positive on target features ("what" knowledge); optional negative = rejection template; gain on the salience channel = singleton-detection mode | Repulsive gain on the proximity channel; attractive gain on the goal channel |
| gain(q): envelope | Functional viewing field around fixation (graded, eccentricity-dependent) | Per-ray sensing envelope k (from ego dynamics/sensing range) |
| gain(q): history h | Presence-driven leaky traces of target (+) and distractor (−) locations | Presence-driven leaky traces of goal (+) and threat (−) bearings (§6) |
| Readout | Softmax sample → first saccade | Softmax expectation → (fx, fy) each step |
| Parameterization | ~8–10 fitted scalars (measurement job) | Learned network weights (competence job), or the parametric potential-field control law |

Note the last row's symmetry: the potential-field expert the agent imitates
*is* the parametric instantiation of the master equation in the action
domain (channels × signed gains → vector readout). Parametric at ~10
parameters when the job is explaining humans; learned at ~261k weights when
the job is acting. Capacity scales with the job; structure does not change.

## 3. Search instantiation, v1 details

- **Front-end**: feature channels kept un-collapsed (the template needs
  channels to weight); per-dimension local contrast summed into a salience
  channel (what task-independent capture rides on — its default weight w_s
  is the Theeuwes/Folk dial). Transients: differential only when the
  paradigm creates them (new objects vs. placeholders); a static display's
  onset transient is spatially uniform.
- **Readout**: conditional logit — P(first saccade → item i) = softmax_i
  F(i)/τ, with τ fixed as the unit of measurement. Motor repetition
  (position priming of the response) is a lagged covariate in the readout,
  quarantined from the history traces. Latency is a conditioning covariate
  only (fast/slow split interacting with source weights); v1 does not
  generate latencies. Averaging (between-item) landings need pre-registered
  assignment/exclusion rules.
- **Excluded from v1**: latency generation, multi-fixation search,
  inhibition of return, foveal verification. The v2 accumulator readout
  (leaky competing accumulators over a motor map; threshold crossing =
  latency, activity-weighted centroid = endpoint; recovers the global
  effect and the latency–capture trade-off) is a swappable module — all
  v1 field-side fits carry over.

## 4. Selection history: presence-driven leaky traces

Location-indexed traces, one per event type (target, distractor), updated
every trial:

h_{t+1}(q) = (1 − η) · h_t(q) + η · e_t(q)

- **Presence-driven (design decision):** e_t(q) marks that a target (+) or
  distractor (−) *appeared* at q — registration by the front-end,
  independent of where the saccade went. Update path: front-end → traces.
  No saccade → trace feedback; the saccade's only history effect is motor
  repetition, which lives in the readout.
- Two traces with separate (η, β); the single signed trace (equal rates and
  weights) is the nested restriction, testable by model comparison. β
  (weight in gain(q)) is separate from η (accrual rate): rate vs. asymptote.
- Spatial spread: updates convolved with a small kernel (fitted width).
- Presence sourced from the contrast maps (item as individuated oddity):
  predicts reduced trace learning for very low-salience items even under
  presence-driven updating.

**Falsifiers of presence-driven updating:** learning rate must be a
property of display statistics (identical sequences → identical traces
regardless of individual capture rates); suppression develops at the same
η for distractors that never capture; acquisition is simple-exponential;
the trial-conditional kernel shows no difference following captured vs.
clean trials at matched history. Violations favor selection-gated or
hybrid update rules, which remain in the family as alternatives.

**Fitting:** η, β, kernel width identified from trial-order dynamics of
first-saccade choices (acquisition curves; lagged kernels decaying as
(1−η)^k; reversal transients). Hierarchical fits make cross-task η
contrasts ("task A induces faster buildup") posterior statements. Public
trial-level datasets (OSF: Gaspelin, Theeuwes / van Moorselaar labs)
suffice; parameter recovery on synthetic data precedes any human fit.

## 5. The transition (search → action), in full

1. The channel list changes with the sensory task: several visual feature
   channels → one proximity channel plus a goal channel.
2. The readout changes with the effector: discrete sample (saccade) →
   continuous expectation (movement vector).
3. The parameterization regime changes with the job: fitted scalars for
   measurement → learned weights for competence — with the reach-avoid
   diagnostics (§8) as evidence that the learned version retains the
   structure.

Everything else — the signed field, the source-blind readout, the
transient-weighted channels, the envelope + history gain map, the
task-silencing principle — is identical by construction.

## 6. Agent-side selection-history experiments

The reach-avoid goal is *known* per trial (it is in the observation), so a
target trace cannot aid localization; what it predicts is **anticipation**:

- **Exposure**: goals spawn preferentially in one region (e.g., 70% one
  quadrant), separated by explicit **goal-free periods** (~100–200 steps
  with no goal present). Online trace over bearings updated each step from
  the observed goal bearing (presence-driven), entering the field as
  +β·h(q). Bolted onto the frozen trained agent; no retraining. During
  goal-free periods a zero goal vector is fed — in GoalEs2Model this
  exactly zeroes the phasic goal field (the alignment term vanishes), so
  behavior is driven by obstacle avoidance + trace alone.
- **Predictions**: (a) during goal-free periods the agent drifts toward the
  frequent region — a pure readout of the trace, uncontaminated by any
  active goal; (b) faster time-to-goal for frequent-region goals at matched
  spawn distance, with the mirror-image collateral cost for rare-region
  goals; (c) both effects persist into an unbiased test block, decaying at
  rate η — persistence despite *cost* (the drift now lengthens paths to
  goals elsewhere), the strongest form of the selection-history signature.
- The trace injects into the same field the frozen action head reads, so
  drift also tests whether the head's field→action mapping generalizes to
  a source it was never trained on — the source-blindness commitment
  cashed out behaviorally.
- **Symmetry run**: the same rule with negative sign in the existing
  hazard-biased world (threat trace) — one presence-driven mechanism
  producing facilitation and suppression, mirroring the target/distractor
  trace pair on the search side.

**Outcome (run 2026-09-10; anticipation_experiment.py, β=0.15, η=0.05, 3
seeds; details in RESULTS_reach_avoid.md):** anticipatory drift confirmed
(goal-free distance to the frequent region falls 162→~110px across
exposure; spawn distances to frequent goals shortened 282→235px), and the
frozen source-blind head translated the never-trained trace source into
coherent behavior. Persistence confirmed with decay over ~1/η unbiased
goals (late-test residual identified as a centering artifact of the
centroid trace). The speed prediction *reversed*: during pursuit the trace
is a competing attractor, slowing frequent-region pursuit (30.2 vs. 13.8
steps/100px).

The reversal sharpens the theory: the payoff structure follows from *where
in the trial the uncertainty sits*. During the goal-free period the agent
knows nothing about the next goal except through the trace — genuine
anticipatory uncertainty, reduced by the prior and cashed out physically as
drift and pre-positioning. At goal onset the goal's location enters the
observation and the prior becomes informationally redundant, so residual
influence during pursuit can only distort. Search differs in that
uncertainty *persists after onset* (the target must still be found), which
is why the same trace mechanism yields within-trial benefits there.
Human-testable prediction (two-part): a fully valid location cue added to a
probability-cueing paradigm should abolish the within-trial
frequent-location benefit while leaving any residual history effect only
pre-onset (e.g., anticipatory gaze bias before display onset).

Separating goals by distance from the trace centroid recovered the
**congruency structure** the region average hid: history-valid goals
(within 100px of the centroid) are reached ~2x *faster* (compound
pre-positioning + tailwind), near-miss goals (100–250px ring) ~2x slower
(endgame attractor competition), clearly-wrong goals pay a mild headwind —
the probability-cueing profile (benefit at the frequent location, cost at
near-misses). The net-negative average is a resolution mismatch: the true
goal distribution is a quadrant but the leaky-centroid trace is a point
prior, so most goals land in its near-miss ring. The spec's trace kernel
width σ_h is the missing component; prediction: net payoff improves as
trace spread approaches the true spawn spread. **Confirmed by the σ_h
follow-up**: a nonparametric spread-matched trace (weighted bumps at
remembered spawn positions) removes ~75% of the excess pursuit cost,
eliminates the far-goal headwind (opposing bumps cancel into a plateau
inside the spread), and fully preserves anticipatory drift — the trace's
spatial resolution, not the history mechanism, was the liability.

**Definitive version — fixation-start trial structure** (`--reset_agent`:
teleport to center = return to fixation, anticipation period, goal
onset), which removes the positional-carryover confound of the continuous
design. Result: active per-trial anticipatory excursion toward the
frequent region from a standardized start (control moves away), building
with exposure and decaying across the unbiased test block; **net faster
acquisition of history-congruent goals** (22.5 vs. 24.6 and 32.8 vs. 38.8
steps) delivered through the excursion's head start (per-distance speed
equal), with cost confined to incongruent goals and the post-bias test
block (persistence-despite-cost). The full probability-cueing
phenomenology, recovered in the action domain.

A β=0.05 follow-up additionally found a gain window bounded below by the
environment's 1px actuation quantization (no drift, residual pursuit
drag). Together these motivate two v2 refinements: **spread-matched
traces** (σ_h) and **pursuit-time normalization** (the prior yields when a
fully observed goal supersedes it) — the latter plausibly why human
history effects are small during explicit goal-directed action.

Combined with the existing behavior-cloning null (§8), the selection-history
story is parallel across domains: *the slot exists (architecture); imitation
cannot fill it (null); one presence-driven rule fills it and the frozen
head expresses it (positive); its payoff sign depends on target uncertainty
(reversal); the same rule's rate is what the human fits estimate (η).*

## 7. Design decisions on record

- **No cued spatial knowledge.** Neither the planned search tasks nor the
  reach-avoid task involves cues; the gain map is envelope + history only.
  This also excludes scene-prior guidance — appropriate for singleton-type
  displays; the architecture has an obvious slot should it ever be needed.
- **Transient weighting: silenced, not deleted.** The planned search tasks
  use static, simultaneous-onset displays, so the transient weighting is
  spatially uniform — nothing to implement or fit in v1 (task-silencing).
  It stays in the master equation's channel definition because it is what
  makes the cross-domain channel correspondence exact (transient-weighted
  proximity = looming, the most distinctively biological ES2 component),
  and it keeps onset-capture paradigms reachable without model changes.
- **Presence-driven trace updating** (not selection-gated); falsifiers in §4.
- **Presence sourced from contrast maps** (weak commitment; see §4).
- **v1 models endpoints only**; latency generation deferred to the v2
  accumulator readout.
- **Task-silencing principle**: the full model is the union of sources; a
  task silences sources through input structure (no differential
  transients under placeholders; flat statistics → flat traces).
  Parameters receiving no variance from the design are fixed, not fitted.
- **Parametric fitting for the search job** (~8–10 named scalars,
  hierarchically pooled; τ fixed as unit): the parameters are the
  scientific deliverable. A flexible neural net fitted to the same data is
  retained as comparison model and misspecification alarm, not instrument.

## 8. Supporting evidence from this repo's simulations

(2D-Escaping-Ball reach-avoid extension; RESULTS_reach_avoid.md.)

- **Competence:** goal-conditioned ES2 matches the expert's Pareto point
  (67.6 goals/min @ 0.4 collisions/min); goal-blind baselines ~0 goals/min.
- **Structure is architectural, not free:** under the goal-swap probe, ES2
  (and a goal Transformer) collapse goal-reaching (−97…−100%) with
  collision rates unchanged; a capacity-comparable MLP on the same data
  entangles the sources (collisions double under a wrong goal).
- **The learning rule is the locus of selection history:** an ES2 agent
  behavior-cloned in a hazard-biased world transfers no directional
  asymmetry (k, field, or behavior) — the expected null for imitation of a
  memoryless expert, motivating the online traces of §6.

## 9. Parameters

**Fitted (search instantiation):** w_s (salience weight), g (template
strength), σ_env (envelope width), η_tgt, η_dist, β_tgt, β_dist, σ_h (trace
kernel width), w_rep (+ decay), ε (lapse). τ fixed as unit. Typical
single-experiment identifiable subset: ~8.

**State, not parameters:** the traces h(q) — deterministic given the trial
sequence and η; estimated never, generated always.

**Set by the task:** channel list, which channels g targets, event
sequences e_t, display geometry.

**Model-comparison forks (discrete):** ⊗ additive vs. multiplicative; two
traces vs. single signed trace; presence-driven vs. selection-gated
(committed, falsifiable); contrast- vs. channel-sourced presence.

## 10. Relation to existing models

Kin to Guided Search (weighted feature channels + bottom-up contrast +
history) and salience/priority-map theory (Itti–Koch; Fecteau & Munoz); the
field-as-valence reading descends from Lewin, and the action instantiation
is an additive goal+obstacle behavioral dynamics model in the Fajen–Warren
lineage. Distinctive claims: one signed source-blind field; a priori
parametric entry weighting from the ego's state; transient-weighted input;
effector-referenced coordinates — and the cross-domain evidence that this
structure, not capacity or data, yields modular, causally testable
goal conditioning.
