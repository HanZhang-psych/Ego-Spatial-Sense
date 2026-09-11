# A signed priority field for search and action

**One model, two instantiations: the eyes navigating a search space, and an
agent navigating an action space.** Derived from the ego spatial sense (ES2)
model (`2D-Escaping-Ball/model/es2.py`, extended in
`2D-Escaping-Ball/model/goal_es2.py`).

Status: **fitted (v2.1 final form)**. The search instantiation is
implemented and fitted in `Visual-Search/` (RESULTS.md there); the
action instantiation and its diagnostics live in `2D-Escaping-Ball/`
(see `README_reach_avoid.md`, `RESULTS_reach_avoid.md`).

**Final model of record (search side), one sentence:** scoped to the
FIRST saccade of each trial (launched from central fixation), one
priority map assembled from minimal goal-weighted evidence —
g_T·relu(target-color contrast) − g_D·relu(distractor-color
contrast): ONE-SIDED rectified channels, chosen for interpretability
(g_T = pure enhancement of the target color, g_D = pure suppression
of the distractor color; the fully linear signed field fits identically to
within noise, and rectify-after-the-gains remains rejected) — plus
the pixel-derived template-shape map and the two world-anchored
leaky location traces rendered as a HISTORY FIELD (each item's
trace value placed at its location, smoothed by a fixed sigma=0.09
kernel — a stated assumption). All of it forms ONE pre-window
priority map over the display; the ego-anchored sigmoid attention
window multiplies that map pixel by pixel (in scope the fitted
window is flat — r0 beyond the display — a shared gain kept on
theoretical definition, not data constraint), and the readout is
each item's sector average (implemented exactly, as precomputed
sector x distance-bin area sums; no ray sampling); no IoR term
(unidentifiable before the second saccade); softmax over the sector
averages; nine fitted weights. In this construction g_D fits to ~0:
suppression is carried by relegation plus location history (see
RESULTS, "One construction everywhere"). The narrowed scope
dissolves the shape-gating fork (one vantage point makes gated and
ungated reparameterizations); the earlier all-saccade tests of that
fork are preserved in RESULTS. The
shape channel is pixel-derived: displays are reconstructed
canonically (target = circle, green items, red singleton — each
subject's template and colors were fixed all session, so only
match/mismatch structure matters) and shape evidence is a
discriminative normalized cross-correlation (circle NCC minus the
best competing shape's). Disclosed limit: the data never record item
shapes, so the reconstruction places the circle at targLoc by
construction — the pixel channel makes the pathway realistic, not
the display's provenance, and its fit sat within 0.005 of the
analytic label when both were tested (all-saccade era, 1.2376 vs
1.2333; the scoped first-fixation numbers live in RESULTS). Every structural choice (traces, IoR,
goal-early assembly, window form, history ordering, the dropped
presence and salience channels, the kept shape term, the
rectification placement) was decided or priced by held-out
comparison; the ledger is `Visual-Search/RESULTS.md`.

## 1. Master equation

Over a generic **effector-referenced space** q — the places the effector can
go: display locations for the eyes, movement directions for the body —

**F(q) = gain(q) ⊗ Σ_c g_c · φ_c(q, t)   →   action = soft readout of F**

- **Channels** φ_c(q,t): evidence maps. *The task supplies the channel
  list.* A channel is temporally differenced only where the world makes
  time meaningful — the agent's proximity channel (looming); the search
  channels are static feature maps, and the search model contains **no
  transient map** (§7). Signed channel gains g_c subsume attraction,
  repulsion, template guidance, and rejection templates in one notation.
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

### 1.1 Terminology, the attention window, and frames of reference

Following the original paper's usage: the **Ego Spatial Attention Field**
is the a priori object — the 360° graded weighting profile (per-direction
sigmoid with learned steepness k); the **Ego Spatial Sense** is what
results when that profile interacts with the actual contents of the
space. The Attention Field is fixation-invariant: it translates rigidly
with the ego and never reshapes. The Sense is fixation-dependent: the
same fixed profile evaluated against new ego-to-item distances after
each saccade yields a different reading.

**The envelope is the attention window.** The Attention Field is the
model's version of the useful/functional field of view, and it plays
exactly the role of Theeuwes' attentional window: bottom-up salience is
weighted by the envelope on entry into F, so salience outside the window
writes ~nothing and cannot capture. Two refinements: it is *graded*
(sigmoid falloff; "shrinking the window" = steepening it, subsuming the
zoom lens) and *directional in principle* (per-direction k). This turns
a verbal debate into a model comparison: reduced capture via a shrunken
envelope (window account) penalizes *all* eccentric items equally;
reduced capture via a down-weighted salience gain (suppression account)
penalizes only the singleton. Distinguishable in first-saccade data.

**The window is still the network's sigmoid block — tied, not replaced.**
Reporting the estimated window as "σ_env, 1–2 numbers" does not mean the
search model swaps the agent's neural envelope for a hand formula. It is
the same per-ray sigmoid block with learnable steepness k, trained
inside the same network by the same MLE; the search version simply
**ties the k's across directions** (weight sharing), because the ring
geometry probes the falloff at only ~3 chord distances per fixation —
enough to identify a shared radial steepness, not 360 independent ones.
Weight-tying is a standard network operation; the architecture is
unchanged, and every fitted "parameter" in this document is a weight of
that network. The parameter counts state how many free weights survive
the task's identifiability constraints. Extension: the tie can be
partially relaxed to a 2-parameter horizontal/vertical split of k —
the human functional viewing field is anisotropic (wider horizontally),
and items land at many ego-relative directions across fixations, so
this is plausibly identifiable. Not v1.

**Two anchoring frames.** The composite a priori gain map — the
**spatial prior**, gain(q) = envelope(q) + h(q), i.e. the attention
window modified by history — mixes components glued to different frames:
the envelope is *ego-anchored* (rides with fixation, shape fixed), the
history traces are *world-anchored* (glued to display/arena locations,
re-projected into ego coordinates whenever the ego moves — the agent
recomputes this projection from its current position every step). So the
prior is *stimulus*-independent but not *fixation*-independent: it
reshapes in ego coordinates on every saccade. For the first saccade
(standardized central fixation) the prior is a single static map per
trial; from saccade 2 onward the model predicts a dissociation —
distance effects follow the eyes, history effects stay glued to display
locations. Since v1 fits saccades 1–5 (§3), this dissociation is
*tested* in v1, not merely predicted.

## 2. Instantiation table

| Component | Visual search (eyes) | Reach-avoid (agent) |
| --- | --- | --- |
| Space q | Display item locations | Movement directions (360 rays) |
| Channels φ_c | Color, orientation, size, ... + per-dimension contrast (salience) channel — static maps, no transient channel | Obstacle proximity, temporally differenced (looming) + goal-presence indicator |
| Channel gains g_c | Template: positive on target features ("what" knowledge); optional negative = rejection template; gain on the salience channel = singleton-detection mode | Repulsive gain on the proximity channel; attractive gain on the goal channel |
| gain(q): envelope | Functional viewing field around fixation (graded, eccentricity-dependent) | Per-ray sensing envelope k (from ego dynamics/sensing range) |
| gain(q): history h | Presence-driven leaky traces of target (+) and distractor (−) locations | Presence-driven leaky traces of goal (+) and threat (−) bearings (§6) |
| Readout | Softmax sample → each saccade (1–5), from the current fixation | Softmax expectation → (fx, fy) each step |
| Parameterization | Same ES2-style gain blocks, behavior-cloned from ~700k pooled human saccades 1–5 (one population-level fit) | Learned network weights, behavior-cloned from expert demonstrations |

Note the last row's symmetry: both instantiations are the *same structured
network* — channelized gain blocks writing into one signed field — trained
the same way (maximum-likelihood behavior cloning), with humans as the
demonstrators on the search side and the potential-field expert on the
action side. The spatial prior is not hand-parameterized: it is **derived
from the trained network by probing** (feed history state and task set
with no display), exactly as the original ES2 paper derives the pure ego
spatial field from the trained agent. The fit is population-level only (no
per-subject layer — individual differences are out of scope, §7);
uncertainty on pooled estimates via bootstrap over subjects.

**Why scalar gains in search but gain networks in the agent
(manuscript-ready justification).** The apparent asymmetry — single
fitted numbers (g_T, g_S) on the search side, small learned networks on
the agent side — is not a difference in model but a difference in what
the two tasks allow the same component to express. A channel gain is,
in general, a function of the channel's input: in the reach-avoid task,
goal distance and obstacle proximity vary continuously, the appropriate
gain differs across that range (attraction tapers near the goal;
repulsion scales with proximity), and the demonstrations exercise the
full input range, so the gain must be, and can be, estimated as a
function — hence the small network. In the search task the display
geometry freezes the gain's input: every item lies at the same
eccentricity, and template match and singleton status are binary, so
the gain function is only ever evaluated at a single point per channel,
and its most general identifiable form *is* one number. The scalar is
not a simplification of the gain block but its value under the task's
input distribution — the same task-silencing principle that leaves the
attention window formally present but inert for first saccades on an
iso-eccentric ring. The degeneracy is reversible, and v1 exercises the
reversal: fitting saccades 1–5 restores the distance axis, which v1
assigns to a single shared envelope falloff (σ_env, estimated) while
the channel gains stay scalar — the separability assumption stated in
§3; channel-specific distance profiles, as the agent's gain blocks
learn, remain an extension requiring no structural change.

## 3. Search instantiation, v1 details

- **Front-end (v2 revision: no shortcut — pixels in).** The model
  receives the rendered display, not role flags. A fixed (unfitted)
  Itti & Koch-style perception module — the analog of the agent's LiDAR
  — computes color-opponency (R–G, B–Y) and intensity feature maps and
  their center-surround contrast, combined into a salience map. All
  maps are sampled along rays from the *current fixation* (a visual
  LiDAR), so eccentricity is implicit in the sensory signal and
  re-centers after every saccade, exactly as the agent's scan does.
  **Goal-early assembly (v2.1, canonical after model comparison):**
  there is no separate task-blind salience channel. The task set
  enters the feature channels *before* the contrast stage — fixed
  opponency axes are rotated into template-referenced coordinates (the
  goal supplies the direction of the target color; the fitted gain
  supplies only strength), and the gain-weighted contrast is rectified
  into one goal-modified salience/priority map. Consequences: (a)
  feature-level suppression can only attenuate/relegate (drive toward
  zero) — negative writing is reserved for the spatial sources (traces,
  IoR); (b) below-baseline "suppression" of the singleton is carried by
  template-color enhancement relegating the mismatching item, which the
  pooled data independently favor (the nested mechanism comparison);
  (c) bottom-up capture is the default-gain path (an unweighted
  presence/intensity term), and learning to ignore is gain adjustment.
  Goal-early beat the goal-late variant (separate channels + salience
  map, gains applied after contrast) decisively out-of-sample at equal
  weight count (RESULTS.md, Visual-Search). Separating
  enhancement-vs-suppression and rejection-vs-salience empirically
  requires displays with ≥3 colors — in two-color displays they are
  structurally unidentifiable. The form-match channel is
  pixel-derived (discriminative NCC on the rendered display), with
  the display-provenance limit stated above. No transient channel
  (§7).
- **Network form (mirrors `goal_es2.py`)**: the display is rendered as ray
  maps over the search ring, one per channel; each channel passes through
  its own small learned gain block (the analog of `goal_gain`), producing
  signed contributions summed into one field over rays. History traces are
  runtime state injected through the same machinery. Both top-down gains
  are static single parameters (g_T, g_S): they encode the task set —
  attend green, willfully ignore red — not learning (§7). The pre-onset
  spatial prior is the network's field with history + task set only (no
  display) — the ES2 pure-field probe.
- **Readout — all saccades 1–5 (as in the source analyses)**: each
  saccade is its own conditional-logit trial from the *current* fixation
  — P(saccade → item i) = softmax_i F(i)/τ, τ fixed as the unit of
  measurement. The bare field → softmax, exactly parallel to the agent's
  field → action head: no motor covariates, no lapse parameter (§7).
  For saccade k the ego position is where saccade k−1 landed, so
  item-to-fixation distances vary and the envelope becomes estimable
  (its radial falloff σ_env, 1–2 numbers — three probed chord distances
  on a 6-ring support a monotone falloff, not the full 360° profile).
  The currently fixated item leaves the choice set; trials truncate at
  target fixation (post-target saccades are responding, not searching);
  later-saccade selection effects (target not yet found) are handled by
  conditioning on the current state. Positions only: latency is neither
  generated nor used as a covariate (§7). Averaging (between-item)
  landings need pre-registered assignment/exclusion rules.
- **Inhibition of return — diagnostic first, parameter only if demanded
  (§7)**: v1 fits with no IoR term (zero new parameters, maximally
  agent-like) and checks predicted vs. observed refixation rates; a
  single visited-item penalty is added only if that diagnostic fails,
  reported as a model comparison.
- **Excluded from v1**: saccade latency (entirely), foveal verification,
  channel-specific distance profiles (v1 assumes one shared envelope
  falloff — separability; per-channel falloffs, as the agent's gain
  blocks have, are an extension). The v2 accumulator readout
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
  No saccade → trace feedback: where the eyes went has no effect on the
  traces, or on anything else (no motor-repetition term — scope decision,
  §7; the target-location trace will absorb any motor-repetition variance,
  a stated caveat on β_tgt).
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
saccade choices (acquisition curves; lagged kernels decaying as
(1−η)^k; reversal transients). One pooled fit per task; cross-task η
contrasts ("task A induces faster buildup") compare pooled estimates
across tasks (bootstrap over subjects for uncertainty). Public
trial-level datasets (OSF: Gaspelin, Theeuwes / van Moorselaar labs)
suffice; parameter recovery on synthetic data precedes any human fit.

## 5. The transition (search → action), in full

1. The channel list changes with the sensory task: several static visual
   feature channels → one temporally differenced proximity channel
   (looming) plus a goal channel.
2. The readout changes with the effector: discrete sample (saccade) →
   continuous expectation (movement vector).
3. The parameterization regime changes with the job: fitted scalars for
   measurement → learned weights for competence — with the reach-avoid
   diagnostics (§8) as evidence that the learned version retains the
   structure.

Everything else — the signed field, the source-blind readout, the
channelized gain blocks, the envelope + history gain map, the
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

- **Limitation: within-session learning is location-based, not
  feature-based.** The only quantities that update across trials are
  the two location-indexed traces h_T/h_D; the feature gains (g_T,
  g_D, g_F) are a static task set, fixed at their fitted asymptote
  from trial 1. Consequence, measured (capture_by_trial.py): the
  observed capture effect deepens across the session (-2.7 over
  trials 1-10 to -7.4 by trials 60+, held-out subjects) while the
  model is flat at ~-4.6 - the right depth, no trajectory. Location
  learning cannot produce this: in roving designs the singleton
  occupies all slots equally, so h_D builds uniformly and cancels
  out of the singleton-vs-baseline contrast. The published
  capture-to-suppression trajectory is what within-session GAIN
  learning on the feature channels would look like (b growing from
  ~0 to asymptote, with initial capture carried by a salience
  channel) - the agent-side thesis "learning to ignore is gain
  adjustment," which this model inherits only as a fitted endpoint.
  Stated as a limitation of the current model, not fitted around.
- **No cued spatial knowledge.** Neither the planned search tasks nor the
  reach-avoid task involves cues; the gain map is envelope + history only.
  This also excludes scene-prior guidance — appropriate for singleton-type
  displays; the architecture has an obvious slot should it ever be needed.
- **No transient map in the search model (revised: removed, not merely
  silenced).** The search displays are static with simultaneous onsets, so
  the search channels are purely static feature/salience maps. Temporal
  differencing exists only where the world makes time meaningful — the
  agent's proximity channel (looming). Onset-capture paradigms would
  require adding a transient channel as an explicit model extension, not
  a re-weighting of something already present.
- **Presence-driven trace updating** (not selection-gated); falsifiers in §4.
- **Presence sourced from contrast maps** (weak commitment; see §4).
- **Positions only (scope decision).** Saccade latency is entirely out of
  scope — neither generated nor used as a covariate or gain input. The
  model explains *where* saccades go, never *when*. (Latency
  generation would be the v2 accumulator readout, deferred.)
- **All saccades 1–5 in scope (revised from first-saccade-only),**
  matching the source paper's own saccade-index analysis. Each saccade
  is a conditional-logit trial from the current fixation; this is what
  makes the attention window estimable (§3) and turns the ego-anchored
  vs. world-anchored dissociation (§1.1) into a tested prediction.
- **IoR: try none first.** v1 carries no inhibition-of-return term; the
  predicted vs. observed refixation rate is a reported diagnostic, and
  a single visited-item penalty is added only if it fails (model
  comparison, not assumption).
- **No individual differences (scope decision).** One population-level
  fit; no per-subject parameter layer. Uncertainty on pooled estimates
  via bootstrap over subjects.
- **Top-down gains are static — no learning curve inside g_S (scope
  decision).** g_T and g_S each encode a willful task set (attend green,
  ignore red): one parameter apiece, constant across the experiment.
  Building the first-encounters capture→suppression curve into g_S would
  confound the goal-driven set with selection history — acquisition
  effects belong to history mechanisms (traces), never to the top-down
  gains. Consequence: fits use experimental blocks (stable set; practice
  excluded, as in the source studies), and the practice-block learning
  curve is outside the fitted model's scope (a possible feature-indexed
  history-trace extension, kept strictly separate from g_S).
- **No readout add-ons (scope decision).** No motor-repetition covariate,
  no lapse rate: the readout is the bare field → softmax, exactly
  parallel to the agent's field → action head. Cost accepted: the
  target-location trace absorbs any motor-repetition variance, and
  stray saccades load onto τ-scaled noise rather than a lapse term.
- **Task-silencing principle**: the full model is the union of sources; a
  task silences sources through input structure (no differential
  transients under placeholders; flat statistics → flat traces).
  Parameters receiving no variance from the design are fixed, not fitted.
- **Neural instantiation for the search model (revised decision).** The
  search model is the same structured network as the agent — channelized
  gain blocks trained by maximum likelihood on pooled human saccades (1–5)
  — rather than a hand-parameterized field. This keeps the search model
  maximally close to ES2 and lets the spatial prior be *derived from the
  trained network* (pre-onset probe), as the original paper derives the
  pure ego spatial field. Enhancement vs. suppression is read from
  channel probes (per-channel field contributions, as `compute_fields`
  does in the agent). The earlier all-scalar parameterization is
  retained as a comparison model.

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

**Trained (one pooled fit, positions only, saccades 1–5):** 7–8 numbers
doing all the work, behavior-cloned by MLE on saccade destinations
across all subjects and studies —
g_T (template gain, static); g_S (salience/rejection gain, static, one
number — task set, not learning, §7); β_tgt, β_dist, η_tgt, η_dist
(location traces); σ_env (attention-window radial falloff, 1–2 numbers
— estimable because saccades 2+ vary item-to-fixation distance; silent
within the first-saccade subset, where all items are iso-eccentric).
τ fixed as unit. No latency terms, no per-subject layer, no
motor-repetition or lapse terms, no IoR term unless the refixation
diagnostic demands one (§7) — the readout is the bare field → softmax,
as in the agent.

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
lineage. Distinctive claims: one signed source-blind field; an a priori spatial
prior derived from the ego's state (probed from the trained network);
effector-referenced coordinates — and the cross-domain evidence that this
structure, not capacity or data, yields modular, causally testable
goal conditioning.
