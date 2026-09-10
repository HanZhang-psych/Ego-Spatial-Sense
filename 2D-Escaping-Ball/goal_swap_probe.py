"""Goal-swap diagnostic: does the agent actually use its goal input?

A causal ablation on the goal channel of the reach-avoid agent, in two
complementary forms (run over several seeds each; wrong goals are mirrored
or independently random so an accidental alignment with the true goal
cannot masquerade as goal-blindness):

  (a) Failure form — feed a goal DIFFERENT from the scored one and score
      against the TRUE goal.  Sensitivity = true-goals/min(correct) minus
      true-goals/min(wrong).  Goal-conditioned agent: large gap (it chases
      the wrong place).  Goal-blind agent: ~zero gap.

  (b) Positive form — score against the FED goal across many randomly
      assigned goals.  A goal-conditioned agent reaches whatever goal it is
      handed; a goal-blind agent reaches the fed goal only at chance rate.

Collision rates are reported per condition: the goal input should steer
DIRECTION, not avoidance, so collisions/min should stay comparable across
conditions.  If they collapse under a wrong goal, that entanglement is
itself a finding.

Usage:
    python goal_swap_probe.py --model_path pretrained/goal_es2.pth --device cpu
    python goal_swap_probe.py --model_path pretrained/es2.pth --device cpu  # goal-blind control
"""

import argparse

from evaluate_reach_avoid import add_common_args, load_model, run_rollout, summarize

KEYS = ["true_goals_per_min", "fed_goals_per_min", "collisions_per_min", "field_asymmetry"]


def run_condition(model, is_goal_model, args, goal_mode):
    results = []
    for i in range(args.num_seeds):
        r = run_rollout(
            model, is_goal_model, args, args.random_seed + i, goal_mode=goal_mode
        )
        results.append(r)
    avg = summarize(results, KEYS[:3])
    asyms = [r["field_asymmetry"] for r in results if r["field_asymmetry"] == r["field_asymmetry"]]
    avg["field_asymmetry"] = sum(asyms) / len(asyms) if asyms else float("nan")
    return avg


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    model, is_goal_model = load_model(args)
    kind = "goal-conditioned" if is_goal_model else "goal-blind (control)"
    print(f"\nProbing {args.model_path} [{kind}] — "
          f"{args.num_seeds} seeds x {args.max_steps} steps per condition\n")

    conditions = ["correct", "opposite", "random"]
    stats = {}
    for mode in conditions:
        stats[mode] = run_condition(model, is_goal_model, args, mode)
        s = stats[mode]
        print(
            f"[{mode:8s}] true-goals/min={s['true_goals_per_min']:6.2f}  "
            f"fed-goals/min={s['fed_goals_per_min']:6.2f}  "
            f"collisions/min={s['collisions_per_min']:5.2f}  "
            f"field_asym={s['field_asymmetry']:.4f}"
        )

    correct = stats["correct"]["true_goals_per_min"]
    print("\n--- Goal-sensitivity (failure form, scored against TRUE goal) ---")
    for mode in ["opposite", "random"]:
        wrong = stats[mode]["true_goals_per_min"]
        rel = (correct - wrong) / correct if correct > 0 else float("nan")
        print(
            f"gap(correct - {mode}): {correct:.2f} - {wrong:.2f} = "
            f"{correct - wrong:.2f} goals/min  (relative drop {rel:.1%})"
        )

    print("\n--- Goal-following (positive form, scored against FED goal) ---")
    print(
        f"random fed goals reached: {stats['random']['fed_goals_per_min']:.2f}/min "
        f"(correct-goal reference: {correct:.2f}/min; a goal-blind agent only "
        f"reaches fed goals at chance rate)"
    )

    print("\n--- Avoidance entanglement check ---")
    c0 = stats["correct"]["collisions_per_min"]
    for mode in ["opposite", "random"]:
        print(
            f"collisions/min correct vs {mode}: {c0:.2f} vs "
            f"{stats[mode]['collisions_per_min']:.2f}"
        )


if __name__ == "__main__":
    main()
