"""
Seed sweep for the adversarial edge-case evaluation.

Runs the adversarial 2D Escaping Ball scenario (see ``evaluate_adversarial.py``)
across many random seeds for each model, then prints a per-model survival-time
summary table -- the same style of comparison the paper reports for its
10/20/30/40-ball conditions, but for the "one ball suddenly pursues the ego"
condition.

For each (model, seed) the trigger time, pursuer, and jitter are fixed by the
seed, so every model faces the identical perturbation on that seed.

Example:
    python3 sweep_adversarial.py --device cpu --num_seeds 10 --num_balls 10
    python3 sweep_adversarial.py --device cpu --seeds 0 1 2 3 4 \
        --pursuer_speed 6 --csv results.csv
"""

import os

# Force headless SDL before pygame is imported anywhere.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import statistics
from types import SimpleNamespace

import torch

import evaluate_adversarial as ea
from model.es2 import Es2Model
from model.mlp import MLPModel
from model.transformer import TransformerModel


def load_model(model_path, ns):
    """Mirror the model-selection logic in evaluate_adversarial.__main__."""
    if "es2" in model_path:
        model = Es2Model(
            num_features=ns.num_features,
            num_actions=ns.num_actions,
            sensing_range=ns.sensing_range,
        ).to(ns.device)
    elif "mlp" in model_path:
        model = MLPModel(
            num_features=ns.num_features,
            num_actions=ns.num_actions,
            sensing_range=ns.sensing_range,
        ).to(ns.device)
    elif "transformer" in model_path:
        model = TransformerModel(
            num_features=ns.num_features,
            num_actions=ns.num_actions,
            d_model=ns.d_model,
            nhead=ns.nhead,
            num_encoder_layers=ns.num_layers,
            dim_feedforward=ns.dim_feedforward,
            sensing_range=ns.sensing_range,
        ).to(ns.device)
    else:
        raise ValueError(f"Undefined model type for path: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=ns.device))
    model.eval()
    return model


def make_args(base, seed):
    """A fresh args namespace for one run (evaluate reads module-global args too)."""
    ns = SimpleNamespace(**vars(base))
    ns.random_seed = seed
    return ns


def run_one(model, base, seed):
    ns = make_args(base, seed)
    # Ball.move reads the module-global `args`; keep it in sync with this run.
    ea.args = ns
    steps = ea.evaluate(model, ns)
    return steps


def main():
    parser = argparse.ArgumentParser()

    # Which models to sweep.
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pretrained/es2.pth", "pretrained/mlp.pth", "pretrained/transformer.pth"],
        help="Model checkpoint paths to compare.",
    )

    # Which seeds to sweep (either an explicit list or a generated range).
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of seeds. Overrides --num_seeds/--seed_start.",
    )
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)

    # Environment / adversarial settings (passed straight through to evaluate).
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--pursuer_speed", type=float, default=0.0)
    parser.add_argument("--pursuit_noise", type=float, default=0.6)
    parser.add_argument("--trigger_min", type=float, default=5.0)
    parser.add_argument("--trigger_max", type=float, default=30.0)
    parser.add_argument("--pursuer_index", type=int, default=-1)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--device", type=str, default="cpu")

    # Transformer-specific parameters.
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)

    # Output.
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print every per-seed result."
    )

    args = parser.parse_args()

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    # Base namespace shared by every run; evaluate() runs headless, no-throttle, quiet.
    base = SimpleNamespace(
        num_features=args.num_features,
        num_actions=args.num_actions,
        frequency=args.frequency,
        num_balls=args.num_balls,
        max_speed=args.max_speed,
        max_steps=args.max_steps,
        pursuer_speed=args.pursuer_speed,
        pursuit_noise=args.pursuit_noise,
        trigger_min=args.trigger_min,
        trigger_max=args.trigger_max,
        pursuer_index=args.pursuer_index,
        sensing_range=args.sensing_range,
        width=args.width,
        height=args.height,
        device=args.device,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        render=False,
        no_throttle=True,
        quiet=True,
        random_seed=0,
    )

    pursuer_speed = args.pursuer_speed if args.pursuer_speed > 0 else args.max_speed
    print(
        f"Adversarial seed sweep | balls={args.num_balls} "
        f"pursuer_speed={pursuer_speed:.2f} (ego max={args.max_speed:.2f}) "
        f"noise=±{args.pursuit_noise:.2f} rad "
        f"trigger=[{args.trigger_min:.0f},{args.trigger_max:.0f}]s "
        f"max={args.max_steps / args.frequency:.0f}s | seeds={seeds}\n"
    )

    freq = args.frequency
    full = args.max_steps
    rows = []  # (model_name, list_of_seconds, list_of_steps, n_completed)
    per_seed_records = []  # (model_name, seed, steps, seconds, completed)

    for model_path in args.models:
        name = os.path.splitext(os.path.basename(model_path))[0]
        model = load_model(model_path, base)
        secs, step_list, completed = [], [], 0
        for seed in seeds:
            steps = run_one(model, base, seed)
            s = steps / freq
            done = steps >= full
            secs.append(s)
            step_list.append(steps)
            completed += int(done)
            per_seed_records.append((name, seed, steps, s, done))
            if args.verbose:
                flag = " [reached max]" if done else ""
                print(f"  {name:<12} seed {seed:<4} -> {s:8.2f} s{flag}")
        rows.append((name, secs, step_list, completed))
        if args.verbose:
            print()

    # ---------------- Summary table ----------------
    print("=" * 74)
    header = f"{'model':<14}{'n':>4}{'mean(s)':>10}{'std':>9}{'min':>9}{'max':>9}{'#maxed':>9}"
    print(header)
    print("-" * 74)
    for name, secs, _steps, completed in rows:
        n = len(secs)
        mean = statistics.mean(secs) if secs else 0.0
        std = statistics.pstdev(secs) if len(secs) > 1 else 0.0
        print(
            f"{name:<14}{n:>4}{mean:>10.2f}{std:>9.2f}"
            f"{min(secs):>9.2f}{max(secs):>9.2f}{completed:>9}"
        )
    print("=" * 74)
    print("mean = average survival (s); #maxed = runs reaching --max_steps without collision")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "seed", "steps", "seconds", "reached_max"])
            for rec in per_seed_records:
                w.writerow(rec)
        print(f"\nWrote per-run results to {args.csv}")


if __name__ == "__main__":
    main()
