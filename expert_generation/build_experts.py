import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from environment import PlantConfig, TorchFivePoleBatch
from expert_generation.expert_tracking import prepare_reference, verify_reference
from expert_generation.trajectory_expert import solve_trajectory


def build(args):
    """Build a library using only trajectories that pass simulator replay."""
    torch.set_num_threads(1)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = PlantConfig()
    initial_states = TorchFivePoleBatch(
        args.count, seed=args.seed, config=config
    ).state.numpy()
    initial_states[0] = 0
    initial_states[0, 2:7] = np.pi

    guess = dict(np.load(args.guess)) if args.guess else None
    accepted_references = []
    attempts = []

    for index, initial_state in enumerate(initial_states):
        start_time = time.time()
        candidate = solve_trajectory(
            initial_state,
            args.horizon,
            args.intervals,
            args.seed + index,
            guess,
            max_iterations=args.iterations,
            config=config,
        )
        record = {
            name: value
            for name, value in candidate.items()
            if name not in {"states", "forces", "accelerations", "plant"}
        }
        record["index"] = index
        record["initial_state"] = initial_state.tolist()

        candidate_is_valid = (
            candidate["solver_success"]
            and candidate["violation"] < 1e-5
            and candidate["max_assistance"] == 0
        )
        if candidate_is_valid:
            reference = prepare_reference(candidate, config)
            replay = verify_reference(
                reference, config, initial=initial_state
            )
            record["replay"] = replay
            if replay["accepted"]:
                accepted_references.append(reference)
                guess = candidate

        record["wall_s"] = time.time() - start_time
        attempts.append(record)
        replay_passed = record.get("replay", {}).get("accepted", False)
        print(
            f"expert {index + 1}/{len(initial_states)} | "
            f"solver {candidate['status']} | replay {replay_passed} | "
            f"{record['wall_s']:.1f}s",
            flush=True,
        )
        output.with_suffix(".json").write_text(
            json.dumps(
                {"attempts": attempts, "accepted": len(accepted_references)},
                indent=2,
            ),
            encoding="utf-8",
        )

    if not accepted_references:
        raise RuntimeError("No trajectory passed simulator replay")

    library = {
        "format": "expert-library-v1",
        "plant": asdict(config),
        "seed": args.seed,
        "reference": torch.tensor(
            np.stack([item["reference"] for item in accepted_references]),
            dtype=torch.float64,
        ),
        "inputs": torch.tensor(
            np.stack([item["inputs"] for item in accepted_references]),
            dtype=torch.float64,
        ),
        "gains": torch.tensor(
            np.stack([item["gains"] for item in accepted_references]),
            dtype=torch.float64,
        ),
        "validation": attempts,
    }
    torch.save(library, output)
    print(f"Saved {len(accepted_references)} experts to {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--intervals", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--guess")
    parser.add_argument("--output", default="experts/library.pt")
    args = parser.parse_args()
    if min(args.count, args.intervals, args.iterations) < 1:
        parser.error("Counts must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
