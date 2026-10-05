import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from controllers import LocalLQR, load_bundle
from environment import TorchFivePoleBatch
from evaluate import run_evaluation
from imitation import ExpertPolicy, ImitationController, ImitationPolicy, fit_balance_imitation, fit_behavioral_cloning


@torch.no_grad()
def collect_demonstrations(expert, config, episodes_per_expert=64, seed=10, device="cpu", student=None, beta=1.0, noise=True):
    # Collect successful expert examples
    expert_count = len(expert.initials)
    episode_count = expert_count * episodes_per_expert
    env = TorchFivePoleBatch(episode_count, device, seed, config)

    routes = torch.arange(expert_count, device=device)
    routes = routes.repeat_interleave(episodes_per_expert)
    teacher = ImitationController(env, expert, routes=routes)
    learner = None
    if student is not None:
        learner = ImitationController(env, student, mode="policy", routes=routes)

    horizon = expert.horizon_steps
    states = torch.empty((episode_count, horizon, 12), device=device, dtype=torch.float64)
    labels = torch.empty((episode_count, horizon), device=device, dtype=torch.float64)
    applied_forces = torch.empty_like(labels)
    invalid = torch.zeros(episode_count, device=device, dtype=torch.bool)
    peak_force = torch.zeros(episode_count, device=device, dtype=torch.float64)

    for step in range(env.max_steps):
        if noise and step > 0 and step < horizon and step % 5 == 0:
            noise_scale = env.tensor([0.0002, 0.002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.002, 0.002, 0.002, 0.002, 0.002])
            if step * env.dt >= 3:
                noise_scale = noise_scale * 0.1
            disturbance = env.randn(episode_count, 12) * noise_scale
            active = (~env.finished)[:, None]
            env.state = env.state + torch.where(active, disturbance, 0)
            env.state[:, 2:7] = env.wrap(env.state[:, 2:7])

        teacher_force = teacher.raw_force()
        teacher_action = (teacher_force / env.max_force).clamp(-1, 1)
        action = teacher_action

        if learner is not None:
            learner_action = learner.action()
            random_values = torch.rand(episode_count, device=device, generator=env.generator)
            teacher_mask = random_values < beta
            action = torch.where(teacher_mask, teacher_action, learner_action)

        if step < horizon:
            states[:, step] = env.state
            labels[:, step] = teacher_force
            applied_forces[:, step] = action * env.max_force

        peak_force = torch.maximum(peak_force, action.abs() * env.max_force)
        _, _, _, info = env.step(action)
        invalid = invalid | info["failed_track"] | info["failed_numeric"]

    full_length = env.elapsed_ticks >= env.max_steps * env.action_repeat
    accepted = env.success_achieved & ~invalid & full_length
    samples = []
    expert_summaries = []

    for expert_index in range(expert_count):
        mask = accepted & (routes == expert_index)
        route_samples = {
            "states": states[mask].cpu(),
            "forces": labels[mask].cpu(),
            "applied": applied_forces[mask].cpu()
        }
        samples.append(route_samples)

        minimum_hold = 0.0
        if mask.any():
            hold_times = env.max_hold_ticks[mask].double() * env.physics_dt
            minimum_hold = float(hold_times.min())

        expert_summary = {
            "expert": expert_index,
            "attempted": episodes_per_expert,
            "accepted": int(mask.sum()),
            "min_hold_s": minimum_hold
        }
        expert_summaries.append(expert_summary)

    summary = {
        "seed": seed,
        "beta": beta,
        "process_disturbances": noise,
        "attempted": episode_count,
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "max_applied_force_N": float(peak_force.max()),
        "experts": expert_summaries
    }
    return samples, summary


@torch.no_grad()
def collect_balance_demonstrations(config, device="cpu", seed=8123, episodes=256):
    # Collect examples near the upright position
    env = TorchFivePoleBatch(episodes, device, seed, config)
    local_controller = LocalLQR(env)

    directions = env.randn(episodes, 12)
    directions = directions / directions.norm(dim=1, keepdim=True)
    factor = torch.linalg.cholesky(local_controller.P)
    initial_states = torch.linalg.solve_triangular(factor.T, directions.T, upper=True)
    initial_states = initial_states.T * (4 * local_controller.rho) ** 0.5

    env.reset(states=initial_states)
    target_forces = local_controller.force(initial_states)
    invalid = torch.zeros(episodes, device=device, dtype=torch.bool)

    for i in range(env.max_steps):
        action = local_controller.action(env.state)
        _, _, _, info = env.step(action)
        invalid = invalid | info["failed_track"] | info["failed_numeric"]

    full_length = env.elapsed_ticks >= env.max_steps * env.action_repeat
    valid = env.success_achieved & ~invalid & full_length
    if valid.sum() < 32:
        raise RuntimeError("Not enough successful balance examples")

    summary = {"attempted": episodes, "accepted": int(valid.sum())}
    return initial_states[valid].cpu(), target_forces[valid].cpu(), summary


def merge_samples(old_samples, new_samples):
    if old_samples is None:
        return new_samples

    merged = []
    for i in range(len(old_samples)):
        old_route = old_samples[i]
        new_route = new_samples[i]
        route = {}
        for name in old_route:
            route[name] = torch.cat([old_route[name], new_route[name]])
        merged.append(route)
    return merged


def train(args):
    torch.set_num_threads(1)
    device = "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"

    expert, _, config, expert_data = load_bundle(args.expert, device)
    if not isinstance(expert, ExpertPolicy):
        raise ValueError("--expert must be a verified expert library")

    default_name = "imitation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or Path("runs") / default_name)
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    all_samples = None
    policy = None
    best_score = (-1.0, -1.0, -1.0)

    if args.resume:
        policy, _, old_config, _ = load_bundle(args.resume, device)
        if not isinstance(policy, ImitationPolicy) or old_config != config:
            raise ValueError("The resume checkpoint does not match this expert")
        dataset_path = Path(args.resume).parent / "dataset.pt"
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
        if not torch.equal(dataset["initials"], expert.initials.cpu()):
            raise ValueError("The resume data came from a different expert")
        all_samples = dataset["samples"]

    balance_states, balance_forces, balance_summary = collect_balance_demonstrations(config, device, args.seed + 8123)
    balance_fit, balance_rmse = fit_balance_imitation(balance_states, balance_forces)
    balance_dataset = {
        "states": balance_states,
        "forces": balance_forces,
        "validation": balance_summary
    }
    torch.save(balance_dataset, output / "balance_dataset.pt")
    print(f"Training on {device} with {len(expert.initials)} expert routes", flush=True)

    for round_index in range(args.dagger_rounds + 1):
        if policy is None:
            teacher_chance = 1.0
        else:
            teacher_chance = max(0.0, 1.0 - (round_index + 1) / (args.dagger_rounds + 1))

        new_samples, collection = collect_demonstrations(
            expert, config, args.rollouts_per_expert, args.seed + round_index * 10000,
            device, policy, teacher_chance, not args.no_noise
        )
        all_samples = merge_samples(all_samples, new_samples)

        for route in all_samples:
            if len(route["states"]) < 16:
                raise RuntimeError("An expert has fewer than 16 valid rollouts. Increase --rollouts-per-expert.")

        fitted, fit_metrics = fit_behavioral_cloning(all_samples, expert.initials, args.ridge)
        fitted.update(balance_fit)
        policy = ImitationPolicy(fitted, device)

        evaluation = run_evaluation(policy, None, config, device, args.eval_episodes, seed=50000, mode="policy")
        summary = evaluation["summary"]
        record = {
            "round": round_index,
            "collection": collection,
            "fit": fit_metrics,
            "balance_validation": balance_summary,
            "balance_rmse_N": balance_rmse,
            "evaluation": summary
        }
        with (output / "training.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")

        total_rmse = 0.0
        for item in fit_metrics:
            total_rmse += item["force_rmse_N"]
        average_rmse = total_rmse / len(fit_metrics)
        print(f"round {round_index}: valid demos {collection['accepted']}/{collection['attempted']}, force RMSE {average_rmse:.5f} N, success {summary['success_rate']:.1%}", flush=True)

        bundle = {
            "format": "imitation-v1",
            "policy": fitted,
            "plant": expert_data["plant"],
            "round": round_index,
            "expert_source": str(Path(args.expert).resolve()),
            "evaluation": evaluation,
            "args": vars(args)
        }
        torch.save(bundle, output / "latest.pt")

        score = (summary["success_and_survived_rate"], summary["success_rate"], summary["mean_max_hold_s"])
        if score > best_score:
            best_score = score
            torch.save(bundle, output / "best.pt")

        dataset = {"samples": all_samples, "initials": expert.initials.cpu()}
        torch.save(dataset, output / "dataset.pt")

    torch.save(bundle, output / "final.pt")
    print(f"Training finished: {output / 'best.pt'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", default="experts/library.pt")
    parser.add_argument("--rollouts-per-expert", type=int, default=64)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-7)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()

    if args.rollouts_per_expert < 1 or args.eval_episodes < 1:
        parser.error("Episode counts must be positive")
    if args.dagger_rounds < 0 or args.ridge <= 0:
        parser.error("DAgger rounds and ridge are invalid")
    return args


if __name__ == "__main__":
    train(parse_args())
