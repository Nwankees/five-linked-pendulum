# Evaluate the controller across multiple episodes
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from environment import TorchFivePoleBatch
from controllers import make_controller, load_bundle


@torch.no_grad()
def run_evaluation(swingup, catch, config, device='cpu', episodes=64, seed=1000,
                   mode='auto', initial_states=None, disturbance=False, trace_path=None):
    env = TorchFivePoleBatch(episodes, device, seed, config)
    if initial_states is not None:
        env.reset(states=initial_states)
    initial = env.state.clone()
    controller = make_controller(env, swingup, catch, mode)
    active = ~env.finished
    zeros = env.tensor(np.zeros(episodes))
    max_up = zeros.clone()
    max_weak = zeros.clone()
    best_speed = zeros.clone()
    best_energy = zeros.clone()
    min_cost = torch.full_like(zeros, float('inf'))
    max_force = zeros.clone()
    saturated = zeros.clone()
    counts = zeros.clone()
    tracks = torch.zeros_like(active, dtype=torch.long)
    numerics = torch.zeros_like(active, dtype=torch.long)
    entries = torch.zeros_like(active, dtype=torch.long)
    switches = torch.zeros_like(active, dtype=torch.long)
    success_time = torch.full_like(zeros, float('nan'))
    trace_states = [initial.cpu().numpy()]
    trace_actions = []
    trace_modes = []
    for step in range(env.max_steps):
        if disturbance and step == env.max_steps // 2:
            env.state[active, 7] += 1.5
        min_cost = torch.minimum(min_cost, torch.where(active, controller.local.cost(env.state), min_cost))
        action = controller.action()
        entries = torch.where(active, controller.entries, entries)
        switches = torch.where(active, controller.switches, switches)
        saturated += (active & (action.abs() >= .999)).double()
        counts += active.double()
        max_force = torch.maximum(max_force, torch.where(active, action.abs()*env.max_force, 0))
        _, _, done, info = env.step(action)
        improved = active & (info['weakest_upright'] > max_weak)
        best_speed = torch.where(improved, env.state[:, 7:].abs().max(1).values, best_speed)
        best_energy = torch.where(improved, info['kinetic_energy'], best_energy)
        max_up = torch.maximum(max_up, torch.where(active, info['upright'], 0))
        max_weak = torch.maximum(max_weak, torch.where(active, info['weakest_upright'], 0))
        tracks |= (active & info['failed_track']).long()
        numerics |= (active & info['failed_numeric']).long()
        new_success = active & info['newly_successful']
        success_time = torch.where(new_success, info['elapsed_time'], success_time)
        if trace_path:
            trace_states.append(env.state.cpu().numpy().copy())
            trace_actions.append(action.cpu().numpy())
            trace_modes.append(controller.mode.cpu().numpy())
        active &= ~done
        if not active.any():
            break
    rows = []
    for i in range(episodes):
        rows.append(dict(batch_seed=seed, episode_index=i, success=bool(env.success_achieved[i]),
                         survived_horizon=not bool(tracks[i] or numerics[i]),
                         success_time=float(success_time[i]) if torch.isfinite(success_time[i]) else None,
                         max_hold_s=float(env.max_hold_ticks[i]*env.physics_dt),
                         max_up=float(max_up[i]), max_weak=float(max_weak[i]),
                         speed_at_best_weak=float(best_speed[i]), kinetic_at_best_weak=float(best_energy[i]),
                         min_capture_cost=float(min_cost[i]), lqr_entries=int(entries[i]),
                         switches=int(switches[i]), saturated_fraction=float(saturated[i]/counts[i].clamp_min(1)),
                         max_force=float(max_force[i]), failed_track=bool(tracks[i]), failed_numeric=bool(numerics[i])))
    summary = dict(episodes=episodes, reset='hanging' if initial_states is None else 'explicit_diagnostic_states',
                   mode=controller.kind, success_rate=float(env.success_achieved.double().mean()),
                   success_and_survived_rate=float((env.success_achieved & ~(tracks.bool()|numerics.bool())).double().mean()),
                   mean_max_hold_s=float((env.max_hold_ticks.double()*env.physics_dt).mean()),
                   best_hold_s=float(env.max_hold_ticks.max()*env.physics_dt),
                   mean_max_up=float(max_up.mean()), mean_max_weak=float(max_weak.mean()),
                   mean_speed_at_best_weak=float(best_speed.mean()),
                   min_capture_cost=float(min_cost.min()), median_min_capture_cost=float(min_cost.median()),
                   lqr_entry_rate=float((entries>0).double().mean()),
                   track_failure_rate=float(tracks.double().mean()), numeric_failure_rate=float(numerics.double().mean()))
    if trace_path:
        np.savez_compressed(trace_path, states=np.asarray(trace_states), actions=np.asarray(trace_actions),
                            modes=np.asarray(trace_modes), dt=env.dt)
    return {'summary': summary, 'episodes': rows, 'lqr': controller.local.report,
            'plant': asdict(config), 'batch_seed': seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/imitation_complete/best.pt')
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--mode', choices=['auto', 'hybrid', 'policy', 'lqr'], default='auto')
    parser.add_argument('--disturbance', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--trace')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(1)
    if torch.cuda.is_available() and not args.cpu:
        device = 'cuda'
    else:
        device = 'cpu'
    swingup, catch, config, _ = load_bundle(args.model, device)
    result = run_evaluation(swingup, catch, config, device, args.episodes, args.seed, args.mode,
                            disturbance=args.disturbance, trace_path=args.trace)
    result['model'] = str(Path(args.model).resolve())
    print(json.dumps(result['summary'], indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
