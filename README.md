# Five-Link Inverted Pendulum

A cart, five connected pendulums, a nonlinear simulator, and a controller
trained by imitation learning using PyTorch neural networks.

![The trained controller swinging up and balancing](assets/demo.gif)

## Run the trained controller

Use Python 3.11 or newer. From this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python play.py
```

Controls: **R** resets, **D** disturbs a joint, **Space** pauses, **Esc** exits.
For manual control, run `python play.py --manual` and use the arrow keys.
The included model is `models/scheduled.pt`.

## Results from this rewrite

| Experiment | Result |
| --- | --- |
| PPO, 28,548 valid simulation steps | 0 successes in 109 completed episodes |
| Ordinary two-hidden-layer imitation MLP | 0/32 successful evaluation episodes |
| Structured neural imitation controller | **512/512** successful episodes |
| Same controller, physics step halved | **64/64** successful episodes |

The PPO run is a small experiment, not evidence that PPO cannot solve this
problem. Its rollouts are truncated after 2.56 seconds and bootstrap the
remaining value; it is not a strong PPO benchmark.

A successful episode must hold **every** link within **10 degrees** of upright
and below **0.5 radians/second** for **two continuous seconds**. Episodes last
15 simulated seconds. Leaving the ±2.4 m track or producing a nonfinite state
at any physics step makes the entire episode fail, even after a successful hold.

The two evaluations use seeds 91000 and 99000. Both start near the downward
configuration, with small random variations in cart position, speed, joint
angles and angular speeds. They do not establish recovery from arbitrary
configurations or large disturbances. The largest observed cart positions
were 1.179 m and 1.144 m respectively.

Reproduce the evaluations:

```bash
python evaluate.py --episodes 512 --seed 91000 --output results/evaluation_512.json
python evaluate.py --episodes 64 --seed 99000 --physics-step 0.0025 --output results/evaluation_fine_64.json
python -m unittest test_physics test_learning
```

The controller still updates every 0.02 seconds in the finer-step test.
Only the physics integration step changes, from 0.005 to 0.0025 seconds.
Reports contain the result, maximum cart position and longest hold for each trial.

## Read the code in this order

1. `play.py`: draw the cart and links; ask the controller for an action.
2. `environment.py`: start an episode, apply a force, check balance and track limits.
3. `physics.py`: work out how the cart and links accelerate, then advance time.
4. `model.py`: the small neural networks and the readable route-selection loop.
5. `train.py`: turn demonstration state/force pairs into trained network weights.
6. `evaluate.py`: run complete episodes and write their outcomes.

`ppo.py` is the separate PPO experiment. `baseline_model.py` and
`train_baseline.py` preserve the conventional MLP attempt that failed.
`WALKTHROUGH.md` explains the important lines and the necessary physics math.

## What imitation learning does here

The training data contains eight demonstrated routes from hanging down to
upright. Each route has 250 control stages. At reset, the controller picks
the route whose starting state is closest to the actual starting state.

Each stage has a tiny **12 → 24 ReLU → 1** neural network. It receives the
difference between the current state and the typical demonstrated state for
that stage. Its output is the cart force. After five seconds, a final network
takes over for balancing. These are real `torch.nn` modules, used directly
for inference; the controller does not call an expert controller or LQR solver.

This is deliberately **structured imitation**, not one universal nonlinear
policy. The hidden layer is fixed: it splits each input into positive and
negative parts. Only the output weights and bias are trained, using supervised
least squares. This architecture can express the expert's local linear feedback
exactly. ReLU is present, but that does not make the learned control law broadly
nonlinear or establish general-purpose neural control. There are 2,001 small
networks, rather than one large network. Their code is shared; only weights differ.

The stage means and input scales are learned from the demonstrations. The
training errors in `models/scheduled.json` are training errors, not held-out
validation scores. The independent physical rollouts above are the performance
test. Predicting demonstration forces alone is insufficient, as the failed MLP shows.

## Train it

```bash
python train.py
python evaluate.py
```

Both demonstration datasets are included under `data/`, so training does not
depend on `old/` or a download. The simulator, learning code, evaluation and
visualization were rewritten; the **expert demonstrations were retained from
the earlier project**, not generated anew. Their provenance is in `data/README.md`.
Regenerating the original optimized expert from scratch is outside this rewrite.

To repeat the other experiments:

```bash
python ppo.py --updates 8
python train_baseline.py --epochs 100
python evaluate.py --model models/imitation.pt --episodes 32 --output results/mlp_baseline.json
```

## Physical model

Each link is a massless rigid rod with a point mass at its tip. The five angles
are absolute angles measured from upright. The model includes cart friction,
relative joint damping, gravity and coupling between all six moving coordinates.
Only the cart is actuated. RK4 integrates motion using four slope estimates.
No joint motors, state resets, scripted motion or assistance are used during a trial.

## Local history

The previous working files are preserved under `old/` and excluded from Git.
The original three commits remain ancestors of the local `readable-rebuild`
branch. New commits record real completed stages with their actual dates.
No push or published-history replacement was performed.

AI assistance was used for this rewrite, including implementation and tests.
