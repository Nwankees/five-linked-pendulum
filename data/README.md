# Demonstration provenance

These are state/force training examples retained from the earlier local
five-link pendulum project, before the code rewrite:

* `demonstrations.pt`: originally `runs/imitation_complete/dataset.pt`.
  Eight routes, each with 124 episodes and 250 time steps, including collected
  corrective examples. Each state has 12 components. Forces are in newtons.
* `balance.pt`: originally `runs/imitation_complete/balance_dataset.pt`.
  512 state/force examples near the upright configuration.

The original expert used numerically optimized swing-up trajectories with
feedback tracking and local balancing. This rewrite does not claim to have
regenerated that expert. `train.py` reads the labelled examples only; it does
not load original fitted policy weights or expert feedback gains.

The expert has eight downward-start routes. Evaluating new random downward
starts tests local generalization around this demonstration distribution,
not unconstrained swing-up from arbitrary states.

`SHA256SUMS.txt` records the exact input files. Training and playback do not
read from the local `old/` backup.
