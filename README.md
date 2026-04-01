# Compliant Mechanism Synthesis

This repository now implements a first point-and-beam graph prototype for 2D
compliant mechanism synthesis.

## Core Representation

A design is represented by:

- node positions `X ∈ R^(N x 2)` in normalized `[0,1] x [0,1]`
- node roles `R ∈ {fixed, mobile, free}^N`
- symmetric connectivity / thickness activations `A ∈ [0,1]^(N x N)`

The diagonal of `A` is always zero. Beam radius is derived from `A`.

## Mechanics

The mechanics module uses a differentiable 2D Euler-Bernoulli frame FEM in
PyTorch.

- each node has DOFs `(ux, uy, theta)`
- each edge is a beam element with axial and bending stiffness
- the 2 fixed nodes are rigidly clamped
- the 2 mobile nodes are tied to a rigid body with `(Ux, Uy, Theta)`
- effective properties are computed from three unit load cases:
  - horizontal force
  - vertical force
  - moment

The reported target properties are:

- `k_x`
- `k_y`
- `k_theta`

## Learning Model

The model is a graph-based iterative refinement network.

- each node is a token
- node tokens use position features and learned role embeddings
- target properties and timestep are injected globally
- attention layers alternate between:
  - connectivity-conditioned self-attention
  - unconditioned self-attention

The model predicts:

- per-node displacement deltas `(dx, dy)`
- per-node latent vectors `u_i`

Connectivity updates are produced only from dot products between these latent
vectors.

The reported mechanical properties remain the raw effective stiffness values
`k_x`, `k_y`, and `k_theta`. The training loop normalizes them internally only
for model conditioning and loss scaling.

## Training Loop

Training starts from a clean synthetic graph sample `(X_0, A_0)` and creates a
noisy state `(X_t, A_t)`.

For each batch:

- add Gaussian noise to free-node positions
- add Gaussian noise to the connectivity matrix
- refine the noisy graph over several learned rollout steps
- apply position updates only to free nodes
- update connectivity through adjacency logits plus node-latent dot products
- evaluate the differentiable frame FEM on intermediate and final states

The loss includes:

- target property loss on `k_x`, `k_y`, `k_theta`
- reconstruction loss on free-node positions
- reconstruction loss on connectivity
- beam material penalty
- connectivity penalty
- binarization encouragement for `A`

## Synthetic Data

Synthetic graph targets contain:

- random valid 2D point sets
- exactly 2 fixed nodes near the bottom boundary
- exactly 2 mobile nodes near the top boundary
- remaining free nodes in the interior
- procedurally generated sparse graph motifs such as chains, arches, braced
  layouts, and fan-like structures

## Train

```bash
uv sync
uv run cms-train --name prototype
```

The default training configuration uses:

- `num_nodes=12`
- `d_model=256`
- `num_layers=6`
- iterative rollout refinement
- canonical TensorBoard evaluations during training

## Sample

```bash
uv run cms-sample \
  --checkpoint-path artifacts/prototype.pt \
  --target-kx 2.0e-3 --target-ky 4.0e-3 --target-ktheta 5.0e-4 \
  --name sample
```

Sampling outputs:

- final node positions
- node roles
- final continuous connectivity matrix
- thresholded sparse graph for visualization
- evaluated properties

## TensorBoard

```bash
uv run tensorboard --logdir runs
```

Each run is written to `runs/<timestamp>-<name>`.

## Visualization

Training and sampling log graph figures showing:

- fixed nodes
- mobile nodes
- free nodes
- active beams with line width proportional to activation

## Notes

- This prototype intentionally favors clarity over aggressive optimization.
- The main learning and mechanics path no longer depends on a pixel grid.
- Thresholding is used for sparse visualization, but training keeps connectivity continuous.
