# Compliant Mechanism Synthesis

This repository now implements a first point-and-beam graph prototype for 2D
compliant mechanism synthesis.

## Core Representation

A design is represented by:

- node positions `X ∈ R^(N x 2)` in normalized `[0,1] x [0,1]`
- node roles `R ∈ {fixed, mobile, free}^N`
- symmetric connectivity / thickness activations `A ∈ [0,1]^(N x N)`

The diagonal of `A` is always zero. Beam radius is derived from `A`.

The normalized design domain maps to a physical `0.2 m x 0.2 m` workspace.
By default, the preferred fabricable beam diameters span `0.2 mm` to `2.0 mm`,
while the internal representation can explore somewhat thicker bars and penalize
them back down.

For model-facing features, positions are re-centered to a symmetric domain
`[-1, 1] x [-1, 1]`, where `0` is the workspace center.

## Mechanics

The mechanics module uses a differentiable 2D Euler-Bernoulli frame FEM in
PyTorch.

- each node has DOFs `(ux, uy, theta)`
- each edge is a beam element with axial and bending stiffness
- the 2 fixed nodes are rigidly clamped
- the 2 mobile nodes are tied to a rigid body with `(Ux, Uy, Theta)`
- effective response is computed from three unit load cases:
  - horizontal force
  - vertical force
  - moment

The target mechanical objective is the full `3x3` generalized response matrix of
the mobile rigid body, including coupling terms between force and moment.

## Learning Model

The model is a graph-based iterative refinement network.

- each node is a token
- node tokens use position features and learned role embeddings
- target response matrix and timestep are injected globally
- attention layers alternate between:
  - connectivity-conditioned self-attention
  - unconditioned self-attention

The model predicts:

- per-node displacement deltas `(dx, dy)`
- per-node latent vectors `u_i`

Connectivity updates are produced only from dot products between these latent
vectors.

The mechanics module reports both the raw mobile-body response matrix and the
derived effective stiffness matrix.

The FEM stays in SI units internally. The learning code uses an explicit
adimensionalization layer built from characteristic scales:

- characteristic length `L_c = 0.5 * workspace_size = 0.1 m`
- characteristic force `F_c = E * pi * r_max^2`
- characteristic moment `M_c = F_c * L_c`
- characteristic stress `sigma_c = yield_stress`

Generalized stiffness, generalized response, nodal translations, nodal stresses,
and material usage are normalized through these scales before they are fed to
the model or used in the main mechanical loss terms.

## Training Loop

For each batch:

- sample an initial graph state from pure noise
- sample a target generalized stiffness matrix from a fixed normalized target
  library and map it back to SI for evaluation
- refine the noisy graph over several learned rollout steps
- inject additional Gaussian noise into positions and connectivity at each
  rollout step during training
- apply position updates only to free nodes
- update connectivity directly in activation space through node-latent dot products
- evaluate the differentiable frame FEM on intermediate and final states

The loss includes:

- target response-matrix loss on the unique symmetric terms of the `3x3` matrix,
  including couplings without double-counting mirrored entries
- monotonic-improvement loss so intermediate rollout steps keep reducing the
  mechanical target error instead of regressing
- beam material penalty
- explicit connectivity sparsity penalty on the average continuous edge activation
- connectivity penalty
- soft beam-length regularization for bars that are too short or too long
- soft diameter regularization for bars that are too thin or too thick
- soft free-node spacing regularization to avoid node collapse
- soft quadratic domain regularization outside the true workspace bounds for
  free nodes

The default geometric regularization thresholds are expressed in physical units:

- beam length between `1 mm` and `20 mm`
- preferred beam diameter between `0.2 mm` and `2.0 mm`

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

- `num_nodes=32`
- `batch_size=256`
- `train_steps=2000` with pure-noise states generated on demand
- `d_model=256`
- `num_layers=6`
- iterative rollout refinement
- pure-noise initial states with no reference-design reconstruction loss
- canonical TensorBoard evaluations during training
- explicit SI-to-model characteristic scaling for mechanics features and losses
- configurable geometric regularization thresholds and weights for beam length
  and diameter

## Sample

```bash
uv run cms-sample \
  --checkpoint-path artifacts/prototype.pt \
  --target-response "2.0e-3,1.0e-4,0.0,1.0e-4,4.0e-3,2.0e-4,0.0,2.0e-4,5.0e-4" \
  --name sample
```

Sampling outputs:

- final node positions
- node roles
- final continuous connectivity matrix
- thresholded sparse graph for visualization
- evaluated response and stiffness matrices

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
