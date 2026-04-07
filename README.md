# Compliant Mechanism Synthesis

This repository is now centered on the first two steps of the new training
strategy:

1. generate good 3D offline examples with an explicit optimizer,
2. use those examples as the substrate for the later supervised stage.

The old RL-first 2D prototype has been removed on purpose.

## Current Status

Implemented:

- 3D point-and-beam graph representation
- differentiable 3D space-frame FEM in PyTorch
- rigid clamping for the fixed anchor group
- rigid-body reduction for the mobile anchor group
- library of structured 3D starting-point primitives
- explicit per-case optimizer over free-node positions and connectivity
- offline dataset generation with TensorBoard logs
- single-case sampling and visualization

Deferred on purpose:

- supervised graph-refinement model
- online refinement / RL
- richer conditioning such as style tokens

## Design State

Each design contains:

- node positions `X in R^(N x 3)`
- node roles `R in {fixed, mobile, free}^N`
- symmetric beam-activation matrix `A in [0, 1]^(N x N)`

The current 3D setup uses:

- 3 fixed anchor nodes
- 3 mobile anchor nodes
- a variable number of free nodes

The anchor triplets are the minimum clean representation for a rigid clamp and a
rigid 3D output body without inventing extra shims.

## Mechanics

The mechanics backend is a differentiable 3D space-frame FEM.

- each node has 6 DOFs: translation plus rotation
- each active edge is a circular beam with axial, torsional, and bending stiffness
- fixed anchors are clamped
- mobile anchors are condensed into a 6-DOF rigid body
- free-node DOFs are statically condensed to obtain the effective output stiffness

The main mechanical target is the `6x6` generalized stiffness matrix of the
mobile rigid body.

## Offline Dataset Pipeline

The current workflow is:

1. sample a structured 3D starting point from the primitive library,
2. evaluate its current effective stiffness,
3. perturb that stiffness into a nearby SPD target,
4. optimize free-node positions and connectivity directly through the FEM,
5. save both the initial and optimized designs as offline training data.

Current starting-point families:

- `straight_lattice_sheet`
- `curved_lattice_sheet`
- `helix_lattice_sheet`
- `straight_beam`
- `curved_beam`
- `path_truss`
- `loose_cloud`

The optimizer logs to TensorBoard from the start so the quality of the generated
examples can be inspected before any supervised model is trained.

## Commands

Install dependencies:

```bash
uv sync
```

Generate an offline dataset:

```bash
uv run cms-dataset-generate \
  --num-cases 32 \
  --num-free-nodes 18 \
  --optimization-steps 120 \
  --logdir datasets
```

By default this also writes the `.pt`, previews and `summary.txt` under the
run directory (something like `datasets/20260406T180000-supervised/`).

Render previews again from an existing dataset:

```bash
uv run cms-dataset-visualize \
  --dataset-path artifacts/dataset.pt \
  --output-dir artifacts/dataset_preview
```

Inspect a single optimized case via `cms-dataset-generate`:

```bash
uv run cms-dataset-generate \
  --just-check-sample \
  --sample-primitive curved_lattice_sheet \
  --sample-num-free-nodes 18 \
  --sample-optimization-steps 120 \
  --sample-output-dir artifacts/sample_case
```

The check writes:

- `initial.png`
- `optimized.png`
- TensorBoard logs under `output-dir/tensorboard`

Open TensorBoard:

```bash
uv run tensorboard --logdir runs
```

## Tests

```bash
uv run pytest
```
