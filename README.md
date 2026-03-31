# Compliant mechanism synthesis

Objective: Synthesize compliant mechanisms that achieve:
- Target rigidities between input and output ports.
- Desired motion paths
- Specified force transmission characteristics

## First approach plan

Initial setup (the simplest):

- 2D (0,0) to (GRID_SIZE,GRID_SIZE) grid of pixels
- Bottom plate is fixed, top plate is free to move
- Target is: top plate has to have a specified k_x, k_y and k_theta

The media:
- 2D grid of pixels, each pixel is either solid or void
- Solid pixels are modeled with FEA. (Steel's mechanical properties, for now)

The model:
- 4x4 patches projected to an embedding space of d_model.
- Diffusion transformer encoder predicts the deltas to create a new design that
more better matches the target k_x, k_y and k_theta.

Training:
- Typical diffusion style training, with a noise scheduler and a loss that
compares the predicted design's k_x, k_y and k_theta to the target.
- Additional loss for minimizing solid's surface area, to encourage simpler
designs. Simply summing up all the interface edges between solid and void
pixels should work.
- Also, conectivity, if the resulting design is not connected, we add a penalty
to the loss.

## Prototype implemented here

This repository now includes a first runnable prototype built around the same
core idea, using a lightweight spring-network FEM evaluator instead of the
earlier heuristic mechanics surrogate.

What the prototype does:

- Generates connected binary 2D designs on a larger square grid.
- Represents occupancies internally in continuous `[0, 1]` space during denoising.
- Computes FEM-based `k_x`, `k_y`, and `k_theta` values from the topology.
- Trains a patch-based transformer denoiser with an elite-search training loop conditioned on target properties.
- Logs losses and generated samples to TensorBoard.
- Samples a design for a requested target triple using FEM scoring and local search.
- Uses 4-neighbor connectivity for topology validity, while the FEM adds weak diagonal springs to reduce grid anisotropy.

This is meant to validate the end-to-end workflow before investing in a more
physical continuum solver and a more faithful diffusion process.

## Occupancy Semantics

The current repository treats occupancies in two different roles on purpose:

- During generation and denoising, the design lives in continuous `[0, 1]` space.
- Values between `0` and `1` are not interpreted as partial physical material.
- The official manufacturable design is the hard-thresholded topology `x >= 0.5`.
- All official FEM property evaluation uses that thresholded binary design.
- Topological connectivity is still judged with 4-neighbor adjacency; diagonal contact alone does not count as a valid bridge.

This lets the model reason in a smooth space internally while still forcing a
clear binary decision whenever we ask the mechanics model for `k_x`, `k_y`, and
`k_theta`.

## Training Loop

The current training loop is closer to policy improvement than plain supervised
reconstruction.

For each batch of target properties:

- Sample candidate designs from the current model.
- Sample additional random structural candidates for exploration.
- Threshold every candidate at `0.5`.
- Evaluate all candidates with the spring-network FEM.
- Keep the best candidate per target as an elite.
- Train the denoiser to reconstruct those elites from noisy continuous inputs.

This means exploration comes from both the model sampler and the random
candidate pool, while learning happens by imitating the best FEM-scored designs
found so far.

During training, the prototype can also run periodic canonical evaluations for
representative in-distribution targets derived from a larger synthetic
reference pool. Those snapshots are written to TensorBoard so you can visually
track whether the model is learning distinct families of mechanisms instead of
collapsing to trivial patches.

## Setup

```bash
uv sync
```

## Train

```bash
uv run cms-train --epochs 8 --dataset-size 256 --batch-size 16 \
  --train-model-candidates 2 --train-random-candidates 6 \
  --train-sample-steps 6 --log-every-steps 5 \
  --canonical-eval-every-steps 20 \
  --name prototype
```

This training loop now also includes a binarization penalty so the continuous
occupancies are pushed away from ambiguous gray values.

The default surface regularization is intentionally light so the search loop is
less tempted to collapse into nearly empty or nearly full patches.

The default connectivity regularization is also intentionally modest. It still
discourages disconnected structures, but it should not dominate the search so
hard that the model prefers trivial fully-solid-like solutions.

To reduce lattice anisotropy without letting corner-touching pixels count as a
full structural connection, the spring-network FEM uses weak diagonal springs in
addition to the main axial ones.

The `dataset-size` here is the size of the synthetic target-property pool used
to define which stiffness triplets the search loop practices against.

Canonical evaluations are controlled with:

```bash
--canonical-eval-every-steps 20
--canonical-model-candidates 2
--canonical-random-candidates 4
--canonical-sample-steps 6
```

The current code computes a single `low=q20` and a single `high=q80` over the
positive property values in the reference pool, then evaluates these three
canonical patterns:

- low `k_x`, higher `k_y` and `k_theta`
- higher `k_x`, low `k_y`, higher `k_theta`
- higher `k_x`, higher `k_y`, low `k_theta`

Those runs are logged under TensorBoard tags like:

```text
canonical/low-kx_high-ky-ktheta/design
canonical/high-kx_low-ky-high-ktheta/design
canonical/high-kx-ky_low-ktheta/design
```

## Sample

```bash
uv run cms-sample --checkpoint-path artifacts/prototype.pt \
  --target-kx 0.20 --target-ky 0.28 --target-ktheta 0.18 \
  --name sample \
  --log-every-steps 2
```

Both commands print periodic progress to stdout so long searches are visible
while they run.

## TensorBoard

```bash
uv run tensorboard --logdir runs
```

Artifacts are written under `artifacts/` and logs under `runs/`.
Each training or sampling run uses `--name` and automatically creates a
timestamped directory like `runs/20260330-154500-prototype`, so runs stay
ordered lexicographically without needing to pass a path-like log name.

## Sampling Budget

The sampler now uses a bounded FEM-guided search by default so it does not sit
for a long time looking stuck.

All candidate evaluation in sampling is done after thresholding at `0.5`.

If you want faster sampling, reduce the search budget:

```bash
uv run cms-sample --checkpoint-path artifacts/prototype.pt \
  --target-kx 0.20 --target-ky 0.28 --target-ktheta 0.18 \
  --model-candidates 2 --random-candidates 6 \
  --search-iterations 6 --proposal-count 8 \
  --name fast-sample \
  --log-every-steps 2
```

## Verified Commands

These are the exact commands used to verify the first end-to-end prototype in this repository.

Install dependencies:

```bash
uv sync
```

Run a verified training pass:

```bash
uv run cms-train --epochs 3 --dataset-size 96 --batch-size 8 \
  --train-model-candidates 2 --train-random-candidates 4 \
  --train-sample-steps 4 --log-every-steps 2 \
  --canonical-eval-every-steps 4 \
  --name fem-verify-train \
  --checkpoint-path artifacts/fem-verify.pt
```

Run a verified sampling pass:

```bash
uv run cms-sample --checkpoint-path artifacts/fem-verify.pt \
  --target-kx 0.20 --target-ky 0.28 --target-ktheta 0.18 \
  --model-candidates 2 --random-candidates 6 \
  --search-iterations 6 --proposal-count 8 --log-every-steps 2 \
  --name fem-verify-sample \
  --output-path artifacts/fem-verify-sample.pt
```

Open TensorBoard for all runs:

```bash
uv run tensorboard --logdir runs
```

The verified sampling run produced:

```text
achieved_properties depends on the current checkpoint and search budget.
```

The current prototype validates the search-and-imitation loop end to end, but
it is still mechanically weak. Better targets usually require either more
training epochs, a larger target pool, or a larger search budget.
