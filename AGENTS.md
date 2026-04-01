## Project Notes

- Use `uv` for all Python environment and run commands.
- Keep Python code under `src/compliant_mechanism_synthesis/`.
- Use TensorBoard for training and sampling logs. Default log directory: `runs/`.
- Preferred entrypoints:
  - `uv run cms-train ...`
  - `uv run cms-sample ...`
- The current prototype is intentionally lightweight and CPU-friendly.
- When extending the project, prefer small incremental changes over broad rewrites.
- This project is green-field: do not add backward-compatibility code, fallback paths, or migration shims unless explicitly requested.

## Prototype Scope

- Designs are point-and-beam graphs in 2D, not pixel grids.
- A design state consists of node positions, node roles, and a symmetric beam-activation matrix.
- Roles are `fixed`, `mobile`, and `free`.
- The mechanics model is a differentiable 2D Euler-Bernoulli frame FEM with axial plus bending stiffness.
- The 2 fixed nodes are rigidly clamped.
- The 2 mobile nodes are tied to a rigid body with generalized DOFs `Ux`, `Uy`, and `Theta`.
- The learning model is a graph-based iterative refinement model.
- Each refinement step predicts free-node displacement deltas and connectivity updates.
- Connectivity updates must come only from dot products between per-node latent vectors.
- Attention layers alternate between connectivity-conditioned and unconditioned self-attention.
- Training uses noisy positions and noisy connectivity, plus property, reconstruction, material, binarization, and connectivity losses.
- Training logs and sample figures are written to TensorBoard.
