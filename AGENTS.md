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

- Binary 2D topology grid with fixed bottom plate and movable top plate.
- Patch-based transformer denoiser conditioned on target `k_x`, `k_y`, `k_theta`.
- Spring-network FEM evaluator for topology stiffness instead of the earlier heuristic surrogate.
- Occupancy values in `[0, 1]` are an internal relaxed representation; all official mechanical evaluation uses a hard threshold at `0.5`.
- Training uses search-and-imitation: sample candidates, evaluate them with FEM, keep elites, and train the denoiser on those elites.
- Training logs and sample images are written to TensorBoard.
