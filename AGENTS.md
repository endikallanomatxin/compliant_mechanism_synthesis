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
- A single differentiable spring-network FEM evaluator in PyTorch is used throughout the project.
- Occupancy values in `[0, 1]` are an internal relaxed representation; training uses the differentiable FEM directly on relaxed occupancies, while candidate selection and final reporting can still threshold at `0.5` when a hard binary design is needed.
- Training starts from random noise and refines it over multiple learned rollout steps, optimized end-to-end with differentiable FEM property loss plus lightweight topology regularization.
- Each target is trained from multiple noise initializations, aggregated with a softmin objective to encourage exploration instead of collapsing to one easy family of solutions.
- Training also uses a monotonic-improvement penalty so later rollout steps are encouraged to improve mechanical error instead of deferring all progress to the end.
- Training logs and sample images are written to TensorBoard.
