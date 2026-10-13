# Project instructions

- Use `uv` for Python environments and commands.
- Keep application Python under `src/compliant_mechanism_synthesis/`.
- Define CLI defaults in config dataclasses; parsers read those defaults.
- Keep computation float64 and differentiable in PyTorch, with CPU and Intel XPU support. Vectorize beam kernels and avoid device transfers in the mechanics hot path.
- Use TensorBoard for optimization and mechanics diagnostics in `runs/`.
- Designs are point-and-beam graphs in 2D or 3D, with fixed, mobile and free roles.
- Clamp fixed nodes exactly and tie mobile nodes to one rigid body.
- Nonlinear equilibrium continuation and the full deformation path are central to analysis and optimization.
- Use direct gradient descent and evolutionary search; do not introduce learned models.
- Keep UI and rendering in web technologies, independent of the mechanics kernel.
- Prefer incremental changes and analytical or finite-difference mechanics checks.
- Do not introduce compatibility layers or migration shims.
