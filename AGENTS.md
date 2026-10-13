# Project instructions

- Use `uv` for Python environments and commands.
- Keep application Python under `src/compliant_mechanism_synthesis/`.
- Define CLI defaults in config dataclasses; parsers read those defaults.
- Keep computation float64 and differentiable in PyTorch, with CPU and Intel XPU support. Vectorize beam kernels and avoid device transfers in the mechanics hot path.
- Use TensorBoard for optimization and mechanics diagnostics in `runs/`.
- Designs are point-and-beam graphs in 3D, with fixed, active, passive and free roles. Active and passive nodes define independent rigid bodies.
- Clamp fixed nodes exactly and tie active/passive nodes to their respective rigid bodies. Prescribe the active pose and solve nonlinear equilibrium for passive and free degrees of freedom.
- Nonlinear equilibrium continuation and the full deformation path are central to analysis and optimization.
- Use direct gradient descent and evolutionary search; do not introduce learned models.
- Keep UI and rendering in web technologies, independent of the mechanics kernel.
- Prefer incremental changes and analytical or finite-difference mechanics checks. Planar kernels may remain as independent verification tools; the application and synthesis API expose 3D only.
- Do not introduce compatibility layers or migration shims.
