# Reusable equilibrium implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Track each step below.

**Goal:** Reduce repeated nonlinear mechanics work while preserving the selected stable 3D path and its first design derivative.

**Architecture:** Introduce a device-local scaled factorization object, a custom adjoint for exact equilibrium states, and explicit parent path hints distinct from exact caches. Keep the current population, objective, beam model and path resolution for comparisons.

**Tech Stack:** Python via uv, PyTorch float64 CPU/Intel XPU, TensorBoard, existing web UI.

**Spec:** ../specs/2026-10-14-reusable-equilibrium-design.md

## Global constraints

Exact fixed clamps and independent active/passive rigid ties. Full nonlinear path, including preloading. Binary topology through evolution only. No learned models, compatibility shims, device transfers in the hot path, or implicit CPU fallback. New commits, no amends. Execution in this existing feature branch as explicitly requested by the user.

## Review focus

1. Unstable/singular/nonfinite systems must remain rejected.
2. Active support loss requires stiffness derivatives in addition to pose derivatives.
3. Private beam coordinates contribute to collision and length gradients.
4. Parent hints cannot bypass Newton or silently select a different stable branch.
5. Contexts must not retain graphs across cycles or unbounded factorization caches.

## Task 1: Scaled systems and native stability

Files: new `src/compliant_mechanism_synthesis/equilibrium_system.py`; modify `spectral.py`, `prescribed_motion.py`, `condensed_mechanics.py`; new `tests/test_equilibrium_system.py`.
Interface: `coordinate_scale(like, dof, length)`, `factor_system(matrix, scale)` returning detached solver context with `solve(rhs)`, validity mask, CPU native Cholesky / XPU LU with existing device-local stability test. `scaled_descent(h,g,dof,length,regularization)` returns physical correction. Both vector and multiple RHS supported.

- [x] Add independent scaled-solve equivalence, native CPU positivity, singular/nonfinite rejection and CPU/XPU tests; run and observe failure.
- [x] Implement scaling/factorization and integrate into Newton and final sensitivity solves without weakening physical residual tolerance.
- [x] Verify rest conditioning improvement, exact result, input derivatives and mechanical checks.

## Task 2: Explicit equilibrium adjoint

Files: new `implicit_equilibrium.py`; modify `prescribed_motion.py`, `condensed_mechanics.py`; extend `tests/test_condensed_integration.py` and new `tests/test_implicit_equilibrium.py`.
Interface: equilibrium attachment receives positions, radii, prescribed body, detached converged graph/private states and final factorization. Outputs full q and private state carrying a first-derivative custom backward. Backward uses saved factorization, contracts residual derivatives through autograd, and includes private load condensation/recovery. Final stiffness evaluated on attached state only if its derivative is required.

- [x] Add finite differences of translation/orientation and private-node objective, active input sensitivity, active/inactive support, CPU/XPU and no repeated tangent during pose-only backward tests; observe failure before implementation.
- [x] Implement exact adjoint; factorization context lasts only for backward, no persistent autograd graphs.
- [x] Compare cached/fresh and condensed/uncondensed gradients; retain independent planar verification behavior.

## Task 3: Explicit parent hints and branch guards

Files: new `path_hint.py`; modify `optimization.py`, `prescribed_motion.py`; new `tests/test_path_hint.py`.
Interface: parent records passed as `initial_path` separately from exact `cached`. Same topology/roles/mesh/input only. Hints provide guesses at common progress, private displacement relative to new reference, and conservative distance bounds. Continue from the child's previous state; use parent guess only when close to its predictor, otherwise discard. Verify against cold continuation before enabling by default; ambiguous cases fall back.

- [x] Test hints still run Newton, incompatible/large changes rejected, preloading retained, topology changes cold, exact cache distinct and path/gradient agreement.
- [x] Propagate hints from gradient parents and focused children without global nearest-cache search. Cache bound by live population/proposals.
- [x] Record hint acceptance/fallback and sensitivity/factorization work in TensorBoard and existing diagnostics.

## Task 4: Verification, measurements and commit

Files: `PERFORMANCE.md`, `README.md`, benchmark under `src/compliant_mechanism_synthesis/` if needed.
- [x] Run meaningful CPU/XPU mechanics tests, entire suite and existing browser checks.
- [x] Compare identical persisted populations: warmed medians (3 repeats), objective and gradients, evaluation/gradient/full cycle separately; report memory and convergence, not kernel speedups as global speedups.
- [x] Review spec coverage and diff; document any measured limitation. Commit new implementation with tests, docs and completed plan.

## Execution ledger

User explicitly approved implementation in this session; implementation method is native, no agents. Plan review performed against the approved specification. Conservative branch behavior takes precedence over enabling hints by default. Higher-order optimizer derivatives are outside scope; stiffness loss still requires differentiating its explicit tangent through the first-derivative equilibrium attachment.


## Final rulings and verification

- Preserved the existing physical residual acceptance criterion while scaling global/adjoint coordinates; this avoids weakening or silently changing the reference accuracy. Private Schur blocks remain exact in physical coordinates and their adjoint solves use scaled coordinates.
- Parent continuation across large design changes is conservatively rejected, as allowed by the spec. Changed topology uses cold continuation. Local predictor/positivity guards are not a global multistability certificate.
- Explicit adjoint first derivatives include private-coordinate objectives and prescribed-input sensitivity. Active stiffness costs retain explicit tangent differentiation. Higher-order optimizer derivatives are not exposed.
- Added a 256 MiB factorization budget and exact-cache replay after initial measurements showed that forward/backward reuse alone left refactorization work between evaluations.
- Verification: whole mechanics/search suite, finite differences on CPU/XPU, exact cached/fresh gradients, direct private-coordinate gradients, memory-budget eviction, signed/preloaded hints and browser controls. Measurements and limits recorded in PERFORMANCE.md.
- Existing workspace retained as requested; user explicitly authorized implementation and new commits. No subagents, worktree relocation, merge or push.
