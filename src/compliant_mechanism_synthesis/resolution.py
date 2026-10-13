"""Explicit, isolated refinement check for the selected deformation path."""

import time
from dataclasses import replace

import torch

from .prescribed_motion import trace_motion


def interpolate(factors, poses, samples):
    """Piecewise linear interpolation of computed equilibria, without extrapolation."""
    if len(factors) < 2 or samples.min() < factors[0] or samples.max() > factors[-1]:
        raise ValueError("A complete path is required for interpolation")
    i = (
        torch.searchsorted(factors, samples, right=True)
        .sub(1)
        .clamp(0, len(factors) - 2)
    )
    t = (samples - factors[i]) / (factors[i + 1] - factors[i])
    return torch.lerp(poses[i], poses[i + 1], t[:, None])


def compare_resolution(search, index=0):
    search.ensure_evaluated()
    if not 0 <= index < search.population.size:
        raise ValueError("Invalid population index")
    design = search.design(index)
    entry = search.path_cache.get(search._path_key(design))
    if entry is None or not entry[0]:
        raise ValueError("Select a structure with a complete converged path")
    started = time.perf_counter()
    config = replace(search.continuation, steps=search.continuation.steps * 2)
    batch = search.batch_design(
        search.population.take(slice(index, index + 1))
    ).detached()
    fine = trace_motion(batch, search.input_trajectory, config)
    base = {
        "index": index,
        "coarse_steps": search.continuation.steps,
        "fine_steps": config.steps,
        "coarse_states": len(entry[1]),
        "fine_states": sum(bool(s.accepted[0]) for s in fine.states),
        "converged": bool(fine.completed[0]),
        "seconds": time.perf_counter() - started,
    }
    if not base["converged"]:
        return base
    dof, dim = len(search.target), search.config.dimension
    factors = search.target.new_tensor([r[0] for r in entry[1]])
    poses = torch.stack([r[1][-2 * dof : -dof] for r in entry[1]]).to(search.device)
    states = [s for s in fine.states if bool(s.accepted[0])]
    refined_factors = torch.stack([s.factors[0] for s in states])
    refined_poses = torch.stack([s.q[0, -2 * dof : -dof] for s in states]).detach()
    samples = torch.unique(
        torch.cat(
            (
                torch.linspace(0, 1, 101, dtype=poses.dtype, device=poses.device),
                factors,
                refined_factors,
                search.trajectory.factors,
            )
        )
    )
    coarse = interpolate(factors, poses, samples)
    refined = interpolate(refined_factors, refined_poses, samples)
    delta = refined - coarse
    position = delta[:, :dim].norm(dim=-1) * 1000
    orientation = delta[:, dim:].norm(dim=-1) * (180 / torch.pi)
    target = search.target_at(samples)
    coarse_loss = float(torch.trapezoid(search.pose_loss(coarse, target), samples))
    fine_loss = float(torch.trapezoid(search.pose_loss(refined, target), samples))
    result = base | {
        "position_max_mm": float(position.max()),
        "orientation_max_deg": float(orientation.max()),
        "weighted_rms_difference": float(
            torch.trapezoid(
                search.pose_loss(delta, torch.zeros_like(delta)), samples
            ).sqrt()
        ),
        "coarse_tracking_cost": coarse_loss,
        "fine_tracking_cost": fine_loss,
        "cost_change_percent": 100
        * (fine_loss - coarse_loss)
        / max(abs(coarse_loss), 1e-12),
        "samples": samples.cpu().tolist(),
        "coarse": coarse.cpu().tolist(),
        "fine": refined.cpu().tolist(),
        "position_error_mm": position.cpu().tolist(),
        "orientation_error_deg": orientation.cpu().tolist(),
    }
    for name in (
        "position_max_mm",
        "orientation_max_deg",
        "weighted_rms_difference",
        "cost_change_percent",
    ):
        search.writer.add_scalar("resolution/" + name, result[name], search.iteration)
    search.writer.flush()
    return result
