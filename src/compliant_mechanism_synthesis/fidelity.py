"""Spatial accuracy checks on common prescribed poses, outside optimization AD."""

import torch

from .nonlinear import rotation, rotation_vector


def diverse_candidates(search, keys):
    # A small viable population can contain cloned survivors; solve each once.
    unique = list(dict.fromkeys(keys))
    candidates = [keys.index(key) for key in unique]
    count = min(search.config.fidelity_check_candidates, len(candidates))
    similarity = search.similarity(search.population).detach()
    available = torch.zeros(
        search.population.size, dtype=torch.bool, device=search.device
    )
    available[candidates] = True
    selected = [candidates[0]]
    available[selected[0]] = False
    for _ in range(1, count):
        closeness = similarity[:, selected].amax(-1)
        closeness[~available] = float("inf")
        selected.append(int(closeness.argmin()))
        available[selected[-1]] = False
    return selected


def mesh_pose_error(records, states, index):
    """Compare only identical inputs, separating mesh error from time sampling."""
    reference = {round(record[0], 10): record[1][-12:-6] for record in records}
    coarse, fine = [], []
    for state in states:
        if not bool(state.accepted[index]):
            continue
        pose = reference.get(round(float(state.factors[index]), 10))
        if pose is not None:
            coarse.append(pose)
            fine.append(state.q[index, -12:-6].detach())
    if len(coarse) < 2:
        return None
    fine = torch.stack(fine)
    coarse = torch.stack(coarse).to(fine)
    position = (fine[:, :3] - coarse[:, :3]).norm(dim=-1).max()
    angles = rotation_vector(
        rotation(coarse[:, 3:]) @ rotation(fine[:, 3:]).transpose(-1, -2)
    )
    orientation = angles.norm(dim=-1).max()
    return float(position), float(orientation), len(coarse)
