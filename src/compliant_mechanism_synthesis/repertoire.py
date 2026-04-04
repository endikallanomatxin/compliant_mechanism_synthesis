from __future__ import annotations

from dataclasses import dataclass

import torch

from compliant_mechanism_synthesis.common import unique_values_to_symmetric_matrix
from compliant_mechanism_synthesis.mechanics import FrameFEMConfig, characteristic_scales
from compliant_mechanism_synthesis.scaling import (
    denormalize_generalized_stiffness_matrix,
    normalize_generalized_stiffness_matrix,
)


COMPONENT_LABELS = [
    "ux_ux",
    "ux_uy",
    "ux_theta",
    "uy_uy",
    "uy_theta",
    "theta_theta",
]

SOURCE_RANDOM_INITIALIZATION = 0
SOURCE_RL = 1


def _upper_triangular_components(matrices: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            matrices[:, 0, 0],
            matrices[:, 0, 1],
            matrices[:, 0, 2],
            matrices[:, 1, 1],
            matrices[:, 1, 2],
            matrices[:, 2, 2],
        ],
        dim=1,
    )


def _source_mask(
    source: torch.Tensor,
    source_codes: tuple[int, ...] | None,
) -> torch.Tensor:
    if source_codes is None:
        return torch.ones_like(source, dtype=torch.bool)
    source_codes_tensor = torch.tensor(source_codes, dtype=source.dtype)
    return (source[:, None] == source_codes_tensor[None, :]).any(dim=1)


def _project_positive_definite(matrices: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    symmetric = 0.5 * (matrices + matrices.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    clipped = eigenvalues.clamp_min(eps)
    projected = eigenvectors @ torch.diag_embed(clipped) @ eigenvectors.transpose(1, 2)
    eye = torch.eye(projected.shape[-1], device=projected.device, dtype=projected.dtype)
    projected = 0.5 * (projected + projected.transpose(1, 2))
    projected_eigenvalues = torch.linalg.eigvalsh(projected)
    min_eigenvalue = projected_eigenvalues.amin(dim=1).clamp_max(0.0).abs()
    return projected + (min_eigenvalue + eps).view(-1, 1, 1) * eye.unsqueeze(0)


def _rarity_weights(components: torch.Tensor, k: int = 8) -> torch.Tensor:
    if components.shape[0] <= 1:
        return torch.ones((components.shape[0],), dtype=components.dtype)
    k = min(max(k, 1), components.shape[0] - 1)
    distances = torch.cdist(components, components)
    inf = torch.tensor(float("inf"), dtype=distances.dtype, device=distances.device)
    distances = distances + torch.diag_embed(torch.full((components.shape[0],), inf, device=distances.device, dtype=distances.dtype))
    neighbor_distances = distances.topk(k=k, largest=False).values
    density_proxy = neighbor_distances[:, -1].clamp_min(1e-6)
    weights = density_proxy / density_proxy.mean().clamp_min(1e-6)
    return weights


def _select_best_index(
    candidate_mask: torch.Tensor,
    component_sum: torch.Tensor,
    used_indices: set[int],
) -> int | None:
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return None
    available = [int(index.item()) for index in candidate_indices if int(index.item()) not in used_indices]
    if available:
        available_indices = torch.tensor(
            available,
            device=component_sum.device,
            dtype=torch.long,
        )
        scores = component_sum.index_select(0, available_indices)
        return int(available_indices[torch.argmax(scores)].item())
    scores = component_sum.index_select(0, candidate_indices)
    return int(candidate_indices[torch.argmax(scores)].item())


def _select_nearest_zero_index(
    absolute_component: torch.Tensor,
    component_sum: torch.Tensor,
    used_indices: set[int],
) -> int | None:
    sorted_indices = torch.argsort(absolute_component, stable=True)
    available = [
        int(index.item())
        for index in sorted_indices
        if int(index.item()) not in used_indices
    ]
    if available:
        available_indices = torch.tensor(
            available,
            device=component_sum.device,
            dtype=torch.long,
        )
        absolute_values = absolute_component.index_select(0, available_indices)
        minimum = absolute_values.amin()
        tie_mask = absolute_values <= (minimum + 1e-9)
        tie_indices = available_indices[tie_mask]
        tie_scores = component_sum.index_select(0, tie_indices)
        return int(tie_indices[torch.argmax(tie_scores)].item())
    if sorted_indices.numel() == 0:
        return None
    absolute_values = absolute_component.index_select(0, sorted_indices)
    minimum = absolute_values.amin()
    tie_mask = absolute_values <= (minimum + 1e-9)
    tie_indices = sorted_indices[tie_mask]
    tie_scores = component_sum.index_select(0, tie_indices)
    return int(tie_indices[torch.argmax(tie_scores)].item())


@dataclass
class SimulationRepertoire:
    positions: torch.Tensor
    roles: torch.Tensor
    adjacency: torch.Tensor
    stiffness: torch.Tensor
    source: torch.Tensor
    max_cases: int

    @classmethod
    def empty(cls, num_nodes: int, max_cases: int) -> SimulationRepertoire:
        return cls(
            positions=torch.empty((0, num_nodes, 2), dtype=torch.float32),
            roles=torch.empty((0, num_nodes), dtype=torch.long),
            adjacency=torch.empty((0, num_nodes, num_nodes), dtype=torch.float32),
            stiffness=torch.empty((0, 3, 3), dtype=torch.float32),
            source=torch.empty((0,), dtype=torch.long),
            max_cases=max_cases,
        )

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def add(
        self,
        positions: torch.Tensor,
        roles: torch.Tensor,
        adjacency: torch.Tensor,
        stiffness: torch.Tensor,
        source_code: int,
    ) -> None:
        finite_mask = (
            torch.isfinite(positions).all(dim=(1, 2))
            & torch.isfinite(adjacency).all(dim=(1, 2))
            & torch.isfinite(stiffness).all(dim=(1, 2))
        )
        if not finite_mask.any():
            return
        positions = positions[finite_mask]
        roles = roles[finite_mask]
        adjacency = adjacency[finite_mask]
        stiffness = stiffness[finite_mask]
        batch_size = positions.shape[0]
        source = torch.full((batch_size,), source_code, dtype=torch.long)
        self.positions = torch.cat([self.positions, positions.detach().cpu()], dim=0)
        self.roles = torch.cat([self.roles, roles.detach().cpu()], dim=0)
        self.adjacency = torch.cat([self.adjacency, adjacency.detach().cpu()], dim=0)
        self.stiffness = torch.cat([self.stiffness, stiffness.detach().cpu()], dim=0)
        self.source = torch.cat([self.source, source], dim=0)
        if len(self) > self.max_cases:
            keep = slice(len(self) - self.max_cases, len(self))
            self.positions = self.positions[keep]
            self.roles = self.roles[keep]
            self.adjacency = self.adjacency[keep]
            self.stiffness = self.stiffness[keep]
            self.source = self.source[keep]

    def normalized_components(
        self,
        frame_config: FrameFEMConfig | None = None,
        source_codes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if len(self) == 0:
            return torch.empty((0, 6), dtype=torch.float32)
        source_mask = _source_mask(self.source, source_codes)
        if not source_mask.any():
            return torch.empty((0, 6), dtype=torch.float32)
        normalized = normalize_generalized_stiffness_matrix(
            self.stiffness[source_mask],
            characteristic_scales(frame_config),
        )
        components = _upper_triangular_components(normalized)
        finite_mask = torch.isfinite(components).all(dim=1)
        return components[finite_mask]

    def sample_cases(
        self,
        batch_size: int,
        device: torch.device,
        source_codes: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise RuntimeError("cannot sample from an empty simulation repertoire")
        candidate_mask = _source_mask(self.source, source_codes)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            candidate_indices = torch.arange(len(self))
        components = self.normalized_components(
            FrameFEMConfig(),
            source_codes=source_codes,
        )
        if components.shape[0] == candidate_indices.shape[0] and components.shape[0] > 1:
            weights = _rarity_weights(components)
            selection = torch.multinomial(weights, batch_size, replacement=True)
        else:
            selection = torch.randint(candidate_indices.shape[0], (batch_size,))
        indices = candidate_indices.index_select(0, selection)
        return (
            self.positions.index_select(0, indices).to(device),
            self.roles.index_select(0, indices).to(device),
            self.adjacency.index_select(0, indices).to(device),
            self.stiffness.index_select(0, indices).to(device),
        )

    def sample_target_stiffness(
        self,
        batch_size: int,
        device: torch.device,
        frame_config: FrameFEMConfig | None = None,
        source_codes: tuple[int, ...] | None = None,
        target_noise_scale: float = 0.0,
    ) -> torch.Tensor:
        if len(self) == 0:
            raise RuntimeError("cannot sample target stiffness from an empty repertoire")
        finite_stiffness_mask = torch.isfinite(self.stiffness).all(dim=(1, 2))
        finite_stiffness_mask = finite_stiffness_mask & _source_mask(self.source, source_codes)
        finite_stiffness = self.stiffness[finite_stiffness_mask]
        if finite_stiffness.shape[0] == 0:
            return self.sample_target_stiffness(
                batch_size,
                device,
                frame_config,
                source_codes=None,
                target_noise_scale=target_noise_scale,
            )
        if finite_stiffness.shape[0] == 1:
            sampled = finite_stiffness[:1].expand(batch_size, -1, -1).to(device)
            if target_noise_scale <= 0.0:
                return sampled
            components = self.normalized_components(frame_config, source_codes)
            component_std = components.std(dim=0, unbiased=False) if components.shape[0] > 0 else torch.ones((6,), dtype=sampled.dtype)
            normalized_sampled = normalize_generalized_stiffness_matrix(
                sampled,
                characteristic_scales(frame_config),
            )
            sampled_components = _upper_triangular_components(normalized_sampled)
            jitter = torch.randn_like(sampled_components) * (
                target_noise_scale * component_std.to(device=sampled.device, dtype=sampled.dtype)
            )
            return _project_positive_definite(
                denormalize_generalized_stiffness_matrix(
                    unique_values_to_symmetric_matrix(sampled_components + jitter, size=3),
                    characteristic_scales(frame_config),
                )
            )
        components = self.normalized_components(frame_config, source_codes)
        if components.shape[0] < 2 or not torch.isfinite(components).all():
            if finite_stiffness.shape[0] > 1:
                normalized = normalize_generalized_stiffness_matrix(
                    finite_stiffness,
                    characteristic_scales(frame_config),
                )
                fallback_components = _upper_triangular_components(normalized)
                weights = _rarity_weights(fallback_components)
                indices = torch.multinomial(weights, batch_size, replacement=True)
            else:
                indices = torch.randint(finite_stiffness.shape[0], (batch_size,))
            sampled = finite_stiffness.index_select(0, indices).to(device)
            if target_noise_scale <= 0.0:
                return sampled
            components = self.normalized_components(frame_config)
            if components.shape[0] == 0:
                return sampled
            normalized_sampled = normalize_generalized_stiffness_matrix(
                sampled,
                characteristic_scales(frame_config),
            )
            sampled_components = _upper_triangular_components(normalized_sampled)
            component_std = components.std(dim=0, unbiased=False).to(
                device=sampled.device,
                dtype=sampled.dtype,
            )
            jitter = torch.randn_like(sampled_components) * (target_noise_scale * component_std)
            return _project_positive_definite(
                denormalize_generalized_stiffness_matrix(
                    unique_values_to_symmetric_matrix(sampled_components + jitter, size=3),
                    characteristic_scales(frame_config),
                )
            )
        mean = components.mean(dim=0)
        centered = components - mean
        covariance = centered.transpose(0, 1) @ centered / max(components.shape[0] - 1, 1)
        covariance = covariance + 1e-3 * torch.eye(
            covariance.shape[0], dtype=covariance.dtype
        )
        if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            indices = torch.randint(finite_stiffness.shape[0], (batch_size,))
            return finite_stiffness.index_select(0, indices).to(device)
        rarity_weights = _rarity_weights(components).to(device)
        source_indices = torch.multinomial(
            rarity_weights,
            batch_size,
            replacement=True,
        )
        sampled_components = components.to(device).index_select(0, source_indices)
        gaussian = torch.distributions.MultivariateNormal(
            torch.zeros_like(mean).to(device),
            covariance_matrix=covariance.to(device),
        )
        sampled_components = sampled_components + gaussian.sample((batch_size,))
        if target_noise_scale > 0.0:
            component_std = components.std(dim=0, unbiased=False).to(device)
            sampled_components = sampled_components + torch.randn_like(sampled_components) * (
                target_noise_scale * component_std
            )
        normalized_targets = unique_values_to_symmetric_matrix(
            sampled_components,
            size=3,
        )
        raw_targets = denormalize_generalized_stiffness_matrix(
            normalized_targets,
            characteristic_scales(frame_config),
        )
        return _project_positive_definite(raw_targets)

    def canonical_specs(
        self,
        device: torch.device,
        frame_config: FrameFEMConfig | None = None,
        max_specs: int | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        if len(self) == 0:
            return []
        components = self.normalized_components(frame_config)
        component_sum = components.sum(dim=1)
        max_abs = components.abs().amax(dim=0).clamp_min(1e-6)
        specs: list[tuple[str, torch.Tensor]] = []
        used_indices: set[int] = set()
        for dim_idx, label in enumerate(COMPONENT_LABELS):
            mask = components[:, dim_idx].abs() <= 0.1 * max_abs[dim_idx]
            best_index = _select_best_index(mask, component_sum, used_indices)
            if best_index is None:
                absolute_component = components[:, dim_idx].abs()
                best_index = _select_nearest_zero_index(
                    absolute_component,
                    component_sum,
                    used_indices,
                )
            if best_index is None:
                best_index = int(torch.argmax(component_sum).item())
            specs.append(
                (
                    f"{dim_idx + 1:02d}_{label}_near_zero",
                    self.stiffness[best_index].to(device),
                )
            )
            used_indices.add(best_index)
        if max_specs is not None:
            specs = specs[:max_specs]
        return specs

    def payload(self) -> dict[str, torch.Tensor | int]:
        return {
            "positions": self.positions,
            "roles": self.roles,
            "adjacency": self.adjacency,
            "stiffness": self.stiffness,
            "source": self.source,
            "max_cases": self.max_cases,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, torch.Tensor | int],
    ) -> SimulationRepertoire:
        return cls(
            positions=payload["positions"],
            roles=payload["roles"],
            adjacency=payload["adjacency"],
            stiffness=payload["stiffness"],
            source=payload["source"],
            max_cases=int(payload["max_cases"]),
        )
