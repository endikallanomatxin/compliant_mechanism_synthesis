"""Small symmetric spectra and first derivatives on CPU and Intel XPU."""

import torch


def symmetric_eigh(matrix):
    """Tiny spectral problems stay on Intel XPU (native eigh falls back to CPU)."""
    if matrix.device.type != "xpu":
        return torch.linalg.eigh(matrix)
    a = matrix.clone()
    n = a.shape[-1]
    vectors = torch.eye(n, dtype=a.dtype, device=a.device).expand_as(a).clone()
    # Disjoint Jacobi pairs rotate simultaneously (round-robin schedule).
    ring = list(range(n if n % 2 == 0 else n + 1))
    rounds = []
    for _ in range(len(ring) - 1):
        pairs = [
            (ring[k], ring[-k - 1])
            for k in range(len(ring) // 2)
            if ring[k] < n and ring[-k - 1] < n
        ]
        i = torch.tensor([p[0] for p in pairs], device=a.device)
        j = torch.tensor([p[1] for p in pairs], device=a.device)
        rounds.append((i, j))
        ring = [ring[0], ring[-1], *ring[1:-1]]
    eye = torch.eye(n, dtype=a.dtype, device=a.device).expand_as(a)
    for _ in range(8):
        for i, j in rounds:
            off = a[..., i, j]
            threshold = (
                torch.finfo(a.dtype).eps
                * a.abs().amax(dim=(-2, -1), keepdim=False).clamp_min(1)[..., None]
            )
            angle = torch.where(
                off.abs() > threshold,
                0.5 * torch.atan2(2 * off, a[..., j, j] - a[..., i, i]),
                0,
            )
            # Use the small rotation; repeated eigenvalue swaps can prevent
            # convergence when several pairs rotate simultaneously.
            angle = torch.where(angle > torch.pi / 4, angle - torch.pi / 2, angle)
            angle = torch.where(angle < -torch.pi / 4, angle + torch.pi / 2, angle)
            c, s = angle.cos(), angle.sin()
            transform = eye.clone()
            transform[..., i, i] = c
            transform[..., j, j] = c
            transform[..., i, j] = s
            transform[..., j, i] = -s
            a = transform.transpose(-1, -2) @ a @ transform
            vectors = vectors @ transform
    values, order = a.diagonal(dim1=-2, dim2=-1).sort(-1)
    return values, vectors.gather(-1, order[..., None, :].expand_as(vectors))


class Spectrum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        values, vectors = symmetric_eigh((matrix + matrix.transpose(-1, -2)) / 2)
        ctx.save_for_backward(vectors)
        return values

    @staticmethod
    def backward(ctx, gradient):
        (vectors,) = ctx.saved_tensors
        return (vectors * gradient[..., None, :]) @ vectors.transpose(-1, -2)


def positive_definite(matrix):
    """Unpivoted LDL positivity check with O(n) device-local Schur updates.

    Avoid a full XPU eigensolve or a CPU-fallback Cholesky in the hot path.
    Only positive pivots are used; an invalid matrix is rejected, not repaired.
    """
    a = matrix.detach().clone()
    valid = torch.isfinite(a).all(dim=(-2, -1))
    for k in range(a.shape[-1]):
        pivot = a[..., k, k]
        valid = valid & (pivot > 0) & torch.isfinite(pivot)
        if k + 1 < a.shape[-1]:
            column = a[..., k + 1 :, k]
            safe = torch.where(
                (pivot > 0) & torch.isfinite(pivot), pivot, torch.ones_like(pivot)
            )
            a[..., k + 1 :, k + 1 :] = (
                a[..., k + 1 :, k + 1 :]
                - column[..., :, None] * column[..., None, :] / safe[..., None, None]
            )
    return valid
