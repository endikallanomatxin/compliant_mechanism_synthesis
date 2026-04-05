from __future__ import annotations

from enum import IntEnum

import torch


class NodeRole(IntEnum):
    FIXED = 0
    MOBILE = 1
    FREE = 2


def role_masks(roles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fixed = roles == int(NodeRole.FIXED)
    mobile = roles == int(NodeRole.MOBILE)
    free = roles == int(NodeRole.FREE)
    return fixed, mobile, free
