import pytest
import torch

from compliant_mechanism_synthesis.trajectory import Trajectory, canonical


def test_smooth_nonlinear_path_interpolates_every_waypoint():
    path = canonical(torch.tensor([-0.002, 0.006, 0.12], dtype=torch.float64))
    path.validate()
    assert torch.allclose(path.at(path.factors), path.poses, atol=1e-14)
    mid = path.at(torch.tensor(0.35, dtype=torch.float64))
    assert mid[0] > 0 and mid[0] != path.poses[-1, 0] * 0.35
    eps = 1e-6
    for t in path.factors[1:-1]:
        left = (path.at(t) - path.at(t - eps)) / eps
        right = (path.at(t + eps) - path.at(t)) / eps
        assert torch.allclose(left, right, atol=1e-5)


def test_rejects_unsorted_waypoints():
    with pytest.raises(ValueError):
        Trajectory(torch.tensor([0.0, 0.8, 0.5, 1.0]), torch.zeros(4, 3)).validate()
