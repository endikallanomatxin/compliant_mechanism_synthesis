import math

import torch

from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import Search, SearchConfig


def test_loss_components_sum_to_objective_without_extra_equilibrium(tmp_path):
    torch.set_num_threads(1)
    search = Search(
        SearchConfig(population=2, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=2),
    )
    try:
        search.snapshot()
        record = search.loss_history[-1]
        keys = (
            "position",
            "orientation",
            "support",
            "material",
            "collisions",
            "lengths",
            "node_spacing",
        )
        assert math.isclose(
            sum(record[key] for key in keys), record["total"], rel_tol=1e-12
        )
        assert all(math.isfinite(record[key]) and record[key] >= 0 for key in keys)
        diagnostics = dict(search.last_diagnostics)
        search._record_loss()
        assert len(search.loss_history) == 1 and search.last_diagnostics == diagnostics
        search.snapshot = lambda *args, **kwargs: {}
        search.step("gradient")
        assert search.loss_history[-1]["iteration"] == 1
        assert math.isclose(
            sum(search.loss_history[-1][key] for key in keys),
            float(search.scores[0]),
            rel_tol=1e-12,
        )
        search.tune(
            {"collision_weight": 0, "beam_length_weight": 0, "material_weight": 0}
        )
        search._record_loss()
        assert search.loss_history[-1]["objective_revision"] == 1
        assert all(
            search.loss_history[-1][key] == 0
            for key in ("collisions", "lengths", "material")
        )
    finally:
        search.writer.close()
