from __future__ import annotations

import pytest

from compliant_mechanism_synthesis.cli import sample_main, train_main


def test_train_main_defers_until_supervised_stage_exists() -> None:
    with pytest.raises(NotImplementedError, match="Supervised training"):
        train_main([])


def test_sample_main_defers_until_sampling_pipeline_exists() -> None:
    with pytest.raises(NotImplementedError, match="Sampling is intentionally paused"):
        sample_main([])
