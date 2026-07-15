"""整数模型融合测试。"""
from functools import partial

import pandas as pd
import pytest

from src.models.ensemble import predict_integer_blend


def fixed_predictor(values, train_samples, forecast_times, group_cols):
    groups = train_samples[group_cols].drop_duplicates()
    result = pd.DataFrame({"time_window": forecast_times}).merge(groups, how="cross")
    result["predicted"] = values
    return result


def test_integer_blend_rounds_once_after_weighting():
    train = pd.DataFrame({"zone": ["核心区"]})
    times = pd.date_range("2018-01-25", periods=2, freq="h")
    result = predict_integer_blend(
        train,
        times,
        ["zone"],
        predictors=[
            partial(fixed_predictor, [0.0, 3.0]),
            partial(fixed_predictor, [2.0, 1.0]),
        ],
        weights=[0.3, 0.7],
    )

    assert result["predicted"].tolist() == [1, 2]
    assert pd.api.types.is_integer_dtype(result["predicted"])


def test_integer_blend_rejects_invalid_weights():
    train = pd.DataFrame({"zone": ["核心区"]})
    times = pd.date_range("2018-01-25", periods=1, freq="h")
    predictor = partial(fixed_predictor, [1.0])

    with pytest.raises(ValueError, match="sum to 1"):
        predict_integer_blend(
            train,
            times,
            ["zone"],
            predictors=[predictor, predictor],
            weights=[0.2, 0.2],
        )
