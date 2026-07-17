"""PureML 树模型训练与递归预测测试。"""
import pandas as pd
import pytest

from src.models.tree import (
    daily_batch_tree_forecast,
    fit_daily_batch_tree_model,
    fit_tree_model,
    predict_pure_ml,
    predict_pure_ml_daily,
)


def make_training_samples() -> pd.DataFrame:
    times = pd.date_range("2018-01-01", periods=4 * 24, freq="h")
    rows = []
    for zone_index, zone in enumerate(["核心区", "近港区"]):
        for time_window in times:
            rows.append({
                "time_window": time_window,
                "zone": zone,
                "vessel_count": (time_window.hour % 5) + zone_index,
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_name", ["lightgbm", "random_forest"])
def test_pure_ml_models_return_complete_integer_predictions(model_name):
    train = make_training_samples()
    forecast_times = pd.date_range("2018-01-05", periods=4, freq="h")

    result = predict_pure_ml(
        train,
        forecast_times,
        ["zone"],
        model_name=model_name,
        random_state=2026,
    )

    assert len(result) == len(forecast_times) * 2
    assert result["predicted"].ge(0).all()
    assert pd.api.types.is_integer_dtype(result["predicted"])
    assert set(result["time_window"]) == set(forecast_times)


def test_random_forest_predictions_are_reproducible():
    train = make_training_samples()
    forecast_times = pd.date_range("2018-01-05", periods=3, freq="h")
    first = predict_pure_ml(
        train,
        forecast_times,
        ["zone"],
        model_name="random_forest",
        random_state=7,
    )
    second = predict_pure_ml(
        train,
        forecast_times,
        ["zone"],
        model_name="random_forest",
        random_state=7,
    )

    pd.testing.assert_frame_equal(first, second)


def test_unknown_tree_model_is_rejected():
    with pytest.raises(ValueError, match="model_name"):
        fit_tree_model(
            make_training_samples(),
            ["zone"],
            model_name="unknown",
        )


def test_daily_batch_model_returns_two_complete_days():
    train = make_training_samples()
    forecast_times = pd.date_range("2018-01-05", periods=2 * 24, freq="h")

    result = predict_pure_ml_daily(
        train,
        forecast_times,
        ["zone"],
        random_state=2026,
    )

    assert len(result) == len(forecast_times) * 2
    assert result["predicted"].ge(0).all()
    assert pd.api.types.is_integer_dtype(result["predicted"])
    assert set(result["time_window"]) == set(forecast_times)


def test_daily_count_model_requires_counts_for_every_forecast_date():
    train = make_training_samples()
    training_counts = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=4, freq="D"),
        "vessel_count": [40, 41, 42, 43],
    })
    fitted = fit_daily_batch_tree_model(
        train,
        ["zone"],
        daily_vessel_counts=training_counts,
    )

    with pytest.raises(ValueError, match="missing forecast dates"):
        daily_batch_tree_forecast(
            fitted,
            train,
            pd.date_range("2018-01-05", periods=24, freq="h"),
            daily_vessel_counts=training_counts,
        )
