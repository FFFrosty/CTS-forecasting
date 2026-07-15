"""每日船舶数校准统计基线测试。"""
import pandas as pd
import pytest

from src.features.temporal import build_daily_vessel_counts
from src.models.calibrated import predict_calibrated_hour_mean
from scripts.train_v3 import validate_prediction_grid


def test_build_daily_vessel_counts_uses_unique_vessels_per_date():
    trajectory = pd.DataFrame({
        "time": pd.to_datetime([
            "2018-01-01 00:00", "2018-01-01 01:00",
            "2018-01-01 02:00", "2018-01-02 00:00",
        ]),
        "mmsi": [1, 1, 2, 1],
    })

    result = build_daily_vessel_counts(trajectory)
    assert result.to_dict("records") == [
        {"date": pd.Timestamp("2018-01-01"), "vessel_count": 2},
        {"date": pd.Timestamp("2018-01-02"), "vessel_count": 1},
    ]


def test_calibrated_prediction_scales_and_returns_integers():
    train = pd.DataFrame({
        "time_window": pd.to_datetime([
            "2018-01-01 00:00", "2018-01-02 00:00",
        ]),
        "zone": ["核心区", "核心区"],
        "vessel_count": [2, 4],
    })
    daily_counts = pd.DataFrame({
        "date": pd.to_datetime([
            "2018-01-01", "2018-01-02", "2018-01-03",
        ]),
        "vessel_count": [10, 20, 30],
    })

    result = predict_calibrated_hour_mean(
        train,
        pd.date_range("2018-01-03", periods=1, freq="h"),
        ["zone"],
        daily_vessel_counts=daily_counts,
        n_days=2,
    )

    assert result.iloc[0]["predicted"] == 6
    assert pd.api.types.is_integer_dtype(result["predicted"])


def test_calibrated_prediction_requires_forecast_daily_counts():
    train = pd.DataFrame({
        "time_window": pd.to_datetime(["2018-01-01 00:00"]),
        "zone": ["核心区"],
        "vessel_count": [2],
    })
    daily_counts = pd.DataFrame({
        "date": pd.to_datetime(["2018-01-01"]),
        "vessel_count": [10],
    })

    with pytest.raises(ValueError, match="forecast dates"):
        predict_calibrated_hour_mean(
            train,
            pd.date_range("2018-01-02", periods=1, freq="h"),
            ["zone"],
            daily_vessel_counts=daily_counts,
        )


def test_submission_grid_rejects_float_predictions():
    forecast_times = pd.date_range("2018-01-25", periods=1, freq="h")
    predictions = pd.DataFrame({
        "time_window": forecast_times,
        "zone": ["核心区"],
        "predicted": [1.0],
    })

    with pytest.raises(ValueError, match="must be integers"):
        validate_prediction_grid(
            predictions,
            forecast_times,
            ["zone"],
            [("核心区",)],
        )
