"""日历回测与官方评分测试。"""
from functools import partial

import pandas as pd
import pytest

from src.evaluation import (
    evaluate_backtest,
    make_calendar_folds,
    predict_group_mean,
    predict_hour_dow_mean,
    predict_recent_hour_mean,
    score_predictions,
)


def test_calendar_folds_are_contiguous_and_skip_excluded_ranges():
    excluded = [(pd.Timestamp("2018-01-05"), pd.Timestamp("2018-01-05 23:00"))]
    folds = make_calendar_folds(
        data_start=pd.Timestamp("2018-01-01 00:00"),
        data_end=pd.Timestamp("2018-01-10 23:00"),
        min_train_days=3,
        horizon_days=2,
        excluded_test_ranges=excluded,
    )

    assert folds
    assert all(len(fold) == 48 for fold in folds)
    assert all((fold[1:] - fold[:-1] == pd.Timedelta(hours=1)).all() for fold in folds)
    assert all(pd.Timestamp("2018-01-05") not in fold.normalize() for fold in folds)


def test_score_predictions_uses_official_sse_components():
    actual_a = pd.DataFrame({
        "time_window": pd.to_datetime(["2018-01-01 00:00", "2018-01-01 01:00"]),
        "zone": ["核心区", "核心区"],
        "vessel_count": [2, 1],
    })
    pred_a = pd.DataFrame({
        "time_window": actual_a["time_window"],
        "zone": actual_a["zone"],
        "predicted": [1, 3],
    })
    actual_b = pd.DataFrame({
        "time_window": pd.to_datetime(["2018-01-01 00:00", "2018-01-01 01:00"]),
        "source_zone": ["核心区", "核心区"],
        "target_zone": ["近港区", "近港区"],
        "vessel_count": [0, 2],
    })
    pred_b = pd.DataFrame({
        "time_window": actual_b["time_window"],
        "source_zone": actual_b["source_zone"],
        "target_zone": actual_b["target_zone"],
        "predicted": [1, 0],
    })

    a_score = score_predictions(actual_a, pred_a, ["zone"])
    b_score = score_predictions(actual_b, pred_b, ["source_zone", "target_zone"])
    assert a_score["sse"] == pytest.approx(5.0)
    assert b_score["sse"] == pytest.approx(5.0)
    assert a_score["sse"] + 3 * b_score["sse"] == pytest.approx(20.0)


def test_score_predictions_rejects_missing_keys():
    actual = pd.DataFrame({
        "time_window": pd.to_datetime(["2018-01-01 00:00", "2018-01-01 01:00"]),
        "zone": ["核心区", "核心区"],
        "vessel_count": [1, 2],
    })
    predictions = pd.DataFrame({
        "time_window": pd.to_datetime(["2018-01-01 00:00"]),
        "zone": ["核心区"],
        "predicted": [1],
    })

    with pytest.raises(ValueError, match="same keys"):
        score_predictions(actual, predictions, ["zone"])


def test_predictors_use_requested_forecast_timestamps():
    train = pd.DataFrame({
        "time_window": pd.to_datetime([
            "2018-01-01 00:00", "2018-01-01 01:00",
            "2018-01-02 00:00", "2018-01-02 01:00",
        ]),
        "zone": ["核心区"] * 4,
        "vessel_count": [1, 2, 3, 4],
    })
    requested = pd.date_range("2018-01-10", periods=24, freq="h")

    for predictor in [
        predict_hour_dow_mean,
        partial(predict_recent_hour_mean, n_days=2),
    ]:
        result = predictor(train, requested, ["zone"])
        assert result["time_window"].tolist() == requested.tolist()
        assert result["predicted"].notna().all()


def test_combined_backtest_uses_shared_folds_and_weighted_sse():
    times = pd.date_range("2018-01-01", periods=4 * 24, freq="h")
    day_value = times.day.astype(float)
    task_a = pd.DataFrame({
        "time_window": times,
        "zone": "核心区",
        "vessel_count": day_value,
    })
    task_b = pd.DataFrame({
        "time_window": times,
        "source_zone": "核心区",
        "target_zone": "近港区",
        "vessel_count": day_value - 1,
    })
    excluded = [(pd.Timestamp("2018-01-03"), pd.Timestamp("2018-01-03 23:00"))]

    result = evaluate_backtest(
        task_a,
        task_b,
        predictor=predict_group_mean,
        min_train_days=1,
        horizon_days=1,
        excluded_test_ranges=excluded,
        a_excluded_train_ranges=excluded,
        b_excluded_train_ranges=excluded,
    )

    assert result["forecast_start"].tolist() == [
        pd.Timestamp("2018-01-02"),
        pd.Timestamp("2018-01-04"),
    ]
    assert (
        result["weighted_sse"]
        == result["a_sse"] + 3 * result["b_sse"]
    ).all()
