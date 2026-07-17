"""PureML 因果特征测试。"""
import numpy as np
import pandas as pd

from src.features.ml import (
    build_causal_features,
    build_daily_batch_features,
    build_point_features,
    daily_batch_numeric_feature_columns,
)


def test_gap_is_not_compressed_when_building_lags():
    samples = pd.DataFrame({
        "time_window": pd.to_datetime([
            "2018-01-01 00:00",
            "2018-01-01 01:00",
            "2018-01-01 03:00",
        ]),
        "zone": ["核心区"] * 3,
        "vessel_count": [1.0, 2.0, 4.0],
    })

    features = build_causal_features(samples, ["zone"])
    at_three = features.loc[
        features["time_window"] == pd.Timestamp("2018-01-01 03:00")
    ].iloc[0]
    assert np.isnan(at_three["lag_1"])
    assert at_three["lag_2"] == 2.0


def test_current_target_does_not_leak_into_its_features():
    samples = pd.DataFrame({
        "time_window": pd.date_range("2018-01-01", periods=4, freq="h"),
        "zone": ["核心区"] * 4,
        "vessel_count": [1.0, 2.0, 3.0, 4.0],
    })
    changed = samples.copy()
    changed.loc[2, "vessel_count"] = 999.0

    before = build_causal_features(samples, ["zone"]).iloc[2]
    after = build_causal_features(changed, ["zone"]).iloc[2]
    feature_cols = [
        column
        for column in before.index
        if column not in {"time_window", "zone", "vessel_count"}
    ]
    pd.testing.assert_series_equal(
        before[feature_cols],
        after[feature_cols],
        check_names=False,
    )


def test_point_features_match_vectorized_causal_features():
    samples = pd.DataFrame({
        "time_window": pd.date_range("2018-01-01", periods=30, freq="h"),
        "zone": ["核心区"] * 30,
        "vessel_count": np.arange(30, dtype=float),
    })
    future_time = pd.Timestamp("2018-01-02 06:00")
    extended = pd.concat([
        samples,
        pd.DataFrame({
            "time_window": [future_time],
            "zone": ["核心区"],
            "vessel_count": [np.nan],
        }),
    ], ignore_index=True)
    vectorized = build_causal_features(extended, ["zone"]).iloc[-1]
    history = samples.set_index("time_window")["vessel_count"]
    point = build_point_features(history, future_time, samples["time_window"].min())

    for column, value in point.items():
        assert np.isclose(vectorized[column], value, equal_nan=True), column


def test_daily_batch_features_do_not_use_same_day_targets():
    samples = pd.DataFrame({
        "time_window": pd.date_range("2018-01-01", periods=4 * 24, freq="h"),
        "zone": ["核心区"] * (4 * 24),
        "vessel_count": np.arange(4 * 24, dtype=float),
    })
    changed = samples.copy()
    target_date = pd.Timestamp("2018-01-04")
    changed.loc[
        changed["time_window"].dt.normalize() == target_date,
        "vessel_count",
    ] += 999.0

    before = build_daily_batch_features(samples, ["zone"])
    after = build_daily_batch_features(changed, ["zone"])
    feature_cols = daily_batch_numeric_feature_columns(include_daily_count=False)
    before_day = before.loc[before["date"] == target_date, feature_cols]
    after_day = after.loc[after["date"] == target_date, feature_cols]
    pd.testing.assert_frame_equal(before_day, after_day)


def test_daily_count_features_are_attached_to_every_hour_of_target_day():
    samples = pd.DataFrame({
        "time_window": pd.date_range("2018-01-01", periods=3 * 24, freq="h"),
        "zone": ["核心区"] * (3 * 24),
        "vessel_count": np.arange(3 * 24, dtype=float),
    })
    daily_counts = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=3, freq="D"),
        "vessel_count": [40, 42, 45],
    })

    features = build_daily_batch_features(
        samples,
        ["zone"],
        daily_vessel_counts=daily_counts,
    )
    final_day = features[features["date"] == pd.Timestamp("2018-01-03")]

    assert len(final_day) == 24
    assert final_day["daily_vessel_count"].eq(45).all()
    assert final_day["daily_vessel_count_delta_1"].eq(3).all()
    assert "daily_vessel_count" in daily_batch_numeric_feature_columns(True)
    assert "daily_vessel_count" not in daily_batch_numeric_feature_columns(False)
