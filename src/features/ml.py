"""PureML 模型使用的严格因果时间序列特征。"""
from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
DEFAULT_ROLLING_WINDOWS = (6, 12, 24, 72, 168)
DEFAULT_SAME_HOUR_WINDOWS = (3, 7, 14)
DAILY_BATCH_LAGS = (24, 48, 72, 168)
DAILY_TOTAL_WINDOWS = (3, 7)


def complete_time_grid(
    samples: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "vessel_count",
) -> pd.DataFrame:
    """补齐每个分组的连续小时网格，缺失标签保留为 NaN。"""
    required = {"time_window", target_col, *group_cols}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"samples are missing required columns: {sorted(missing)}")

    work = samples[["time_window", *group_cols, target_col]].copy()
    work["time_window"] = pd.to_datetime(work["time_window"])
    keys = ["time_window", *group_cols]
    if work.duplicated(keys).any():
        raise ValueError("samples must contain unique time/group keys")
    if work.empty:
        raise ValueError("samples must not be empty")
    if not work["time_window"].eq(work["time_window"].dt.floor("h")).all():
        raise ValueError("time_window values must be aligned to whole hours")

    groups = work[group_cols].drop_duplicates().sort_values(group_cols)
    times = pd.DataFrame({
        "time_window": pd.date_range(
            work["time_window"].min(),
            work["time_window"].max(),
            freq="h",
        )
    })
    grid = times.merge(groups, how="cross")
    return (
        grid.merge(work, on=keys, how="left", validate="one_to_one")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def numeric_feature_columns(
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    same_hour_windows: Sequence[int] = DEFAULT_SAME_HOUR_WINDOWS,
) -> list[str]:
    """返回 PureML 数值特征的稳定列顺序。"""
    columns = [
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "day_index",
    ]
    columns.extend(f"lag_{lag}" for lag in lags)
    for window in rolling_windows:
        columns.extend([
            f"rolling_{window}_mean",
            f"rolling_{window}_std",
            f"rolling_{window}_count",
        ])
    for window in same_hour_windows:
        columns.extend([
            f"same_hour_{window}d_mean",
            f"same_hour_{window}d_std",
            f"same_hour_{window}d_count",
        ])
    return columns


def _add_calendar_features(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    result = frame.copy()
    time = result["time_window"]
    result["hour"] = time.dt.hour
    result["day_of_week"] = time.dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["day_index"] = (
        (time - origin) / pd.Timedelta(days=1)
    ).astype(float)
    return result


def build_causal_features(
    samples: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "vessel_count",
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    same_hour_windows: Sequence[int] = DEFAULT_SAME_HOUR_WINDOWS,
) -> pd.DataFrame:
    """按真实小时轴生成只依赖当前时刻之前标签的训练特征。"""
    frame = complete_time_grid(samples, group_cols, target_col)
    origin = frame["time_window"].min()
    frame = _add_calendar_features(frame, origin)

    grouped = frame.groupby(group_cols, sort=False)[target_col]
    for lag in lags:
        frame[f"lag_{lag}"] = grouped.shift(lag)

    for window in rolling_windows:
        frame[f"rolling_{window}_mean"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
        frame[f"rolling_{window}_std"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).std()
        )
        frame[f"rolling_{window}_count"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).count()
        )

    same_hour_groups = [*group_cols, "hour"]
    same_hour = frame.groupby(same_hour_groups, sort=False)[target_col]
    for window in same_hour_windows:
        frame[f"same_hour_{window}d_mean"] = same_hour.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
        frame[f"same_hour_{window}d_std"] = same_hour.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).std()
        )
        frame[f"same_hour_{window}d_count"] = same_hour.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).count()
        )
    return frame


def build_point_features(
    history: pd.Series,
    time_window: pd.Timestamp,
    origin: pd.Timestamp,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    same_hour_windows: Sequence[int] = DEFAULT_SAME_HOUR_WINDOWS,
) -> dict[str, float]:
    """从单个分组的历史序列生成一个未来时刻的递归预测特征。"""
    time_window = pd.Timestamp(time_window)
    hour = time_window.hour
    day_of_week = time_window.dayofweek
    features: dict[str, float] = {
        "hour": float(hour),
        "day_of_week": float(day_of_week),
        "is_weekend": float(day_of_week in (5, 6)),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
        "day_index": float((time_window - origin) / pd.Timedelta(days=1)),
    }

    for lag in lags:
        features[f"lag_{lag}"] = float(
            history.get(time_window - pd.Timedelta(hours=lag), np.nan)
        )

    for window in rolling_windows:
        start = time_window - pd.Timedelta(hours=window)
        end = time_window - pd.Timedelta(hours=1)
        values = history.loc[start:end].dropna()
        features[f"rolling_{window}_mean"] = float(values.mean())
        features[f"rolling_{window}_std"] = float(values.std())
        features[f"rolling_{window}_count"] = float(values.count())

    for window in same_hour_windows:
        timestamps = [
            time_window - pd.Timedelta(days=days)
            for days in range(1, window + 1)
        ]
        values = pd.Series(
            [history.get(timestamp, np.nan) for timestamp in timestamps],
            dtype=float,
        ).dropna()
        features[f"same_hour_{window}d_mean"] = float(values.mean())
        features[f"same_hour_{window}d_std"] = float(values.std())
        features[f"same_hour_{window}d_count"] = float(values.count())
    return features


def daily_batch_numeric_feature_columns(
    include_daily_count: bool,
    lags: Sequence[int] = DAILY_BATCH_LAGS,
    same_hour_windows: Sequence[int] = DEFAULT_SAME_HOUR_WINDOWS,
    daily_total_windows: Sequence[int] = DAILY_TOTAL_WINDOWS,
) -> list[str]:
    """返回每日批量预测使用的、在目标日开始前可获得的特征。"""
    columns = [
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "day_index",
    ]
    columns.extend(f"lag_{lag}" for lag in lags)
    for window in same_hour_windows:
        columns.extend([
            f"same_hour_{window}d_mean",
            f"same_hour_{window}d_std",
            f"same_hour_{window}d_count",
        ])
    columns.append("previous_day_total")
    for window in daily_total_windows:
        columns.extend([
            f"previous_{window}d_total_mean",
            f"previous_{window}d_total_std",
            f"previous_{window}d_total_count",
        ])
    if include_daily_count:
        columns.extend([
            "daily_vessel_count",
            "daily_vessel_count_delta_1",
            "daily_vessel_count_ratio_3d",
            "daily_vessel_count_ratio_7d",
        ])
    return columns


def _prepare_daily_count_features(
    daily_vessel_counts: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    required = {"date", "vessel_count"}
    missing = required.difference(daily_vessel_counts.columns)
    if missing:
        raise ValueError(
            f"daily_vessel_counts are missing required columns: {sorted(missing)}"
        )
    counts = daily_vessel_counts[["date", "vessel_count"]].copy()
    counts["date"] = pd.to_datetime(counts["date"]).dt.normalize()
    if counts["date"].duplicated().any():
        raise ValueError("daily_vessel_counts must contain one row per date")

    dates = pd.DataFrame({
        "date": pd.date_range(start_date, end_date, freq="D")
    })
    counts = dates.merge(counts, on="date", how="left")
    previous = counts["vessel_count"].shift(1)
    counts["daily_vessel_count_delta_1"] = counts["vessel_count"] - previous
    for window in (3, 7):
        reference = previous.rolling(window, min_periods=1).mean()
        counts[f"daily_vessel_count_ratio_{window}d"] = np.where(
            reference > 0,
            counts["vessel_count"] / reference,
            np.nan,
        )
    return counts.rename(columns={"vessel_count": "daily_vessel_count"})


def build_daily_batch_features(
    samples: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "vessel_count",
    daily_vessel_counts: pd.DataFrame | None = None,
    lags: Sequence[int] = DAILY_BATCH_LAGS,
    same_hour_windows: Sequence[int] = DEFAULT_SAME_HOUR_WINDOWS,
    daily_total_windows: Sequence[int] = DAILY_TOTAL_WINDOWS,
) -> pd.DataFrame:
    """构造同一天 24 小时可同时获得的直接预测特征。

    所有目标历史特征均来自目标日前一天或更早。同一天任一小时的
    ``vessel_count`` 都不会进入当天其他小时的特征。
    """
    frame = complete_time_grid(samples, group_cols, target_col)
    origin = frame["time_window"].min()
    frame = _add_calendar_features(frame, origin)
    frame["date"] = frame["time_window"].dt.normalize()

    grouped = frame.groupby(group_cols, sort=False)[target_col]
    for lag in lags:
        frame[f"lag_{lag}"] = grouped.shift(lag)

    same_hour_groups = [*group_cols, "hour"]
    same_hour = frame.groupby(same_hour_groups, sort=False)[target_col]
    for window in same_hour_windows:
        frame[f"same_hour_{window}d_mean"] = same_hour.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
        frame[f"same_hour_{window}d_std"] = same_hour.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).std()
        )
        frame[f"same_hour_{window}d_count"] = same_hour.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).count()
        )

    daily = (
        frame.groupby([*group_cols, "date"], as_index=False)[target_col]
        .agg(daily_total="sum", labeled_hours="count")
    )
    daily.loc[daily["labeled_hours"] < 24, "daily_total"] = np.nan
    daily_grouped = daily.groupby(group_cols, sort=False)["daily_total"]
    daily["previous_day_total"] = daily_grouped.shift(1)
    for window in daily_total_windows:
        daily[f"previous_{window}d_total_mean"] = daily_grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
        daily[f"previous_{window}d_total_std"] = daily_grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).std()
        )
        daily[f"previous_{window}d_total_count"] = daily_grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).count()
        )
    daily_feature_cols = [
        *group_cols,
        "date",
        "previous_day_total",
        *[
            f"previous_{window}d_total_{stat}"
            for window in daily_total_windows
            for stat in ("mean", "std", "count")
        ],
    ]
    frame = frame.merge(
        daily[daily_feature_cols],
        on=[*group_cols, "date"],
        how="left",
        validate="many_to_one",
    )

    if daily_vessel_counts is not None:
        count_features = _prepare_daily_count_features(
            daily_vessel_counts,
            frame["date"].min(),
            frame["date"].max(),
        )
        frame = frame.merge(
            count_features,
            on="date",
            how="left",
            validate="many_to_one",
        )
    return frame
