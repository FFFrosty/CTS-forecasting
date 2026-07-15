"""日历感知的预测基线与官方指标回测。"""
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


TimeRange = tuple[pd.Timestamp, pd.Timestamp]
Predictor = Callable[[pd.DataFrame, pd.DatetimeIndex, list[str]], pd.DataFrame]

A_GROUP_COLS = ["zone"]
B_GROUP_COLS = ["source_zone", "target_zone"]


def exclude_time_ranges(
    df: pd.DataFrame,
    ranges: Sequence[TimeRange],
) -> pd.DataFrame:
    """删除落在任一闭区间内的时间窗口。"""
    keep = pd.Series(True, index=df.index)
    for start, end in ranges:
        keep &= ~df["time_window"].between(start, end)
    return df.loc[keep].copy()


def make_calendar_folds(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    min_train_days: int,
    horizon_days: int,
    step_days: int = 1,
    excluded_test_ranges: Sequence[TimeRange] = (),
) -> list[pd.DatetimeIndex]:
    """生成连续日历预测窗口，跳过覆盖不可信标签的窗口。"""
    if min_train_days < 1 or horizon_days < 1 or step_days < 1:
        raise ValueError("min_train_days, horizon_days and step_days must be positive")

    first_forecast = data_start.normalize() + pd.Timedelta(days=min_train_days)
    last_forecast = data_end.normalize() - pd.Timedelta(days=horizon_days - 1)
    if first_forecast > last_forecast:
        return []

    folds = []
    for forecast_start in pd.date_range(
        first_forecast, last_forecast, freq=f"{step_days}D"
    ):
        forecast_times = pd.date_range(
            forecast_start, periods=horizon_days * 24, freq="h"
        )
        if forecast_times[-1] > data_end:
            continue
        overlaps_excluded = any(
            (forecast_times[0] <= end) and (forecast_times[-1] >= start)
            for start, end in excluded_test_ranges
        )
        if not overlaps_excluded:
            folds.append(forecast_times)
    return folds


def _prediction_grid(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
) -> pd.DataFrame:
    groups = train_df[group_cols].drop_duplicates().sort_values(group_cols)
    times = pd.DataFrame({"time_window": forecast_times})
    return times.merge(groups, how="cross")


def _add_group_fallback(
    predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    group_mean = (
        train_df.groupby(group_cols, as_index=False)["vessel_count"]
        .mean()
        .rename(columns={"vessel_count": "group_mean"})
    )
    predictions = predictions.merge(group_mean, on=group_cols, how="left")
    predictions["predicted"] = predictions["predicted"].fillna(
        predictions["group_mean"]
    )
    predictions["predicted"] = predictions["predicted"].fillna(0.0)
    return predictions.drop(columns="group_mean")


def predict_group_mean(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
) -> pd.DataFrame:
    """用每个圈层或迁移方向的全历史均值预测。"""
    grid = _prediction_grid(train_df, forecast_times, group_cols)
    grid["predicted"] = np.nan
    result = _add_group_fallback(grid, train_df, group_cols)
    return result[["time_window"] + group_cols + ["predicted"]]


def predict_hour_dow_mean(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    round_predictions: bool = False,
) -> pd.DataFrame:
    """同星期同时刻均值；缺少组合时依次退化到同小时和分组均值。"""
    work = train_df.copy()
    work["hour"] = work["time_window"].dt.hour
    work["day_of_week"] = work["time_window"].dt.dayofweek

    exact = (
        work.groupby(group_cols + ["hour", "day_of_week"], as_index=False)[
            "vessel_count"
        ]
        .mean()
        .rename(columns={"vessel_count": "predicted"})
    )
    hourly = (
        work.groupby(group_cols + ["hour"], as_index=False)["vessel_count"]
        .mean()
        .rename(columns={"vessel_count": "hour_mean"})
    )

    result = _prediction_grid(work, forecast_times, group_cols)
    result["hour"] = result["time_window"].dt.hour
    result["day_of_week"] = result["time_window"].dt.dayofweek
    result = result.merge(
        exact, on=group_cols + ["hour", "day_of_week"], how="left"
    )
    result = result.merge(hourly, on=group_cols + ["hour"], how="left")
    result["predicted"] = result["predicted"].fillna(result["hour_mean"])
    result = _add_group_fallback(result, work, group_cols)
    if round_predictions:
        result["predicted"] = np.rint(result["predicted"])
    return result[["time_window"] + group_cols + ["predicted"]]


def predict_recent_hour_mean(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    n_days: int = 10,
    round_predictions: bool = False,
) -> pd.DataFrame:
    """用最近 N 个有效日期中相同小时的均值预测。"""
    work = train_df.copy()
    work["date"] = work["time_window"].dt.normalize()
    work["hour"] = work["time_window"].dt.hour
    recent_dates = sorted(work["date"].unique())[-n_days:]
    recent = work[work["date"].isin(recent_dates)]
    means = (
        recent.groupby(group_cols + ["hour"], as_index=False)["vessel_count"]
        .mean()
        .rename(columns={"vessel_count": "predicted"})
    )

    result = _prediction_grid(work, forecast_times, group_cols)
    result["hour"] = result["time_window"].dt.hour
    result = result.merge(means, on=group_cols + ["hour"], how="left")
    result = _add_group_fallback(result, work, group_cols)
    if round_predictions:
        result["predicted"] = np.rint(result["predicted"])
    return result[["time_window"] + group_cols + ["predicted"]]


def predict_daily_profile(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    n_days: int = 10,
    round_predictions: bool = True,
) -> pd.DataFrame:
    """最近 N 个有效日的日总量均值，按全历史小时比例分配。"""
    work = train_df.copy()
    work["date"] = work["time_window"].dt.normalize()
    work["hour"] = work["time_window"].dt.hour
    recent_dates = sorted(work["date"].unique())[-n_days:]

    daily = work.groupby(group_cols + ["date"], as_index=False)[
        "vessel_count"
    ].sum()
    daily_mean = (
        daily[daily["date"].isin(recent_dates)]
        .groupby(group_cols, as_index=False)["vessel_count"]
        .mean()
        .rename(columns={"vessel_count": "pred_daily"})
    )

    hourly = work.groupby(group_cols + ["hour"], as_index=False)[
        "vessel_count"
    ].sum()
    hourly["group_total"] = hourly.groupby(group_cols)["vessel_count"].transform(
        "sum"
    )
    hourly["proportion"] = np.where(
        hourly["group_total"] > 0,
        hourly["vessel_count"] / hourly["group_total"],
        0.0,
    )

    result = _prediction_grid(work, forecast_times, group_cols)
    result["hour"] = result["time_window"].dt.hour
    result = result.merge(daily_mean, on=group_cols, how="left")
    result = result.merge(
        hourly[group_cols + ["hour", "proportion"]],
        on=group_cols + ["hour"],
        how="left",
    )
    result["predicted"] = (
        result["pred_daily"].fillna(0.0) * result["proportion"].fillna(0.0)
    ).clip(lower=0.0)
    if round_predictions:
        result["predicted"] = np.rint(result["predicted"])
    return result[["time_window"] + group_cols + ["predicted"]]


def score_predictions(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    group_cols: list[str],
) -> dict[str, float | int]:
    """按键严格对齐后计算单任务 SSE、MAE 和 RMSE。"""
    keys = ["time_window"] + group_cols
    if actual.duplicated(keys).any() or predictions.duplicated(keys).any():
        raise ValueError("actual and predictions must have unique keys")

    merged = actual[keys + ["vessel_count"]].merge(
        predictions[keys + ["predicted"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("actual and predictions do not contain the same keys")
    if merged[["vessel_count", "predicted"]].isna().any().any():
        raise ValueError("actual and predictions must not contain missing values")

    error = merged["predicted"] - merged["vessel_count"]
    return {
        "sse": float(np.square(error).sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "n": int(len(merged)),
    }


def evaluate_backtest(
    task_a: pd.DataFrame,
    task_b: pd.DataFrame,
    predictor: Predictor,
    min_train_days: int = 7,
    horizon_days: int = 3,
    step_days: int = 1,
    excluded_test_ranges: Sequence[TimeRange] = (),
    a_excluded_train_ranges: Sequence[TimeRange] = (),
    b_excluded_train_ranges: Sequence[TimeRange] = (),
) -> pd.DataFrame:
    """在相同日历折上同时评估 A/B，并计算官方加权 SSE。"""
    data_start = max(task_a["time_window"].min(), task_b["time_window"].min())
    data_end = min(task_a["time_window"].max(), task_b["time_window"].max())
    folds = make_calendar_folds(
        data_start=data_start,
        data_end=data_end,
        min_train_days=min_train_days,
        horizon_days=horizon_days,
        step_days=step_days,
        excluded_test_ranges=excluded_test_ranges,
    )

    results = []
    for forecast_times in folds:
        forecast_start = forecast_times[0]
        a_train = exclude_time_ranges(
            task_a[task_a["time_window"] < forecast_start],
            a_excluded_train_ranges,
        )
        b_train = exclude_time_ranges(
            task_b[task_b["time_window"] < forecast_start],
            b_excluded_train_ranges,
        )
        a_train_days = a_train["time_window"].dt.normalize().nunique()
        b_train_days = b_train["time_window"].dt.normalize().nunique()
        if a_train_days < min_train_days or b_train_days < min_train_days:
            continue

        a_actual = task_a[task_a["time_window"].isin(forecast_times)]
        b_actual = task_b[task_b["time_window"].isin(forecast_times)]
        a_pred = predictor(a_train, forecast_times, A_GROUP_COLS)
        b_pred = predictor(b_train, forecast_times, B_GROUP_COLS)
        a_score = score_predictions(a_actual, a_pred, A_GROUP_COLS)
        b_score = score_predictions(b_actual, b_pred, B_GROUP_COLS)

        weighted_sse = a_score["sse"] + 3.0 * b_score["sse"]
        weighted_n = a_score["n"] + 3 * b_score["n"]
        results.append({
            "forecast_start": forecast_start,
            "forecast_end": forecast_times[-1],
            "a_sse": a_score["sse"],
            "b_sse": b_score["sse"],
            "weighted_sse": weighted_sse,
            "weighted_mse": weighted_sse / weighted_n,
            "a_mae": a_score["mae"],
            "b_mae": b_score["mae"],
            "a_rmse": a_score["rmse"],
            "b_rmse": b_score["rmse"],
            "a_n": a_score["n"],
            "b_n": b_score["n"],
        })
    return pd.DataFrame(results)
