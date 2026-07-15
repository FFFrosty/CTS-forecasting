"""利用已知每日船舶数校准的整数统计基线。"""
import numpy as np
import pandas as pd


def predict_calibrated_hour_mean(
    train_samples: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    daily_vessel_counts: pd.DataFrame,
    n_days: int = 10,
) -> pd.DataFrame:
    """按每日船舶数缩放最近 N 个有效日的同小时均值。

    缩放公式为：
        同小时均值 × 预测日船舶数 / 最近 N 个有效日平均船舶数

    预测结果四舍五入为非负整数，以满足提交格式要求。
    """
    if n_days < 1:
        raise ValueError("n_days must be positive")

    work = train_samples.copy()
    work["date"] = work["time_window"].dt.normalize()
    work["hour"] = work["time_window"].dt.hour
    counts = daily_vessel_counts[["date", "vessel_count"]].copy()
    counts["date"] = pd.to_datetime(counts["date"]).dt.normalize()
    if counts["date"].duplicated().any():
        raise ValueError("daily_vessel_counts must contain one row per date")

    recent_dates = sorted(work["date"].unique())[-n_days:]
    recent_counts = counts[counts["date"].isin(recent_dates)]
    if len(recent_counts) != len(recent_dates):
        raise ValueError("daily vessel counts are missing for recent training dates")
    reference_count = recent_counts["vessel_count"].mean()
    if reference_count <= 0:
        raise ValueError("mean daily vessel count must be positive")

    recent = work[work["date"].isin(recent_dates)]
    hourly_mean = (
        recent.groupby(group_cols + ["hour"], as_index=False)["vessel_count"]
        .mean()
        .rename(columns={"vessel_count": "base_prediction"})
    )

    groups = work[group_cols].drop_duplicates().sort_values(group_cols)
    result = pd.DataFrame({"time_window": forecast_times}).merge(groups, how="cross")
    result["date"] = result["time_window"].dt.normalize()
    result["hour"] = result["time_window"].dt.hour
    result = result.merge(hourly_mean, on=group_cols + ["hour"], how="left")
    result = result.merge(
        counts.rename(columns={"vessel_count": "daily_vessel_count"}),
        on="date",
        how="left",
    )

    if result["daily_vessel_count"].isna().any():
        raise ValueError("daily vessel counts are missing for forecast dates")
    if result["base_prediction"].isna().any():
        raise ValueError("recent training data do not cover every group and hour")

    scaled = (
        result["base_prediction"]
        * result["daily_vessel_count"]
        / reference_count
    )
    result["predicted"] = np.rint(scaled).clip(lower=0).astype(int)
    return result[["time_window"] + group_cols + ["predicted"]]
