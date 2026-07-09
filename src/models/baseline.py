"""基线模型：历史均值外推。"""
import pandas as pd
import numpy as np


def historical_mean_predict(
    train_samples: pd.DataFrame,
    forecast_horizon: int = 168,
    group_cols: list[str] | None = None,
    target_col: str = "vessel_count",
) -> pd.DataFrame:
    """用训练集同星期同时刻的历史均值作为预测值。

    Parameters
    ----------
    train_samples : pd.DataFrame
        训练集聚合样本，含 time_window, hour, day_of_week 和目标列。
    forecast_horizon : int
        预测步数（7天 × 24小时 = 168）。
    group_cols : list[str]
        分组列，如 ["zone"] 或 ["source_zone", "target_zone"]。

    Returns
    -------
    pd.DataFrame
        含 time_window, group_cols, predicted。
    """
    if group_cols is None:
        group_cols = ["zone"]

    df = train_samples.copy()
    # 同星期同时刻的历史均值
    mean_by_hour_dow = df.groupby(group_cols + ["hour", "day_of_week"])[target_col].mean()
    mean_by_hour_dow = mean_by_hour_dow.reset_index(name="predicted")

    # 生成验证集的时间窗口序列
    val_start = pd.Timestamp("2018-01-25")
    time_windows = pd.date_range(val_start, periods=forecast_horizon, freq="1h")

    # 构建预测帧
    predictions = []
    for tw in time_windows:
        h = tw.hour
        dow = tw.dayofweek
        matched = mean_by_hour_dow[
            (mean_by_hour_dow["hour"] == h)
            & (mean_by_hour_dow["day_of_week"] == dow)
        ].copy()
        matched["time_window"] = tw
        predictions.append(matched)

    result = pd.concat(predictions, ignore_index=True)
    return result[["time_window"] + group_cols + ["predicted"]]


def naive_rolling_mean_predict(
    train_samples: pd.DataFrame,
    forecast_horizon: int = 168,
    group_cols: list[str] | None = None,
    target_col: str = "vessel_count",
    window: int = 24,
) -> pd.DataFrame:
    """朴素前向填充：用训练集最后 24h 的均值作为常值外推。

    Parameters
    ----------
    window : int
        取训练集最后 window 个窗口的均值。
    """
    if group_cols is None:
        group_cols = ["zone"]

    df = train_samples.copy()
    last_24h_mean = (
        df.groupby(group_cols)
        .apply(lambda g: g[target_col].iloc[-window:].mean())
        .reset_index(name="predicted")
    )

    val_start = pd.Timestamp("2018-01-25")
    time_windows = pd.date_range(val_start, periods=forecast_horizon, freq="1h")

    predictions = []
    for tw in time_windows:
        entry = last_24h_mean.copy()
        entry["time_window"] = tw
        predictions.append(entry)

    return pd.concat(predictions, ignore_index=True)
