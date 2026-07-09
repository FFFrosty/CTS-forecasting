"""时间特征：滞后特征与滚动统计。"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, time_col: str = "time_window") -> pd.DataFrame:
    """提取基础时间特征。

    Returns
    -------
    pd.DataFrame
        附加 hour, day_of_week, is_weekend 列。
    """
    df = df.copy()
    t = df[time_col]
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "vessel_count",
    group_cols: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """为每组的时序添加滞后特征。

    Parameters
    ----------
    df : pd.DataFrame
        含 time_window 和目标列，已按时间排序。
    target_col : str
        目标列名。
    group_cols : list[str]
        分组依据，如 ["zone"]。
    lags : list[int]
        滞后步数（以时间窗口为单位），默认 [1, 2, 3, 6, 12, 24]。

    Returns
    -------
    pd.DataFrame
        附加 lag_{n} 列。
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]
    if group_cols is None:
        group_cols = ["zone"]

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(group_cols)[target_col]
            .shift(lag)
        )
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "vessel_count",
    group_cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """为每组的时序添加滚动统计特征。

    Parameters
    ----------
    windows : list[int]
        滚动窗口大小，默认 [6, 12, 24]。

    Returns
    -------
    pd.DataFrame
        附加 rolling_{w}_mean, rolling_{w}_std 列。
    """
    if windows is None:
        windows = [6, 12, 24]
    if group_cols is None:
        group_cols = ["zone"]

    df = df.copy()
    for w in windows:
        rolled = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_{w}_mean"] = rolled

        rolled_std = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        )
        df[f"rolling_{w}_std"] = rolled_std

    return df
