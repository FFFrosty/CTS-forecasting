"""时间窗口聚合：活跃状态标注、赛题A/B样本构建。"""
import pandas as pd
import numpy as np


def label_active_rows(df: pd.DataFrame, sog_min: float = 2.0, sog_max: float = 10.0) -> pd.DataFrame:
    """逐行标记是否处于活跃作业状态。

    Parameters
    ----------
    df : pd.DataFrame
        含 sog 列。
    sog_min, sog_max : float
        活跃航速区间（节）。

    Returns
    -------
    pd.DataFrame
        附加 is_active_row 布尔列。
    """
    df = df.copy()
    df["is_active_row"] = df["sog"].between(sog_min, sog_max)
    return df


def build_window_labels(
    df: pd.DataFrame,
    min_records: int = 3,
    freq: str = "1h",
) -> pd.DataFrame:
    """按时间窗口聚合，判定每窗口每条船是否处于活跃状态。

    活跃判定：该船在该小时内 is_active_row=True 的记录数 >= min_records。

    Returns
    -------
    pd.DataFrame
        每窗口每条船一条记录，含 is_active_vessel 列。
    """
    df = df.copy()
    df["time_window"] = df["time"].dt.floor(freq)

    active_counts = (
        df.groupby(["mmsi", "time_window"])["is_active_row"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "active_rows", "count": "total_rows"})
        .reset_index()
    )
    active_counts["is_active_vessel"] = active_counts["active_rows"] >= min_records
    return active_counts


def build_task_a_samples(
    vessel_labels: pd.DataFrame,
    zone_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建赛题A样本：每个时间窗口 × 圈层的活跃拖轮数量。

    Parameters
    ----------
    vessel_labels : pd.DataFrame
        build_window_labels 的输出，含 mmsi, time_window, is_active_vessel。
    zone_df : pd.DataFrame
        原始数据附加 zone 列（每个 mmsi 在窗口内的圈层归属）。

    Returns
    -------
    pd.DataFrame
        含 time_window, zone, vessel_count。
    """
    # 每窗口每条船的圈层（取该窗口内最后一条记录的圈层位置）
    last_zone = (
        zone_df.groupby(["mmsi", "time_window"])["zone"]
        .last()
        .reset_index()
    )

    merged = vessel_labels.merge(last_zone, on=["mmsi", "time_window"])
    active_only = merged[merged["is_active_vessel"]]

    task_a = (
        active_only.groupby(["time_window", "zone"])["mmsi"]
        .nunique()
        .reset_index(name="vessel_count")
    )
    return task_a


def build_task_b_samples(
    vessel_labels: pd.DataFrame,
    zone_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建赛题B样本：相邻时间窗口间的圈层迁移量。

    若船在 t 窗口位于 zone_A，在 t+1 窗口位于 zone_B（A≠B），
    则记为一次 zone_A → zone_B 迁移。

    Returns
    -------
    pd.DataFrame
        含 time_window, source_zone, target_zone, vessel_count。
    """
    last_zone = (
        zone_df.groupby(["mmsi", "time_window"])["zone"]
        .last()
        .reset_index()
    )

    # 每个窗口的活动船
    active = vessel_labels[["mmsi", "time_window", "is_active_vessel"]]
    merged = active.merge(last_zone, on=["mmsi", "time_window"])

    # 对每艘船按时间排序，取相邻窗口
    merged = merged.sort_values(["mmsi", "time_window"])
    merged["next_zone"] = merged.groupby("mmsi")["zone"].shift(-1)
    merged["next_window"] = merged.groupby("mmsi")["time_window"].shift(-1)

    # 迁移判定：连续窗口且zone不同
    time_delta = (merged["next_window"] - merged["time_window"]).dt.total_seconds()
    merged["is_adjacent"] = time_delta == 3600.0  # 1h

    migrations = merged[
        merged["is_adjacent"]
        & (merged["zone"] != merged["next_zone"])
        & merged["is_active_vessel"]
    ]

    task_b = (
        migrations.groupby(["next_window", "zone", "next_zone"])["mmsi"]
        .nunique()
        .reset_index(name="vessel_count")
    )
    task_b = task_b.rename(columns={
        "next_window": "time_window",
        "zone": "source_zone",
        "next_zone": "target_zone",
    })
    return task_b
