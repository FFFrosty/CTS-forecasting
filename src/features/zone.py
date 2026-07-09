"""时间窗口聚合：活跃状态标注、赛题A/B样本构建。

zone_state 编码（3位二进制，每位代表该圈层是否活跃）：
    bit0 (1): 外围区
    bit1 (2): 近港区
    bit2 (4): 核心区
    例: 6 (110) = 核心区+近港区同时活跃, 0 = 无圈层活跃
"""
import pandas as pd
import numpy as np

ZONE_BIT = {"外围区": 1, "近港区": 2, "核心区": 4}
ZONE_ORDER = ["核心区", "近港区", "外围区"]
ZONE_MIGRATION_PAIRS = [
    ("核心区", "近港区"),
    ("核心区", "外围区"),
    ("近港区", "核心区"),
    ("近港区", "外围区"),
    ("外围区", "核心区"),
    ("外围区", "近港区"),
]


def label_active_rows(df: pd.DataFrame, sog_min: float = 2.0, sog_max: float = 10.0) -> pd.DataFrame:
    """逐行标记是否处于活跃作业状态。"""
    df = df.copy()
    df["is_active_row"] = df["sog"].between(sog_min, sog_max)
    return df


def build_window_labels(
    df: pd.DataFrame,
    min_records: int = 3,
    freq: str = "1h",
) -> pd.DataFrame:
    """按时间窗口聚合，判定每窗口每条船是否处于活跃状态。

    活跃判定：该船在该小时内 is_active_row=True 的记录数 >= min_records（总数）。
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


def build_vessel_state_table(
    vessel_labels: pd.DataFrame,
    zone_df: pd.DataFrame,
    min_active_per_zone: int = 3,
) -> pd.DataFrame:
    """构建每条船每小时的二进制 zone_state。

    对每个 (mmsi, 小时, 圈层) 统计 SOG 在 [2,10] 的记录数。
    若 >= min_active_per_zone，则该圈层 bit 置 1。
    一条船可以在同一小时内多个圈层活跃 → zone_state 可以是 001~111。

    zone_state = 0 表示该船在此时未在任何圈层内有足够活跃记录（但可能仍
    满足总体活跃条件 is_active_vessel，因为活跃判定是全局记录数）。

    primary_zone: 从 zone_state 推导的唯一圈层（用于赛题B迁移判定）。
        优先取内圈：若核心区活跃 → 核心区；
        否则若近港活跃 → 近港区；
        否则若外围活跃 → 外围区；
        否则取该小时出现最频繁的圈层。

    Returns
    -------
    pd.DataFrame
        含 mmsi, time_window, zone_state, primary_zone, is_active, prev_state, is_migrated
    """
    # 每窗口每条船在每个圈层的活跃记录数
    per_zone_active = (
        zone_df[zone_df["is_active_row"]]
        .groupby(["mmsi", "time_window", "zone"])["is_active_row"]
        .count()
        .reset_index(name="zone_active_rows")
    )

    # 只保留 >= min 的圈层
    per_zone_active["zone_bit"] = per_zone_active["zone"].map(ZONE_BIT)
    per_zone_active = per_zone_active[per_zone_active["zone_active_rows"] >= min_active_per_zone]

    # 聚合为 zone_state
    zone_state = (
        per_zone_active.groupby(["mmsi", "time_window"])["zone_bit"]
        .sum()
        .reset_index(name="zone_state")
    )

    # 与 vessel_labels 合并（保留所有船-小时组合，zone_state NaN → 0）
    merged = vessel_labels.merge(zone_state, on=["mmsi", "time_window"], how="left")
    merged["zone_state"] = merged["zone_state"].fillna(0).astype(int)

    # 推导 primary_zone
    def state_to_primary(s):
        if s & 4:
            return "核心区"
        if s & 2:
            return "近港区"
        if s & 1:
            return "外围区"
        return "港外"

    merged["primary_zone"] = merged["zone_state"].apply(state_to_primary)

    # 对于 zone_state=0 的船，用该小时内出现最频繁的圈层
    # （如果没有活跃圈层记录，仍需要一个位置用于迁移判定）
    no_zone = merged["zone_state"] == 0
    if no_zone.any():
        zone_mode = (
            zone_df.groupby(["mmsi", "time_window"])["zone"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[-1])
            .reset_index(name="zone_mode")
        )
        merged = merged.merge(zone_mode, on=["mmsi", "time_window"], how="left")
        merged.loc[no_zone, "primary_zone"] = merged.loc[no_zone, "zone_mode"].fillna("港外")
        merged = merged.drop(columns=["zone_mode"])

    # 上一小时状态 + 迁移判定（基于 primary_zone）
    merged = merged.sort_values(["mmsi", "time_window"]).reset_index(drop=True)
    merged["prev_state"] = merged.groupby("mmsi")["zone_state"].shift(1).fillna(0).astype(int)
    merged["prev_primary"] = merged.groupby("mmsi")["primary_zone"].shift(1)
    merged["prev_window"] = merged.groupby("mmsi")["time_window"].shift(1)
    merged["is_consecutive"] = (
        (merged["time_window"] - merged["prev_window"]).dt.total_seconds() == 3600
    )
    merged["is_migrated"] = (
        merged["is_consecutive"]
        & (merged["primary_zone"] != merged["prev_primary"])
    )

    keep_cols = ["mmsi", "time_window", "zone_state", "primary_zone",
                 "is_active_vessel", "prev_state", "is_migrated"]
    return merged[keep_cols].rename(columns={"is_active_vessel": "is_active"})


def build_task_a_samples(vessel_state: pd.DataFrame) -> pd.DataFrame:
    """从个体船状态表构建赛题A样本：每窗口每圈层的活跃拖轮数量。

    使用二进制 zone_state 解码：一条船在多个圈层活跃 → 每个圈层各+1。

    Returns
    -------
    pd.DataFrame
        含 time_window, zone, vessel_count。
    """
    rows = []
    for zone, bit in ZONE_BIT.items():
        active_mask = (vessel_state["zone_state"] & bit) > 0
        counts = (
            vessel_state[active_mask]
            .groupby("time_window")["mmsi"]
            .nunique()
            .reset_index(name="vessel_count")
        )
        counts["zone"] = zone
        rows.append(counts)

    return pd.concat(rows, ignore_index=True)[["time_window", "zone", "vessel_count"]]


def build_task_b_samples(vessel_state: pd.DataFrame) -> pd.DataFrame:
    """从个体船状态表构建赛题B样本：相邻窗口间的圈层迁移量。

    基于 primary_zone 判定迁移（单圈层归属）。
    若 zone_state 在相邻小时的主圈层发生变化 → 记为一次迁移。

    Returns
    -------
    pd.DataFrame
        含 time_window, source_zone, target_zone, vessel_count。
    """
    vs = vessel_state.sort_values(["mmsi", "time_window"])
    vs["next_primary"] = vs.groupby("mmsi")["primary_zone"].shift(-1)
    vs["next_window"] = vs.groupby("mmsi")["time_window"].shift(-1)

    time_delta = (vs["next_window"] - vs["time_window"]).dt.total_seconds()
    vs["is_adjacent"] = time_delta == 3600.0

    migrations = vs[
        vs["is_adjacent"]
        & (vs["primary_zone"] != vs["next_primary"])
        & (vs["primary_zone"] != "港外")
        & (vs["next_primary"] != "港外")
    ]

    task_b = (
        migrations.groupby(["next_window", "primary_zone", "next_primary"])["mmsi"]
        .nunique()
        .reset_index(name="vessel_count")
    )
    return task_b.rename(columns={
        "next_window": "time_window",
        "primary_zone": "source_zone",
        "next_primary": "target_zone",
    })
