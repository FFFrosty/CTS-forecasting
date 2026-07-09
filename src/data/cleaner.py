"""数据清洗：船型过滤、哨兵值处理。"""
import pandas as pd
import numpy as np


def filter_tug_vessels(df: pd.DataFrame) -> pd.DataFrame:
    """只保留拖轮/拖带/供应拖船类船舶。

    过滤掉 Pilot Vessel, Container Ship, Salvage Ship,
    Well Stimulation Vessel 等非拖轮类型。
    """
    tug_keywords = ("tug", "towing", "tow")
    mask = df["ship_type"].str.lower().apply(
        lambda s: any(kw in s for kw in tug_keywords)
    )
    return df[mask].copy()


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """将 AIS 协议哨兵值替换为 NaN。

    - true_heading == 511  → NaN（AIS 标准：不可用）
    - abs(rot) >= 720      → NaN（ROT 解码异常/饱和）
    """
    df = df.copy()

    # True Heading: 511 = not available (ITU-R M.1371-5)
    df.loc[df["true_heading"] == 511, "true_heading"] = np.nan

    # ROT: |rot| >= 720 → 超量程/哨兵，置 NaN
    df.loc[df["rot"].abs() >= 720, "rot"] = np.nan

    return df


def remove_outlier_positions(df: pd.DataFrame) -> pd.DataFrame:
    """移除明显超出港区范围的异常位置记录。

    保留距离中心点约 50km 以内的记录。
    """
    # 粗略过滤：经度 117.5~118.5, 纬度 38.7~39.2
    mask = (
        (df["x"].between(117.5, 118.5))
        & (df["y"].between(38.7, 39.2))
    )
    return df[mask].copy()
