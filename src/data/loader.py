"""训练与验证数据加载。"""
import pandas as pd
from pathlib import Path


def load_training_data(path: str | Path) -> pd.DataFrame:
    """加载训练集 AIS 原始 CSV。

    Parameters
    ----------
    path : str or Path
        CSV 文件路径。

    Returns
    -------
    pd.DataFrame
        包含列: mmsi, x, y, cog, true_heading, sog, rot, time, source_dataset, ship_type
    """
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["time"])
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_validation_count(path: str | Path) -> pd.DataFrame:
    """加载验证集每日船舶数量 CSV。

    Returns
    -------
    pd.DataFrame
        包含列: date, vessel_count
    """
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_submission_template(path: str | Path) -> pd.DataFrame:
    """加载赛题提交模板 CSV。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    return df
