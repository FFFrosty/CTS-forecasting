"""空间特征：距离计算与圈层判定。"""
import numpy as np
import pandas as pd


def haversine_distance(
    lon1: float, lat1: float,
    lon2: np.ndarray | pd.Series, lat2: np.ndarray | pd.Series,
) -> np.ndarray:
    """计算两组经纬度之间的 Haversine 距离（km）。

    Parameters
    ----------
    lon1, lat1 : float
        参考点经纬度（度）。
    lon2, lat2 : array-like
        目标点经纬度（度）。

    Returns
    -------
    np.ndarray
        距离数组，单位 km。
    """
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c


def classify_zone(
    df: pd.DataFrame,
    center_lon: float,
    center_lat: float,
    radii: dict[str, float],
    distance_col: str = "distance_km",
    zone_col: str = "zone",
) -> pd.DataFrame:
    """按距离中心点的远近给每条记录分配圈层。

    Parameters
    ----------
    df : pd.DataFrame
        含 x, y 列（经度、纬度）。
    center_lon, center_lat : float
        港口中心坐标。
    radii : dict
        {"core": 3.0, "near_port": 10.0, "outer": 30.0}
    distance_col : str
        距离列名。
    zone_col : str
        圈层列名。

    Returns
    -------
    pd.DataFrame
        附加 distance_km 和 zone 列。
    """
    df = df.copy()
    df[distance_col] = haversine_distance(
        center_lon, center_lat,
        df["x"].values, df["y"].values,
    )

    conditions = [
        df[distance_col] <= radii["core"],
        df[distance_col] <= radii["near_port"],
        df[distance_col] <= radii["outer"],
    ]
    choices = ["核心区", "近港区", "外围区"]
    df[zone_col] = np.select(conditions, choices, default="港外")

    # 丢弃港外记录
    df = df[df[zone_col] != "港外"].copy()
    return df


ZONE_ORDER = ["核心区", "近港区", "外围区"]
ZONE_MIGRATION_PAIRS = [
    ("核心区", "近港区"),
    ("核心区", "外围区"),
    ("近港区", "核心区"),
    ("近港区", "外围区"),
    ("外围区", "核心区"),
    ("外围区", "近港区"),
]
