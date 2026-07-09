"""数据清洗模块测试。"""
import pandas as pd
import numpy as np
from src.data.cleaner import filter_tug_vessels, clean_sentinels


def test_filter_tug_vessels():
    df = pd.DataFrame({
        "mmsi": [1, 2, 3, 4, 5],
        "ship_type": [
            "Tug",
            "Anchor Handling Tug Supply",
            "Pilot Vessel",
            "Container Ship (Fully Cellular)",
            "Towing;Tug",
        ],
    })
    result = filter_tug_vessels(df)
    assert len(result) == 3
    assert set(result["mmsi"]) == {1, 2, 5}


def test_clean_sentinels_heading():
    df = pd.DataFrame({
        "true_heading": [511, 180, 0, 511, 270],
        "rot": [0.0, 10.0, -5.0, 720.0, -731.0],
    })
    result = clean_sentinels(df)

    # Heading 511 → NaN
    assert pd.isna(result.loc[0, "true_heading"])
    assert pd.isna(result.loc[3, "true_heading"])
    assert result.loc[1, "true_heading"] == 180
    assert result.loc[4, "true_heading"] == 270

    # ROT ±720, ±731 → NaN
    assert pd.isna(result.loc[3, "rot"])
    assert pd.isna(result.loc[4, "rot"])
    assert result.loc[1, "rot"] == 10.0


def test_clean_sentinels_preserves_structure():
    df = pd.DataFrame({
        "mmsi": [1119, 3019],
        "x": [117.79, 117.78],
        "y": [38.97, 38.98],
        "cog": [295.2, 317.0],
        "true_heading": [21, 209],
        "sog": [0.0, 2.1],
        "rot": [0.0, 0.0],
        "time": ["2018/1/1 2:41", "2018/1/1 3:41"],
        "source_dataset": ["china_coastal", "china_coastal"],
        "ship_type": ["Tug", "Tug"],
    })
    result = clean_sentinels(df)
    assert list(result.columns) == list(df.columns)
    assert len(result) == 2
