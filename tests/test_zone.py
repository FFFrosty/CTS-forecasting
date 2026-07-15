"""赛题 A/B 官方标签口径测试。"""
import pandas as pd

from scripts.preprocess import expand_a_index, expand_b_index
from src.features.zone import (
    build_task_a_samples,
    build_task_b_samples,
    build_vessel_repr_table,
    build_vessel_state_table,
    build_window_labels,
)


def test_task_a_counts_one_vessel_in_multiple_zones():
    """同船同小时在多个圈层各有至少 3 条活跃记录时应分别计数。"""
    rows = []
    for minute, zone in [
        (1, "核心区"), (2, "核心区"), (3, "核心区"),
        (4, "近港区"), (5, "近港区"), (6, "近港区"),
    ]:
        rows.append({
            "mmsi": 1,
            "time": pd.Timestamp(2018, 1, 1, 0, minute),
            "zone": zone,
            "is_active_row": True,
        })

    # 总活跃记录达到 3 条，但任一圈层均不足 3 条，不能计入 A 题。
    for minute, zone in [(10, "核心区"), (11, "核心区"), (12, "近港区")]:
        rows.append({
            "mmsi": 2,
            "time": pd.Timestamp(2018, 1, 1, 0, minute),
            "zone": zone,
            "is_active_row": True,
        })

    trajectory = pd.DataFrame(rows)
    trajectory["time_window"] = trajectory["time"].dt.floor("1h")
    labels = build_window_labels(trajectory, min_records=3)
    states = build_vessel_state_table(labels, trajectory, min_active_per_zone=3)

    vessel_1 = states.loc[states["mmsi"] == 1].iloc[0]
    vessel_2 = states.loc[states["mmsi"] == 2].iloc[0]
    assert vessel_1["zone_state"] == 6  # 核心区(4) + 近港区(2)
    assert vessel_2["is_active"]
    assert vessel_2["zone_state"] == 0

    samples = build_task_a_samples(states)
    counts = samples.set_index("zone")["vessel_count"].to_dict()
    assert counts == {"核心区": 1, "近港区": 1}


def test_task_b_tie_uses_zone_with_latest_record():
    """区域记录数并列时，最后出现时间更晚的区域作为代表区域。"""
    trajectory = pd.DataFrame({
        "mmsi": [1, 1, 1, 1],
        "time_window": [pd.Timestamp("2018-01-01 00:00")] * 4,
        "time": pd.to_datetime([
            "2018-01-01 00:05",
            "2018-01-01 00:10",
            "2018-01-01 00:06",
            "2018-01-01 00:20",
        ]),
        "zone": ["核心区", "核心区", "近港区", "近港区"],
        # B 题不使用 SOG 活跃条件。
        "sog": [0.0, 0.0, 30.0, 30.0],
    })

    result = build_vessel_repr_table(trajectory)
    assert result.iloc[0]["repr_zone"] == "近港区"


def test_task_b_uses_source_hour_and_requires_adjacent_windows():
    """B 题以源小时为标签，且只统计严格相邻小时的迁移。"""
    repr_table = pd.DataFrame({
        "mmsi": [1, 1, 1, 2, 2, 3, 3],
        "time_window": pd.to_datetime([
            "2018-01-01 00:00", "2018-01-01 01:00", "2018-01-01 03:00",
            "2018-01-01 00:00", "2018-01-01 01:00",
            "2018-01-01 00:00", "2018-01-01 01:00",
        ]),
        "repr_zone": [
            "核心区", "外围区", "近港区",
            "近港区", "近港区",
            "港外", "核心区",
        ],
    })

    result = build_task_b_samples(repr_table)
    assert len(result) == 1
    migration = result.iloc[0]
    assert migration["time_window"] == pd.Timestamp("2018-01-01 00:00")
    assert migration["source_zone"] == "核心区"
    assert migration["target_zone"] == "外围区"
    assert migration["vessel_count"] == 1


def test_training_grids_include_zero_count_combinations():
    """聚合标签必须补齐全部小时、圈层和迁移方向。"""
    task_a = pd.DataFrame({
        "time_window": [pd.Timestamp("2018-01-01 00:00")],
        "zone": ["核心区"],
        "vessel_count": [2],
    })
    expanded_a = expand_a_index(task_a)
    assert len(expanded_a) == 24 * 24 * 3
    assert expanded_a["vessel_count"].isna().sum() == 0
    assert (expanded_a["vessel_count"] == 0).sum() == len(expanded_a) - 1

    task_b = pd.DataFrame({
        "time_window": [pd.Timestamp("2018-01-01 00:00")],
        "source_zone": ["核心区"],
        "target_zone": ["近港区"],
        "vessel_count": [1],
    })
    expanded_b = expand_b_index(task_b)
    assert len(expanded_b) == 24 * 24 * 6
    assert expanded_b["vessel_count"].isna().sum() == 0
    assert (expanded_b["vessel_count"] == 0).sum() == len(expanded_b) - 1
