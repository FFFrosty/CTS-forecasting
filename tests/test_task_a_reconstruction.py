import pandas as pd

from src.data.task_a_reconstruction import (
    build_daily_activity_statistics,
    build_sparse_observations,
    is_low_coverage,
)


def test_low_coverage_mask_uses_observed_hour_boundaries():
    times = pd.Series(pd.to_datetime([
        "2018-01-12 06:00",
        "2018-01-12 07:00",
        "2018-01-19 09:00",
        "2018-01-19 10:00",
    ]))

    assert is_low_coverage(times).tolist() == [False, True, True, False]


def test_sparse_observations_follow_task_a_per_zone_threshold():
    rows = []
    for minute in (0, 10, 20):
        rows.append({
            "mmsi": 1,
            "time": pd.Timestamp(2018, 1, 1, 0, minute),
            "sog": 5.0,
            "zone": "核心区",
        })
    for minute in (30, 40):
        rows.append({
            "mmsi": 2,
            "time": pd.Timestamp(2018, 1, 1, 0, minute),
            "sog": 5.0,
            "zone": "近港区",
        })
    rows.append({
        "mmsi": 3,
        "time": pd.Timestamp("2018-01-01 00:50"),
        "sog": 12.0,
        "zone": "核心区",
    })

    sparse, coverage = build_sparse_observations(pd.DataFrame(rows))

    assert sparse.to_dict("records") == [{
        "time_window": pd.Timestamp("2018-01-01 00:00"),
        "zone": "核心区",
        "sparse_count": 1,
    }]
    assert coverage.loc[0, "secondary_records"] == 6
    assert coverage.loc[0, "secondary_vessels"] == 3


def test_daily_statistics_decode_multi_zone_activity():
    vessel_state = pd.DataFrame({
        "mmsi": [1, 2, 3],
        "time_window": pd.to_datetime([
            "2018-01-01 00:00",
            "2018-01-01 01:00",
            "2018-01-01 02:00",
        ]),
        "zone_state": [4, 6, 0],
        "is_active": [True, True, True],
    })
    daily_counts = pd.DataFrame({
        "date": [pd.Timestamp("2018-01-01")],
        "vessel_count": [5],
    })

    result = build_daily_activity_statistics(vessel_state, daily_counts)
    zone_counts = result.set_index("zone")["zone_daily_unique"].to_dict()

    assert zone_counts == {"核心区": 2, "近港区": 1, "外围区": 0}
    assert result["active_vessels"].eq(3).all()
    assert result["daily_vessel_count"].eq(5).all()
