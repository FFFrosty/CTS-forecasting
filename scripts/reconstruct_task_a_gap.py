"""利用辅助AIS数据源和日级活跃统计复原A题低采样小时。"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaner import remove_outlier_positions
from src.data.task_a_reconstruction import (
    SECONDARY_SOURCES,
    build_daily_activity_statistics,
    build_reconstruction_features,
    build_sparse_observations,
    reconstruct_task_a,
)
from src.features.spatial import classify_zone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--pseudo-label-weight", type=float, default=0.35)
    return parser.parse_args()


def load_secondary_rows(raw_path: Path) -> pd.DataFrame:
    columns = [
        "mmsi",
        "x",
        "y",
        "sog",
        "time",
        "source_dataset",
        "ship_type",
    ]
    parts = []
    for chunk in pd.read_csv(
        raw_path,
        encoding="utf-8-sig",
        usecols=columns,
        parse_dates=["time"],
        chunksize=500_000,
    ):
        tug = chunk["ship_type"].fillna("").str.lower().str.contains(
            "tug|towing|tow",
            regex=True,
        )
        source = chunk["source_dataset"].isin(SECONDARY_SOURCES)
        parts.append(chunk.loc[tug & source].copy())
    if not parts:
        raise RuntimeError("no secondary AIS rows were found")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    raw_path = next(RAW_DIR.glob("*0124*AIS.csv"))
    with open(PROJECT_ROOT / "configs" / "settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Loading secondary AIS sources...")
    secondary = load_secondary_rows(raw_path)
    secondary = remove_outlier_positions(secondary)
    secondary = classify_zone(
        secondary,
        center_lon=config["center"]["lon"],
        center_lat=config["center"]["lat"],
        radii=config["zone_radii"],
    )
    sparse, coverage = build_sparse_observations(secondary)

    task_a = pd.read_csv(
        PROCESSED_DIR / "task_a_train.csv",
        encoding="utf-8-sig",
        parse_dates=["time_window"],
        usecols=["time_window", "zone", "vessel_count"],
    )
    vessel_state = pd.read_csv(
        PROCESSED_DIR / "vessel_state.csv",
        encoding="utf-8-sig",
        parse_dates=["time_window"],
        usecols=["mmsi", "time_window", "zone_state", "is_active"],
    )
    daily_counts = pd.read_csv(
        PROCESSED_DIR / "daily_vessel_counts.csv",
        encoding="utf-8-sig",
        parse_dates=["date"],
    )
    daily_statistics = build_daily_activity_statistics(
        vessel_state,
        daily_counts,
    )
    features = build_reconstruction_features(
        task_a,
        sparse,
        coverage,
        daily_statistics,
    )
    reconstructed = reconstruct_task_a(
        features,
        random_state=args.random_state,
        pseudo_label_weight=args.pseudo_label_weight,
    )

    reconstructed_path = PROCESSED_DIR / "task_a_train_reconstructed.csv"
    features_path = PROCESSED_DIR / "task_a_reconstruction_features.csv"
    daily_path = PROCESSED_DIR / "task_a_reconstruction_daily.csv"
    reconstructed.to_csv(reconstructed_path, index=False, encoding="utf-8-sig")
    features.to_csv(features_path, index=False, encoding="utf-8-sig")

    daily = (
        reconstructed.assign(
            original=lambda x: x["original_vessel_count"],
            reconstructed=lambda x: x["vessel_count"],
        )
        .groupby(reconstructed["time_window"].dt.normalize())[
            ["original", "reconstructed"]
        ]
        .sum()
        .reset_index(names="date")
    )
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    gap = reconstructed[reconstructed["is_imputed"]]
    print(f"Secondary rows: {len(secondary):,}")
    print(f"Imputed A rows: {len(gap):,}")
    print(f"Original gap total: {gap['original_vessel_count'].sum():.1f}")
    print(f"Reconstructed gap total: {gap['vessel_count'].sum():.1f}")
    print(f"Reconstructed labels: {reconstructed_path}")
    print(f"Reconstruction features: {features_path}")
    print(f"Daily diagnostics: {daily_path}")


if __name__ == "__main__":
    main()
