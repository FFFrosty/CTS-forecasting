"""选择多个版本并绘制 A/B 预测曲线。"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submission"
OUTPUT_DIR = SUBMISSION_DIR / "PureML_v5" / "plots"

MODEL_DIRS = {
    "v2": SUBMISSION_DIR,
    "v4": SUBMISSION_DIR / "v4",
    "LightGBM": SUBMISSION_DIR / "PureML_v5" / "lightgbm",
    "RF": SUBMISSION_DIR / "PureML_v5" / "random_forest",
    "Daily-NoCount": (
        SUBMISSION_DIR / "PureML_v5" / "daily_batch" / "no_daily_count"
    ),
}
COLORS = {
    "v2": "#4C78A8",
    "v4": "#E45756",
    "LightGBM": "#F58518",
    "RF": "#54A24B",
    "Daily-NoCount": "#E45756",
}
LINE_STYLES = {
    "v2": "--",
    "v4": "-",
    "LightGBM": "-.",
    "RF": ":",
    "Daily-NoCount": "-",
}
ZONES = ["核心区", "近港区", "外围区"]
DIRECTIONS = [
    ("核心区", "近港区"),
    ("核心区", "外围区"),
    ("近港区", "核心区"),
    ("近港区", "外围区"),
    ("外围区", "核心区"),
    ("外围区", "近港区"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_DIRS,
        default=["v2", "v4", "LightGBM", "RF"],
    )
    parser.add_argument("--output-prefix", default="prediction_comparison")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def load_predictions(
    task: str,
    model_names: list[str],
) -> dict[str, pd.DataFrame]:
    filename = f"task_{task}_submission.csv"
    predictions = {}
    expected_keys = None
    key_cols = ["time_window", "zone"] if task == "a" else [
        "time_window",
        "source_zone",
        "target_zone",
    ]
    for model in model_names:
        directory = MODEL_DIRS[model]
        frame = pd.read_csv(
            directory / filename,
            encoding="utf-8-sig",
            parse_dates=["time_window"],
        ).sort_values(key_cols)
        keys = set(frame[key_cols].itertuples(index=False, name=None))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"{model} 的 Task {task.upper()} 预测网格不一致")
        predictions[model] = frame
    return predictions


def format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlim(pd.Timestamp("2018-01-25"), pd.Timestamp("2018-01-31 23:00"))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)


def plot_task_a(
    predictions: dict[str, pd.DataFrame],
    output_prefix: str,
) -> Path:
    fig, axes = plt.subplots(
        len(ZONES),
        1,
        figsize=(16, 11),
        sharex=True,
    )
    for ax, zone in zip(axes, ZONES):
        for model, frame in predictions.items():
            subset = frame[frame["zone"] == zone].sort_values("time_window")
            ax.plot(
                subset["time_window"],
                subset["vessel_count"],
                label=model,
                color=COLORS[model],
                linestyle=LINE_STYLES[model],
                linewidth=2.0 if model in {"v4", "Daily-NoCount"} else 1.5,
                alpha=0.9,
            )
        ax.set_title(zone, loc="left")
        ax.set_ylabel("预测拖轮数")
        format_time_axis(ax)

    axes[-1].set_xlabel("时间窗口")
    fig.suptitle("Task A：各圈层小时预测曲线", fontsize=17, y=0.995)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUTPUT_DIR / f"{output_prefix}_task_a.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_task_b(
    predictions: dict[str, pd.DataFrame],
    output_prefix: str,
) -> Path:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(17, 12),
        sharex=True,
    )
    for ax, (source, target) in zip(axes.flat, DIRECTIONS):
        for model, frame in predictions.items():
            subset = frame[
                frame["source_zone"].eq(source)
                & frame["target_zone"].eq(target)
            ].sort_values("time_window")
            ax.step(
                subset["time_window"],
                subset["vessel_count"],
                where="post",
                label=model,
                color=COLORS[model],
                linestyle=LINE_STYLES[model],
                linewidth=2.0 if model in {"v4", "Daily-NoCount"} else 1.4,
                alpha=0.9,
            )
        ax.set_title(f"{source} → {target}", loc="left")
        ax.set_ylabel("预测迁移量")
        format_time_axis(ax)

    for ax in axes[-1]:
        ax.set_xlabel("时间窗口")
    fig.suptitle("Task B：各迁移方向小时预测曲线", fontsize=17, y=0.995)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUTPUT_DIR / f"{output_prefix}_task_b.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_overview(
    task_a: dict[str, pd.DataFrame],
    task_b: dict[str, pd.DataFrame],
    output_prefix: str,
) -> Path:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        constrained_layout=True,
    )
    for ax, task_name, predictions in [
        (axes[0], "Task A 三圈层合计", task_a),
        (axes[1], "Task B 六方向合计", task_b),
    ]:
        for model, frame in predictions.items():
            hourly = frame.groupby("time_window", as_index=False)["vessel_count"].sum()
            total = int(hourly["vessel_count"].sum())
            ax.plot(
                hourly["time_window"],
                hourly["vessel_count"],
                label=f"{model}（7日总量 {total}）",
                color=COLORS[model],
                linestyle=LINE_STYLES[model],
                linewidth=2.0 if model in {"v4", "Daily-NoCount"} else 1.5,
                alpha=0.9,
            )
        ax.set_title(task_name, loc="left")
        ax.set_ylabel("每小时预测合计")
        ax.legend(ncol=2)
        format_time_axis(ax)

    axes[-1].set_xlabel("时间窗口")
    model_label = " / ".join(task_a)
    fig.suptitle(f"{model_label} 预测总量对比", fontsize=17)
    path = OUTPUT_DIR / f"{output_prefix}_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_a = load_predictions("a", args.models)
    task_b = load_predictions("b", args.models)
    paths = [
        plot_overview(task_a, task_b, args.output_prefix),
        plot_task_a(task_a, args.output_prefix),
        plot_task_b(task_b, args.output_prefix),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
