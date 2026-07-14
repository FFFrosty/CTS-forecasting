"""可视化：每天一张角色分布热力图，分A/B题口径。

A题 → role_daily_a/：底色=primary_zone, 数字=zone_state, 黑框=is_active
B题 → role_daily_b/：底色=repr_zone（纯色块，无数字无黑框）

每天组内按"加权中心度"排序。
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_A = PROCESSED_DIR / "viz" / "role_daily_a"
OUT_B = PROCESSED_DIR / "viz" / "role_daily_b"
OUT_A.mkdir(parents=True, exist_ok=True)
OUT_B.mkdir(parents=True, exist_ok=True)

DOW_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 区域颜色映射
ZONE_CODE = {"无数据": 0, "港外": 1, "外围区": 2, "近港区": 3, "核心区": 4}
ZONE_COLORS = ["#ffffff", "#cccccc", "#fff176", "#ff9800", "#e53935"]
ZONE_TEXT_COLOR = {0: "#888888", 1: "#000000", 2: "#000000", 3: "#000000", 4: "#ffffff"}
cmap_zone = ListedColormap(ZONE_COLORS)
norm_zone = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

ZONE_WEIGHT = {"核心区": 3, "近港区": 2, "外围区": 1, "港外": 0, "无数据": -1}
GAP_DATE_STRS = {str(d.date()) for d in pd.date_range("2018-01-13", "2018-01-18")}


def load_task_a():
    """从 vessel_state.csv 加载 A 题口径数据。"""
    df = pd.read_csv(PROCESSED_DIR / "vessel_state.csv", parse_dates=["time_window"])
    df["date"] = df["time_window"].dt.date
    df["hour"] = df["time_window"].dt.hour
    print(f"A题: {len(df)} 行, {df['mmsi'].nunique()} 艘船")
    return df


def load_task_b():
    """从 vessel_repr.csv 加载 B 题口径数据。"""
    df = pd.read_csv(PROCESSED_DIR / "vessel_repr.csv", parse_dates=["time_window"])
    df["date"] = df["time_window"].dt.date
    df["hour"] = df["time_window"].dt.hour
    print(f"B题: {len(df)} 行, {df['mmsi'].nunique()} 艘船")
    return df


def fill_24h_grid(df, zone_col):
    """补全每船每天 24 小时网格，缺失小时填'无数据'。"""
    existing_keys = set(map(tuple, df[["mmsi", "date", "hour"]].drop_duplicates().values))
    missing = []
    for m, d in df[["mmsi", "date"]].drop_duplicates().itertuples(index=False):
        for h in range(24):
            if (m, d, h) not in existing_keys:
                missing.append({"mmsi": m, "date": d, "hour": h, zone_col: "无数据"})
    if missing:
        df = pd.concat([df, pd.DataFrame(missing)], ignore_index=True)
    return df


def build_daily_scores(df, zone_col):
    """每船每天加权中心度，用于排序。"""
    df["zone_weight"] = df[zone_col].map(ZONE_WEIGHT).fillna(-1)
    return df.groupby(["mmsi", "date"])["zone_weight"].mean().reset_index(name="score")


def draw_day(date, day_group, state, zone_col, out_dir, extra_cols=None,
             draw_text=True, draw_border=True, title_suffix=""):
    """绘制一天的热力图。

    Parameters
    ----------
    extra_cols : list of str, optional
        额外列名，会按顺序传入 matrix 构造。第一个用于 text，第二个用于 border。
    """
    n_vessels = len(day_group)
    zone_matrix = np.zeros((n_vessels, 24), dtype=int)
    text_matrix = np.zeros((n_vessels, 24), dtype=int) if draw_text else None
    border_matrix = np.zeros((n_vessels, 24), dtype=bool) if draw_border else None
    mmsi_labels = []

    for i, m in enumerate(day_group["mmsi"]):
        one = state[(state["mmsi"] == m) & (state["date"] == date)].sort_values("hour")
        zone_vec = one[zone_col].map(ZONE_CODE).fillna(0).values
        zone_matrix[i] = zone_vec
        mmsi_labels.append(str(m))

        if extra_cols:
            if draw_text and len(extra_cols) >= 1:
                text_matrix[i] = one[extra_cols[0]].fillna(0).astype(int).values
            if draw_border and len(extra_cols) >= 2:
                border_matrix[i] = one[extra_cols[1]].eq(True).values

    fig, ax = plt.subplots(figsize=(12, max(8, n_vessels * 0.3)))
    ax.imshow(zone_matrix, aspect="auto", cmap=cmap_zone, norm=norm_zone, interpolation="nearest")

    # 格子内文字
    if draw_text and text_matrix is not None:
        for i in range(n_vessels):
            for j in range(24):
                val = text_matrix[i, j]
                if val > 0:
                    zc = zone_matrix[i, j]
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7, color=ZONE_TEXT_COLOR.get(zc, "#000000"), fontweight="bold")

    # 黑框
    if draw_border and border_matrix is not None:
        for i in range(n_vessels):
            for j in range(24):
                if border_matrix[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="black", linewidth=1.0))

    # Y轴
    ylabels = [f"{mmsi_labels[i]} ({day_group.iloc[i]['score']:.1f})" for i in range(n_vessels)]
    ax.set_yticks(range(n_vessels))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_xlabel("小时")

    # 背景
    is_gap = str(date) in GAP_DATE_STRS
    is_weekend = date.weekday() in [5, 6]
    bg_note = ""
    if is_gap:
        ax.axhspan(-0.5, n_vessels - 0.5, color="red", alpha=0.06, zorder=0)
        bg_note = " [断层期]"
    elif is_weekend:
        ax.axhspan(-0.5, n_vessels - 0.5, color="orange", alpha=0.04, zorder=0)
        bg_note = " [周末]"

    score_min = day_group["score"].min()
    score_max = day_group["score"].max()
    ax.set_title(
        f"{date.strftime('%Y-%m-%d')} {DOW_NAMES[date.weekday()]}{bg_note}  "
        f"({n_vessels}艘, score {score_min:.1f}~{score_max:.1f}){title_suffix}"
    )

    plt.tight_layout()
    plt.savefig(out_dir / f"{date}.png", dpi=120, bbox_inches="tight")
    plt.close()


def main():
    # ===== A 题 =====
    print("=== A 题 ===")
    state_a = load_task_a()
    state_a = fill_24h_grid(state_a, "primary_zone")
    scores_a = build_daily_scores(state_a, "primary_zone")

    all_dates = sorted(state_a["date"].unique())
    for date in all_dates:
        day_group = scores_a[scores_a["date"] == date].sort_values("score", ascending=False)
        draw_day(date, day_group, state_a, zone_col="primary_zone",
                 out_dir=OUT_A, extra_cols=["zone_state", "is_active"],
                 draw_text=True, draw_border=True,
                 title_suffix="\n[ A题 ] 底色=活跃区域(primary_zone), 数字=zone_state(1=外围/2=近港/4=核心/6=核+近...), 黑框=活跃作业")

    print(f"A题完成: {len(all_dates)} 张图 → {OUT_A}")

    # ===== B 题 =====
    print("=== B 题 ===")
    state_b = load_task_b()
    state_b = fill_24h_grid(state_b, "repr_zone")
    scores_b = build_daily_scores(state_b, "repr_zone")

    all_dates_b = sorted(state_b["date"].unique())
    for date in all_dates_b:
        day_group = scores_b[scores_b["date"] == date].sort_values("score", ascending=False)
        draw_day(date, day_group, state_b, zone_col="repr_zone",
                 out_dir=OUT_B, extra_cols=None,
                 draw_text=False, draw_border=False,
                 title_suffix="\n[ B题 ] 底色=代表区域(repr_zone, 全部记录众数)")

    print(f"B题完成: {len(all_dates_b)} 张图 → {OUT_B}")


if __name__ == "__main__":
    main()