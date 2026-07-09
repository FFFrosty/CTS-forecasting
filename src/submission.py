"""生成竞赛提交文件。"""
import pandas as pd
from pathlib import Path


def fill_template(
    template_path: str | Path,
    predictions: pd.DataFrame,
    output_path: str | Path,
    target_col: str = "predicted",
) -> pd.DataFrame:
    """将预测值填入提交模板并保存。

    Parameters
    ----------
    template_path : str or Path
        赛题提供的提交模板 CSV。
    predictions : pd.DataFrame
        预测结果，需含与模板匹配的键列和预测列。
    output_path : str or Path
        输出路径。
    target_col : str
        预测值列名，模板中对应 vessel_count。

    Returns
    -------
    pd.DataFrame
        填好值的提交文件内容。
    """
    template = pd.read_csv(template_path, encoding="utf-8-sig")
    template.columns = template.columns.str.strip().str.lower()

    # 确保时间窗口格式一致
    if "time_window" in predictions.columns:
        predictions["time_window"] = pd.to_datetime(predictions["time_window"])
    if "time_window" in template.columns:
        template["time_window"] = pd.to_datetime(template["time_window"])

    # 找到匹配列
    merge_keys = [c for c in template.columns if c != "vessel_count"]

    filled = template.drop(columns=["vessel_count"]).merge(
        predictions[merge_keys + [target_col]],
        on=merge_keys,
        how="left",
    )
    filled.rename(columns={target_col: "vessel_count"}, inplace=True)

    # 确保列顺序与模板一致
    filled = filled[template.columns]

    filled.to_csv(output_path, index=False, encoding="utf-8-sig")
    return filled


def generate_submissions(
    task_a_predictions: pd.DataFrame,
    task_b_predictions: pd.DataFrame,
    template_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """同时生成赛题A和赛题B的提交文件。

    Returns
    -------
    tuple[Path, Path]
        (task_a_output_path, task_b_output_path)
    """
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_a_template = template_dir / "提交结果1_区域活跃拖轮数量.csv"
    task_b_template = template_dir / "提交结果2_圈层间拖轮迁移量.csv"

    path_a = output_dir / "task_a_submission.csv"
    path_b = output_dir / "task_b_submission.csv"

    fill_template(task_a_template, task_a_predictions, path_a)
    fill_template(task_b_template, task_b_predictions, path_b)

    return path_a, path_b
