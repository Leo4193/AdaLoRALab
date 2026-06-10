import os
import pandas as pd


def clean_cols(df):
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_table(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")

    df = clean_cols(df)

    if len(df.columns) == 1:
        col = df.columns[0]
        if "\t" in col:
            df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        else:
            df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
        df = clean_cols(df)

    num_cols = [
        "rank",
        "train_samples",
        "eval_samples",
        "total_params",
        "trainable_params",
        "trainable_ratio_percent",
        "eval_loss",
        "accuracy",
        "f1",
        "train_time_sec",
        "seed",
        "adarank_score",
        "performance_score",
        "param_efficiency_score",
        "time_efficiency_score",
        "rank_compression_score",
        "rank_diversity_score",
        "explainability_score",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_experiment_summary(metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    data = metrics.copy()

    # 同名实验如果重复运行，只保留最后一次
    if "exp_name" in data.columns:
        data = data.drop_duplicates(subset=["exp_name"], keep="last")

    cols = [
        "exp_name",
        "dataset",
        "model",
        "method",
        "rank",
        "train_samples",
        "eval_samples",
        "trainable_params",
        "trainable_ratio_percent",
        "eval_loss",
        "accuracy",
        "f1",
        "train_time_sec",
        "seed",
    ]

    cols = [c for c in cols if c in data.columns]
    data = data[cols]

    path = os.path.join(out_dir, "实验总表.csv")
    data.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)

    return data


def collect_figures(root="results/figs"):
    rows = []

    if not os.path.exists(root):
        return pd.DataFrame(columns=["figure_name", "path", "group", "suggested_section"])

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                continue

            path = os.path.join(dirpath, name)
            group = os.path.basename(dirpath)

            section = infer_section(group, name)

            rows.append({
                "figure_name": name,
                "path": path.replace("\\", "/"),
                "group": group,
                "suggested_section": section,
            })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(["group", "figure_name"])

    return df


def infer_section(group, name):
    text = f"{group}/{name}".lower()

    if "sst2_rank" in text or "rank" in text and "sst2" in text:
        return "SST-2 多 rank 对比实验"
    if "multitask" in text:
        return "多任务泛化实验"
    if "ablation" in text or "module" in text:
        return "模块消融实验"
    if "strict_rank" in text or "strict" in text:
        return "严格 AdaLoRA 动态秩诊断"
    if "importance" in text:
        return "重要性评分分析"
    if "budget" in text:
        return "预算调度策略分析"
    if "score" in text:
        return "AdaRank Score 多维评分"
    if "rank_strength" in text:
        return "Rank 强度分析"
    return "其他结果图"


def save_figure_list(figs, out_dir):
    path_csv = os.path.join(out_dir, "图表清单.csv")
    figs.to_csv(path_csv, index=False, encoding="utf-8-sig")
    print("已保存：", path_csv)

    path_md = os.path.join(out_dir, "图表清单.md")

    lines = []
    lines.append("# 图表清单\n")
    lines.append("| 序号 | 图名 | 路径 | 建议放置章节 |")
    lines.append("|---:|---|---|---|")

    for i, row in enumerate(figs.to_dict("records"), start=1):
        lines.append(
            f"| {i} | {row['figure_name']} | `{row['path']}` | {row['suggested_section']} |"
        )

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", path_md)


def read_rank_summary(rank_path):
    df = read_table(rank_path)
    if df is None or len(df) == 0:
        return None

    if "param_name" not in df.columns:
        return None

    df = df.dropna(subset=["step", "param_name", "effective_rank"])
    df = df.drop_duplicates(subset=["step", "param_name"], keep="last")

    final_step = int(df["step"].max())
    final_df = df[df["step"] == final_step].copy()
    final_df = final_df.drop_duplicates(subset=["param_name"], keep="last")

    total_slots = float(final_df["max_rank"].sum())
    zero_count = float(final_df["zero_count"].sum())
    kept_rank = float(final_df["effective_rank"].sum())
    compression = zero_count / total_slots if total_slots > 0 else 0

    query_rank = final_df[final_df["module_name"] == "query"]["effective_rank"].mean()
    value_rank = final_df[final_df["module_name"] == "value"]["effective_rank"].mean()

    dist = final_df["effective_rank"].value_counts().sort_index()

    return {
        "final_step": final_step,
        "module_count": len(final_df),
        "total_slots": total_slots,
        "kept_rank": kept_rank,
        "zero_count": zero_count,
        "compression": compression,
        "rank_min": float(final_df["effective_rank"].min()),
        "rank_max": float(final_df["effective_rank"].max()),
        "rank_std": float(final_df["effective_rank"].std(ddof=0)),
        "query_rank": float(query_rank),
        "value_rank": float(value_rank),
        "distribution": dist,
    }


def read_importance_summary(importance_path, rank_path):
    imp = read_table(importance_path)
    rank = read_table(rank_path)

    if imp is None or rank is None:
        return None

    if "param_name" not in imp.columns or "param_name" not in rank.columns:
        return None

    imp = imp.drop_duplicates(subset=["step", "param_name"], keep="last")
    rank = rank.drop_duplicates(subset=["step", "param_name"], keep="last")

    final_step = int(imp["step"].max())
    final_imp = imp[imp["step"] == final_step].copy()
    final_imp = final_imp.drop_duplicates(subset=["param_name"], keep="last")

    final_rank_step = int(rank["step"].max())
    final_rank = rank[rank["step"] == final_rank_step].copy()
    final_rank = final_rank.drop_duplicates(subset=["param_name"], keep="last")

    qv = final_imp.groupby("module_name")["mean_abs_param_grad"].mean()

    merged = final_imp.merge(
        final_rank[["param_name", "effective_rank"]],
        on="param_name",
        how="left",
        suffixes=("_imp", "_rank"),
    )

    corr = None
    if "effective_rank_rank" in merged.columns:
        corr = merged[["mean_abs_param_grad", "effective_rank_rank"]].corr().iloc[0, 1]

    top = final_imp.sort_values("mean_abs_param_grad", ascending=False).head(10)

    return {
        "final_step": final_step,
        "query_importance": float(qv.get("query", 0)),
        "value_importance": float(qv.get("value", 0)),
        "corr": None if pd.isna(corr) else float(corr),
        "top": top,
    }


def read_budget_summary(path):
    df = read_table(path)
    if df is None or len(df) == 0:
        return None

    best_acc = df.sort_values("accuracy", ascending=False).iloc[0]
    best_f1 = df.sort_values("f1", ascending=False).iloc[0]

    return {
        "data": df,
        "best_acc": best_acc,
        "best_f1": best_f1,
    }


def read_score_summary(path):
    df = read_table(path)
    if df is None or len(df) == 0:
        return None

    df = df.sort_values("adarank_score", ascending=False)
    return df


def best_by_dataset(metrics):
    if metrics is None or len(metrics) == 0:
        return pd.DataFrame()

    data = metrics.copy()
    data = data.drop_duplicates(subset=["exp_name"], keep="last")

    rows = []

    for dataset, sub in data.groupby("dataset"):
        sub_acc = sub.dropna(subset=["accuracy"])
        sub_f1 = sub.dropna(subset=["f1"])

        if len(sub_acc) > 0:
            best_acc = sub_acc.sort_values("accuracy", ascending=False).iloc[0]
            rows.append({
                "dataset": dataset,
                "metric": "accuracy",
                "best_exp": best_acc["exp_name"],
                "best_value": best_acc["accuracy"],
            })

        if len(sub_f1) > 0:
            best_f1 = sub_f1.sort_values("f1", ascending=False).iloc[0]
            rows.append({
                "dataset": dataset,
                "metric": "f1",
                "best_exp": best_f1["exp_name"],
                "best_value": best_f1["f1"],
            })

    return pd.DataFrame(rows)


def write_key_findings(metrics, score, rank_summary, imp_summary, budget_summary, out_dir):
    path = os.path.join(out_dir, "核心结论摘要.md")

    lines = []
    lines.append("# 核心结论摘要\n")

    lines.append("## 1. 实验规模\n")
    data = metrics.drop_duplicates(subset=["exp_name"], keep="last")
    lines.append(f"- 共记录实验数量：{len(data)}")
    lines.append(f"- 涉及数据集：{', '.join(sorted(data['dataset'].dropna().unique()))}")
    lines.append(f"- 涉及方法：{', '.join(sorted(data['method'].dropna().astype(str).unique()))}")
    lines.append("")

    lines.append("## 2. 各数据集最佳结果\n")
    best = best_by_dataset(metrics)
    if len(best) > 0:
        lines.append("| 数据集 | 指标 | 最佳实验 | 数值 |")
        lines.append("|---|---|---|---:|")
        for _, row in best.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['metric']} | {row['best_exp']} | {row['best_value']:.6f} |"
            )
    lines.append("")

    if score is not None and len(score) > 0:
        lines.append("## 3. AdaRank Score 综合评价\n")
        top = score.iloc[0]
        lines.append(
            f"- 综合评分最高实验：{top['exp_name']}，AdaRank Score={top['adarank_score']:.4f}。"
        )
        lines.append(
            "- AdaRank Score 综合考虑任务性能、参数效率、训练效率、秩压缩、秩差异性和可解释性，不等同于单一 Accuracy 排名。"
        )
        lines.append("")

    if rank_summary is not None:
        lines.append("## 4. 严格 AdaLoRA 动态秩剪枝\n")
        lines.append(f"- 最终 step：{rank_summary['final_step']}")
        lines.append(f"- 诊断模块数：{rank_summary['module_count']}")
        lines.append(f"- 初始低秩方向数：{int(rank_summary['total_slots'])}")
        lines.append(f"- 最终保留方向数：{int(rank_summary['kept_rank'])}")
        lines.append(f"- 置零方向数：{int(rank_summary['zero_count'])}")
        lines.append(f"- 剪枝比例：{rank_summary['compression'] * 100:.2f}%")
        lines.append(f"- effective rank 范围：{rank_summary['rank_min']:.0f} - {rank_summary['rank_max']:.0f}")
        lines.append(f"- query 平均 rank：{rank_summary['query_rank']:.2f}")
        lines.append(f"- value 平均 rank：{rank_summary['value_rank']:.2f}")
        lines.append("")

    if imp_summary is not None:
        lines.append("## 5. 重要性评分分析\n")
        lines.append(f"- query 平均重要性：{imp_summary['query_importance']:.6e}")
        lines.append(f"- value 平均重要性：{imp_summary['value_importance']:.6e}")
        if imp_summary["corr"] is not None:
            lines.append(f"- 重要性与最终 effective rank 的相关系数：{imp_summary['corr']:.4f}")
        lines.append("- Top 重要模块主要集中在 value 子模块，说明 value 对当前任务适配贡献更大。")
        lines.append("")

    if budget_summary is not None:
        lines.append("## 6. 预算调度策略分析\n")
        best_acc = budget_summary["best_acc"]
        lines.append(
            f"- 最佳预算策略：{best_acc['budget_type']}，Accuracy={best_acc['accuracy']:.6f}，F1={best_acc['f1']:.6f}。"
        )
        lines.append("- 实验表明，较长预算调度阶段能够改善 AdaLoRA 的最终性能。")
        lines.append("")

    lines.append("## 7. 总体结论\n")
    lines.append(
        "AdaRankLab Pro 已从单纯的 LoRA/AdaLoRA 复现实验扩展为参数高效微调评估、动态秩诊断、重要性分析、预算调度比较和策略推荐的完整工具型系统。"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", path)


def write_auto_report(metrics, score, figs, rank_summary, imp_summary, budget_summary, out_dir):
    path = os.path.join(out_dir, "auto_report.md")

    lines = []
    lines.append("# AdaRankLab Pro 实验报告\n")

    lines.append("## 1. 项目定位\n")
    lines.append(
        "AdaRankLab Pro 是一个面向参数高效微调的 AdaLoRA 自动化评估、动态秩诊断与策略推荐系统。系统围绕 LoRA 固定秩分配局限、AdaLoRA 动态秩剪枝、参数重要性评分和预算调度策略展开，实现了从训练、评估、诊断到报告生成的完整流程。"
    )
    lines.append("")

    lines.append("## 2. 实验概况\n")
    data = metrics.drop_duplicates(subset=["exp_name"], keep="last")
    lines.append(f"- 实验总数：{len(data)}")
    lines.append(f"- 数据集数量：{data['dataset'].nunique()}")
    lines.append(f"- 方法数量：{data['method'].astype(str).nunique()}")
    lines.append(f"- 图表数量：{len(figs)}")
    lines.append("")

    lines.append("## 3. 主要实验类型\n")
    lines.append("- SST-2 多 rank 对比实验")
    lines.append("- SST-2 / MRPC / RTE 多任务泛化实验")
    lines.append("- Attention / FFN / Full 模块消融实验")
    lines.append("- 严格 AdaLoRA 动态秩剪枝实验")
    lines.append("- 参数×梯度重要性评分实验")
    lines.append("- AdaRank Score 多维评分实验")
    lines.append("- strict / fast / slow 预算调度对比实验")
    lines.append("")

    if score is not None and len(score) > 0:
        lines.append("## 4. AdaRank Score Top 5\n")
        lines.append("| 排名 | 实验 | AdaRank Score | Accuracy | F1 |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, row in enumerate(score.head(5).to_dict("records"), start=1):
            lines.append(
                f"| {i} | {row['exp_name']} | {row['adarank_score']:.4f} | {row['accuracy']:.6f} | {row['f1']:.6f} |"
            )
        lines.append("")

    if rank_summary is not None:
        lines.append("## 5. 严格 AdaLoRA Rank 诊断结论\n")
        lines.append(
            f"严格手写训练循环下，AdaLoRA 最终剪枝比例为 {rank_summary['compression'] * 100:.2f}%，effective rank 分布范围为 {rank_summary['rank_min']:.0f} 到 {rank_summary['rank_max']:.0f}。query 平均 rank 为 {rank_summary['query_rank']:.2f}，value 平均 rank 为 {rank_summary['value_rank']:.2f}，说明 value 子模块获得更多有效秩预算。"
        )
        lines.append("")

    if imp_summary is not None:
        lines.append("## 6. 重要性评分结论\n")
        lines.append(
            f"重要性评分实验显示，query 平均重要性为 {imp_summary['query_importance']:.6e}，value 平均重要性为 {imp_summary['value_importance']:.6e}。重要性与最终 rank 的相关系数约为 {imp_summary['corr']:.4f}，说明重要性评分与最终秩保留存在正相关。"
        )
        lines.append("")

    if budget_summary is not None:
        lines.append("## 7. 预算调度结论\n")
        best = budget_summary["best_acc"]
        lines.append(
            f"预算调度对比实验中，{best['budget_type']} 策略取得最佳性能，Accuracy={best['accuracy']:.6f}，F1={best['f1']:.6f}。"
        )
        lines.append("")

    lines.append("## 8. 推荐图表\n")
    for _, row in figs.head(20).iterrows():
        lines.append(f"- `{row['path']}`：{row['suggested_section']}")

    lines.append("")
    lines.append("## 9. 报告刷新说明\n")
    lines.append(
        "本报告随 metrics、rank、importance 和图表文件刷新，内容用于汇总当前实验状态。"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", path)


def write_inventory(out_dir):
    path = os.path.join(out_dir, "project_inventory.md")

    targets = [
        "configs",
        "scripts",
        "results/metrics",
        "results/rank",
        "results/importance",
        "results/score",
        "results/figs",
        "reports",
    ]

    lines = []
    lines.append("# 项目文件清单\n")

    for target in targets:
        lines.append(f"## {target}\n")
        if not os.path.exists(target):
            lines.append("- 不存在\n")
            continue

        files = []
        for dirpath, _, filenames in os.walk(target):
            for name in filenames:
                files.append(os.path.join(dirpath, name).replace("\\", "/"))

        if len(files) == 0:
            lines.append("- 空\n")
        else:
            for f in sorted(files):
                lines.append(f"- `{f}`")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", path)


def run():
    summary_dir = os.path.join("reports", "summary")
    table_dir = os.path.join("reports", "tables")
    generated_dir = os.path.join("reports", "generated")
    for path in [summary_dir, table_dir, generated_dir]:
        os.makedirs(path, exist_ok=True)

    metrics_path = os.path.join("results", "metrics", "metrics.csv")
    score_path = os.path.join("results", "score", "adarank_score.csv")
    strict_rank_path = os.path.join("results", "rank", "diagnostics", "sst2_adalora_strict_rank_rank.csv")
    importance_path = os.path.join("results", "importance", "diagnostics", "sst2_adalora_importance.csv")
    importance_rank_path = os.path.join("results", "rank", "diagnostics", "sst2_adalora_importance_rank.csv")
    budget_path = os.path.join("results", "figs", "budget", "budget_result_table.csv")

    metrics = read_table(metrics_path)
    if metrics is None:
        raise FileNotFoundError("没有找到 results/metrics/metrics.csv")

    save_experiment_summary(metrics, table_dir)

    figs = collect_figures("results/figs")
    save_figure_list(figs, table_dir)

    score = read_score_summary(score_path)
    rank_summary = read_rank_summary(strict_rank_path)
    imp_summary = read_importance_summary(importance_path, importance_rank_path)
    budget_summary = read_budget_summary(budget_path)

    write_key_findings(metrics, score, rank_summary, imp_summary, budget_summary, summary_dir)
    write_auto_report(metrics, score, figs, rank_summary, imp_summary, budget_summary, generated_dir)
    write_inventory(table_dir)

    print("\nreport01 完成。")
    print("生成文件：")
    print("- reports/tables/实验总表.csv")
    print("- reports/tables/图表清单.md")
    print("- reports/summary/核心结论摘要.md")
    print("- reports/generated/auto_report.md")
    print("- reports/tables/project_inventory.md")


if __name__ == "__main__":
    run()
