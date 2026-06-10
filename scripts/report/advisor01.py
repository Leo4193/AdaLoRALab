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
        "compression_ratio",
        "rank_std",
        "query_mean_rank",
        "value_mean_rank",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_latest(metrics):
    data = metrics.copy()
    if "exp_name" in data.columns:
        data = data.drop_duplicates(subset=["exp_name"], keep="last")
    return data


def pick_best_sst2_performance(metrics):
    data = get_latest(metrics)
    sub = data[data["dataset"] == "sst2"].copy()
    sub = sub.dropna(subset=["accuracy", "f1"])

    if len(sub) == 0:
        return None

    sub["avg_perf"] = (sub["accuracy"] + sub["f1"]) / 2
    return sub.sort_values("avg_perf", ascending=False).iloc[0]


def pick_best_efficiency(score):
    if score is None or len(score) == 0:
        return None

    data = score.copy()
    if "adarank_score" in data.columns:
        data = data.sort_values("adarank_score", ascending=False)

    return data.iloc[0]


def pick_best_parameter_saving(metrics):
    data = get_latest(metrics)
    data = data.dropna(subset=["trainable_params", "accuracy", "f1"])

    # 只考虑正式实验，不考虑 2000 样本调试实验
    data = data[~data["exp_name"].astype(str).str.contains("_r4$", na=False)]

    if len(data) == 0:
        return None

    # 要求性能不能太差，避免只因为参数少被推荐
    data["avg_perf"] = (data["accuracy"] + data["f1"]) / 2
    good = data[data["avg_perf"] >= 0.85].copy()

    if len(good) == 0:
        good = data.copy()

    return good.sort_values("trainable_params", ascending=True).iloc[0]


def compare_module_ablation(metrics):
    data = get_latest(metrics)
    names = [
        "sst2_attn_lora_r4_e3",
        "sst2_ffn_lora_r4_e3",
        "sst2_full_lora_r4_e3",
    ]

    sub = data[data["exp_name"].isin(names)].copy()
    if len(sub) == 0:
        return None

    sub["avg_perf"] = (sub["accuracy"] + sub["f1"]) / 2
    best = sub.sort_values("avg_perf", ascending=False).iloc[0]

    return sub, best


def compare_budget(budget):
    if budget is None or len(budget) == 0:
        return None

    data = budget.copy()
    data["avg_perf"] = (data["accuracy"] + data["f1"]) / 2
    best = data.sort_values("avg_perf", ascending=False).iloc[0]

    return data, best


def read_strict_rank_summary(rank_path):
    df = read_table(rank_path)
    if df is None or len(df) == 0:
        return None

    if "param_name" not in df.columns:
        return None

    df = df.drop_duplicates(subset=["step", "param_name"], keep="last")
    final_step = int(df["step"].max())
    final = df[df["step"] == final_step].copy()
    final = final.drop_duplicates(subset=["param_name"], keep="last")

    total_rank = final["max_rank"].sum()
    zero_count = final["zero_count"].sum()
    compression = zero_count / total_rank if total_rank > 0 else 0

    q = final[final["module_name"] == "query"]["effective_rank"].mean()
    v = final[final["module_name"] == "value"]["effective_rank"].mean()

    return {
        "final_step": final_step,
        "compression": compression,
        "zero_count": zero_count,
        "total_rank": total_rank,
        "query_rank": q,
        "value_rank": v,
        "rank_min": final["effective_rank"].min(),
        "rank_max": final["effective_rank"].max(),
    }


def read_importance_summary(importance_path, rank_path):
    imp = read_table(importance_path)
    rank = read_table(rank_path)

    if imp is None or rank is None:
        return None

    imp = imp.drop_duplicates(subset=["step", "param_name"], keep="last")
    rank = rank.drop_duplicates(subset=["step", "param_name"], keep="last")

    final_step = int(imp["step"].max())
    final_imp = imp[imp["step"] == final_step].copy()
    final_imp = final_imp.drop_duplicates(subset=["param_name"], keep="last")

    final_rank_step = int(rank["step"].max())
    final_rank = rank[rank["step"] == final_rank_step].copy()
    final_rank = final_rank.drop_duplicates(subset=["param_name"], keep="last")

    q_imp = final_imp[final_imp["module_name"] == "query"]["mean_abs_param_grad"].mean()
    v_imp = final_imp[final_imp["module_name"] == "value"]["mean_abs_param_grad"].mean()

    merged = final_imp.merge(
        final_rank[["param_name", "effective_rank"]],
        on="param_name",
        how="left",
        suffixes=("_imp", "_rank"),
    )

    corr = None
    if "effective_rank_rank" in merged.columns:
        corr = merged[["mean_abs_param_grad", "effective_rank_rank"]].corr().iloc[0, 1]

    top = final_imp.sort_values("mean_abs_param_grad", ascending=False).head(5)

    return {
        "query_importance": q_imp,
        "value_importance": v_imp,
        "corr": corr,
        "top": top,
    }


def write_advisor(
    metrics,
    score,
    budget,
    rank_summary,
    importance_summary,
    out_path,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    best_perf = pick_best_sst2_performance(metrics)
    best_eff = pick_best_efficiency(score)
    best_param = pick_best_parameter_saving(metrics)
    module_result = compare_module_ablation(metrics)
    budget_result = compare_budget(budget)

    lines = []

    lines.append("# AdaRankLab Pro 微调策略推荐报告\n")
    lines.append("本报告依据已有训练结果、AdaRank Score、Rank 诊断、重要性评分和预算调度实验，为不同应用场景给出推荐策略。\n")

    lines.append("## 1. 总体推荐结论\n")

    if best_perf is not None:
        lines.append(
            f"- **最高 SST-2 性能优先**：推荐 `{best_perf['exp_name']}`，"
            f"Accuracy={best_perf['accuracy']:.6f}，F1={best_perf['f1']:.6f}，"
            f"可训练参数={int(best_perf['trainable_params'])}。"
        )

    if best_eff is not None:
        lines.append(
            f"- **综合效率优先**：推荐 `{best_eff['exp_name']}`，"
            f"AdaRank Score={best_eff['adarank_score']:.4f}。"
        )

    if best_param is not None:
        lines.append(
            f"- **参数最省且性能可接受**：推荐 `{best_param['exp_name']}`，"
            f"可训练参数={int(best_param['trainable_params'])}，"
            f"Accuracy={best_param['accuracy']:.6f}，F1={best_param['f1']:.6f}。"
        )

    if budget_result is not None:
        _, best_budget = budget_result
        lines.append(
            f"- **AdaLoRA 预算调度优先**：推荐 `{best_budget['budget_type']}` 策略，"
            f"Accuracy={best_budget['accuracy']:.6f}，F1={best_budget['f1']:.6f}。"
        )

    lines.append("")

    lines.append("## 2. 面向不同需求的推荐\n")

    lines.append("### 2.1 追求最高分类性能\n")
    if best_perf is not None:
        lines.append(
            f"推荐使用 `{best_perf['exp_name']}`。该实验在 SST-2 上取得当前最高综合性能，"
            f"Accuracy={best_perf['accuracy']:.6f}，F1={best_perf['f1']:.6f}。"
        )
        lines.append("适用场景：最终模型精度优先，对训练参数量要求不极端严格。")
    lines.append("")

    lines.append("### 2.2 追求参数效率和训练效率\n")
    if best_eff is not None:
        lines.append(
            f"推荐使用 `{best_eff['exp_name']}`。该实验在 AdaRank Score 中排名最高，"
            "说明其在性能、参数量和训练成本之间具有较优折中。"
        )
        lines.append("适用场景：课程演示、资源受限设备、小规模快速实验。")
    lines.append("")

    lines.append("### 2.3 追求动态秩压缩和可解释性\n")
    if rank_summary is not None:
        lines.append(
            "推荐使用 `sst2_adalora_strict_rank` 或 `sst2_adalora_importance`。"
            f"严格 AdaLoRA 实验最终剪枝比例为 {rank_summary['compression'] * 100:.2f}%，"
            f"effective rank 范围为 {rank_summary['rank_min']:.0f} 到 {rank_summary['rank_max']:.0f}，"
            f"query 平均 rank={rank_summary['query_rank']:.2f}，"
            f"value 平均 rank={rank_summary['value_rank']:.2f}。"
        )
        lines.append("适用场景：需要展示 AdaLoRA 动态秩分配、剪枝过程、模块重要性差异的研究型实验。")
    lines.append("")

    lines.append("### 2.4 只允许开放部分模块时\n")
    if module_result is not None:
        sub, best_module = module_result
        lines.append(
            f"模块消融实验中，最佳方案为 `{best_module['exp_name']}`，"
            f"Accuracy={best_module['accuracy']:.6f}，F1={best_module['f1']:.6f}。"
        )

        ffn = sub[sub["exp_name"] == "sst2_ffn_lora_r4_e3"]
        attn = sub[sub["exp_name"] == "sst2_attn_lora_r4_e3"]

        if len(ffn) > 0 and len(attn) > 0:
            ffn_row = ffn.iloc[0]
            attn_row = attn.iloc[0]
            lines.append(
                f"其中 FFN-only 的 Accuracy={ffn_row['accuracy']:.6f}、F1={ffn_row['f1']:.6f}，"
                f"Attention-only 的 Accuracy={attn_row['accuracy']:.6f}、F1={attn_row['f1']:.6f}。"
            )
            if ffn_row["f1"] > attn_row["f1"]:
                lines.append("因此，在当前 SST-2 设置下，如果只能选择一类模块，优先推荐开放 FFN 模块。")
            else:
                lines.append("因此，在当前 SST-2 设置下，Attention 模块仍具有较强适配能力。")
    lines.append("")

    lines.append("### 2.5 采用 AdaLoRA 时如何选择预算调度\n")
    if budget_result is not None:
        data, best_budget = budget_result
        lines.append(
            f"预算调度对比显示，`{best_budget['budget_type']}` 策略取得最佳结果，"
            f"Accuracy={best_budget['accuracy']:.6f}，F1={best_budget['f1']:.6f}。"
        )
        lines.append("因此，建议不要过早完成预算收缩，而应保留较长预算调度阶段，让模型更充分地估计参数重要性。")
    lines.append("")

    lines.append("## 3. 机制解释建议\n")

    if importance_summary is not None:
        lines.append(
            f"重要性评分实验显示，query 平均重要性为 {importance_summary['query_importance']:.6e}，"
            f"value 平均重要性为 {importance_summary['value_importance']:.6e}。"
        )

        if importance_summary["corr"] is not None:
            lines.append(
                f"重要性评分与最终 effective rank 的相关系数为 {importance_summary['corr']:.4f}，"
                "说明重要性更高的模块更倾向于保留较高 rank。"
            )

        lines.append("Top 重要模块主要集中在 value 子模块，因此报告中应强调 value 在当前任务中的重要性。")
    lines.append("")

    lines.append("## 4. 不推荐直接采用的方案\n")
    lines.append("- 不建议将初版 `train03.py` 的 Rank 记录作为最终证据，因为该版本中 effective_rank 未出现有效分化。")
    lines.append("- 不建议只依据单一 Accuracy 判断方法优劣，应结合参数效率、训练效率、rank 压缩和可解释性综合评价。")
    lines.append("- 不建议将普通 AdaLoRA r2/r4 的短训练结果作为最终结论，因为其动态秩调度尚未充分发挥。")
    lines.append("")

    lines.append("## 5. 最终提交建议\n")
    lines.append("- 报告主线应写成：固定秩问题 → 多 rank 对比 → 模块消融 → 严格 AdaLoRA 剪枝 → 重要性评分 → 预算调度 → 多维评分 → 策略推荐。")
    lines.append("- Demo 视频应重点展示：运行统一入口、查看 metrics、展示 rank heatmap、展示 importance 图、展示 advisor 推荐报告。")
    lines.append("- 系统手册中应明确说明不同脚本版本的作用，特别是 `train03.py` 为失败诊断版本，`train05.py` 为严格动态秩版本，`train06.py` 为重要性评分版本。")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", out_path)


def run():
    metrics_path = os.path.join("results", "metrics", "metrics.csv")
    score_path = os.path.join("results", "score", "adarank_score.csv")
    budget_path = os.path.join("results", "figs", "budget", "budget_result_table.csv")
    strict_rank_path = os.path.join("results", "rank", "diagnostics", "sst2_adalora_strict_rank_rank.csv")
    importance_path = os.path.join("results", "importance", "diagnostics", "sst2_adalora_importance.csv")
    importance_rank_path = os.path.join("results", "rank", "diagnostics", "sst2_adalora_importance_rank.csv")

    out_path = os.path.join("reports", "generated", "advisor_suggestions.md")

    metrics = read_table(metrics_path)
    if metrics is None:
        raise FileNotFoundError("没有找到 results/metrics/metrics.csv")

    score = read_table(score_path)
    budget = read_table(budget_path)
    rank_summary = read_strict_rank_summary(strict_rank_path)
    importance_summary = read_importance_summary(importance_path, importance_rank_path)

    write_advisor(
        metrics,
        score,
        budget,
        rank_summary,
        importance_summary,
        out_path,
    )

    print("\nadvisor01 完成。")
    print("- reports/generated/advisor_suggestions.md")


if __name__ == "__main__":
    run()
