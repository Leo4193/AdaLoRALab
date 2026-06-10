import os
import math
import pandas as pd
import matplotlib.pyplot as plt


def clean_cols(df):
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_metrics(path):
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
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def read_rank_summary(path):
    # 读取严格 AdaLoRA rank 日志，并计算压缩率和 rank 差异性
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_cols(df)

    num_cols = [
        "step",
        "layer",
        "effective_rank",
        "zero_count",
        "max_rank",
        "mean_abs_lora_E",
        "max_abs_lora_E",
        "min_abs_lora_E",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "param_name", "effective_rank", "zero_count", "max_rank"])

    # 去除重复记录
    df = df.drop_duplicates(subset=["step", "param_name"], keep="last")

    final_step = int(df["step"].max())
    final_df = df[df["step"] == final_step].copy()
    final_df = final_df.drop_duplicates(subset=["param_name"], keep="last")

    total_rank_slots = float(final_df["max_rank"].sum())
    zero_count = float(final_df["zero_count"].sum())
    kept_rank = float(final_df["effective_rank"].sum())

    compression_ratio = zero_count / total_rank_slots if total_rank_slots > 0 else 0.0

    rank_min = float(final_df["effective_rank"].min())
    rank_max = float(final_df["effective_rank"].max())
    rank_range = rank_max - rank_min

    rank_std = float(final_df["effective_rank"].std(ddof=0))
    max_rank = float(final_df["max_rank"].max())

    # 归一化的 rank 差异性，越大说明越不是平均分配
    rank_diversity = rank_std / max_rank if max_rank > 0 else 0.0

    query_mean = final_df[final_df["module_name"] == "query"]["effective_rank"].mean()
    value_mean = final_df[final_df["module_name"] == "value"]["effective_rank"].mean()

    return {
        "final_step": final_step,
        "module_count": int(len(final_df)),
        "total_rank_slots": total_rank_slots,
        "kept_rank": kept_rank,
        "zero_count": zero_count,
        "compression_ratio": compression_ratio,
        "rank_min": rank_min,
        "rank_max": rank_max,
        "rank_range": rank_range,
        "rank_std": rank_std,
        "rank_diversity": rank_diversity,
        "query_mean_rank": float(query_mean),
        "value_mean_rank": float(value_mean),
    }


def select_score_rows(df):
    # 选入最终报告中有代表性的实验，不把最小调试实验纳入总评分
    keep_names = [
        # SST-2 多 rank
        "sst2_lora_r2_e3",
        "sst2_lora_r4_e3",
        "sst2_lora_r8_e3",
        "sst2_lora_r16_e3",
        "sst2_adalora_r2_e3",
        "sst2_adalora_r4_e3",
        "sst2_adalora_r8_e3",
        "sst2_adalora_r16_e3",

        # 多任务
        "mrpc_lora_r4_e3",
        "mrpc_adalora_r4_e3",
        "rte_lora_r4_e3",
        "rte_adalora_r4_e3",

        # 模块消融
        "sst2_attn_lora_r4_e3",
        "sst2_ffn_lora_r4_e3",
        "sst2_full_lora_r4_e3",

        # 严格 AdaLoRA 动态 rank
        "sst2_adalora_strict_rank",
    ]

    data = df[df["exp_name"].isin(keep_names)].copy()

    # 去重，避免重复运行同一实验时重复计分
    data = data.drop_duplicates(subset=["exp_name"], keep="last")

    return data


def minmax_benefit(series):
    # 越大越好
    vals = pd.to_numeric(series, errors="coerce")
    mn = vals.min()
    mx = vals.max()

    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series([50.0] * len(vals), index=series.index)

    return (vals - mn) / (mx - mn) * 100


def minmax_cost(series):
    # 越小越好
    vals = pd.to_numeric(series, errors="coerce")
    mn = vals.min()
    mx = vals.max()

    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series([50.0] * len(vals), index=series.index)

    return (mx - vals) / (mx - mn) * 100


def add_rank_scores(data, strict_summary):
    data["rank_compression_score"] = 0.0
    data["rank_diversity_score"] = 0.0
    data["explainability_score"] = 30.0

    # LoRA 有性能和效率，但没有动态 rank 诊断
    data.loc[data["method"].astype(str).str.contains("lora", case=False, na=False), "explainability_score"] = 40.0

    # 普通 AdaLoRA 有一定可解释性，但如果没有严格 rank 日志，不给满
    data.loc[data["method"].astype(str).str.contains("adalora", case=False, na=False), "explainability_score"] = 60.0

    # 严格 AdaLoRA 有完整 rank 日志和剪枝曲线
    strict_name = "sst2_adalora_strict_rank"

    if strict_summary is not None and strict_name in set(data["exp_name"]):
        idx = data["exp_name"] == strict_name

        compression = strict_summary["compression_ratio"]
        diversity = strict_summary["rank_diversity"]

        # 压缩率直接映射到 0-100
        data.loc[idx, "rank_compression_score"] = compression * 100

        # rank_diversity 通常较小，这里按 0.25 作为较强差异性参考上界
        data.loc[idx, "rank_diversity_score"] = min(diversity / 0.25 * 100, 100)

        data.loc[idx, "explainability_score"] = 100.0

    return data


def compute_scores(data, strict_summary):
    data = data.copy()

    # 性能分：Accuracy 与 F1 均值
    data["performance_score"] = (data["accuracy"] * 0.5 + data["f1"] * 0.5) * 100

    # 参数效率分：可训练参数越少越好
    data["param_efficiency_score"] = minmax_cost(data["trainable_params"])

    # 训练效率分：训练时间越短越好
    data["time_efficiency_score"] = minmax_cost(data["train_time_sec"])

    data = add_rank_scores(data, strict_summary)

    data["adarank_score"] = (
        0.35 * data["performance_score"]
        + 0.20 * data["param_efficiency_score"]
        + 0.10 * data["time_efficiency_score"]
        + 0.15 * data["rank_compression_score"]
        + 0.10 * data["rank_diversity_score"]
        + 0.10 * data["explainability_score"]
    )

    score_cols = [
        "performance_score",
        "param_efficiency_score",
        "time_efficiency_score",
        "rank_compression_score",
        "rank_diversity_score",
        "explainability_score",
        "adarank_score",
    ]

    for col in score_cols:
        data[col] = data[col].round(4)

    return data


def save_score_table(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cols = [
        "exp_name",
        "dataset",
        "method",
        "rank",
        "accuracy",
        "f1",
        "trainable_params",
        "train_time_sec",
        "performance_score",
        "param_efficiency_score",
        "time_efficiency_score",
        "rank_compression_score",
        "rank_diversity_score",
        "explainability_score",
        "adarank_score",
    ]

    table = data[cols].sort_values("adarank_score", ascending=False)

    path = os.path.join(out_dir, "adarank_score.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)

    return table


def show_score_bar(table, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    top = table.head(12).copy()
    labels = top["exp_name"].tolist()

    plt.figure(figsize=(11, 6))
    plt.bar(labels, top["adarank_score"])

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Experiment")
    plt.ylabel("AdaRank Score")
    plt.title("AdaRank Score Ranking")
    plt.tight_layout()

    path = os.path.join(fig_dir, "adarank_score_bar.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_dimension_bar(table, fig_dir):
    top = table.head(8).copy()

    score_cols = [
        "performance_score",
        "param_efficiency_score",
        "time_efficiency_score",
        "rank_compression_score",
        "rank_diversity_score",
        "explainability_score",
    ]

    x = list(range(len(top)))
    bottom = [0.0] * len(top)

    plt.figure(figsize=(12, 6))

    for col in score_cols:
        vals = top[col].tolist()
        plt.bar(x, vals, bottom=bottom, label=col.replace("_score", ""))
        bottom = [bottom[i] + vals[i] for i in range(len(vals))]

    plt.xticks(x, top["exp_name"], rotation=45, ha="right")
    plt.xlabel("Experiment")
    plt.ylabel("Dimension Score Sum")
    plt.title("AdaRank Dimension Scores")
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(fig_dir, "adarank_dimension_bar.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_radar(table, fig_dir):
    # 选取代表性实验画雷达图
    candidates = [
        "sst2_lora_r4_e3",
        "sst2_lora_r8_e3",
        "sst2_full_lora_r4_e3",
        "sst2_adalora_strict_rank",
    ]

    data = table[table["exp_name"].isin(candidates)].copy()

    if len(data) == 0:
        print("没有找到雷达图候选实验，跳过。")
        return

    labels = [
        "Performance",
        "ParamEff",
        "TimeEff",
        "Compression",
        "Diversity",
        "Explainability",
    ]

    score_cols = [
        "performance_score",
        "param_efficiency_score",
        "time_efficiency_score",
        "rank_compression_score",
        "rank_diversity_score",
        "explainability_score",
    ]

    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for _, row in data.iterrows():
        vals = [float(row[col]) for col in score_cols]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, label=row["exp_name"])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title("AdaRank Score Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    path = os.path.join(fig_dir, "adarank_score_radar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def write_summary(table, strict_summary, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    best = table.iloc[0]
    lines = []

    lines.append("# AdaRank Score 综合评价报告\n")
    lines.append("## 1. 综合评分设计\n")
    lines.append(
        "AdaRank Score 从任务性能、参数效率、训练效率、秩压缩效果、秩分配差异性和可解释性完整度六个维度评价不同微调实验。"
    )
    lines.append("")
    lines.append("权重设置如下：性能 35%，参数效率 20%，训练效率 10%，秩压缩 15%，秩分配差异性 10%，可解释性 10%。")
    lines.append("")

    lines.append("## 2. 综合排名第一实验\n")
    lines.append(f"- 实验名称：{best['exp_name']}")
    lines.append(f"- 数据集：{best['dataset']}")
    lines.append(f"- 方法：{best['method']}")
    lines.append(f"- AdaRank Score：{best['adarank_score']}")
    lines.append(f"- Accuracy：{best['accuracy']}")
    lines.append(f"- F1：{best['f1']}")
    lines.append(f"- 可训练参数量：{best['trainable_params']}")
    lines.append("")

    if strict_summary is not None:
        lines.append("## 3. 严格 AdaLoRA 动态秩诊断摘要\n")
        lines.append(f"- 最终 step：{strict_summary['final_step']}")
        lines.append(f"- 诊断模块数：{strict_summary['module_count']}")
        lines.append(f"- 初始 rank 方向数：{int(strict_summary['total_rank_slots'])}")
        lines.append(f"- 最终保留方向数：{int(strict_summary['kept_rank'])}")
        lines.append(f"- 最终置零方向数：{int(strict_summary['zero_count'])}")
        lines.append(f"- 剪枝比例：{strict_summary['compression_ratio'] * 100:.2f}%")
        lines.append(f"- effective rank 范围：{strict_summary['rank_min']:.0f} - {strict_summary['rank_max']:.0f}")
        lines.append(f"- query 平均 rank：{strict_summary['query_mean_rank']:.2f}")
        lines.append(f"- value 平均 rank：{strict_summary['value_mean_rank']:.2f}")
        lines.append("")

    lines.append("## 4. 结果解释\n")
    lines.append(
        "该评分体系并不只比较单一 Accuracy，而是将参数效率、训练成本和动态秩可解释性纳入统一评价。"
    )
    lines.append(
        "因此，LoRA 类实验可能在任务性能和训练效率上占优，而严格 AdaLoRA 实验则在秩压缩、秩分配差异性和可解释性方面获得更高分。"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", out_path)


def run():
    metric_path = os.path.join("results", "metrics", "metrics.csv")
    strict_rank_path = os.path.join("results", "rank", "diagnostics", "sst2_adalora_strict_rank_rank.csv")

    score_dir = os.path.join("results", "score")
    fig_dir = os.path.join("results", "figs", "score")
    summary_path = os.path.join("reports", "summary", "adarank_score_summary.md")

    df = read_metrics(metric_path)
    data = select_score_rows(df)

    if len(data) == 0:
        raise ValueError("没有筛选到可评分实验。")

    strict_summary = read_rank_summary(strict_rank_path)

    if strict_summary is not None:
        print("严格 AdaLoRA rank 摘要：")
        for k, v in strict_summary.items():
            print(k, ":", v)
    else:
        print("没有找到严格 AdaLoRA rank 日志，rank 相关分数将为 0。")

    scored = compute_scores(data, strict_summary)
    table = save_score_table(scored, score_dir)

    print("\nAdaRank Score 排名前 10：")
    print(table[["exp_name", "adarank_score", "performance_score", "param_efficiency_score", "rank_compression_score", "rank_diversity_score", "explainability_score"]].head(10))

    show_score_bar(table, fig_dir)
    show_dimension_bar(table, fig_dir)
    show_radar(table, fig_dir)
    write_summary(table, strict_summary, summary_path)

    print("\nscore01 完成。")


if __name__ == "__main__":
    run()
