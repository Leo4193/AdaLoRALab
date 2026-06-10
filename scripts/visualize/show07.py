import os
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


def read_rank(path):
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

    df = df.dropna(subset=["step", "param_name", "effective_rank"])
    df["step"] = df["step"].astype(int)
    df["layer"] = df["layer"].astype(int)

    # 去重，避免最终 step 被记录两次
    df = df.drop_duplicates(subset=["step", "param_name"], keep="last").copy()

    return df


def get_final_rank(rank_df):
    final_step = rank_df["step"].max()
    final_df = rank_df[rank_df["step"] == final_step].copy()
    final_df = final_df.drop_duplicates(subset=["param_name"], keep="last").copy()
    return final_step, final_df


def collect_budget_data():
    items = [
        {
            "name": "strict",
            "exp_name": "sst2_adalora_strict_rank",
            "rank_path": os.path.join("results", "rank", "diagnostics", "sst2_adalora_strict_rank_rank.csv"),
        },
        {
            "name": "fast",
            "exp_name": "sst2_adalora_budget_fast",
            "rank_path": os.path.join("results", "rank", "budget", "sst2_adalora_budget_fast_rank.csv"),
        },
        {
            "name": "slow",
            "exp_name": "sst2_adalora_budget_slow",
            "rank_path": os.path.join("results", "rank", "budget", "sst2_adalora_budget_slow_rank.csv"),
        },
    ]

    metric_path = os.path.join("results", "metrics", "metrics.csv")
    metrics = read_metrics(metric_path)

    rows = []
    rank_logs = {}

    for item in items:
        exp_name = item["exp_name"]
        sub = metrics[metrics["exp_name"] == exp_name].copy()

        if len(sub) == 0:
            print("缺少 metrics 结果：", exp_name)
            continue

        # 如果重复运行过同一个实验，取最后一条
        m = sub.iloc[-1].to_dict()

        if not os.path.exists(item["rank_path"]):
            print("缺少 rank 文件：", item["rank_path"])
            continue

        rank_df = read_rank(item["rank_path"])
        final_step, final_rank = get_final_rank(rank_df)

        total_slots = float(final_rank["max_rank"].sum())
        zero_count = float(final_rank["zero_count"].sum())
        kept_rank = float(final_rank["effective_rank"].sum())
        compression_ratio = zero_count / total_slots if total_slots > 0 else 0.0
        rank_std = float(final_rank["effective_rank"].std(ddof=0))
        rank_min = float(final_rank["effective_rank"].min())
        rank_max = float(final_rank["effective_rank"].max())

        q_rank = final_rank[final_rank["module_name"] == "query"]["effective_rank"].mean()
        v_rank = final_rank[final_rank["module_name"] == "value"]["effective_rank"].mean()

        rows.append({
            "budget_type": item["name"],
            "exp_name": exp_name,
            "accuracy": float(m["accuracy"]),
            "f1": float(m["f1"]),
            "eval_loss": float(m["eval_loss"]),
            "train_time_sec": float(m["train_time_sec"]),
            "final_step": int(final_step),
            "total_rank_slots": total_slots,
            "kept_rank": kept_rank,
            "zero_count": zero_count,
            "compression_ratio": compression_ratio,
            "rank_min": rank_min,
            "rank_max": rank_max,
            "rank_std": rank_std,
            "query_mean_rank": float(q_rank),
            "value_mean_rank": float(v_rank),
        })

        rank_logs[item["name"]] = rank_df

    return pd.DataFrame(rows), rank_logs


def save_table(data, out_dir):
    path = os.path.join(out_dir, "budget_result_table.csv")
    data.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)


def show_accuracy_f1(data, out_dir):
    x = range(len(data))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], data["accuracy"], width=width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], data["f1"], width=width, label="F1")

    plt.xticks(list(x), data["budget_type"])
    plt.xlabel("Budget Schedule")
    plt.ylabel("Score")
    plt.title("Budget Schedule: Accuracy and F1")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "budget_accuracy_f1.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def show_zero_count_curve(rank_logs, out_dir):
    plt.figure(figsize=(8, 5))

    for name, df in rank_logs.items():
        data = (
            df.groupby("step")["zero_count"]
            .sum()
            .reset_index()
            .sort_values("step")
        )

        plt.plot(data["step"], data["zero_count"], marker="o", label=name)

    plt.xlabel("Training Step")
    plt.ylabel("Total Zero Count")
    plt.title("Budget Schedule: Zeroed Rank Directions")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "budget_zero_count_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def show_rank_distribution(rank_logs, out_dir):
    for name, df in rank_logs.items():
        _, final_df = get_final_rank(df)
        counts = final_df["effective_rank"].value_counts().sort_index()

        plt.figure(figsize=(7, 5))
        plt.bar(counts.index.astype(str), counts.values)

        for i, v in enumerate(counts.values):
            plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

        plt.xlabel("Effective Rank")
        plt.ylabel("Number of Modules")
        plt.title(f"Final Rank Distribution: {name}")
        plt.tight_layout()

        path = os.path.join(out_dir, f"budget_rank_distribution_{name}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print("已保存：", path)


def show_query_value_rank(data, out_dir):
    x = range(len(data))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], data["query_mean_rank"], width=width, label="query")
    plt.bar([i + width / 2 for i in x], data["value_mean_rank"], width=width, label="value")

    plt.xticks(list(x), data["budget_type"])
    plt.xlabel("Budget Schedule")
    plt.ylabel("Average Effective Rank")
    plt.title("Query / Value Rank under Different Budgets")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "budget_query_value_rank.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def write_summary(data, out_dir):
    path = os.path.join(out_dir, "budget_summary.md")

    lines = []
    lines.append("# AdaLoRA 预算调度对比摘要\n")

    if len(data) == 0:
        lines.append("没有可用预算调度结果。")
    else:
        best_acc = data.sort_values("accuracy", ascending=False).iloc[0]
        best_f1 = data.sort_values("f1", ascending=False).iloc[0]
        best_compress = data.sort_values("compression_ratio", ascending=False).iloc[0]

        lines.append("## 1. 最佳 Accuracy\n")
        lines.append(f"- {best_acc['budget_type']}: Accuracy={best_acc['accuracy']:.6f}, F1={best_acc['f1']:.6f}\n")

        lines.append("## 2. 最佳 F1\n")
        lines.append(f"- {best_f1['budget_type']}: Accuracy={best_f1['accuracy']:.6f}, F1={best_f1['f1']:.6f}\n")

        lines.append("## 3. 最大压缩率\n")
        lines.append(f"- {best_compress['budget_type']}: compression={best_compress['compression_ratio'] * 100:.2f}%\n")

        lines.append("## 4. 全部结果\n")
        for _, row in data.iterrows():
            lines.append(
                f"- {row['budget_type']}: acc={row['accuracy']:.6f}, "
                f"f1={row['f1']:.6f}, zero={int(row['zero_count'])}, "
                f"compression={row['compression_ratio'] * 100:.2f}%, "
                f"query_rank={row['query_mean_rank']:.2f}, "
                f"value_rank={row['value_mean_rank']:.2f}"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", path)


def run():
    out_dir = os.path.join("results", "figs", "budget")
    os.makedirs(out_dir, exist_ok=True)

    data, rank_logs = collect_budget_data()

    if len(data) == 0:
        raise ValueError("没有找到可用的预算调度实验结果。")

    order = {"strict": 0, "fast": 1, "slow": 2}
    data["order"] = data["budget_type"].map(order)
    data = data.sort_values("order").drop(columns=["order"])

    print("预算调度对比数据：")
    print(data)

    save_table(data, out_dir)
    show_accuracy_f1(data, out_dir)
    show_zero_count_curve(rank_logs, out_dir)
    show_rank_distribution(rank_logs, out_dir)
    show_query_value_rank(data, out_dir)
    write_summary(data, out_dir)

    print("\nshow07 完成。")


if __name__ == "__main__":
    run()
