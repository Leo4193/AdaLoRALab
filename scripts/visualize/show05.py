import os
import pandas as pd
import matplotlib.pyplot as plt


def read_rank(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

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

    df = df.dropna(subset=["step", "layer", "effective_rank"])
    df["step"] = df["step"].astype(int)
    df["layer"] = df["layer"].astype(int)

    return df


def save_final_table(final_df, out_dir):
    path = os.path.join(out_dir, "strict_final_rank_table.csv")
    final_df.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)


def show_rank_distribution(final_df, out_dir):
    counts = final_df["effective_rank"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index.astype(str), counts.values)

    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    plt.xlabel("Effective Rank")
    plt.ylabel("Number of Modules")
    plt.title("Final Effective Rank Distribution")
    plt.tight_layout()

    path = os.path.join(out_dir, "strict_rank_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def show_query_value_rank(final_df, out_dir):
    data = (
        final_df.groupby("module_name")["effective_rank"]
        .mean()
        .reindex(["query", "value"])
        .reset_index()
    )

    plt.figure(figsize=(6, 5))
    plt.bar(data["module_name"], data["effective_rank"])

    for i, v in enumerate(data["effective_rank"]):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Attention Submodule")
    plt.ylabel("Average Effective Rank")
    plt.title("Average Effective Rank: Query vs Value")
    plt.tight_layout()

    path = os.path.join(out_dir, "strict_query_value_rank.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def show_rank_heatmap(final_df, out_dir):
    pivot = final_df.pivot_table(
        index="layer",
        columns="module_name",
        values="effective_rank",
        aggfunc="mean",
    )

    pivot = pivot.reindex(columns=["query", "value"])
    pivot = pivot.sort_index()

    plt.figure(figsize=(6, 6))
    im = plt.imshow(pivot.values, aspect="auto")

    plt.colorbar(im, label="Effective Rank")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), [f"L{int(i)}" for i in pivot.index])

    plt.xlabel("Submodule")
    plt.ylabel("Layer")
    plt.title("Strict AdaLoRA Effective Rank Heatmap")
    plt.tight_layout()

    path = os.path.join(out_dir, "strict_rank_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def show_rank_change_curve(df, out_dir):
    data = (
        df.groupby(["step", "module_name"])["effective_rank"]
        .mean()
        .reset_index()
        .sort_values("step")
    )

    plt.figure(figsize=(8, 5))

    for name in ["query", "value"]:
        sub = data[data["module_name"] == name]
        plt.plot(
            sub["step"],
            sub["effective_rank"],
            marker="o",
            label=name,
        )

    plt.xlabel("Training Step")
    plt.ylabel("Average Effective Rank")
    plt.title("AdaLoRA Effective Rank Evolution")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "strict_rank_change_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def show_zero_count_curve(df, out_dir):
    data = (
        df.groupby("step")["zero_count"]
        .sum()
        .reset_index()
        .sort_values("step")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(data["step"], data["zero_count"], marker="o")

    plt.xlabel("Training Step")
    plt.ylabel("Total Zero Count")
    plt.title("Zeroed Singular Directions during Training")
    plt.tight_layout()

    path = os.path.join(out_dir, "strict_zero_count_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def show_layer_rank(final_df, out_dir):
    data = (
        final_df.groupby("layer")["effective_rank"]
        .mean()
        .reset_index()
        .sort_values("layer")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(data["layer"], data["effective_rank"])

    plt.xlabel("Layer")
    plt.ylabel("Average Effective Rank")
    plt.title("Average Effective Rank by Layer")
    plt.xticks(data["layer"])
    plt.tight_layout()

    path = os.path.join(out_dir, "strict_layer_rank.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("已保存：", path)


def run():
    rank_path = os.path.join(
        "results",
        "rank",
        "diagnostics",
        "sst2_adalora_strict_rank_rank.csv",
    )

    out_dir = os.path.join("results", "figs", "diagnostics", "strict_rank")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(rank_path):
        raise FileNotFoundError(f"找不到 Rank 文件：{rank_path}")

    df = read_rank(rank_path)

    # 去重：同一个 step、同一个参数名只保留最后一次
    df = df.drop_duplicates(
        subset=["step", "param_name"],
        keep="last",
    ).copy()

    final_step = df["step"].max()
    final_df = df[df["step"] == final_step].copy()

    # 最终 step 内部再次去重，确保只有 24 个模块
    final_df = final_df.drop_duplicates(
        subset=["param_name"],
        keep="last",
    ).copy()

    print("严格 AdaLoRA rank 日志读取成功")
    print("总行数：", len(df))
    print("记录 step 数：", df["step"].nunique())
    print("最终 step：", final_step)
    print("最终记录数：", len(final_df))

    print("\n最终 effective_rank 分布：")
    print(final_df["effective_rank"].value_counts().sort_index())

    print("\nquery/value 平均 rank：")
    print(final_df.groupby("module_name")["effective_rank"].mean())

    print("\n最终 zero_count 总和：", int(final_df["zero_count"].sum()))

    save_final_table(final_df, out_dir)
    show_rank_distribution(final_df, out_dir)
    show_query_value_rank(final_df, out_dir)
    show_rank_heatmap(final_df, out_dir)
    show_rank_change_curve(df, out_dir)
    show_zero_count_curve(df, out_dir)
    show_layer_rank(final_df, out_dir)

    print("\nshow05 完成。")


if __name__ == "__main__":
    run()
