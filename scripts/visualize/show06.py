import os
import pandas as pd
import matplotlib.pyplot as plt


def clean_cols(df):
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_cols(df)

    num_cols = [
        "step",
        "layer",
        "effective_rank",
        "zero_count",
        "max_rank",
        "mean_abs_param",
        "mean_abs_grad",
        "mean_abs_param_grad",
        "max_abs_param_grad",
        "sum_abs_param_grad",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "layer", "param_name"])
    df["step"] = df["step"].astype(int)
    df["layer"] = df["layer"].astype(int)

    return df


def remove_duplicates(df):
    return df.drop_duplicates(
        subset=["step", "param_name"],
        keep="last",
    ).copy()


def get_final(df):
    final_step = df["step"].max()
    final_df = df[df["step"] == final_step].copy()
    final_df = final_df.drop_duplicates(subset=["param_name"], keep="last")
    return final_step, final_df


def read_rank(path):
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

    df = df.dropna(subset=["step", "layer", "param_name"])
    df["step"] = df["step"].astype(int)
    df["layer"] = df["layer"].astype(int)
    df = df.drop_duplicates(subset=["step", "param_name"], keep="last")

    final_step = df["step"].max()
    final_df = df[df["step"] == final_step].copy()
    final_df = final_df.drop_duplicates(subset=["param_name"], keep="last")

    return final_df


def save_final_table(final_df, out_dir):
    path = os.path.join(out_dir, "final_importance_table.csv")
    final_df.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)


def show_importance_heatmap(final_df, out_dir):
    pivot = final_df.pivot_table(
        index="layer",
        columns="module_name",
        values="mean_abs_param_grad",
        aggfunc="mean",
    )

    pivot = pivot.reindex(columns=["query", "value"])
    pivot = pivot.sort_index()

    plt.figure(figsize=(6, 6))
    im = plt.imshow(pivot.values, aspect="auto")

    plt.colorbar(im, label="Mean |param × grad|")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), [f"L{int(i)}" for i in pivot.index])

    plt.xlabel("Attention Submodule")
    plt.ylabel("Layer")
    plt.title("AdaLoRA Importance Heatmap")
    plt.tight_layout()

    path = os.path.join(out_dir, "importance_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_query_value_importance(final_df, out_dir):
    data = (
        final_df.groupby("module_name")["mean_abs_param_grad"]
        .mean()
        .reindex(["query", "value"])
        .reset_index()
    )

    plt.figure(figsize=(6, 5))
    plt.bar(data["module_name"], data["mean_abs_param_grad"])

    for i, v in enumerate(data["mean_abs_param_grad"]):
        plt.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Attention Submodule")
    plt.ylabel("Mean |param × grad|")
    plt.title("Average Importance: Query vs Value")
    plt.tight_layout()

    path = os.path.join(out_dir, "query_value_importance.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_importance_curve(df, out_dir):
    data = (
        df.groupby(["step", "module_name"])["mean_abs_param_grad"]
        .mean()
        .reset_index()
        .sort_values("step")
    )

    plt.figure(figsize=(8, 5))

    for name in ["query", "value"]:
        sub = data[data["module_name"] == name]
        plt.plot(
            sub["step"],
            sub["mean_abs_param_grad"],
            marker="o",
            label=name,
        )

    plt.xlabel("Training Step")
    plt.ylabel("Mean |param × grad|")
    plt.title("Importance Evolution during Training")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "importance_change_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_importance_rank_scatter(final_importance, final_rank, out_dir):
    if final_rank is None:
        print("没有 rank 文件，跳过 importance-rank 散点图。")
        return

    rank_cols = ["param_name", "effective_rank", "zero_count"]
    merged = final_importance.merge(
        final_rank[rank_cols],
        on="param_name",
        how="left",
        suffixes=("_imp", "_rank"),
    )

    # 优先使用 rank 文件里的最终 effective_rank
    if "effective_rank_rank" in merged.columns:
        merged["final_effective_rank"] = merged["effective_rank_rank"]
    else:
        merged["final_effective_rank"] = merged["effective_rank"]

    plt.figure(figsize=(7, 5))

    for name in ["query", "value"]:
        sub = merged[merged["module_name"] == name]
        plt.scatter(
            sub["mean_abs_param_grad"],
            sub["final_effective_rank"],
            label=name,
        )

    plt.xlabel("Mean |param × grad|")
    plt.ylabel("Final Effective Rank")
    plt.title("Importance vs Final Effective Rank")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "importance_vs_rank.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)

    corr = merged[["mean_abs_param_grad", "final_effective_rank"]].corr().iloc[0, 1]
    print("importance 与 final rank 的相关系数：", corr)

    table_path = os.path.join(out_dir, "importance_rank_merged.csv")
    merged.to_csv(table_path, index=False, encoding="utf-8-sig")
    print("已保存：", table_path)


def show_top_importance(final_df, out_dir):
    data = final_df.sort_values("mean_abs_param_grad", ascending=False).head(12).copy()

    labels = [
        f"L{int(row['layer'])}-{row['module_name']}"
        for _, row in data.iterrows()
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, data["mean_abs_param_grad"])

    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Layer-Module")
    plt.ylabel("Mean |param × grad|")
    plt.title("Top-12 Important AdaLoRA Modules")
    plt.tight_layout()

    path = os.path.join(out_dir, "top12_importance_modules.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)

    table_path = os.path.join(out_dir, "top12_importance_modules.csv")
    data.to_csv(table_path, index=False, encoding="utf-8-sig")
    print("已保存：", table_path)


def show_layer_importance(final_df, out_dir):
    data = (
        final_df.groupby("layer")["mean_abs_param_grad"]
        .mean()
        .reset_index()
        .sort_values("layer")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(data["layer"], data["mean_abs_param_grad"])

    plt.xlabel("Layer")
    plt.ylabel("Mean |param × grad|")
    plt.title("Layer-wise AdaLoRA Importance")
    plt.xticks(data["layer"])
    plt.tight_layout()

    path = os.path.join(out_dir, "layer_importance.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def write_summary(final_df, final_rank, out_dir):
    qv = final_df.groupby("module_name")["mean_abs_param_grad"].mean()
    top = final_df.sort_values("mean_abs_param_grad", ascending=False).head(5)

    lines = []
    lines.append("# AdaLoRA 重要性评分分析摘要\n")
    lines.append("## 1. Query / Value 平均重要性\n")

    for name, value in qv.items():
        lines.append(f"- {name}: {value:.6e}")

    lines.append("\n## 2. Top 5 重要模块\n")
    for _, row in top.iterrows():
        lines.append(
            f"- L{int(row['layer'])}-{row['module_name']}: {row['mean_abs_param_grad']:.6e}"
        )

    if final_rank is not None:
        merged = final_df.merge(
            final_rank[["param_name", "effective_rank"]],
            on="param_name",
            how="left",
            suffixes=("_imp", "_rank"),
        )
        corr = merged[["mean_abs_param_grad", "effective_rank_rank"]].corr().iloc[0, 1]
        lines.append("\n## 3. 重要性与最终 rank 关系\n")
        lines.append(f"- Pearson 相关系数：{corr:.4f}")

    path = os.path.join(out_dir, "importance_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("已保存：", path)


def run():
    importance_path = os.path.join(
        "results",
        "importance",
        "diagnostics",
        "sst2_adalora_importance.csv",
    )

    rank_path = os.path.join(
        "results",
        "rank",
        "diagnostics",
        "sst2_adalora_importance_rank.csv",
    )

    out_dir = os.path.join("results", "figs", "diagnostics", "importance")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(importance_path):
        raise FileNotFoundError(f"找不到重要性评分文件：{importance_path}")

    raw_df = read_csv(importance_path)
    df = remove_duplicates(raw_df)

    final_step, final_df = get_final(df)
    final_rank = read_rank(rank_path)

    print("重要性评分日志读取成功")
    print("原始行数：", len(raw_df))
    print("去重后行数：", len(df))
    print("记录 step 数：", df["step"].nunique())
    print("最终 step：", final_step)
    print("最终记录数：", len(final_df))

    print("\n最终 query/value 平均重要性：")
    print(final_df.groupby("module_name")["mean_abs_param_grad"].mean())

    print("\nTop 10 重要模块：")
    print(
        final_df.sort_values("mean_abs_param_grad", ascending=False)[
            ["layer", "module_name", "effective_rank", "mean_abs_param_grad"]
        ].head(10)
    )

    save_final_table(final_df, out_dir)
    show_importance_heatmap(final_df, out_dir)
    show_query_value_importance(final_df, out_dir)
    show_importance_curve(df, out_dir)
    show_importance_rank_scatter(final_df, final_rank, out_dir)
    show_top_importance(final_df, out_dir)
    show_layer_importance(final_df, out_dir)
    write_summary(final_df, final_rank, out_dir)

    print("\nshow06 完成。")


if __name__ == "__main__":
    run()
