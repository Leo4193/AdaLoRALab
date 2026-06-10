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
        "max_rank",
        "mean_abs_lora_E",
        "max_abs_lora_E",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_final(df):
    final_step = df["step"].max()
    final_df = df[df["step"] == final_step].copy()
    return final_step, final_df


def save_table(final_df, out_dir):
    path = os.path.join(out_dir, "final_loraE_strength_table.csv")
    final_df.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)


def show_strength_heatmap(final_df, out_dir):
    module_order = ["query", "value", "intermediate", "output"]

    pivot = final_df.pivot_table(
        index="layer",
        columns="module_name",
        values="mean_abs_lora_E",
        aggfunc="mean",
    )

    pivot = pivot.reindex(columns=[m for m in module_order if m in pivot.columns])
    pivot = pivot.sort_index()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, aspect="auto")

    plt.colorbar(im, label="Mean |lora_E|")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), [f"L{int(i)}" for i in pivot.index])

    plt.xlabel("Module")
    plt.ylabel("Layer")
    plt.title("AdaLoRA Singular Value Strength Heatmap")
    plt.tight_layout()

    path = os.path.join(out_dir, "loraE_strength_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_module_type_strength(final_df, out_dir):
    data = (
        final_df.groupby("module_type")["mean_abs_lora_E"]
        .mean()
        .reset_index()
    )

    order = {"Attention": 0, "FFN": 1}
    data["order"] = data["module_type"].map(order)
    data = data.sort_values("order")

    plt.figure(figsize=(6, 5))
    plt.bar(data["module_type"], data["mean_abs_lora_E"])

    for i, v in enumerate(data["mean_abs_lora_E"]):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Module Type")
    plt.ylabel("Mean |lora_E|")
    plt.title("Average AdaLoRA Strength: Attention vs FFN")
    plt.tight_layout()

    path = os.path.join(out_dir, "module_type_strength.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_module_name_strength(final_df, out_dir):
    order = ["query", "value", "intermediate", "output"]

    data = (
        final_df.groupby("module_name")["mean_abs_lora_E"]
        .mean()
        .reindex(order)
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.bar(data["module_name"], data["mean_abs_lora_E"])

    for i, v in enumerate(data["mean_abs_lora_E"]):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Module")
    plt.ylabel("Mean |lora_E|")
    plt.title("Average AdaLoRA Strength by Module")
    plt.tight_layout()

    path = os.path.join(out_dir, "module_name_strength.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_layer_group_strength(final_df, out_dir):
    def group_layer(layer):
        if layer <= 3:
            return "Low layers (0-3)"
        elif layer <= 7:
            return "Middle layers (4-7)"
        else:
            return "High layers (8-11)"

    data = final_df.copy()
    data["layer_group"] = data["layer"].apply(group_layer)

    group_order = ["Low layers (0-3)", "Middle layers (4-7)", "High layers (8-11)"]

    summary = (
        data.groupby("layer_group")["mean_abs_lora_E"]
        .mean()
        .reindex(group_order)
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(summary["layer_group"], summary["mean_abs_lora_E"])

    for i, v in enumerate(summary["mean_abs_lora_E"]):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Layer Group")
    plt.ylabel("Mean |lora_E|")
    plt.title("AdaLoRA Strength across Layer Groups")
    plt.tight_layout()

    path = os.path.join(out_dir, "layer_group_strength.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_layer_total_strength(final_df, out_dir):
    data = (
        final_df.groupby("layer")["mean_abs_lora_E"]
        .sum()
        .reset_index()
        .sort_values("layer")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(data["layer"], data["mean_abs_lora_E"])

    plt.xlabel("Layer")
    plt.ylabel("Total Mean |lora_E|")
    plt.title("AdaLoRA Strength by Layer")
    plt.xticks(data["layer"])
    plt.tight_layout()

    path = os.path.join(out_dir, "layer_total_strength.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_top_modules(final_df, out_dir):
    data = final_df.sort_values("mean_abs_lora_E", ascending=False).head(12).copy()

    labels = [
        f"L{int(row['layer'])}-{row['module_name']}"
        for _, row in data.iterrows()
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, data["mean_abs_lora_E"])

    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Layer-Module")
    plt.ylabel("Mean |lora_E|")
    plt.title("Top-12 AdaLoRA Strongest Modules")
    plt.tight_layout()

    path = os.path.join(out_dir, "top12_strength_modules.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)

    table_path = os.path.join(out_dir, "top12_strength_modules.csv")
    data.to_csv(table_path, index=False, encoding="utf-8-sig")
    print("已保存：", table_path)


def run():
    rank_path = os.path.join(
        "results",
        "rank",
        "diagnostics",
        "sst2_full_adalora_r4_rank_e3_rank.csv",
    )

    out_dir = os.path.join("results", "figs", "diagnostics", "rank_strength")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(rank_path):
        raise FileNotFoundError(f"找不到文件：{rank_path}")

    df = read_rank(rank_path)
    final_step, final_df = get_final(df)

    print("Rank日志读取成功")
    print("最终 step:", final_step)
    print("记录数量:", len(final_df))
    print("effective_rank 唯一值:", sorted(final_df["effective_rank"].dropna().unique()))
    print("\n模块类型平均强度：")
    print(final_df.groupby("module_type")["mean_abs_lora_E"].mean())
    print("\n具体模块平均强度：")
    print(final_df.groupby("module_name")["mean_abs_lora_E"].mean())
    print("\nTop 10 强模块：")
    print(final_df.sort_values("mean_abs_lora_E", ascending=False)[
        ["layer", "module_type", "module_name", "mean_abs_lora_E", "max_abs_lora_E"]
    ].head(10))

    save_table(final_df, out_dir)
    show_strength_heatmap(final_df, out_dir)
    show_module_type_strength(final_df, out_dir)
    show_module_name_strength(final_df, out_dir)
    show_layer_group_strength(final_df, out_dir)
    show_layer_total_strength(final_df, out_dir)
    show_top_modules(final_df, out_dir)

    print("\nshow04_fix 完成。")


if __name__ == "__main__":
    run()
