import os
import pandas as pd
import matplotlib.pyplot as plt


def clean_cols(df):
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_metrics(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        df = clean_cols(df)
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


def get_ablation_data(df):
    names = [
        "sst2_attn_lora_r4_e3",
        "sst2_ffn_lora_r4_e3",
        "sst2_full_lora_r4_e3",
    ]

    data = df[df["exp_name"].isin(names)].copy()

    label_map = {
        "sst2_attn_lora_r4_e3": "Attention-only",
        "sst2_ffn_lora_r4_e3": "FFN-only",
        "sst2_full_lora_r4_e3": "Attention+FFN",
    }

    order_map = {
        "Attention-only": 0,
        "FFN-only": 1,
        "Attention+FFN": 2,
    }

    data["module_type"] = data["exp_name"].map(label_map)
    data["order"] = data["module_type"].map(order_map)
    data = data.sort_values("order")

    return data


def save_table(data, out_dir):
    cols = [
        "exp_name",
        "module_type",
        "rank",
        "trainable_params",
        "trainable_ratio_percent",
        "eval_loss",
        "accuracy",
        "f1",
        "train_time_sec",
    ]

    path = os.path.join(out_dir, "module_ablation_result.csv")
    data[cols].to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)


def show_metric_bar(data, metric, out_dir):
    plt.figure(figsize=(7, 5))

    plt.bar(data["module_type"], data[metric])

    for i, v in enumerate(data[metric]):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Target Modules")
    plt.ylabel(metric.upper())
    plt.title(f"Module Ablation on SST-2: {metric.upper()}")
    plt.tight_layout()

    path = os.path.join(out_dir, f"module_ablation_{metric}.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_param_acc(data, out_dir):
    plt.figure(figsize=(7, 5))

    plt.plot(
        data["trainable_params"],
        data["accuracy"],
        marker="o",
    )

    for _, row in data.iterrows():
        plt.text(
            row["trainable_params"],
            row["accuracy"],
            row["module_type"],
            fontsize=9,
        )

    plt.xlabel("Trainable Parameters")
    plt.ylabel("Accuracy")
    plt.title("Module Ablation: Parameters vs Accuracy")
    plt.tight_layout()

    path = os.path.join(out_dir, "module_param_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_time_bar(data, out_dir):
    plt.figure(figsize=(7, 5))

    plt.bar(data["module_type"], data["train_time_sec"])

    for i, v in enumerate(data["train_time_sec"]):
        plt.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Target Modules")
    plt.ylabel("Training Time / s")
    plt.title("Module Ablation: Training Time")
    plt.tight_layout()

    path = os.path.join(out_dir, "module_train_time.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def run():
    metric_path = os.path.join("results", "metrics", "metrics.csv")
    out_dir = os.path.join("results", "figs", "comparison", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    df = read_metrics(metric_path)
    data = get_ablation_data(df)

    if len(data) == 0:
        raise ValueError("没有找到模块消融实验结果。")

    print("\n模块消融实验数据：")
    print(data[["exp_name", "module_type", "accuracy", "f1", "trainable_params", "train_time_sec"]])

    save_table(data, out_dir)
    show_metric_bar(data, "accuracy", out_dir)
    show_metric_bar(data, "f1", out_dir)
    show_param_acc(data, out_dir)
    show_time_bar(data, out_dir)

    print("\nshow03 完成。")


if __name__ == "__main__":
    run()