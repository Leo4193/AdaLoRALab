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

    print("读取到的列名：")
    print(list(df.columns))

    return df


def get_multitask_data(df):
    # 只取 r4_e3 的正式实验结果
    names = [
        "sst2_lora_r4_e3",
        "sst2_adalora_r4_e3",
        "mrpc_lora_r4_e3",
        "mrpc_adalora_r4_e3",
        "rte_lora_r4_e3",
        "rte_adalora_r4_e3",
    ]

    data = df[df["exp_name"].isin(names)].copy()
    data["method_label"] = data["method"].map({
        "lora": "LoRA",
        "adalora": "AdaLoRA",
    })

    task_order = {"sst2": 0, "mrpc": 1, "rte": 2}
    method_order = {"lora": 0, "adalora": 1}

    data["task_order"] = data["dataset"].map(task_order)
    data["method_order"] = data["method"].map(method_order)

    data = data.sort_values(["task_order", "method_order"])

    return data


def save_table(data, out_dir):
    cols = [
        "exp_name",
        "dataset",
        "method",
        "rank",
        "trainable_params",
        "trainable_ratio_percent",
        "eval_loss",
        "accuracy",
        "f1",
        "train_time_sec",
    ]

    path = os.path.join(out_dir, "multitask_result.csv")
    data[cols].to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)


def show_metric_bar(data, metric, out_dir):
    tasks = ["sst2", "mrpc", "rte"]
    x = list(range(len(tasks)))
    width = 0.35

    lora = data[data["method"] == "lora"].set_index("dataset")
    adalora = data[data["method"] == "adalora"].set_index("dataset")

    lora_vals = [lora.loc[t, metric] if t in lora.index else 0 for t in tasks]
    adalora_vals = [adalora.loc[t, metric] if t in adalora.index else 0 for t in tasks]

    plt.figure(figsize=(8, 5))

    plt.bar(
        [i - width / 2 for i in x],
        lora_vals,
        width=width,
        label="LoRA-r4",
    )

    plt.bar(
        [i + width / 2 for i in x],
        adalora_vals,
        width=width,
        label="AdaLoRA-r4",
    )

    plt.xticks(x, [t.upper() for t in tasks])
    plt.xlabel("Task")
    plt.ylabel(metric.upper())
    plt.title(f"Multi-task {metric.upper()} Comparison")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, f"multitask_{metric}.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_gap(data, out_dir):
    tasks = ["sst2", "mrpc", "rte"]
    gaps = []

    for task in tasks:
        sub = data[data["dataset"] == task]
        lora_acc = sub[sub["method"] == "lora"]["accuracy"].values
        adalora_acc = sub[sub["method"] == "adalora"]["accuracy"].values

        if len(lora_acc) == 0 or len(adalora_acc) == 0:
            gaps.append(0)
        else:
            gaps.append(float(lora_acc[0] - adalora_acc[0]))

    plt.figure(figsize=(7, 5))
    plt.bar([t.upper() for t in tasks], gaps)

    plt.axhline(0, linewidth=1)
    plt.xlabel("Task")
    plt.ylabel("Accuracy Gap: LoRA - AdaLoRA")
    plt.title("Performance Gap across Tasks")
    plt.tight_layout()

    path = os.path.join(out_dir, "multitask_accuracy_gap.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def show_time_bar(data, out_dir):
    labels = [
        f"{row['dataset'].upper()}-{row['method_label']}"
        for _, row in data.iterrows()
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, data["train_time_sec"])
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Experiment")
    plt.ylabel("Training Time / s")
    plt.title("Training Time Comparison across Tasks")
    plt.tight_layout()

    path = os.path.join(out_dir, "multitask_train_time.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def run():
    metric_path = os.path.join("results", "metrics", "metrics.csv")
    out_dir = os.path.join("results", "figs", "comparison", "multitask")
    os.makedirs(out_dir, exist_ok=True)

    df = read_metrics(metric_path)
    data = get_multitask_data(df)

    if len(data) == 0:
        raise ValueError("没有筛选到多任务 r4_e3 实验结果。")

    print("\n多任务正式实验数据：")
    print(data[["exp_name", "dataset", "method", "accuracy", "f1", "train_time_sec"]])

    save_table(data, out_dir)
    show_metric_bar(data, "accuracy", out_dir)
    show_metric_bar(data, "f1", out_dir)
    show_gap(data, out_dir)
    show_time_bar(data, out_dir)

    print("\nshow02 完成。")


if __name__ == "__main__":
    run()