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
        "trainable_params",
        "trainable_ratio_percent",
        "eval_loss",
        "accuracy",
        "f1",
        "train_time_sec",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_rank_data(df):
    # 只保留基础多 rank 实验
    keep_names = [
        "sst2_lora_r2_e3",
        "sst2_lora_r4_e3",
        "sst2_lora_r8_e3",
        "sst2_lora_r16_e3",
        "sst2_adalora_r2_e3",
        "sst2_adalora_r4_e3",
        "sst2_adalora_r8_e3",
        "sst2_adalora_r16_e3",
    ]

    data = df[df["exp_name"].isin(keep_names)].copy()
    data = data.drop_duplicates(subset=["exp_name"], keep="last")

    data = data.sort_values(["method", "rank"])

    print("SST-2 基础多 rank 实验数据：")
    print(data[[
        "exp_name",
        "method",
        "rank",
        "accuracy",
        "f1",
        "eval_loss",
        "trainable_params",
        "train_time_sec",
    ]])

    return data


def save_table(data, out_dir):
    path = os.path.join(out_dir, "sst2_rank_result.csv")
    data.to_csv(path, index=False, encoding="utf-8-sig")
    print("已保存：", path)


def plot_metric(data, metric, ylabel, title, filename, out_dir):
    pivot = data.pivot_table(
        index="rank",
        columns="method",
        values=metric,
        aggfunc="mean",
    ).sort_index()

    ranks = list(pivot.index)
    x = list(range(len(ranks)))
    width = 0.35

    plt.figure(figsize=(8, 5))

    if "lora" in pivot.columns:
        plt.bar(
            [i - width / 2 for i in x],
            pivot["lora"],
            width=width,
            label="LoRA",
        )

    if "adalora" in pivot.columns:
        plt.bar(
            [i + width / 2 for i in x],
            pivot["adalora"],
            width=width,
            label="AdaLoRA",
        )

    plt.xticks(x, [str(r) for r in ranks])
    plt.xlabel("Rank")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def plot_pareto(data, out_dir):
    plt.figure(figsize=(7, 5))

    for method in ["lora", "adalora"]:
        sub = data[data["method"] == method].sort_values("trainable_params")
        if len(sub) == 0:
            continue

        plt.plot(
            sub["trainable_params"],
            sub["accuracy"],
            marker="o",
            label=method,
        )

        for _, row in sub.iterrows():
            plt.text(
                row["trainable_params"],
                row["accuracy"],
                f"r{int(row['rank'])}",
                fontsize=8,
            )

    plt.xlabel("Trainable Parameters")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Trainable Parameters")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "sst2_pareto_acc.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("已保存：", path)


def run():
    metric_path = os.path.join("results", "metrics", "metrics.csv")
    out_dir = os.path.join("results", "figs", "comparison", "sst2_rank")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"找不到指标文件：{metric_path}")

    df = read_metrics(metric_path)
    data = get_rank_data(df)

    if len(data) == 0:
        raise ValueError("没有找到 SST-2 基础多 rank 实验数据。")

    save_table(data, out_dir)

    plot_metric(
        data,
        metric="accuracy",
        ylabel="Accuracy",
        title="SST-2 Accuracy under Different Ranks",
        filename="sst2_acc_rank.png",
        out_dir=out_dir,
    )

    plot_metric(
        data,
        metric="f1",
        ylabel="F1",
        title="SST-2 F1 under Different Ranks",
        filename="sst2_f1_rank.png",
        out_dir=out_dir,
    )

    plot_metric(
        data,
        metric="eval_loss",
        ylabel="Eval Loss",
        title="SST-2 Eval Loss under Different Ranks",
        filename="sst2_eval_loss_curve.png",
        out_dir=out_dir,
    )

    plot_metric(
        data,
        metric="train_time_sec",
        ylabel="Train Time (s)",
        title="Training Time under Different Ranks",
        filename="sst2_train_time.png",
        out_dir=out_dir,
    )

    plot_pareto(data, out_dir)

    print("\nshow01_clean 完成。")


if __name__ == "__main__":
    run()