import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import hf_compat

import re
import csv
import time
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from peft import (
    AdaLoraConfig,
    TaskType,
    get_peft_model,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_params(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    ratio = trainable / total * 100
    return total, trainable, ratio


def load_data(cfg, tokenizer):
    print("正在加载数据集...")

    raw = load_dataset(cfg["dataset_name"], cfg["task_name"])

    train_data = raw["train"]
    eval_data = raw["validation"]

    if cfg.get("max_train_samples") is not None:
        n = min(int(cfg["max_train_samples"]), len(train_data))
        train_data = train_data.select(range(n))

    if cfg.get("max_eval_samples") is not None:
        n = min(int(cfg["max_eval_samples"]), len(eval_data))
        eval_data = eval_data.select(range(n))

    task = cfg["task_name"]

    def tokenize(batch):
        if task == "sst2":
            out = tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=int(cfg["max_length"]),
            )
        elif task in ["mrpc", "rte"]:
            out = tokenizer(
                batch["sentence1"],
                batch["sentence2"],
                truncation=True,
                max_length=int(cfg["max_length"]),
            )
        else:
            raise ValueError(f"暂不支持任务：{task}")

        out["labels"] = batch["label"]
        return out

    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)

    keep_cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    train_remove = [c for c in train_data.column_names if c not in keep_cols]
    eval_remove = [c for c in eval_data.column_names if c not in keep_cols]

    train_data = train_data.remove_columns(train_remove)
    eval_data = eval_data.remove_columns(eval_remove)

    print(f"训练样本数：{len(train_data)}")
    print(f"验证样本数：{len(eval_data)}")

    return train_data, eval_data


def build_model(cfg):
    print("正在加载模型...")

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
    )

    if cfg.get("target_modules_regex") is not None:
        target_modules = cfg["target_modules_regex"]
    else:
        target_modules = cfg["target_modules"]

    peft_config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=int(cfg["init_r"]),
        target_r=int(cfg["target_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
        modules_to_save=["classifier"],

        total_step=int(cfg["total_step"]),
        tinit=int(cfg["tinit"]),
        tfinal=int(cfg["tfinal"]),
        deltaT=int(cfg["deltaT"]),
        orth_reg_weight=float(cfg["orth_reg_weight"]),
    )

    model = get_peft_model(base, peft_config)

    print("可训练参数如下：")
    model.print_trainable_parameters()

    if not hasattr(model, "update_and_allocate"):
        raise RuntimeError("当前模型没有 update_and_allocate，AdaLoRA 动态秩更新无法执行。")

    return model


def make_loader(data, tokenizer, batch_size, shuffle):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
    )


def parse_module(name):
    layer_match = re.search(r"encoder\.layer\.(\d+)", name)
    if layer_match is None:
        return None, None, None

    layer = int(layer_match.group(1))

    if ".attention.self.query." in name:
        module_type = "Attention"
        module_name = "query"
    elif ".attention.self.value." in name:
        module_type = "Attention"
        module_name = "value"
    elif ".intermediate.dense." in name:
        module_type = "FFN"
        module_name = "intermediate"
    elif re.search(r"encoder\.layer\.\d+\.output\.dense\.", name):
        module_type = "FFN"
        module_name = "output"
    else:
        module_type = "Other"
        module_name = "other"

    return layer, module_type, module_name


def collect_rank(model, step, threshold):
    rows = []

    for name, param in model.named_parameters():
        if "lora_E" not in name:
            continue

        layer, module_type, module_name = parse_module(name)
        if layer is None:
            continue

        values = param.detach().float().cpu().view(-1)

        effective_rank = int((values.abs() > threshold).sum().item())
        zero_count = int((values.abs() <= threshold).sum().item())
        max_rank = int(values.numel())
        mean_abs = float(values.abs().mean().item())
        max_abs = float(values.abs().max().item())
        min_abs = float(values.abs().min().item())

        rows.append({
            "step": step,
            "layer": layer,
            "module_type": module_type,
            "module_name": module_name,
            "param_name": name,
            "effective_rank": effective_rank,
            "zero_count": zero_count,
            "max_rank": max_rank,
            "mean_abs_lora_E": mean_abs,
            "max_abs_lora_E": max_abs,
            "min_abs_lora_E": min_abs,
        })

    return rows

def collect_importance(model, step, threshold):
    # 记录 lora_E 的参数、梯度和 |参数 × 梯度| 重要性
    rows = []

    for name, param in model.named_parameters():
        if "lora_E" not in name:
            continue

        layer, module_type, module_name = parse_module(name)
        if layer is None:
            continue

        if param.grad is None:
            continue

        values = param.detach().float().cpu().view(-1)
        grads = param.grad.detach().float().cpu().view(-1)

        abs_param = values.abs()
        abs_grad = grads.abs()
        abs_pg = (values * grads).abs()

        effective_rank = int((abs_param > threshold).sum().item())
        zero_count = int((abs_param <= threshold).sum().item())
        max_rank = int(values.numel())

        rows.append({
            "step": step,
            "layer": layer,
            "module_type": module_type,
            "module_name": module_name,
            "param_name": name,
            "effective_rank": effective_rank,
            "zero_count": zero_count,
            "max_rank": max_rank,
            "mean_abs_param": float(abs_param.mean().item()),
            "mean_abs_grad": float(abs_grad.mean().item()),
            "mean_abs_param_grad": float(abs_pg.mean().item()),
            "max_abs_param_grad": float(abs_pg.max().item()),
            "sum_abs_param_grad": float(abs_pg.sum().item()),
        })

    return rows


def save_importance_rows(rows, path):
    if len(rows) == 0:
        print("没有 importance 记录可保存。")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys = [
        "step",
        "layer",
        "module_type",
        "module_name",
        "param_name",
        "effective_rank",
        "zero_count",
        "max_rank",
        "mean_abs_param",
        "mean_abs_grad",
        "mean_abs_param_grad",
        "max_abs_param_grad",
        "sum_abs_param_grad",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("重要性评分日志已保存到：", path)


def summarize_importance(rows, step):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"step={step} 没有 importance 信息。")
        return

    cur = df[df["step"] == step]
    if len(cur) == 0:
        return

    print("-" * 70)
    print(f"重要性评分检查 step={step}")
    print("记录模块数：", len(cur))

    print("按模块名称 mean_abs_param_grad 平均值：")
    print(cur.groupby("module_name")["mean_abs_param_grad"].mean())

    print("Top 5 重要模块：")
    print(
        cur.sort_values("mean_abs_param_grad", ascending=False)[
            ["layer", "module_name", "effective_rank", "mean_abs_param_grad"]
        ].head(5)
    )
    print("-" * 70)


def save_rank_rows(rows, path):
    if len(rows) == 0:
        print("没有 rank 记录可保存。")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys = [
        "step",
        "layer",
        "module_type",
        "module_name",
        "param_name",
        "effective_rank",
        "zero_count",
        "max_rank",
        "mean_abs_lora_E",
        "max_abs_lora_E",
        "min_abs_lora_E",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Rank 日志已保存到：", path)


def summarize_rank(rows, step):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"step={step} 没有 rank 信息。")
        return

    cur = df[df["step"] == step]
    if len(cur) == 0:
        return

    print("-" * 70)
    print(f"Rank 检查 step={step}")
    print("记录模块数：", len(cur))
    print("effective_rank 分布：")
    print(cur["effective_rank"].value_counts().sort_index())

    print("按模块类型平均 effective_rank：")
    print(cur.groupby("module_type")["effective_rank"].mean())

    print("按模块名称平均 effective_rank：")
    print(cur.groupby("module_name")["effective_rank"].mean())

    print("zero_count 总和：", int(cur["zero_count"].sum()))
    print("-" * 70)


def evaluate(model, loader, device):
    model.eval()

    losses = []
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss
            logits = out.logits

            losses.append(float(loss.item()))

            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()

            preds_all.extend(preds.tolist())
            labels_all.extend(labels.tolist())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all)

    return {
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(acc),
        "f1": float(f1),
    }


def save_metrics(cfg, total_params, trainable_params, trainable_ratio, result, train_time):
    path = cfg["metric_file"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    row = {
        "exp_name": cfg["exp_name"],
        "dataset": cfg["task_name"],
        "model": cfg["model_name"],
        "method": cfg.get("method", "adalora_importance"),
        "rank": cfg["target_r"],
        "train_samples": cfg.get("max_train_samples"),
        "eval_samples": cfg.get("max_eval_samples"),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio_percent": round(trainable_ratio, 4),
        "eval_loss": round(float(result["eval_loss"]), 6),
        "accuracy": round(float(result["accuracy"]), 6),
        "f1": round(float(result["f1"]), 6),
        "train_time_sec": round(float(train_time), 2),
        "seed": cfg["seed"],
    }

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("指标已保存到：", path)
    print("本次实验结果：")
    for k, v in row.items():
        print(f"{k}: {v}")


def save_train_log(log_rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(log_rows) == 0:
        return

    keys = sorted(set().union(*(row.keys() for row in log_rows)))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_rows)

    print("训练日志已保存到：", path)


def train(cfg, model, train_loader, eval_loader, device):
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    total_step = int(cfg["total_step"])
    warmup_steps = int(cfg.get("warmup_steps", max(1, total_step // 10)))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        remain = total_step - step
        total_decay = max(1, total_step - warmup_steps)
        return max(0.0, float(remain) / float(total_decay))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rank_rows = []
    importance_rows = []
    log_rows = []

    threshold = float(cfg.get("rank_threshold", 1e-8))
    rank_log_every = int(cfg.get("rank_log_every", 100))
    eval_every = int(cfg.get("eval_every", 300))
    print_every = int(cfg.get("print_every", 50))
    importance_log_every = int(cfg.get("importance_log_every", rank_log_every))

    global_step = 0
    start_time = time.time()

    # 训练前先记录一次 rank
    rank_rows.extend(collect_rank(model, 0, threshold))
    summarize_rank(rank_rows, 0)

    model.train()

    for epoch in range(int(cfg["num_train_epochs"])):
        print("=" * 80)
        print(f"开始 Epoch {epoch + 1}/{cfg['num_train_epochs']}")

        for batch in train_loader:
            global_step += 1

            batch = {k: v.to(device) for k, v in batch.items()}

            model.train()
            out = model(**batch)
            loss = out.loss

            loss.backward()

            # 关键：梯度仍存在时，先记录重要性评分
            if global_step % importance_log_every == 0:
                imp_rows = collect_importance(model, global_step, threshold)
                importance_rows.extend(imp_rows)
                summarize_importance(importance_rows, global_step)

            # 再执行 AdaLoRA 动态预算更新
            model.update_and_allocate(global_step)  

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % print_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"step={global_step}/{total_step}, loss={loss.item():.4f}, lr={lr:.8f}, update_and_allocate=OK")

            if global_step % rank_log_every == 0:
                rows = collect_rank(model, global_step, threshold)
                rank_rows.extend(rows)
                summarize_rank(rank_rows, global_step)

            if global_step % eval_every == 0:
                result = evaluate(model, eval_loader, device)
                print(f"验证 step={global_step}: loss={result['eval_loss']:.4f}, acc={result['accuracy']:.4f}, f1={result['f1']:.4f}")
                log_rows.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": float(loss.item()),
                    "eval_loss": result["eval_loss"],
                    "accuracy": result["accuracy"],
                    "f1": result["f1"],
                    "lr": scheduler.get_last_lr()[0],
                })

            if global_step >= total_step:
                break

        if global_step >= total_step:
            break

    train_time = time.time() - start_time

    # 训练结束再记录一次
    rows = collect_rank(model, global_step, threshold)
    rank_rows.extend(rows)
    summarize_rank(rank_rows, global_step)

    result = evaluate(model, eval_loader, device)

    return result, train_time, rank_rows, log_rows, importance_rows


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["logging_dir"], exist_ok=True)
    os.makedirs(os.path.join("results", "rank", "diagnostics"), exist_ok=True)
    os.makedirs(os.path.join("results", "importance", "diagnostics"), exist_ok=True)

    set_seed(int(cfg["seed"]))

    print("=" * 80)
    print("严格 AdaLoRA 动态秩实验")
    print("实验名称：", cfg["exp_name"])
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_data, eval_data = load_data(cfg, tokenizer)

    train_loader = make_loader(
        train_data,
        tokenizer,
        int(cfg["train_batch_size"]),
        shuffle=True,
    )

    eval_loader = make_loader(
        eval_data,
        tokenizer,
        int(cfg["eval_batch_size"]),
        shuffle=False,
    )

    model = build_model(cfg)

    total_params, trainable_params, trainable_ratio = count_params(model)

    result, train_time, rank_rows, log_rows, importance_rows = train(
        cfg,
        model,
        train_loader,
        eval_loader,
        device,
    )

    print("保存模型...")
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    save_metrics(cfg, total_params, trainable_params, trainable_ratio, result, train_time)

    rank_file = cfg.get(
        "rank_file",
        os.path.join("results", "rank", "diagnostics", f"{cfg['exp_name']}_rank.csv"),
    )
    save_rank_rows(rank_rows, rank_file)

    importance_file = cfg.get(
        "importance_file",
        os.path.join("results", "importance", "diagnostics", f"{cfg['exp_name']}_importance.csv"),
    )
    save_importance_rows(importance_rows, importance_file)

    log_file = os.path.join("results", "logs", f"{cfg['exp_name']}_history.csv")
    save_train_log(log_rows, log_file)

    print("严格 AdaLoRA 动态秩实验完成。")


if __name__ == "__main__":
    run()
