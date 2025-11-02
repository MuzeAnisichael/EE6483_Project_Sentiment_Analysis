# train_sentiment.py
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    BertTokenizerFast, BertModel,           #BERT分词器和BERT
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW               #优化器

from dataset import SentimentDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter


def set_seed(seed: int = 42):
    #固定种子为42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SentimentClassifier(nn.Module):

    #BERT + Dropout + Linear 二分类
    
    def __init__(self, pretrained_name="bert-base-uncased", dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)  
        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None, loss_fn=None):
        #bert前向
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(self.dropout(cls_emb))

        #可选返回loss
        loss = None
        if labels is not None and loss_fn is not None:
            loss = loss_fn(logits, labels)
        return logits, loss


def load_train_dataframe(train_path):

    # 读取JSON，转为DataFrame并做基础清洗

    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    assert "reviews" in df.columns and "sentiments" in df.columns, "train.json 缺少必要字段"
    # 清洗最基本的空白
    df["reviews"] = df["reviews"].astype(str).fillna("").str.strip()
    df["sentiments"] = df["sentiments"].astype(int)
    return df


def make_dataloaders(df, tokenizer, max_length, batch_size, val_ratio, seed=1234):
    texts = df["reviews"].tolist()
    labels = df["sentiments"].tolist()
    labels_np = np.array(labels)

    # 分层抽样划分（保持0/1比例一致）
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(texts, labels_np))

    texts_train = [texts[i] for i in train_idx]
    labels_train = [labels[i] for i in train_idx]
    texts_val   = [texts[i] for i in val_idx]
    labels_val  = [labels[i] for i in val_idx]

    train_ds = SentimentDataset(texts_train, labels_train, tokenizer, max_length, is_test=False)
    val_ds   = SentimentDataset(texts_val,   labels_val,   tokenizer, max_length, is_test=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, (texts_val, labels_val)


def build_loss_fn(labels, use_class_weight: bool): 
    # 构建交叉熵损失
    if not use_class_weight:
        return nn.CrossEntropyLoss()

    counter = Counter(labels)
    # 按出现次数的反比做权重
    total = sum(counter.values())
    w0 = total / (2.0 * counter.get(0, 1))
    w1 = total / (2.0 * counter.get(1, 1))
    weight = torch.tensor([w0, w1], dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weight)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, max_grad_norm=1.0):
    # 单epoch训练:前向 - 反传 - 梯度裁剪 - 更新参数
    model.train()
    all_losses = []
    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits, loss = model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             loss_fn=loss_fn)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        all_losses.append(loss.item())

    return float(np.mean(all_losses))


@torch.no_grad()
def eval_model(model, dataloader, device, loss_fn):
    # 评估：loss，ACC，F1
    model.eval()
    all_labels, all_preds, all_losses = [], [], []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits, _ = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=None,
                          loss_fn=None)

        loss = loss_fn(logits, labels)
        all_losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return {
        "val_loss": float(np.mean(all_losses)),
        "val_acc": acc,
        "val_f1_macro": f1,
        "labels": all_labels,
        "preds": all_preds
    }


def maybe_save_misclass_examples(val_texts, labels, preds, out_dir, top_k=50):
    # 可能错分样本
    df_err = []
    for t, y, p in zip(val_texts, labels, preds):
        if y != p:
            df_err.append({"review": t, "label": y, "pred": p})
    if df_err:
        df_err = pd.DataFrame(df_err)
        # 只保留前top_k行
        df_err.head(top_k).to_csv(os.path.join(out_dir, "val_misclassified.csv"), index=False)


def main(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading train.json ...")
    df = load_train_dataframe(args.train_path)

    print(f"Num samples = {len(df)} | label distribution = {df['sentiments'].value_counts().to_dict()}")

    print("Loading tokenizer ...")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    print("Building dataloaders ...")
    train_loader, val_loader, (val_texts, val_labels) = make_dataloaders(
        df, tokenizer, args.max_length, args.batch_size, args.val_ratio, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Building model ...")
    model = SentimentClassifier(pretrained_name=args.model_name, dropout_prob=args.dropout).to(device)

    # 构建交叉熵损失，缓解类别不平衡
    loss_fn = build_loss_fn(df["sentiments"].tolist(), use_class_weight=args.class_weight).to(device)

    # 优化器 & 学习率调度
    num_training_steps = len(train_loader) * args.epochs
    num_warmup = int(args.warmup_ratio * num_training_steps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training_steps
    )

    best_metric = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history, patience_counter = [], 0

    print("Start training ...")
    for epoch in range(1, args.epochs + 1):
        # 训练一轮
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn, max_grad_norm=args.max_grad_norm
        )
        # 验证集评估
        metrics = eval_model(model, val_loader, device, loss_fn)

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={metrics['val_loss']:.4f} "
              f"val_acc={metrics['val_acc']:.4f} "
              f"val_f1_macro={metrics['val_f1_macro']:.4f}")

        # 训练曲线
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
            "val_f1_macro": metrics["val_f1_macro"]
        })

        # 早停依据：优先看 val_f1_macro，其次 val_acc
        current = (metrics["val_f1_macro"], metrics["val_acc"])
        if current > (best_metric, -1.0):  # tuple比较，先 f1 再 acc
            best_metric = metrics["val_f1_macro"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_name": args.model_name,
                "max_length": args.max_length
            }, best_path)
            patience_counter = 0
            print(f"  -> New best saved to {best_path} (val_f1_macro={best_metric:.4f})")
            # 保存一次错分样本（用于(f)分析）
            maybe_save_misclass_examples(val_texts, metrics["labels"], metrics["preds"], args.out_dir, top_k=100)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # 训练日志
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)

    # 打印最后一轮的分类报告
    print("\nValidation classification report (last-epoch):")
    print(classification_report(metrics["labels"], metrics["preds"], digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 基础路径
    parser.add_argument("--train_path", type=str, default="train.json")
    parser.add_argument("--out_dir", type=str, default="saved_model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")

    # 超参
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--early_stop", type=int, default=2, help="patience for early stopping")

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--class_weight", action="store_true", help="use class-weighted CE loss if set")
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()
    main(args)
