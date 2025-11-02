# predict_sentiment.py
import os
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from dataset import SentimentDataset
from train_sentiment import SentimentClassifier


@torch.no_grad()
def infer(model, dataloader, device):
    # 逐批前向计算，取最大logit的类别作为预测
    model.eval()
    preds_all = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, loss_fn=None)
        pred_class = torch.argmax(logits, dim=1)  # 0 or 1
        preds_all.extend(pred_class.cpu().tolist())
    return preds_all


def load_test_dataframe(test_path):
    
    #读取 JSON 数组

    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    assert "reviews" in df.columns, "test.json 缺少 'reviews' 字段"
    df["reviews"] = df["reviews"].astype(str).fillna("").str.strip()
    return df


def main(args):
    # 加载最优权重文件
    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 从权重中还原分词器名称与长度设置，保持一致
    tokenizer_name = ckpt["tokenizer_name"]
    max_length = ckpt["max_length"]

    # 准备分词器与设备
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建模型并加载权重
    model = SentimentClassifier(pretrained_name=tokenizer_name)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)

    # 读取测试数据
    df_test = load_test_dataframe(args.test_path)
    test_texts = df_test["reviews"].tolist()

    # 组装dataset
    test_dataset = SentimentDataset(
        texts=test_texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 推理得到预测表
    preds = infer(model, test_loader, device)

    # 与样本顺序一一对应输出，仅一列sentiments
    sub_df = pd.DataFrame({"sentiments": preds})
    sub_df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")
    print(sub_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="test.json")
    parser.add_argument("--model_dir", type=str, default="saved_model")
    parser.add_argument("--out_csv", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
