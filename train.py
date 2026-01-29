'''
    train是调度器
    整体的目标是预测用户下次会买什么 然后进行推荐
'''

import os
import argparse
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.transformer_rec import TransformerRec
from model.gru_rec import GRURec
import json
from datetime import datetime

# -- padding的itemid为0，即不参与gru或者transformer的计算之中 --
PAD_IDX = 0

# -- 读取vocabulary --
def load_vocab(vocab_path):
    vocab = pd.read_csv(vocab_path)
    item2idx = {row.item_id: int(row.item_idx) for _, row in vocab.iterrows()}
    return item2idx, len(item2idx) + 1  # +1 for PAD(0)

# -- 从父类Dataset进行继承 --
'''
    getitem为默认的主要函数，意思就是给定一个i，对应获得哪个样本以及标签
'''
class NextItemDataset(Dataset):
    def __init__(self, csv_path, item2idx, max_seq_len=100):
        self.df = pd.read_csv(csv_path) # 数据
        self.item2idx = item2idx # item对应的字典
        self.max_seq_len = max_seq_len # 序列最大长度

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i] # 获取第i个序列
        hist = r["hist"].split() #获取history
        hist = hist[-self.max_seq_len:] # 最大长度的限制
        x = [self.item2idx[h] for h in hist] # 映射成编码
        y = self.item2idx[str(r["target"])] #索引出target 同时进行编码映射
        return torch.tensor(x, dtype=torch.long), len(x), torch.tensor(y, dtype=torch.long) # 返回张量以及非padding的序列长度

# -- 批处理补序列长度padding（4231→4444） --
def collate_fn(batch):
    xs, lens, ys = zip(*batch)
    max_len = max(lens)
    B = len(xs)
    x_pad = torch.zeros(B, max_len, dtype=torch.long)
    for i, (x, l) in enumerate(zip(xs, lens)):
        x_pad[i, :l] = x
    return x_pad, torch.tensor(lens, dtype=torch.long), torch.stack(ys)

# -- 检验指标为recall以及ndcgs -- 
'''
    装饰器防止误用
'''
@torch.no_grad()
def eval_metrics(model, loader, device, K=20):
    model.eval() # 评估模式
    hits, ndcgs, n = 0, 0.0, 0 # 赋予初值
    for x, lens, y in loader: # loader数据
        x, lens, y = x.to(device), lens.to(device), y.to(device) # tensor
        logits = model(x, lens) #进行预测 得到若干商品打分
        topk = torch.topk(logits, K, dim=1).indices  # (B, K) #得到得分高的前k个商品
        for i in range(y.size(0)):
            n += 1 #计数
            if y[i] in topk[i]: # 如果标签在topk里面的话
                hits += 1 #hits加1
                rank = (topk[i] == y[i]).nonzero(as_tuple=True)[0].item() + 1 #得到排名
                ndcgs += 1.0 / math.log2(rank + 1) #得到ndcg得分 公式来的
    return hits / n, ndcgs / n

# -- 生成文件名 --
def make_run_name(args):
    if args.model_type == "gru":
        return f"gru_e{args.epochs}_bs{args.batch_size}_dim{args.embed_dim}_hid{args.hidden_dim}"
    else:
        return f"tf_e{args.epochs}_bs{args.batch_size}_dim{args.embed_dim}_h{args.n_heads}_l{args.n_layers}_ff{args.ff_dim}"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_v3")
    ap.add_argument("--max_seq_len", type=int, default=100)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_type", choices=["gru", "transformer"], default="gru")
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--ff_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--loss_type", choices=["full", "inbatch"], default="full")

    args = ap.parse_args()

    # -- path 后续用于读取以及区别保存 --
    vocab_path = os.path.join(args.data_dir, "item_vocab.csv")
    train_path = os.path.join(args.data_dir, "train_next.csv")
    val_path   = os.path.join(args.data_dir, "val_next.csv")

    item2idx, num_items = load_vocab(vocab_path)
    print("[INFO] num_items:", num_items)
    
    run_name = args.run_name.strip() or make_run_name(args)
    run_name = run_name + f"_loss{args.loss_type}"
    out_dir = ensure_dir(os.path.join("outputs", run_name))
    run_dir = os.path.join("outputs", run_name)


    # 保存 config（含时间戳，便于复现）
    cfg = vars(args).copy() # 这里的var是要做成字典，是因为本身arg是namespace的格式
    cfg.update({
        "run_name": run_name,
        "num_items": num_items,
        "data_dir_resolved": os.path.abspath(args.data_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })
    # -- 保存静态配置 --
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # -- 保存动态指标用jsonl --
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    best_path = os.path.join(out_dir, "best.pt")
    best_ndcg = -1.0

    # -- 构造dataset（一般必须要有特征以及标签） -- 
    train_ds = NextItemDataset(train_path, item2idx, args.max_seq_len)
    val_ds   = NextItemDataset(val_path, item2idx, args.max_seq_len)

    # -- 使用官方的dataloader（一般不自己写，不然容易都算完了下一批数据还没存进来） --
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
    ) 

    # -- 按需选择model -- 
    if args.model_type == "gru":
        model = GRURec(
            num_items=num_items,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            padding_idx=PAD_IDX,
        )
    else:
        model = TransformerRec(
            num_items=num_items,
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            padding_idx=PAD_IDX,
            max_seq_len=args.max_seq_len,
        )
    # -- 使用gpu 还是cpu --
    model.to(args.device)
    # -- 优化器以及loss标准 --
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # -- 按照epoch进行循环 --
    for ep in range(1, args.epochs + 1):
        # -- train的模式（计算梯度与loss) --
        model.train()
        total_loss = 0.0
        for x, lens, y in train_loader:
            x, lens, y = x.to(args.device), lens.to(args.device), y.to(args.device) # totensor
            logits = model(x, lens) # 预测  # (B, num_items)
            
            # -- 这里判断是使用full的ce 还是使用负样本采样 --
            if args.loss_type == "full":
                loss = criterion(logits, y)
            else:
                # In-batch negative: use other samples' targets as negatives
                # logits: (B, num_items), y: (B,)
                # score[i, j] = logits[i, y[j]]
                score = logits[:, y]  # (B, B) advanced indexing

                labels = torch.arange(score.size(0), device=score.device)  # (B,)
                loss = torch.nn.functional.cross_entropy(score, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # -- 计算loss --
        avg_loss = total_loss / max(1, len(train_loader))
        r20, n20 = eval_metrics(model, val_loader, args.device, K=20)

        row = {
            "epoch": ep,
            "loss": float(avg_loss),
            "recall@20": float(r20),
            "ndcg@20": float(n20),
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 保存最佳模型（以 ndcg@20 为准）
        if n20 > best_ndcg:
            best_ndcg = n20
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": ep,
                    "best_ndcg@20": float(best_ndcg),
                    "run_name": run_name,
                },
                best_path,
            )

        print(f"[EPOCH {ep}] loss={avg_loss:.4f}  Recall@20={r20:.4f}  NDCG@20={n20:.4f}  (best_ndcg@20={best_ndcg:.4f})")


    print("[DONE]")
    # -- 追加对比表 --
    compare_path = os.path.join("outputs", "compare_runs.csv") # 定义路径
    header = not os.path.exists(compare_path) # 判断是不是第一次写
    summary = pd.DataFrame([{
        "run_name": run_name,
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embed_dim": args.embed_dim,
        "hidden_dim": getattr(args, "hidden_dim", None),
        "n_heads": getattr(args, "n_heads", None),
        "n_layers": getattr(args, "n_layers", None),
        "ff_dim": getattr(args, "ff_dim", None),
        "dropout": getattr(args, "dropout", None),
        "best_ndcg@20": float(best_ndcg),
        "out_dir": out_dir,
    }]) # 构建这次试验的summary
    ensure_dir("outputs") # 确保保存的目录存在
    summary.to_csv(compare_path, mode="a", header=header, index=False)#追加写入csv
    print("[SAVE] compare:", compare_path)
    print("[SAVE] run_dir:", out_dir)
    print("[SAVE] best_ckpt:", best_path)


if __name__ == "__main__":
    main()
