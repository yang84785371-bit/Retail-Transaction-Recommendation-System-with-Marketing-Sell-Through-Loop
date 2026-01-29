# eval_baselines.py
import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


def metrics_recall_ndcg_at_k(targets, topk_lists, k=20):
    hit = 0
    ndcg = 0.0
    for t, topk in zip(targets, topk_lists):
        t = str(t)
        if t in topk:
            hit += 1
            rank = topk.index(t) + 1
            ndcg += 1.0 / np.log2(rank + 1)
    n = len(targets)
    return hit / n, ndcg / n


def build_pop_topk(train_targets, k=20):
    cnt = train_targets.astype(str).value_counts()
    return cnt.head(k).index.tolist(), cnt


def build_markov_from_train(train_path):
    # 读取 hist + target，用最后一个 hist item 作为 prev
    df = pd.read_csv(train_path, usecols=["hist", "target"])
    trans = defaultdict(Counter)  # prev -> Counter(next)
    for h, t in zip(df["hist"].astype(str), df["target"].astype(str)):
        hs = h.split()
        if len(hs) == 0:
            continue
        prev = hs[-1]
        trans[prev][t] += 1
    return trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_v3")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--baseline", choices=["pop", "markov"], default="pop")
    ap.add_argument("--save_compare", action="store_true")
    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, "train_next.csv")
    val_path = os.path.join(args.data_dir, "val_next.csv")

    train_t = pd.read_csv(train_path, usecols=["target"])["target"]
    val = pd.read_csv(val_path, usecols=["hist", "target"])
    targets = val["target"].astype(str).tolist()

    pop_topk, pop_cnt = build_pop_topk(train_t, k=args.k)
    print(f"[INFO] pop_top10: {pop_topk[:10]}")

    if args.baseline == "pop":
        topk_lists = [pop_topk] * len(targets)
        r, n = metrics_recall_ndcg_at_k(targets, topk_lists, k=args.k)
        print(f"[POP@{args.k}] Recall@{args.k}={r:.4f}  NDCG@{args.k}={n:.4f}")
        return

    # markov
    trans = build_markov_from_train(train_path)

    topk_lists = []
    for h in val["hist"].astype(str).tolist():
        hs = h.split()
        prev = hs[-1] if len(hs) > 0 else None

        recs = []
        if prev is not None and prev in trans:
            recs = [it for it, _ in trans[prev].most_common(args.k)]

        # 不够 K 用 popularity 补齐（去重）
        if len(recs) < args.k:
            seen = set(recs)
            for it in pop_topk:
                if it not in seen:
                    recs.append(it)
                    seen.add(it)
                if len(recs) >= args.k:
                    break

        topk_lists.append(recs[:args.k])

    r, n = metrics_recall_ndcg_at_k(targets, topk_lists, k=args.k)
    print(f"[MARKOV@{args.k}] Recall@{args.k}={r:.4f}  NDCG@{args.k}={n:.4f}")
    if args.save_compare:
        os.makedirs("outputs", exist_ok=True)
        compare_path = os.path.join("outputs", "compare_runs.csv")
        header = not os.path.exists(compare_path)
        row = pd.DataFrame([{
            "run_name": f"{args.baseline}_k{args.k}",
            "model_type": args.baseline,
            "epochs": None,
            "batch_size": None,
            "embed_dim": None,
            "hidden_dim": None,
            "n_heads": None,
            "n_layers": None,
            "ff_dim": None,
            "dropout": None,
            "best_ndcg@20": float(n),
            "out_dir": args.data_dir,
        }])
        row.to_csv(compare_path, mode="a", header=header, index=False)
        print("[SAVE] compare:", compare_path)   

if __name__ == "__main__":
    main()
