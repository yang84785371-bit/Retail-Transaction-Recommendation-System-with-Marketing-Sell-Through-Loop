# feature/build_train_samples.py
import os
import argparse
import pandas as pd


def make_samples_from_seq(items, max_seq_len=100, min_hist_len=1):
    """
    items: list[str]  某个用户的完整序列（按时间排序）
    生成样本：(hist -> target)
    - hist 截断到最近 max_seq_len
    - 使用 sliding window：每个位置都生成一条 next-item 样本
    """
    samples = []
    if not isinstance(items, list) or len(items) < (min_hist_len + 1):
        return samples

    # sliding window：t 从 1 到 len(items)-1
    for t in range(1, len(items)):
        hist = items[:t]
        target = items[t]

        # 截断：只保留最近 max_seq_len
        if len(hist) > max_seq_len:
            hist = hist[-max_seq_len:]

        if len(hist) >= min_hist_len:
            samples.append((hist, target))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_path", default="data/processed/sequences.csv")
    ap.add_argument("--out_path", default="data/processed/train_samples.csv")
    ap.add_argument("--max_seq_len", type=int, default=100)
    ap.add_argument("--min_hist_len", type=int, default=1)
    ap.add_argument("--max_users", type=int, default=0, help="0 means all users (debug用)")
    args = ap.parse_args()

    seq = pd.read_csv(args.seq_path)

    # seq_item_ids 在 csv 里是字符串形式的 list，需要还原
    # 例如: "['85048', '79323P', ...]"
    # 用 ast.literal_eval 安全解析
    import ast
    seq["seq_item_ids"] = seq["seq_item_ids"].apply(ast.literal_eval)

    if args.max_users and args.max_users > 0:
        seq = seq.head(args.max_users)

    rows = []
    total = 0
    for _, r in seq.iterrows():
        user_id = int(r["user_id"])
        items = r["seq_item_ids"]
        samples = make_samples_from_seq(
            items,
            max_seq_len=args.max_seq_len,
            min_hist_len=args.min_hist_len,
        )
        for hist, target in samples:
            rows.append({
                "user_id": user_id,
                "hist": " ".join(map(str, hist)),   # 用空格连接，后面再 split
                "hist_len": len(hist),
                "target": str(target),
            })
        total += len(samples)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out_df.to_csv(args.out_path, index=False)

    print("[OK] saved:", args.out_path)
    print("[INFO] users:", len(seq), "samples:", total)
    print(out_df.head(3))


if __name__ == "__main__":
    main()
