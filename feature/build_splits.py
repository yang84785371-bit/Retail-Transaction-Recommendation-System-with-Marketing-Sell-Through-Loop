'''
    这个主要是切分训练集和测试集合，直接看traindf和valdf会直观点
    把序列 → 监督样本（hist → target）
'''


import os
import argparse
import pandas as pd
import ast


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_path", default="data/processed_v3/sequences.csv")
    ap.add_argument("--out_dir", default="data/processed_v3")
    ap.add_argument("--max_seq_len", type=int, default=100)
    args = ap.parse_args()

    # -- 生成输出文件夹 --
    os.makedirs(args.out_dir, exist_ok=True)

    # -- 读取seq文件 --
    seq = pd.read_csv(args.seq_path)
    seq["seq_item_ids"] = seq["seq_item_ids"].apply(ast.literal_eval) # 将可能的伪list变成真list 不用eval

    train_rows, val_rows = [], []
    dropped = 0
    # -- 开始循环并且按行建立可迭代对象 --
    for _, r in seq.iterrows():
        user_id = int(r["user_id"])
        items = r["seq_item_ids"]
        items = [str(it).strip() for it in items if str(it).strip() != "" and str(it) != "nan"] #洗脏值，对于列表内非nan和非空的，str之后去掉首尾空格，形成新列表
        
        # -- item太短的话不要 --
        if len(items) < 2:
            dropped += 1
            continue

        # -- train: 到倒数第二跳（每个用户除最后一次外的所有 next-item） --
        # -- 生成阶梯状序列作为训练样本 --
        for t in range(1, len(items) - 1):
            hist = items[:t]
            target = items[t]
            hist = hist[-args.max_seq_len:]
            train_rows.append({
                "user_id": user_id,
                "hist": " ".join(map(str, hist)),
                "hist_len": len(hist),
                "target": str(target),
            })


        # -- val: 最后一跳 --
        # -- 与训练样本类似，但只保留最后一个作为target，其他作为hist --
        hist = items[:-1][-args.max_seq_len:]
        target = items[-1]
        val_rows.append({
            "user_id": user_id,
            "hist": " ".join(map(str, hist)),
            "hist_len": len(hist),
            "target": str(target),
        })

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_path = os.path.join(args.out_dir, "train_next.csv")
    val_path = os.path.join(args.out_dir, "val_next.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("[OK] saved:")
    print(" -", train_path, "rows=", len(train_df))
    print(" -", val_path,   "rows=", len(val_df))
    print("[INFO] dropped_users_len<2:", dropped)
    print("\n[val head]")
    print(val_df.head(3))


if __name__ == "__main__":
    main()
