'''
    build 之后我们手里有这个user_id | recency_days | frequency | monetary_qty
    但是 需要考虑一个问题就是无法对业务进行执行策略
    segment_rfm.py主要回答以下几个问题 目前的用户按照其价值该分成哪几类？
    主要分为 核心用户 champion  近期有买 买的频繁 数量又多
    潜在用户 近期有买 但是频率和数量都一般
    风险用户 买的频繁 数量多 但是很久没买了 证明失去粘性
    无价值用户 很久没买 就算挽回 价值可能也不高
'''

import os
import argparse
import pandas as pd

# -- 给rfm分成若干个档次 -- 
def qcut_rank(s: pd.Series, q=5, ascending=True):
    s2 = s.rank(method="first", ascending=ascending) # 排序 给编号
    return pd.qcut(s2, q=q, labels=False) + 1 # 按照编号进行q分位数等分


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfm_path", default="data/marketing/user_rfm.csv")
    ap.add_argument("--out_path", default="data/marketing/user_segments.csv")
    args = ap.parse_args()

    # -- 读取用户价值表 --
    df = pd.read_csv(args.rfm_path)

    # 分箱：R 越小越好，所以 ascending=True；F/M 越大越好，所以 ascending=False
    df["r_bin"] = qcut_rank(df["recency_days"], q=5, ascending=True) # ascending 降序 代表数值越小越在前面 越是高分
    df["f_bin"] = qcut_rank(df["frequency"], q=5, ascending=False) # 类似但相反
    df["m_bin"] = qcut_rank(df["monetary_qty"], q=5, ascending=False)

    '''
        | Segment   | 含义                 |
        | --------- | ------------------ |
        | Champions | 最近买、买得多、买得大 → 核心用户 |
        | Potential | 最近活跃，但频次或金额还没拉满    |
        | At_Risk   | 很久没买，但历史价值高        |
        | Low_Value | 不活跃 + 价值低          |
    '''
    def assign(row):
        r, f, m = row["r_bin"], row["f_bin"], row["m_bin"]
        # 这里的阈值：1-2 视为“好”，4-5 视为“差/高”
        if r <= 2 and (f <= 2 and m <= 2):
            return "Champions"
        if r >= 4 and (f <= 2 or m <= 2):
            return "At_Risk"
        if r <= 2 and (3 <= f <= 4 or 3 <= m <= 4):
            return "Potential"
        return "Low_Value"

    df["segment"] = df.apply(assign, axis=1) # 按行执行 每个用户各判别一次

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_csv(args.out_path, index=False)

    print("[SAVE]", args.out_path)
    print(df["segment"].value_counts())
    print("\n[HEAD]")
    print(df.head(5))


if __name__ == "__main__":
    main()
