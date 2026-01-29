'''
    这个文件主要是检验策略与实际的占比是否是一致的
'''

# marketing/eval_strategy_sellthrough_mix.py
import os
import argparse
import pandas as pd


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_strategy_path", default="data/marketing/user_strategy_items.csv")
    ap.add_argument("--sellthrough_path", default="data/marketing/item_sellthrough.csv")
    ap.add_argument("--out_path", default="data/marketing/strategy_sellthrough_mix.csv")
    ap.add_argument("--k", type=int, default=20, help="top-k items per user (split by space)")
    args = ap.parse_args()
    # --读取文件 --
    us = pd.read_csv(args.user_strategy_path) # 策略以及推荐商品
    st = pd.read_csv(args.sellthrough_path) # 商品画像

    # -- 复制 以及清洗一下 --
    st = st[["item_id", "sell_flag"]].copy()
    st["item_id"] = st["item_id"].astype(str)
    st["sell_flag"] = st["sell_flag"].astype(str)

    # -- 将商品与画像从列表做成一个字典 ==
    item2flag = dict(zip(st["item_id"], st["sell_flag"]))

    rows = []
    for _, r in us.iterrows():
        strat = str(r["strategy"]) # 取策略
        items = str(r["items"]).split() # 取推荐商品
        items = items[: args.k] # 取前k个

        flags = [item2flag.get(it, "unknown") for it in items] # 这里是工程托底 就是获得item 对应的画像
        total = len(flags) if len(flags) > 0 else 1

        rows.append({
            "strategy": strat,
            "fast_cnt": sum(f == "fast" for f in flags),
            "slow_cnt": sum(f == "slow" for f in flags),
            "dead_cnt": sum(f == "dead" for f in flags),
            "unknown_cnt": sum(f == "unknown" for f in flags),
            "k": len(items),
        }) # 计数并且假如到rows里面

    df = pd.DataFrame(rows)

    # -- 这里是除以item的个数获得三个占比 --
    agg = df.groupby("strategy", as_index=False).sum(numeric_only=True) # 对每个策略进行聚合 并且求sum
    agg["fast_ratio"] = agg["fast_cnt"] / agg["k"]
    agg["slow_ratio"] = agg["slow_cnt"] / agg["k"]
    agg["dead_ratio"] = agg["dead_cnt"] / agg["k"]
    agg["unknown_ratio"] = agg["unknown_cnt"] / agg["k"]

    # 排序：看“清库存倾向”强不强（slow_ratio 高）
    agg = agg.sort_values(["slow_ratio", "fast_ratio"], ascending=False)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    agg.to_csv(args.out_path, index=False)

    print("[SAVE]", args.out_path)
    print("\n[STRATEGY MIX]")
    print(agg[["strategy","k","fast_ratio","slow_ratio","dead_ratio","unknown_ratio"]])


if __name__ == "__main__":
    main()
