'''
    将商品从简单的id变成了经营状态 也就是回答哪些货物卖的快 哪些货物卖的慢 哪些是死库存
    核心的产物是商品经营画像表 其实就是圈定那些商品是快消 哪些商品是慢消 哪些商品是死库存这样
'''
import os
import argparse
import pandas as pd


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_path", default="data/processed_v3/interactions.csv")
    ap.add_argument("--out_path", default="data/marketing/item_sellthrough.csv")
    ap.add_argument("--window_days", type=int, default=30, help="rolling window for sell-through")
    ap.add_argument("--fast_quantile", type=float, default=0.75, help="fast threshold quantile on sales_30d")
    args = ap.parse_args()

    # --读取mature data 事实行为表 --
    df = pd.read_csv(args.interactions_path)

    # -- 必要字段检查（营销这块必须有 qty） --
    required = {"item_id", "ts", "qty"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"interactions missing columns: {miss}. Please rebuild interactions with qty/ts/item_id.")

    # -- 检查一下数据 --
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["item_id", "ts", "qty"])

    # -- 只看正销量 避免出错 虽然之前已经清洗过了 --
    df = df[df["qty"] > 0]

    # -- 检查一下，是否是空 --
    if df.empty:
        raise ValueError("No valid rows after cleaning (qty>0). Check your interactions.csv.")

    # -- 起始日期 可以自己设定 但一般用最大日期 --
    ref_ts = df["ts"].max()
    start_ts = ref_ts - pd.Timedelta(days=args.window_days) # 这里是窗口开始时间

    # -- 窗口内动销：近 window_days 的销量 --
    w = df[df["ts"] >= start_ts].copy() #只保留窗口时间内的订单

    # -- sales_30d：窗口销量；last_sale_days：距离最近售出日期（全量数据上看 last sale 更合理）
    sales = w.groupby("item_id", as_index=False)["qty"].sum().rename(columns={"qty": f"sales_{args.window_days}d"}) # 注意他是窗口时间里
    last_sale = df.groupby("item_id", as_index=False)["ts"].max().rename(columns={"ts": "last_sale_ts"}) # 他是总的不约束的最后售出时间

    # -- 连接成表 --
    out = last_sale.merge(sales, on="item_id", how="left")
    sales_col = f"sales_{args.window_days}d" # 命名
    out[sales_col] = out[sales_col].fillna(0.0) # 对NA进行填充 因为有部分商品以前卖过 窗口时间内没卖过 就会NA

    out["last_sale_days"] = (ref_ts - out["last_sale_ts"]).dt.days.astype(int) #得到一个特征就是距离上次购买隔了多少天

    # -- 动销分层：fast / slow / dead --
    # fast 阈值：对“有销量的商品”做分位数，避免 dead 把阈值拉低
    positive = out[out[sales_col] > 0][sales_col] # 只拿近期买过的商品销量
    fast_th = float(positive.quantile(args.fast_quantile)) if len(positive) > 0 else 0.0 # 在这些近期卖过的商品中算一个分位数阈值

    # -- 根据阈值我们判断武平是快消慢消 还是死库存
    def tag(row):
        s = row[sales_col]
        if s <= 0:
            return "dead"
        return "fast" if s >= fast_th else "slow"

    out["sell_flag"] = out.apply(tag, axis=1) #按行来 每一个商品都判别一次

    # -- 排序：先 dead（越久未售越靠前），再 slow（销量小/久未售），再 fast（销量大） -- 
    out = out.sort_values(
        by=["sell_flag", "last_sale_days", sales_col], # 按照卖不出去的再到卖得出去得排序
        ascending=[True, False, True]
    )

    # 输出
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out = out[["item_id", sales_col, "last_sale_days", "sell_flag", "last_sale_ts"]]
    out.to_csv(args.out_path, index=False)

    # 简单汇报（便于你截图写报告/简历）
    print(f"[INFO] ref_date: {ref_ts.date()} window_days={args.window_days} fast_q={args.fast_quantile} fast_th={fast_th:.2f}")
    print("[SAVE]", args.out_path)
    print(out["sell_flag"].value_counts())
    print("\n[HEAD]")
    print(out.head(10))


if __name__ == "__main__":
    main()
