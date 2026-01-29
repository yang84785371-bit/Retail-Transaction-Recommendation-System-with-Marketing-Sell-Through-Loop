'''
    将用户的历史交易行为压缩成用户的价值行为
    主要就是去考察用户是否重要 是否活跃 是否有价值
    输出是用户画像价值表
    user_id | recency_days | frequency | monetary_qty
'''
import os
import argparse
import pandas as pd


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_path", default="data/processed_v3/interactions.csv")
    ap.add_argument("--out_path", default="data/marketing/user_rfm.csv")
    ap.add_argument("--ref_date", default="", help="reference date YYYY-MM-DD, default=last ts in data")
    args = ap.parse_args()

    # -- 读取数据 -- 
    df = pd.read_csv(args.interactions_path) # 这里可以看到rfm的数据首先是基础事实的交易行为

    # -- 时间转换 --
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce") # 转换失败的变成NA
    df = df.dropna(subset=["user_id", "invoice", "ts"]) # 去掉NA

    # -- 托底 必须要有数量 --
    if "qty" not in df.columns:
        raise ValueError(
            "interactions.csv has no 'qty' column. "
            "Please rebuild interactions to include qty, or switch M to 'item_count'."
        )

    # -- qty类型转换 --
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    # -- 相对日期 --
    '''
        两种模式，要么你指定要么用最大日期
    '''
    if args.ref_date.strip():
        ref = pd.to_datetime(args.ref_date)
    else:
        ref = df["ts"].max()

    #  -- 用户最后的购买日期 --
    last_ts = df.groupby("user_id")["ts"].max()

    # -- 用户购买的频率 --
    freq = df.groupby("user_id")["invoice"].nunique()

    # -- 用户购买的数量 --
    mon = df.groupby("user_id")["qty"].sum()

    # -- 汇编成df --
    out = pd.DataFrame({
        "user_id": last_ts.index.astype(int),
        "recency_days": (ref - last_ts).dt.days.astype(int),
        "frequency": freq.values.astype(int),
        "monetary_qty": mon.values.astype(float),
    })

    out = out.sort_values(["frequency", "monetary_qty"], ascending=False) # 排序 按照频率和数量排序
    out = out.reset_index(drop=True) # 重置索引（无用的）
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True) # mkdir
    out.to_csv(args.out_path, index=False) #存储到csv

    print("[SAVE]", args.out_path)
    print("[INFO] users:", len(out), "ref_date:", ref.date())
    print(out.head(5))


if __name__ == "__main__":
    main()
