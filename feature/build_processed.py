'''
    此文件用于将raw data process成模型可用的feature
    现实世界 → 行为序列
'''
import os
import re
import argparse
import pandas as pd

# --从interaction生成vocab文件 -- 
def build_vocab_from_interactions(inter: pd.DataFrame) -> pd.DataFrame:
    # 0 留给 PAD，所以从 1 开始
    items = sorted(inter["item_id"].astype(str).unique().tolist())
    return pd.DataFrame({"item_id": items, "item_idx": range(1, len(items) + 1)})


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", default="data/raw/online_retail_II.xlsx")
    ap.add_argument("--out_dir", default="data/processed_v3")
    ap.add_argument("--bad_item_pattern", default="(TEST|POST|SAMPLE)")
    args = ap.parse_args()

    # -- 生成输出文件夹 --
    os.makedirs(args.out_dir, exist_ok=True)

    # -- 读取表格 --
    xls = pd.ExcelFile(args.raw_path)

    # -- 将表格中的两个sheet合成一个
    df = pd.concat([pd.read_excel(args.raw_path, sheet_name=s) for s in xls.sheet_names], ignore_index=True)

    # -- 统一字段 --
    df = df.rename(columns={
        "Customer ID": "user_id",
        "StockCode": "item_id",
        "InvoiceDate": "ts",
        "Invoice": "invoice",
        "Quantity": "qty",
        "Price": "price",
    })

    # -- 基础清洗 -- 
    df = df.dropna(subset=["user_id", "item_id", "ts", "invoice"]) #将na的去掉
    df["user_id"] = df["user_id"].astype("int64") # 格式转换
    df["item_id"] = df["item_id"].astype(str).str.strip() #格式转换
    df["invoice"] = df["invoice"].astype(str).str.strip() #格式转换
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce") #将times转换成datetime的格式
    df = df.dropna(subset=["ts"]) #转化失败的话就不要

    # -- 筛选走数量小于0，价格小于0，退货的 --
    df = df[df["qty"].astype(float) > 0]
    df = df[df["price"].astype(float) > 0]
    df = df[~df["invoice"].str.startswith("C", na=False)]

    # --过滤异常 item（TEST/POST/SAMPLE）--
    pat = re.compile(args.bad_item_pattern, re.IGNORECASE) # 构建一个正则字符串
    mask_bad = df["item_id"].apply(lambda x: bool(pat.search(x))) #进行匹配
    print("[INFO] bad_item_rows:", int(mask_bad.sum()))
    df = df[~mask_bad]

    # -- 过滤非商品类编码（如 BANK CHARGES / POSTAGE / ADJUST 等）--
    non_product_pat = re.compile(r"(BANK|CHARGES|POSTAGE|ADJUST|FEE)", re.IGNORECASE)
    df = df[~df["item_id"].str.contains(non_product_pat, na=False)]


    # -- 订单内去重：同一用户同一订单同一商品只算一次 --
    df = df.drop_duplicates(subset=["user_id", "invoice", "item_id"])

    # -- 生成序列 --
    df = df.sort_values(["user_id", "ts", "invoice", "item_id"]) # sort vlue 多级排序 生成我们需要的sequence 

    # -- 给营销用：保留 qty/price 方向A只需要用到前4者 -- 
    keep_cols = ["user_id", "invoice", "ts", "item_id", "qty", "price"] # invoice是订单号
    inter = df[keep_cols].copy()

    seq = inter.groupby("user_id")["item_id"].apply(list).reset_index(name="seq_item_ids") #按照userid进行聚合提取itemid变成list然后命名兵变会df
    seq["seq_len"] = seq["seq_item_ids"].apply(len) # 计算序列长度

    print("[OK] interactions:", inter.shape)
    print("[OK] sequences:", seq.shape)
    print(seq["seq_len"].describe())

    # -- 保存 --
    inter_path = os.path.join(args.out_dir, "interactions.csv")
    seq_path = os.path.join(args.out_dir, "sequences.csv")
    inter.to_csv(inter_path, index=False)
    seq.to_csv(seq_path, index=False)
    print("[SAVE]", inter_path)
    print("[SAVE]", seq_path)

    # -- vocab（基于 interactions 的 item_id）进行编码 --
    # -- 因为itemid作为字符串稀疏离散不可计算 --
    vocab = build_vocab_from_interactions(inter)
    vocab_path = os.path.join(args.out_dir, "item_vocab.csv")
    vocab.to_csv(vocab_path, index=False)
    print("[INFO] unique_items:", len(vocab))
    print("[SAVE]", vocab_path)
    print(vocab.head(5))




if __name__ == "__main__":
    main()
