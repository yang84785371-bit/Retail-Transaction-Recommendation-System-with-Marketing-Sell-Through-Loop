'''
    该文件主要用户生成用户主要爱买什么商品
    不然 就算想要对champion进行推荐 对at risk进行唤回 我们也不知道用什么商品 吸引他们来进行购买
    这里我们希望得到不同的用户画像购买的最多的topk个商品
'''
# marketing/segment_item_pref.py
import argparse
import pandas as pd

def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_path", required=True)
    ap.add_argument("--segments_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    # -- 读取数据 --
    inter = pd.read_csv(args.interactions_path) # mature data 事实行为 interaction
    seg = pd.read_csv(args.segments_path) # 用户画像分类

    df = inter.merge(seg[["user_id", "segment"]], on="user_id", how="inner")
    # -- 提取群体画像
    rows = []
    for seg_name, g in df.groupby("segment"):
        top_items = (
            g["item_id"]
            .value_counts() # 计算数量
            .head(args.topk) # 选topk
            .reset_index() # 重置索引
        )
        top_items.columns = ["item_id", "count"] # 列名
        top_items["segment"] = seg_name # 用户群体
        rows.append(top_items)

    out = pd.concat(rows, ignore_index=True) # 主要是消除多余列名 
    out.to_csv(args.out_path, index=False) # 保存

    print("[SAVE]", args.out_path)
    print("\n[PREVIEW]")
    print(out.groupby("segment").head(5))

if __name__ == "__main__":
    main()
