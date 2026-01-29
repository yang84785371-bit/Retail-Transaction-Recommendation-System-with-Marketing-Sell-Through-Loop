# marketing/build_user_strategy_items.py
import os
import argparse
import pandas as pd

# -- mix items 说的是如何进行推荐商品 关于模型 人群分类以及物品画像分类的混合 --
'''
    这里就是混合器 将原本路线B用户的 进行一定的fast的补充以及slow的补充 做了动销的约束
'''
def mix_items(primary, slow_pool, fast_pool, topk, slow_ratio):
    slow_k = int(topk * slow_ratio) # 确认慢消池数量
    fast_k = topk - slow_k # 确认快消池的数量

    items = [] # 新建一个item 来容纳 该用户的推荐商品
    items += primary[: min(len(primary), fast_k)] # 按照主要推荐商品 以及 fast的min值来判断 固定fast为某个数值 默认用此数值 如果主推太少就用fast补充
    if len(items) < fast_k:
        items += fast_pool[: fast_k - len(items)]

    items += slow_pool[: slow_k] #补充慢消商品 填满k个

    # 去重 + 截断
    seen = set() # 设定一个集合 作为见过的
    out = [] # 设定一个字典 作为输出的
    # --这是逐个取 不重复的并入seen
    for it in items: # 取每个item
        if it not in seen: # 如果没有见过
            out.append(it) # 那就并入输出
            seen.add(it) # 并且并入出现过了
        if len(out) >= topk: # 工程保险罢了
            break
    return out

# -- 这函数其实就是提取各个人群前k个推荐商品 也就是历史上人群里卖得最好的商品 --
def load_top_items_by_segment(path: str, topk: int) -> dict:
    df = pd.read_csv(path) # 读取数据
    # -- 兜底 看是否报错 --
    if "segment" not in df.columns or "item_id" not in df.columns:
        raise ValueError("segment_top_items.csv must contain columns: segment, item_id")
    out = {}# 新建一个字典用于筛选
    for seg in df["segment"].unique(): # 明显 这里是取人群的分类 4个
        sub = df[df["segment"] == seg].head(topk) # 选每个人群的top4个推荐商品
        out[seg] = sub["item_id"].astype(str).tolist() # 只取商品的id 并且做成字典
    return out


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_path", default="data/marketing/user_segments.csv") # 人群画像分层
    ap.add_argument("--sellthrough_path", default="data/marketing/item_sellthrough.csv") # 商品画像分层
    ap.add_argument("--segment_top_items_path", default="data/marketing/segment_top_items.csv") # 人群商品推荐
    ap.add_argument("--out_path", default="data/marketing/user_strategy_items.csv") # 输出路径
    ap.add_argument("--slow_mix_ratio", type=float, default=0.2,
                    help="ratio of slow items mixed into non-clearance strategies") # slow的混合比例
    ap.add_argument("--topk", type=int, default=20) # topk 前k个
    ap.add_argument("--slow_pool_k", type=int, default=30, help="global slow pool size for clearance") # 慢消参数
    ap.add_argument("--fast_pool_k", type=int, default=30, help="global fast pool size for safe conversion") # 快消参数
    ap.add_argument("--interactions_path", default="data/processed_v3/interactions.csv")
    ap.add_argument("--user_hist_topk", default=50)
    args = ap.parse_args()

    # -- 读取文件 -- 
    seg = pd.read_csv(args.segments_path) # 用户人群分类
    st = pd.read_csv(args.sellthrough_path) # 商品分类

    # -- 基础清洗（类型转换或确认） -- 
    seg["user_id"] = seg["user_id"].astype(int)
    seg["segment"] = seg["segment"].astype(str)

    st["item_id"] = st["item_id"].astype(str)
    st["sell_flag"] = st["sell_flag"].astype(str)

    # -- 局商品池：slow / fast（按销量排序） --
    sales_col = [c for c in st.columns if c.startswith("sales_") and c.endswith("d")] # 寻找sale的列名 就是寻找三十天内物品销售的数量 但仅仅是列名 这里这么写和别名类似 是不为了被锁死
    if not sales_col:
        raise ValueError("sellthrough csv must contain sales_{X}d column.") # 兜底 如果sales col 为空那必须要报错
    sales_col = sales_col[0]

    # -- 构建双池 一个是快消池 一个是慢消池 --
    slow_pool = st[st["sell_flag"] == "slow"].sort_values(sales_col, ascending=True)["item_id"].head(args.slow_pool_k).tolist()
    fast_pool = st[st["sell_flag"] == "fast"].sort_values(sales_col, ascending=False)["item_id"].head(args.fast_pool_k).tolist()

    # -- 这里面 我们根据我们已经生成的文件 去提取前k个 各个人群的推荐商品 即历史最高的k个商品--
    seg_top = load_top_items_by_segment(args.segment_top_items_path, topk=args.topk)
    # -- 为at risk 做准备 --
    inter = pd.read_csv(args.interactions_path) # 读取interaction 
    inter["user_id"] = inter["user_id"].astype(int) # 确认类型不出错 
    inter["item_id"] = inter["item_id"].astype(str) #

    # -- 方案：历史高频（按出现次数）
    user_hist = (
        inter.groupby(["user_id", "item_id"])
        .size() # 这里用到size 就是直接计算双id 同时出现的次数 就是 某个用户购买某个商品的历史数量
        .reset_index(name="cnt") # 列名
        .sort_values(["user_id", "cnt"], ascending=[True, False]) # 进行排序 前者无所谓 后者的话 因为我们要取高频的 因此后者按照降序 选择false
    )

    user2hist = (
        user_hist.groupby("user_id")["item_id"]
        .apply(lambda s: s.head(int(args.user_hist_topk)).tolist()) # 取top k个
        .to_dict()
    )

    # -- 新建一个列表用于填充
    rows = []
    # -- 这里比较明显 索引用占位符占位 然后取其他有用信息为r --
    for _, r in seg.iterrows():
        uid = int(r["user_id"])
        s = r["segment"]

        '''

        '''

        # --按照人群不同制定不同的策略 --
        if s == "Champions":
            strategy = "model"
            # 先用 segment top 做一个占位（后续可替换成模型推理 topK）
            items = mix_items(
                primary=seg_top.get(s, []),
                slow_pool=slow_pool,
                fast_pool=fast_pool,
                topk=args.topk,
                slow_ratio=args.slow_mix_ratio
            )
        # -- 这里也使用了占位的思想 --
        # -- 对于hybrid的strategy来说 primary更复杂 一般来说 是使用 模型进行topk个推荐 以及使用用户画像人群的 topk个推荐 然后选择策略进行混合
        elif s == "Potential":
            strategy = "hybrid"
            items = mix_items(
                primary=seg_top.get(s, []),
                slow_pool=slow_pool,
                fast_pool=fast_pool,
                topk=args.topk,
                slow_ratio=args.slow_mix_ratio
            )

        elif s == "At_Risk":
            strategy = "winback_clearance"
            primary = user2hist.get(uid, [])   # 该用户历史高频
            items = mix_items(
                primary=primary,
                slow_pool=slow_pool,
                fast_pool=fast_pool,
                topk=args.topk,
                slow_ratio=args.slow_mix_ratio  # 你也可以单独给 At_Risk 更高一点
            )


        else:  # Low_Value
            strategy = "pop_fast"
            # 用 segment 热销，但尽量落在 fast（更易卖出）
            items = mix_items(
                primary=seg_top.get("Low_Value", []), # 人群推荐商为主 slow与fast为辅
                slow_pool=slow_pool,
                fast_pool=fast_pool,
                topk=args.topk,
                slow_ratio=args.slow_mix_ratio
            )


        rows.append({
            "user_id": uid,
            "segment": s,
            "strategy": strategy,
            "items": " ".join(items),
        }) # 逐个并入到我们row里面

    out = pd.DataFrame(rows) # 转换成pd 
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out.to_csv(args.out_path, index=False)

    print("[SAVE]", args.out_path)
    print(out["strategy"].value_counts())
    print("\n[HEAD]")
    print(out.head(5))


if __name__ == "__main__":
    main()
