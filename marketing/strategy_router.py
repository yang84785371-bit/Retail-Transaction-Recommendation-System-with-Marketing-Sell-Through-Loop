'''
    该文件用于做最后的决策
'''
'''
    | segment   | strategy  | 业务含义         |
    | --------- | --------- | ------------ |
    | Champions | `model`   | 完全相信模型，追求相关性 |
    | Potential | `hybrid`  | 模型 + 人群偏好    |
    | At_Risk   | `history` | 用历史高频商品做唤回   |
    | Low_Value | `pop`     | 推大众款 / 清仓款   |
'''
import pandas as pd


def route_strategy(user_segment):
    if user_segment == "Champions":
        return "model" # 主要是因为这类人 行为稳定 可以用模型
    elif user_segment == "Potential":
        return "hybrid" # 不太稳定 虽然近期有购买 但是总体买的频率和数量都比较少 所以从数据信号分布来说 未收敛 全模型容易过拟合 全规则容易过度保守 所以用模型加人群兜底
    #模型加规则就是模型算出topk 规则给出历史topk 合并进候选池 按照模型给分 要么就2者结果加权混合 要么 按照阈值来 模型置信度高用模型 低就用规则
    elif user_segment == "At_Risk":
        return "history" # 有钱但买的少 一般用历史进行唤回
    else:
        return "pop" # 对于低价值人群就是大众加清仓货


if __name__ == "__main__":
    seg = pd.read_csv("foreign_trade_reco/data/marketing/user_segments.csv")
    seg["strategy"] = seg["segment"].apply(route_strategy)

    print(seg["strategy"].value_counts()) # 这里主要是分布检查 看是否极端

'''
    该数据集5成low value 2成核心用户 大概3成是潜在用户以及粘性风险用户（可营销的地方） 比较正常
'''