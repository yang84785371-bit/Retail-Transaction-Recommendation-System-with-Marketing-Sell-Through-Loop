'''
    Docstring for foreign_trade_reco.model.gru_rec
'''
'''
    其实就是最终的隐状态 等于 前一时刻隐状态和候选隐状态加权 前者的权重是1-更新门 那就是不更新门就是沿用门 后者就是更新门 
    那候选隐状态怎么选 也很简单 我们考虑前一时刻我们需要考虑多少 就是重置门 这个和当下继续加权就可以了 sigmoid是因为是比重 tanh是因为状态本身有正有负
    更新门代表有多少需要更新 重置的话 是否考虑过去的兴趣
'''


import torch
import torch.nn as nn

# -- 从父类nnModule里定义一个类类型--
class GRURec(nn.Module):
    def __init__(self, num_items, embed_dim=64, hidden_dim=128, padding_idx=0):
        # -- 初始化一些值 --
        super().__init__()
        self.embedding = nn.Embedding(
            num_items, embed_dim, padding_idx=padding_idx
        ) # embedding
        self.gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True
        ) # gru神经网络
        self.fc = nn.Linear(hidden_dim, num_items) # 全连接神经网络

    def forward(self, x, lengths):
        """
        x: (B, T) item_idx 序列
        lengths: (B,) 每条序列的真实长度
        """
        # -- embedding --
        emb = self.embedding(x)  # (B, T, D)

        # -- 对数量少的序列进行padding，并且告诉模型哪些是padding的，459257 → 999999，因为要矩阵运算
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # --走gru，加上了门控状态的rnn --
        _, h = self.gru(packed)  # h: (1, B, H)
        h = h.squeeze(0)         # (B, H)

        logits = self.fc(h)      # (B, num_items)
        return logits
