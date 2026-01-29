# model/transformer_rec.py
import torch
import torch.nn as nn


class TransformerRec(nn.Module):
    def __init__(
        self,
        num_items,
        embed_dim=64,
        n_heads=4,
        n_layers=2,
        ff_dim=256,
        dropout=0.1,
        padding_idx=0,
        max_seq_len=100,
    ):
        super().__init__()
        self.num_items = num_items
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_items)

    def forward(self, x, lengths):
        """
        x: (B, T) item_idx
        lengths: (B,) valid length
        """
        device = x.device # 设备
        B, T = x.shape # 形状

        # -- 给个编号 为pos embedding做准备 --
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)

        # -- 融合 以及dropout减少过拟合 --
        h = self.item_emb(x) + self.pos_emb(pos)  # (B, T, D)
        h = self.dropout(h)

        # -- padding --
        key_padding_mask = (x == self.padding_idx)  # (B, T)

        # -- encoder 自注意力更新特征 --
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B, T, D)

        # -- 取最后一个hist --
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        out = h[torch.arange(B, device=device), last_idx]  # (B, D) 高级索引

        logits = self.fc(out)  # (B, num_items)
        return logits
