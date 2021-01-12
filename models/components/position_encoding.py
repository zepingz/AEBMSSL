import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        assert x is not None
        pos = x.cumsum(-1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            pos = pos / (pos[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos = pos[:, :, None] / dim_t
        pos[:, 0::2] = pos[:, 0::2].sin()
        pos[:, 1::2] = pos[:, 1::2].cos()
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=64):
        super().__init__()
        self.embed = nn.Embedding(5, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x):
        bs, frame_num, _ = x.shape
        i = torch.arange(frame_num, device=x.device)
        emb = self.embed(i)
        pos = emb[None, :, :].repeat(bs, 1, 1)
        return pos


class PositionEmbeddingLearned3D(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=64):
        super().__init__()
        self.row_embed = nn.Embedding(10, num_pos_feats // 4)
        self.col_embed = nn.Embedding(10, num_pos_feats // 4)
        self.time_embed = nn.Embedding(10, num_pos_feats // 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.time_embed.weight)

    def forward(self, x):
        bs, frame_num, h, w = x.shape[:4]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        k = torch.arange(frame_num, device=x.device)

        x_emb = self.row_embed(i)[None, None, :, :].repeat(frame_num, h, 1, 1)
        y_emb = self.col_embed(j)[None, :, None, :].repeat(frame_num, 1, w, 1)
        time_emb = self.time_embed(k)[:, None, None, :].repeat(1, h, w, 1)

        pos = torch.cat([x_emb, y_emb, time_emb], dim=-1).unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        return pos
