import torch
import torch.nn as nn

from typing import Type

from prostate.segment_anything.modeling.adapter.Filter import ChannelFilter
from prostate.segment_anything.modeling.modules.reins import Reins  # 🆕 导入 Reins
from prostate.segment_anything.modeling.modules.vHeat import Heat2D



class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.channel_filter = ChannelFilter()
        self.skip_connect = skip_connect
        self.heat2d = Heat2D(dim=D_features, hidden_dim=D_features)
        self.freq_embed = nn.Parameter(torch.randn(14, 14, D_features))
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        self.reins = Reins(
            num_layers=1,
            embed_dims=D_features,
            patch_size=16,
            token_length=64,
            use_softmax=True
        )

    def forward(self, x, embedding_feature, layer_id=None):
        """
        Args:
            x: [B, H, W, C] 通常是空间 token
            embedding_feature: [B, C, H, W]
        """

        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)  # Flatten spatial

        x = x + self.reins(x, layer=layer_id)

        x = x.view(B, H, W, C)


        x = x + embedding_feature.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        x = self.channel_filter(x)

        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs

        return x
