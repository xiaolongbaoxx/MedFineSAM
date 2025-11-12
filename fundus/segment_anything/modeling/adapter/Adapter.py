#
# import torch
# import torch.nn as nn
#
# from typing import Type
#
# from fundus.segment_anything.modeling.adapter.Filter import ChannelFilter
#
#
# class Adapter(nn.Module):
#     def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
#         super().__init__()
#         self.channel_filter = ChannelFilter()
#
#         self.skip_connect = skip_connect
#         D_hidden_features = int(D_features * mlp_ratio)
#         self.act = act_layer()
#         self.D_fc1 = nn.Linear(D_features, D_hidden_features)
#         self.D_fc2 = nn.Linear(D_hidden_features, D_features)
#
#     def forward(self, x, embedding_feature):#
#         x = x + embedding_feature.permute(0, 2, 3, 1)
#         x = self.channel_filter(x)
#
#         # x is (BT, HW+1, D)
#         xs = self.D_fc1(x)
#         xs = self.act(xs)
#         xs = self.D_fc2(xs)
#         if self.skip_connect:
#             x = x + xs
#         else:
#             x = xs
#         return x

import torch
import torch.nn as nn
from typing import Type

from fundus.segment_anything.modeling.adapter.Filter import ChannelFilter
from fundus.segment_anything.modeling.modules.reins import Reins  # 🆕 导入 Reins
from fundus.segment_anything.modeling.modules.vHeat import Heat2D



class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.channel_filter = ChannelFilter()
        self.skip_connect = skip_connect
        self.heat2d = Heat2D(dim=D_features, hidden_dim=D_features)
        self.freq_embed = nn.Parameter(torch.randn(14, 14, D_features))  # 可换成动态
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        # 🆕 初始化 Reins 模块（串联用）
        self.reins = Reins(
            num_layers=1,
            embed_dims=D_features,
            patch_size=16,         # 可根据主干 patch 设置
            token_length=64,       # 可调
            use_softmax=True
        )

    def forward(self, x, embedding_feature, layer_id=None):
        """
        Args:
            x: [B, H, W, C] 通常是空间 token
            embedding_feature: [B, C, H, W]
        """

        # ✅ 1. reshape x → [B, N, C]，才能给 Reins 用
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)  # Flatten spatial

        # ✅ 2. Reins 特征增强
        x = x + self.reins(x, layer=layer_id)

        # ✅ 3. reshape 回 [B, H, W, C]，以便下面做 image_feature 融合
        x = x.view(B, H, W, C)


        # 继续原来的 adapter 操作
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
    # def forward(self, x, embedding_feature, layer_id=None):
    #     """
    #     Args:
    #         x: [B, H, W, C]
    #         embedding_feature: [B, C, H, W]
    #     """
    #     B, C, H, W = embedding_feature.shape
    #
    #     # # === Heat2D 动态扩散增强 ===
    #     # if self.freq_embed.shape[:2] != (H, W):
    #     #     freq_embed = nn.functional.interpolate(
    #     #         self.freq_embed.permute(2, 0, 1).unsqueeze(0),
    #     #         size=(H, W),
    #     #         mode='bicubic',
    #     #         align_corners=False
    #     #     ).squeeze(0).permute(1, 2, 0).contiguous()
    #     # else:
    #     #     freq_embed = self.freq_embed
    #     #
    #     # heat_feature = self.heat2d(embedding_feature, freq_embed)
    #     # feature = embedding_feature + heat_feature
    #     # —— 只在第 0 层 使用 Heat2D ——
    #
    #     if layer_id == 0:
    #         if self.freq_embed.shape[:2] != (H, W):
    #             freq_embed = nn.functional.interpolate(
    #                     self.freq_embed.permute(2, 0, 1).unsqueeze(0),
    #                     size=(H, W),
    #                     mode='bicubic',
    #                     align_corners=False
    #                 ).squeeze(0).permute(1, 2, 0).contiguous()
    #         else:
    #             freq_embed = self.freq_embed
    #         heat_feature = self.heat2d(embedding_feature, freq_embed)
    #         feature = embedding_feature + heat_feature
    #     else:
    #             feature = embedding_feature
    #
    #     # === 融合 feature 到 token 上 ===
    #     x = x + feature.permute(0, 2, 3, 1)  # [B, H, W, C]
    #     x_flat = x.view(B, H * W, C)
    #
    #     # === ReINS 增强 token 表达 ===
    #     x_flat = x_flat + self.reins(x_flat, layer=layer_id)
    #     x = x_flat.view(B, H, W, C)
    #
    #     # === 通道注意力与 MLP ===
    #     x = self.channel_filter(x)
    #
    #     xs = self.D_fc1(x)
    #     xs = self.act(xs)
    #     xs = self.D_fc2(xs)
    #
    #     if self.skip_connect:
    #         x = x + xs
    #     else:
    #         x = xs
    #
    #     return x


