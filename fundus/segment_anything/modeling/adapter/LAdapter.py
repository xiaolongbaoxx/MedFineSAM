import math

import numpy as np
import torch
from numpy import random
from torch import nn

import torch.nn.functional as F
from torch.nn.init import _no_grad_trunc_normal_

from fundus.segment_anything.modeling.adapter.Filter import ChannelFilter


device = "cuda" if torch.cuda.is_available() else "cpu"

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class LAdapter(nn.Module):
    def __init__(self, scale_factor, embed_dim,  depth):
        """
        Args:
        """
        super(LAdapter, self).__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.depth = depth
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim)

        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor),
                nn.GELU(),
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim)
            )
            channel_filter = ChannelFilter()
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)
            setattr(self, 'channel_filter_{}'.format(str(i)), channel_filter)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _no_grad_trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape
        x = x.reshape(N, C, H*W).permute(0, 2, 1)
        embedding_feature = self.embedding_generator(x)
        embedding_feature = embedding_feature.permute(0, 2, 1).reshape(N, -1, H , W)
        return embedding_feature

    def forward(self, i, feature, embedding_feature):#
        N, C, H, W = embedding_feature.shape
        feature = feature.permute(0, 3, 1, 2)
        channel_filter = getattr(self, 'channel_filter_{}'.format(str(i)))
        filtered_feature = channel_filter(0.5*feature+embedding_feature)
        filtered_feature = filtered_feature.reshape(N, C, H * W).permute(0, 2, 1)
        lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
        prompt = lightweight_mlp(filtered_feature)
        return prompt
#
#
# import math
#
# import numpy as np
# import torch
# from numpy import random
# from torch import nn
#
# import torch.nn.functional as F
# from torch.nn.init import _no_grad_trunc_normal_
# from fundus.segment_anything.modeling.modules.reins import Reins
# from fundus.segment_anything.modeling.adapter.Filter import ChannelFilter
# from fundus.segment_anything.modeling.modules.vHeat import Heat2D  # 顶部添加导入
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# def calc_mean_std(feat, eps=1e-5):
#     # eps is a small value added to the variance to avoid divide-by-zero.
#     size = feat.size()
#     assert (len(size) == 4)
#     N, C = size[:2]
#     feat_var = feat.view(N, C, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(N, C, 1, 1)
#     feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#     return feat_mean, feat_std
#
#
# class LAdapter(nn.Module):
#     def __init__(self, scale_factor, embed_dim,  depth):
#         """
#         Args:
#         """
#         super(LAdapter, self).__init__()
#         self.scale_factor = scale_factor
#         self.embed_dim = embed_dim
#         self.depth = depth
#         self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim)
#         # 初始化 Reins（用 depth 来控制层级）
#         # self.heat2d = Heat2D(dim=embed_dim, hidden_dim=embed_dim)
#         self.heat2d = Heat2D(dim=self.embed_dim, hidden_dim=self.embed_dim)
#         # self.freq_embed = nn.Parameter(torch.randn(14, 14, embed_dim))  # 可换成动态生成
#         self.freq_embed_template = nn.Parameter(torch.randn(1, 14, 14, self.embed_dim))  # ✅ 新的模板
#
#         self.reins = Reins(
#             num_layers=depth,
#             embed_dims=self.embed_dim,
#             patch_size=16,
#             token_length=32,  # 可调，16/32 较轻量
#             use_softmax=True
#         )
#         print("[LOADING LAdapter]  embed_dim arg:", embed_dim)
#
#         for i in range(self.depth):
#             lightweight_mlp = nn.Sequential(
#                 nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor),
#                 nn.GELU(),
#                 nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim)
#             )
#             channel_filter = ChannelFilter()
#             setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)
#             setattr(self, 'channel_filter_{}'.format(str(i)), channel_filter)
#
#         self.apply(self._init_weights)
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             _no_grad_trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def init_embeddings(self, x):
#         N, C, H, W = x.permute(0, 3, 1, 2).shape
#         x = x.reshape(N, C, H*W).permute(0, 2, 1)
#         embedding_feature = self.embedding_generator(x)
#         embedding_feature = embedding_feature.permute(0, 2, 1).reshape(N, -1, H , W)
#         return embedding_feature
#
#     def forward(self, i, feature, embedding_feature):
#         """
#         i: 当前 LAdapter 在堆叠中的层级 (从 0 开始)
#         feature: [B, H*W, C] 的 token 序列
#         embedding_feature: [B, C, H, W] 的图像特征
#         """
#         B, C, H, W = embedding_feature.shape
#
#         # 1. 默认不增强
#         enhanced = embedding_feature
#
#         # 2. 只在最浅层 (i == 0) 做 Heat2D 频率扩散
#         if i == 0:
#             # 构造 [B, C, 14, 14]
#             freq = self.freq_embed_template.expand(B, -1, -1, -1).permute(0, 3, 1, 2)
#             # 插值到当前特征尺寸 [H, W]
#             freq = F.interpolate(freq, size=(H, W),
#                                  mode='bicubic', align_corners=False)
#             # 恢复到 [B, H, W, C]
#             freq = freq.permute(0, 2, 3, 1).contiguous()
#             # 计算热扩散特征并做残差融合
#             heat = self.heat2d(embedding_feature, freq)  # [B, C, H, W]
#             enhanced = embedding_feature + heat
#
#         # 3. 将 token feature 从 [B,HW,C] 转回 [B,C,H,W]
#         x = feature.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
#
#         # 4. 融合增强特征
#         fused = 0.5 * x + enhanced                          # [B, C, H, W]
#         # 通道注意力需要 [B, H, W, C]
#         fused = fused.permute(0, 2, 3, 1).contiguous()      # [B, H, W, C]
#         # 调用对应层的 ChannelGate
#         channel_gate = getattr(self, f'channel_filter_{i}')
#         fused = channel_gate(fused)                         # [B, H, W, C]
#
#         # 5. 展平送入 MLP + ReINS
#         fused = fused.view(B, H * W, C)                     # [B, HW, C]
#         mlp = getattr(self, f'lightweight_mlp_{i}')
#         prompt = mlp(fused)                                 # [B, HW, C]
#         prompt = self.reins(prompt, layer=i,
#                              batch_first=True,
#                              has_cls_token=False)          # [B, HW, C]
#
#         return prompt
#     # def forward(self, i, feature, embedding_feature):#
#     #     B, C, H, W = embedding_feature.shape
#     #     feature = feature.permute(0, 3, 1, 2)
#     #     channel_filter = getattr(self, 'channel_filter_{}'.format(str(i)))
#     #     # filtered_feature = channel_filter(0.5*feature+embedding_feature)
#     #     # ==== 1. Heat2D 频率扩散增强 embedding_feature ====
#     #     # if self.freq_embed.shape[:2] != (H, W):
#     #     #     resized_freq = nn.functional.interpolate(
#     #     #         self.freq_embed.permute(2, 0, 1).unsqueeze(0),
#     #     #         size=(H, W),
#     #     #         mode='bicubic',
#     #     #         align_corners=False
#     #     #     )
#     #     #     freq_embed = resized_freq.squeeze(0).permute(1, 2, 0).contiguous()
#     #     # else:
#     #     #     freq_embed = self.freq_embed
#     #     # === 生成 batch-wise 的 freq_embed ===
#     #     B = embedding_feature.shape[0]
#     #     freq_embed = self.freq_embed_template.expand(B, -1, -1, -1)  # [B, 14, 14, C]
#     #     freq_embed = freq_embed.permute(0, 3, 1, 2)  # → [B, C, 14, 14]
#     #     freq_embed = nn.functional.interpolate(
#     #         freq_embed,
#     #         size=(H, W),
#     #         mode='bicubic',
#     #         align_corners=False
#     #     )
#     #     freq_embed = freq_embed.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
#     #
#     #     heat_feature = self.heat2d(embedding_feature, freq_embed)
#     #     enhanced_embedding = embedding_feature + heat_feature
#     #
#     #     # ==== 2. 然后进入 channel_filter ====
#     #     filtered_feature = channel_filter(0.5 * feature + enhanced_embedding)
#     #
#     #     filtered_feature = filtered_feature.reshape(B, C, H * W).permute(0, 2, 1)
#     #     lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
#     #     prompt = lightweight_mlp(filtered_feature)
#     #     # 使用 Reins 增强 prompt
#     #     prompt = self.reins(prompt, layer=i, batch_first=True, has_cls_token=False)
#     #
#     #     return prompt








