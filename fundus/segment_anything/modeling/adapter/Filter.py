
import torch
import torch.nn as nn

from typing import Type
import torch
import torch.fft as fft
import pywt



class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        output = self.sigmoid(max_result + avg_result)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class SpatialFilter(nn.Module):
    def __init__(self, kernel_size=7, channel=3072,reduction=16):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # SpatialAttention
        output = self.sa(x)
        x = x*output
        return x

class ChannelFilter(nn.Module):
    def __init__(self, channel=768):
        super().__init__()
        self.ca = ChannelAttention(channel)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # SpatialAttention
        output = self.ca(x)
        x = x*output
        x = x.permute(0, 2, 3, 1)
        return x

# import torch
# import torch.nn as nn

# class ChannelAttention(nn.Module):
#     def __init__(self, reduction=16):
#         super().__init__()
#         self.reduction = reduction
#         self.mlp = None
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         batch, channel, _, _ = x.shape  # 获取输入的通道数

#         # 确保最小通道数 >= 1，防止 Conv2d(0, x, 1) 的错误
#         hidden_dim = max(1, channel // self.reduction)

#         if self.mlp is None or self.mlp[0].in_channels != channel:
#             self.mlp = nn.Sequential(
#                 nn.Conv2d(channel, hidden_dim, 1, bias=False),  # 确保 hidden_dim >= 1
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(hidden_dim, channel, 1, bias=False)
#             ).to(x.device)

#         max_out = self.mlp(nn.AdaptiveMaxPool2d(1)(x))
#         avg_out = self.mlp(nn.AdaptiveAvgPool2d(1)(x))
#         out = max_out + avg_out
#         return self.sigmoid(out)



# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#         self.bn = nn.BatchNorm2d(1)  # 增加 Batch Normalization
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_result, _ = torch.max(x, dim=1, keepdim=True)
#         avg_result = torch.mean(x, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], dim=1)
#         output = self.conv(result)
#         output = self.bn(output)  # 归一化
#         return self.sigmoid(output)


# class SpatialFilter(nn.Module):
#     def __init__(self, kernel_size=7, channel=3072, reduction=16):
#         super().__init__()
#         self.sa = SpatialAttention(kernel_size=kernel_size)

#     def forward(self, x):
#         output = self.sa(x)
#         x = x * output
#         return x


# class ChannelFilter(nn.Module):
#     def __init__(self, channel=768):
#         super().__init__()
#         self.ca = ChannelAttention(channel)

#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
#         output = self.ca(x)
#         x = x * output
#         x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
#         return x

