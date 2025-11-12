# cam_utils.py
# -*- coding: utf-8 -*-
"""
Grad-CAM for SAM: 使用 image_encoder 最后一层特征图 (neck[-1]) 生成热力图
适配思路：
- target layer = model_sam.image_encoder.neck[-1] (LayerNorm2d 或紧邻返回的卷积层)
- 目标标量 S = (上采样到特征图尺寸的所选掩膜 logit) 与其自身概率权重的逐像素乘积，再求和
- 反传得到梯度，做通道 GAP 得到权重，线性加权激活 → ReLU → 归一化 → 上采样到原图 → 叠加显示
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ========== 可选的图像反归一化 & 叠加工具 ==========

def denorm_image(t: torch.Tensor,
                 mean: Optional[Tuple[float, float, float]] = None,
                 std: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    t: Tensor[C,H,W], C=1 or 3
    return: np.uint8 RGB (H,W,3) in [0,255]
    """
    assert t.ndim == 3, f"expect CHW, got {t.shape}"
    x = t.detach().cpu().float()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    if (mean is not None) and (std is not None):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        x = x * std + mean

    # 兼容 [-1,1] 或 [0,1]
    t_min, t_max = float(x.min()), float(x.max())
    if t_min < -0.01 or t_max > 1.01:
        x = (x + 1.0) / 2.0

    x = x.clamp(0, 1)
    img = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def overlay_cam_on_image(rgb_uint8: np.ndarray,
                         cam01: np.ndarray,
                         alpha: float = 0.5,
                         colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    rgb_uint8: np.uint8 (H,W,3)
    cam01: np.float32/float64 (H,W) in [0,1]
    return: np.uint8 (H,W,3)
    """
    h, w, _ = rgb_uint8.shape
    cam01 = np.clip(cam01, 0.0, 1.0)
    heat = cv2.applyColorMap((cam01 * 255).astype(np.uint8), colormap)[:, :, ::-1]  # to RGB
    out = (alpha * heat + (1 - alpha) * rgb_uint8).clip(0, 255).astype(np.uint8)
    return out


# ========== 核心：Grad-CAM 实现 ==========

class _FeatureHook:
    """Forward/Backward hook：抓取 target layer 的激活和梯度"""
    def __init__(self, module: torch.nn.Module):
        self.fmap = None
        self.grad = None
        self.h1 = module.register_forward_hook(self._fhook)
        # PyTorch 1.10+ 推荐 full backward hook
        self.h2 = module.register_full_backward_hook(self._bhook)

    def _fhook(self, m, inp, out):
        self.fmap = out  # 不 detach，后面需要和权重同设备做运算

    def _bhook(self, m, gin, gout):
        # gout 是 tuple；我们要的是当前层输出的梯度
        self.grad = gout[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()


@torch.no_grad()
def _sigmoid01(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x).clamp(0, 1)


def _make_spatial_weight(mask_logits: torch.Tensor,
                         H: int, W: int, index: int = 0) -> torch.Tensor:
    """
    mask_logits: [B, M, 256, 256] 或 [B, 1, 256, 256]
    返回与 image feature 同尺寸的概率权重 [B,1,H,W]
    """
    prob = _sigmoid01(mask_logits[:, index:index+1])
    w = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)
    return w


def grad_cam_from_last_feature(model_sam,
                               image_tensor: torch.Tensor,
                               img_size: int,
                               mask_index: int = 0,
                               logits_key: str = "masks",
                               use_relu: bool = True) -> np.ndarray:
    """
    使用 SAM 的 image_encoder 最后一层特征图 (neck[-1]) 生成 Grad-CAM。
    关键点：只做一次完整前向 (model_sam(image_tensor, False, img_size))，
           hook 的激活与随后反传在同一次计算图上，保证对齐。
    """
    assert image_tensor.shape[0] == 1, "只支持单张图"
    device = next(model_sam.parameters()).device

    model_sam.train(False)                      # eval 行为，但允许梯度
    model_sam.zero_grad(set_to_none=True)
    image_tensor = image_tensor.to(device).float()

    # 1) 在 image_encoder 的返回前最后一层挂 hook（紧邻返回的 LN/Conv）
    target_layer = model_sam.image_encoder.neck[-1]
    hk = _FeatureHook(target_layer)

    # 2) 单次完整前向：与你验证阶段相同的接口
    #    注意：这里不能被 no_grad 包裹！外层请用 with torch.enable_grad():
    out = model_sam(image_tensor, False, img_size)   # e.g. 返回 {'masks': [1,M,256,256], ...}

    # 3) 取掩膜 logits，并根据 hook 的 fmap 尺寸对齐权重
    if isinstance(out, dict) and logits_key in out:
        mask_logits = out[logits_key]                # [1,M,256,256] or [1,1,256,256]
    else:
        mask_logits = out

    # fmap 由同一次前向得到，尺寸 [1,C,H,W]
    fmap = hk.fmap
    assert fmap is not None, "Hook 没抓到激活，检查 target_layer 是否正确"
    _, _, H, W = fmap.shape

    # 4) 标量目标：掩膜 logit 上采样到 (H,W)，用自身概率做空间权重
    w = _make_spatial_weight(mask_logits, H, W, index=mask_index)  # [1,1,H,W]
    logit_up = F.interpolate(mask_logits[:, mask_index:mask_index+1],
                             size=(H, W), mode='bilinear', align_corners=False)
    S = (logit_up * w).sum()

    # 5) 反传，取梯度
    S.backward(retain_graph=False)
    grad = hk.grad
    hk.close()

    # 6) GAP 通道权重 → 线性加权 → ReLU → 归一化
    weights = grad.mean(dim=(2, 3), keepdim=True)          # [1,C,1,1]
    cam = (weights * fmap).sum(dim=1, keepdim=True)        # [1,1,H,W]
    if use_relu:
        cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    # 7) 上采样回原图大小并转 numpy
    H0, W0 = image_tensor.shape[-2:]
    cam_up = F.interpolate(cam, size=(H0, W0), mode='bilinear', align_corners=False)[0, 0]
    return cam_up.detach().cpu().numpy().astype(np.float32)



# ========== 一步式：计算 + 叠加 + 保存 ==========

def save_cam_overlay(image_tensor: torch.Tensor,
                     cam01: np.ndarray,
                     save_path: str,
                     img_mean=None, img_std=None,
                     alpha: float = 0.5,
                     colormap: int = cv2.COLORMAP_JET):
    """
    把 cam 叠到原图并保存
    """
    assert image_tensor.ndim == 4 and image_tensor.shape[0] == 1
    rgb = denorm_image(image_tensor[0].cpu(), mean=img_mean, std=img_std)  # (H,W,3) uint8
    vis = overlay_cam_on_image(rgb, cam01, alpha=alpha, colormap=colormap)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(vis).save(save_path)


def generate_and_save_cam(model_sam,
                          image_tensor: torch.Tensor,
                          save_path: str,
                          mask_index: int = 0,
                          logits_key: str = "masks",
                          img_mean=None, img_std=None,
                          alpha: float = 0.5):
    """
    一行调用：生成 Grad-CAM → 叠加 → 保存
    """
    with torch.enable_grad():
        cam_np = grad_cam_from_last_feature(
            model_sam=model_sam,
            image_tensor=image_tensor,
            mask_index=mask_index,
            logits_key=logits_key
        )
    save_cam_overlay(
        image_tensor=image_tensor,
        cam01=cam_np,
        save_path=save_path,
        img_mean=img_mean, img_std=img_std,
        alpha=alpha
    )
