# -*- coding: utf-8 -*-
"""
Minimal CAM/Attention visualization for SAM ViT-B with adapters.
Run:
python fundus/segment_anything/modeling/visualize_cam.py \
  --ckpt D:\learn\research\eye_data\RIGAPlus\output\output_prompt_rein\RIGA_source_domain_1_BinRushed_train_512_pretrain_vit_b_epo200_bs8_lr0.0005\epoch_159.pth \
  --image D:\learn\research\eye_data\RIGAPlus\Site_C\image\gdrishtiGS_001.png \
  --outdir ./viz_out \
  --image-size 512
"""

import os
import math
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------
# 1) 激活装饰器（务必在导入模型代码之前）
# ---------------------------------------------------------
from visualizer import get_local
get_local.activate()

# 之后再导入你项目里的模块（带 Adapter 的 image encoder）
from fundus.segment_anything.build_sam import build_sam_vit_b
try:
    from fundus.segment_anything.utils.transforms import ResizeLongestSide
except Exception:
    from fundus.segment_anything.utils.transforms import ResizeLongestSide

warnings.filterwarnings("once", category=FutureWarning)


# ----------------------------
# io helpers
# ----------------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_checkpoint_into(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # 一些训练脚本里会包含 channel_gate / classifier 等，与可视化无关的键
    drop_prefixes = (
        "channel_gate.", "classifier.", "mask_decoder.", "prompt_encoder."
    )
    state = {k: v for k, v in state.items()
             if not any(k.startswith(pf) for pf in drop_prefixes)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] strict=False | loaded={len(state)} | "
          f"missing={len(missing)} | unexpected={len(unexpected)}")


# ----------------------------
# SAM-like preprocess
# ----------------------------
def to_rgb(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img


def preprocess(pil_img: Image.Image, image_size: int, pixel_mean: torch.Tensor,
               pixel_std: torch.Tensor, device: torch.device):
    t = ResizeLongestSide(image_size)
    img_np = np.asarray(pil_img)  # H, W, 3 (uint8)
    x = t.apply_image(img_np)
    x = torch.as_tensor(x, device=device).permute(2, 0, 1).contiguous()[None]  # 1,3,H',W'
    x = x.float() / 255.0

    mean = pixel_mean.to(device)
    std = pixel_std.to(device)
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    x = (x - mean) / std
    return (img_np.astype(np.float32) / 255.0), x  # (0..1) for overlay


# ----------------------------
# CAM aggregation helpers
# ----------------------------
def tensor_to_img01(t: torch.Tensor) -> np.ndarray:
    t = t.detach().float()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t.cpu().numpy()


def overlay_rgb(base01: np.ndarray, heat01: np.ndarray, alpha=0.45):
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat01)[..., :3]
    out = (1 - alpha) * base01 + alpha * heat_rgb
    return np.clip(out, 0, 1)


def aggregate_attention_maps(attn_tensors, raw_h, raw_w):
    """
    attn_tensors: List[Tensor] with shape (B, heads, N, N), N = tokens per call.
    For windowed attention, each call covers一个窗口；这里做一个简单的平均聚合。
    """
    if len(attn_tensors) == 0:
        raise RuntimeError("No attention maps captured. "
                           "Please ensure @get_local('attn_map') is in Attention.forward.")

    heat_acc = None
    for A in attn_tensors:
        # ✅ 保证是 torch.Tensor
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A)
        # A: (B, heads, N, N)
        A = A[0]  # take batch 0
        A = A.mean(dim=0)        # -> (heads, N, N)
        A = A.mean(dim=1)        # -> (heads, N)  (mean over query tokens)
        A = A.mean(dim=0)        # -> (N,)       (mean over heads)

        N = A.numel()
        side = int(math.sqrt(N))
        if side * side != N:
            # 非正方形的话，跳过（或 reshape 失败）；一般不会发生
            continue
        A = A.view(1, side, side)                 # (1, h, w)
        A = F.interpolate(A[None], size=(raw_h, raw_w),
                          mode="bilinear", align_corners=False)[0, 0]  # (H, W)
        heat_acc = A if heat_acc is None else (heat_acc + A)

    if heat_acc is None:
        raise RuntimeError("All captured attention windows were invalid for visualization.")
    heat_acc = heat_acc / len(attn_tensors)
    return heat_acc


# ----------------------------
# main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=r"/root/autodl-tmp/RIGAPlus/output_heat/RIGA_Site_C_512_pretrain_vit_b_epo200_bs8_lr0.0005/epoch_159.pth")
    ap.add_argument("--image", type=str, default= r"/root/autodl-tmp/RIGAPlus/Site_C/image/ndrishtiGS_101.png")
    ap.add_argument("--outdir", type=str, default="./viz_out")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-classes", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 构建 ViT-B（带 adapter 的实现）；不要让 builder 自动加载权重
    sam = build_sam_vit_b(image_size=args.image_size, num_classes=args.num_classes)
    # 某些仓库返回 (model, transform)，做个兼容
    if isinstance(sam, (tuple, list)):
        sam = sam[0]

    # 3) 加载权重
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    # # 简单的 zip 检测（防止把日志/压缩包误传）
    # with open(args.ckpt, "rb") as f:
    #     sig = f.read(4)
    # if sig == b"PK\x03\x04":
    #     raise RuntimeError(
    #         "This looks like a zip/npz file, not a plain torch checkpoint. "
    #         "Please pass a valid *.pth produced by your training script."
    #     )

    load_checkpoint_into(sam, args.ckpt)

    sam.eval().to(device)

    # 4) 预处理
    pil = to_rgb(Image.open(args.image))
    base_np01, x = preprocess(pil, args.image_size, sam.pixel_mean, sam.pixel_std, device)

    # 5) 前向一次（只跑 image encoder）
    with torch.no_grad():
        _ = sam.image_encoder(x)

    # 6) 从装饰器缓存里取注意力
    cache = get_local.cache
    if "Attention.forward" not in cache or len(cache["Attention.forward"]) == 0:
        # 兼容某些版本的装饰器，把 key 切成类名.函数名保存
        keys = list(cache.keys())
        raise RuntimeError(
            f"No attention captured. Found keys in cache: {keys}. "
            f"Make sure @get_local('attn_map') decorates Attention.forward."
        )
    attn_list = cache["Attention.forward"]  # List[Tensor(B, heads, N, N)]

    # 7) 聚合 -> heatmap
    H, W = base_np01.shape[:2]
    heat = aggregate_attention_maps(attn_list, H, W)     # Tensor(H, W)
    heat01 = tensor_to_img01(heat)

    # 8) 保存
    heat_path = os.path.join(args.outdir, "attn_heat.png")
    overlay_path = os.path.join(args.outdir, "attn_overlay.png")
    plt.imsave(heat_path, heat01, cmap="jet")
    plt.imsave(overlay_path, overlay_rgb(base_np01, heat01))
    print(f"[done] saved:\n  {heat_path}\n  {overlay_path}")


if __name__ == "__main__":
    main()



# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# from fundus.segment_anything.build_sam import build_sam_vit_b
# from fundus.segment_anything.utils.transforms import ResizeLongestSide
# from fundus.segment_anything.modeling.visualizer import get_local

# def preprocess(pil_img, image_size, pixel_mean, pixel_std, device):
#     t = ResizeLongestSide(image_size)
#     img = np.asarray(pil_img.convert("RGB"))
#     img_resized = t.apply_image(img)  # H' x W' x 3, uint8
#     x = torch.as_tensor(img_resized, device=device).permute(2, 0, 1)[None].float() / 255.0
#     pm = pixel_mean.to(device).view(1, -1, 1, 1)
#     ps = pixel_std.to(device).view(1, -1, 1, 1)
#     x = (x - pm) / ps
#     return img.astype(np.float32) / 255.0, x

# def save_heat(outdir, name, heat, base_img):
#     os.makedirs(outdir, exist_ok=True)
#     heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
#     plt.imsave(os.path.join(outdir, f"{name}_heat.png"), heat, cmap="jet")
#     cmap = plt.get_cmap("jet")(heat)[..., :3]
#     overlay = 0.55 * cmap + 0.45 * base_img
#     overlay = np.clip(overlay, 0, 1)
#     plt.imsave(os.path.join(outdir, f"{name}_overlay.png"), overlay)

# def main():
#     # # === 路径参数 ===
#     # ckpt_path = "D:/learn/research/eye_data/RIGAPlus/output/output_heat/epoch_159.pth"
#     # image_path = "D:/learn/research/code/sam/dapsam_rein_heat/sample1.jpg"
#     # outdir = "D:\learn\\research\code\sam\dapsam_rein_heat\sample1_heat_out"
#     # image_size = 512
#     # num_classes = 2
#     #     # === 路径参数 ===
#     ckpt_path = "/root/autodl-tmp/RIGAPlus/output_heat/RIGA_Site_C_512_pretrain_vit_b_epo200_bs8_lr0.0005/epoch_159.pth"
#     image_path = "/root/autodl-tmp/RIGAPlus/Site_C/image/ndrishtiGS_101.png"
#     outdir = "/root/autodl-tmp/RIGAPlus/output_heat/sample_out"
#     image_size = 512
#     num_classes = 2


#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 1) 构建 SAM（vit_b）
#     sam = build_sam_vit_b(image_size=image_size, num_classes=num_classes)
#     if isinstance(sam, (tuple, list)):
#         sam = sam[0]
#     sam.eval().to(device)

#     # 2) 加载权重（过滤和你之前一样）
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     state = ckpt.get("model", ckpt.get("state_dict", ckpt))
#     # 常见无关/通道门控之类的可过滤；没有就不需要
#     filt = {k: v for k, v in state.items() if not k.startswith("channel_gate.")}
#     missing, unexpected = sam.load_state_dict(filt, strict=False)
#     print(f"[load] strict=False | loaded={len(filt)} | missing={len(missing)} | unexpected={len(unexpected)}")

#     # 3) 预处理
#     raw_np, x = preprocess(Image.open(image_path), image_size, sam.pixel_mean, sam.pixel_std, device)

#     # 4) 开启捕获，然后前向一次
#     get_local.activate(True)
#     with torch.no_grad():
#         _ = sam.image_encoder(x)
#     attn_list = get_local.pop_all("attention_map")
#     print(f"[capture] got {len(attn_list)} attention maps")

#     if not attn_list:
#         raise RuntimeError("没有捕捉到 attention map。确认 Attention.forward 里已调用 get_local.push('attention_map', attn)。")

#     # 5) 简单可视化：按层平均 heads，再把 token map 上采样到原图大小
#     # 取最后一层为例
#     A = attn_list[-1]          # (B, heads, N, N) on CPU
#     A = A[0].mean(0)           # (N, N)
#     imp = A.mean(0)            # 每个 token 被关注的平均程度 -> (N,)
#     side = int(np.sqrt(imp.numel()))
#     token_map = imp[:side*side].reshape(side, side).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)

#     token_map = torch.nn.functional.interpolate(
#         token_map, size=raw_np.shape[:2], mode="bilinear", align_corners=False
#     )[0, 0].numpy()

#     save_heat(outdir, "last_layer_mean", token_map, raw_np)
#     print(f"[done] saved to: {outdir}")

# if __name__ == "__main__":
#     main()
