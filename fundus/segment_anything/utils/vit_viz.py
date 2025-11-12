# segment_anything/utils/vit_viz.py
import os
import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from fundus.segment_anything.build_sam import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b

# ---- 引入 SAM 的 ResizeLongestSide 预处理 ----
try:
    from fundus.segment_anything.utils.transforms import ResizeLongestSide
except Exception:
    from segment_anything.utils.transforms import ResizeLongestSide


# ----------------------------
# 工具函数
# ----------------------------
def to_3ch(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def tensor_to_np_img(t):
    # t: (H, W) or (1, H, W)
    t = t.detach().cpu().numpy()
    if t.ndim == 3:
        t = t[0]
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t

def overlay(img_np, heat_np, alpha=0.45):
    # img_np: (H,W,3) [0,1]; heat_np: (H,W) [0,1]
    cmap = plt.get_cmap('jet')
    heat_color = cmap(heat_np)[..., :3]
    out = (1 - alpha) * img_np + alpha * heat_color
    out = np.clip(out, 0, 1)
    return out

def attention_rollout(attn_list, discard_ratio=0.0):
    """
    将多层注意力做 rollout（Abnar & Zuidema, 2020）
    attn_list: [L] of tensors with shape (B, Heads, T, T), 取 B=1
    """
    result = None
    for A in attn_list:
        A = A[0]                     # (H, T, T)
        A = A.mean(dim=0)            # (T, T)
        if discard_ratio > 0:
            flat = A.flatten()
            num = flat.numel()
            k = int(num * discard_ratio)
            if k > 0:
                idx = torch.argsort(flat)[:k]
                flat[idx] = 0
                A = flat.view_as(A)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        I = torch.eye(A.size(0), device=A.device)
        A = A + I
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        result = A if result is None else A @ result
    return result  # (T, T)


# ----------------------------
# Hook 管理
# ----------------------------
class ViTInspector:
    def __init__(self, image_encoder):
        self.enc = image_encoder
        self.handles = []
        self.tokens_by_layer = []
        self.attns_by_layer = []

        # hook 1: 保存每层 tokens（取 block 输出）
        def hook_tokens(idx):
            def _hook(module, inp, out):
                self.tokens_by_layer.append((idx, out.detach()))
            return _hook

        # hook 2: 保存注意力矩阵（兼容 3D/4D/tuple 输入）
        for i, blk in enumerate(self.enc.blocks):
            # tokens
            self.handles.append(blk.register_forward_hook(hook_tokens(i)))

            # wrap attention forward
            attn = blk.attn
            orig_forward = attn.forward

            def make_forward(attn_module, orig_f):
                def new_forward(*args, **kwargs):
                    # --- 取出真正的 x（可能被 tuple 打包）---
                    if len(args) == 0:
                        return orig_f(*args, **kwargs)
                    if isinstance(args[0], tuple):
                        x = args[0][0]
                        rest_tuple = args[0][1:]
                        extra_args = tuple()
                        packed_tuple = True
                    else:
                        x = args[0]
                        rest_tuple = tuple()
                        extra_args = args[1:]
                        packed_tuple = False

                    if not isinstance(x, torch.Tensor):
                        return orig_f(*args, **kwargs)

                    dim = getattr(attn_module.qkv, "in_features", None)
                    if dim is None:
                        # 有些实现 qkv 是 Linear，能直接读到 in_features；读不到就回退
                        return orig_f(*args, **kwargs)

                    B = x.shape[0]
                    orig_restore = None  # 用于 4D 还原

                    # --- 将任意形状整理为 (B, N, dim) ---
                    if x.ndim == 4:
                        # 可能是 (B, C, H, W) 或 (B, H, W, C)
                        if x.shape[-1] == dim:
                            # (B, H, W, C) -> (B, N, C)
                            H, W = x.shape[1], x.shape[2]
                            x_tokens = x.reshape(B, H * W, dim)
                            orig_restore = ("NHWC", (H, W))
                        elif x.shape[1] == dim:
                            # (B, C, H, W) -> (B, N, C)
                            C, H, W = x.shape[1], x.shape[2], x.shape[3]
                            if C != dim:
                                # 通道不是 dim，无法确定，回退
                                return orig_f(*args, **kwargs)
                            x_tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, dim)
                            orig_restore = ("NCHW", (H, W))
                        else:
                            # 其它排列不支持，回退
                            return orig_f(*args, **kwargs)

                    elif x.ndim == 3:
                        # 可能是 (B, N, C) 或 (B, C, N)
                        if x.shape[-1] == dim:
                            x_tokens = x  # (B, N, dim)
                        elif x.shape[1] == dim:
                            # (B, dim, N) -> (B, N, dim)
                            x_tokens = x.transpose(1, 2).contiguous()
                        else:
                            # 尝试把最后两维中较大的当 N，较小的当 dim（仅当能匹配）
                            c1, c2 = x.shape[1], x.shape[2]
                            if c1 == dim:
                                x_tokens = x.transpose(1, 2).contiguous()
                            elif c2 == dim:
                                x_tokens = x
                            else:
                                return orig_f(*args, **kwargs)
                    else:
                        # 其它维度不处理
                        return orig_f(*args, **kwargs)

                    N = x_tokens.shape[1]  # token 数
                    head_dim = dim // attn_module.num_heads

                    # --- 标准 QKV 注意力 ---
                    qkv = attn_module.qkv(x_tokens).chunk(3, dim=-1)
                    q, k, v = [
                        t.reshape(B, N, attn_module.num_heads, head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
                        for t in qkv
                    ]
                    attn_prob = (q @ k.transpose(-2, -1)) * attn_module.scale
                    attn_prob = attn_prob.softmax(dim=-1)
                    # 收集注意力
                    self.attns_by_layer.append(attn_prob.detach())

                    x_out = (attn_prob @ v).transpose(1, 2).reshape(B, N, dim)  # (B, N, dim)
                    x_out = attn_module.proj(x_out)

                    # --- 若输入是 4D，尽量还原回原状 ---
                    if orig_restore is not None:
                        kind, (H, W) = orig_restore
                        if kind == "NHWC":
                            x_out = x_out.reshape(B, H, W, dim)
                        elif kind == "NCHW":
                            x_out = x_out.reshape(B, H, W, dim).permute(0, 3, 1, 2).contiguous()

                    # --- 维持原 forward 的接口 ---
                    if packed_tuple:
                        return (x_out,) + rest_tuple
                    else:
                        if len(extra_args) > 0:
                            return (x_out,) + extra_args
                        return x_out
                return new_forward


    def remove(self):
        for h in self.handles:
            h.remove()


# ----------------------------
# 预处理
# ----------------------------
def sam_preprocess(transform, pil_img: Image.Image, device, pixel_mean, pixel_std):
    img = np.asarray(pil_img)  # uint8
    if img.ndim == 2:  # 灰度 -> 3 通道
        img = np.stack([img, img, img], axis=-1)

    img_resized = transform.apply_image(img)  # uint8, H'×W'×3

    x = torch.as_tensor(img_resized, device=device).permute(2, 0, 1).contiguous()[None, ...]  # (1,3,H',W')
    x = x.float() / 255.0  # 转 float32, [0,1]

    # 用 SAM 的均值/方差归一化
    pm = pixel_mean.to(device)
    ps = pixel_std.to(device)
    if pm.ndim == 1:
        pm = pm.view(1, -1, 1, 1)
    if ps.ndim == 1:
        ps = ps.view(1, -1, 1, 1)
    x = (x - pm) / ps

    return (img.astype(np.float32) / 255.0), x  # 原图(0..1) + 网络输入张量


# ----------------------------
# 主流程（固定 ViT-B）
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="vit_b", choices=["vit_h","vit_l","vit_b"])  # 默认 vit_b
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./viz_out")
    parser.add_argument("--layers", type=int, nargs="*", default=[0,3,6,9,12])
    parser.add_argument("--show-heads", type=int, nargs="*", default=[0,1])
    parser.add_argument("--discard", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # ---- 构建 ViT-B（只取真正的 model）----
    def _build_only_model(builder, image_size, num_classes):
        ret = builder(image_size=image_size, num_classes=num_classes)
        sam = ret
        while isinstance(sam, (tuple, list)):
            sam = sam[0]
        return sam

    # 强制使用 vit_b（即便传了其它，也以你现在的需求为准）
    sam = _build_only_model(build_sam_vit_b, args.image_size, args.num_classes)

    # ---- 加载权重：过滤无关分支 + 按形状匹配加载 ----
    ckpt = torch.load(args.sam_checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    target_sd = sam.state_dict()

    filtered = {}
    drop_prefixes = [
        "channel_gate.",
        "image_encoder.l_adapter.",
        "image_encoder.neck.",
        # 其它你明确不想加载的自定义分支也可以加在这里
    ]

    for k, v in state.items():
        # 1) 丢弃不需要的前缀
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        # 2) 只加载和当前模型 shape 完全一致的参数
        if k in target_sd and tuple(v.shape) == tuple(target_sd[k].shape):
            filtered[k] = v

    missing, unexpected = sam.load_state_dict(filtered, strict=False)
    print("[vit_viz] strict=False | loaded:", len(filtered), "| missing:", len(missing), "| unexpected:", len(unexpected))

    sam.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)

    # 创建 ResizeLongestSide 预处理器
    transform = ResizeLongestSide(args.image_size)

    # 读图并预处理
    img = to_3ch(Image.open(args.image).convert("RGB"))
    raw_np, inp_t = sam_preprocess(transform, img, device, sam.pixel_mean, sam.pixel_std)
    raw_h, raw_w = raw_np.shape[:2]
    vis_img = raw_np

    # 注册 ViT hooks
    inspector = ViTInspector(sam.image_encoder)

    # 前向只过 image encoder
    with torch.no_grad():
        _ = sam.image_encoder(inp_t)

    # 解析 tokens / attention
    layer_tokens = {idx: tok for (idx, tok) in inspector.tokens_by_layer}
    attn_list = inspector.attns_by_layer

    # rollout 热力图
    rollout = attention_rollout(attn_list, discard_ratio=args.discard)
    T = rollout.size(-1)
    token_importance = rollout.mean(dim=0)
    side = int(math.sqrt(T))
    token_map = token_importance[:side*side].reshape(1, side, side)
    token_map = F.interpolate(token_map[None, ...], size=(raw_h, raw_w),
                              mode="bilinear", align_corners=False)[0, 0]
    token_map_np = tensor_to_np_img(token_map)
    plt.imsave(os.path.join(args.outdir, "0_rollout_heat.png"), token_map_np, cmap="jet")
    plt.imsave(os.path.join(args.outdir, "0_rollout_overlay.png"), overlay(vis_img, token_map_np))

    # 逐层/逐 head 注意力
    for li in args.layers:
        A = attn_list[li][0]  # (H, T, T)
        Hh = A.shape[0]
        for hh in args.show_heads:
            if hh >= Hh:
                continue
            Ah = A[hh]
            imp = Ah.mean(dim=0)
            imp = imp[:side*side].reshape(1, side, side)
            imp = F.interpolate(imp[None, ...], size=(raw_h, raw_w),
                                mode="bilinear", align_corners=False)[0, 0]
            imp_np = tensor_to_np_img(imp)
            plt.imsave(os.path.join(args.outdir, f"layer{li}_head{hh}_heat.png"), imp_np, cmap="jet")
            plt.imsave(os.path.join(args.outdir, f"layer{li}_head{hh}_overlay.png"), overlay(vis_img, imp_np))

    # Patch 特征强度（L2）
    for li in args.layers:
        tok = layer_tokens[li][0]  # (T, C) 或更高维被 flatten 过的输出
        t = tok[:side*side] if tok.size(0) >= side*side else tok
        l2 = torch.norm(t, dim=-1)
        l2 = l2.reshape(1, side, side)
        l2 = F.interpolate(l2[None, ...], size=(raw_h, raw_w),
                           mode="bilinear", align_corners=False)[0, 0]
        l2_np = tensor_to_np_img(l2)
        plt.imsave(os.path.join(args.outdir, f"layer{li}_featL2_heat.png"), l2_np, cmap="jet")
        plt.imsave(os.path.join(args.outdir, f"layer{li}_featL2_overlay.png"), overlay(vis_img, l2_np))

    inspector.remove()
    print(f"[Done] Visualizations saved to: {args.outdir}")


if __name__ == "__main__":
    main()
