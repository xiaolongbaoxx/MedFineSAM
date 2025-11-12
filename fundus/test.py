import os
import sys
from cam_utils import grad_cam_from_last_feature
import cv2
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from datasets.fundus.RIGA_dataloader import RIGA_labeled_set
from datasets.fundus.convert_csv_to_list import convert_labeled_list
from datasets.fundus.dice import get_hard_dice
from datasets.fundus.transform import collate_fn_ts
from segment_anything.utils.metrics import calculate_metrics
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry


def save_vis(image_np, mask_np, save_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np.transpose(1, 2, 0))  # 通道顺序视情况调整
    plt.imshow(mask_np, alpha=0.5, cmap='Reds')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fundus.segment_anything.modeling.modules.reins import Reins

def _toggle_reins_logging(model, flag=True, reset=False):
    for m in model.modules():
        if isinstance(m, Reins):
            m.enable_usage_logging(flag)
            if reset:
                m.reset_usage()

def _collect_reins_usage(model, save_dir):
    import os, json, numpy as np
    os.makedirs(save_dir, exist_ok=True)
    all_sum = []
    all_top1 = []
    idx = 0
    for m in model.modules():
        if isinstance(m, Reins):
            usage_sum, usage_top1 = m.get_usage()  # [L, M], [L, M]
            import csv

            # usage_sum
            with open(os.path.join(save_dir, f"reins_usage_{idx:02d}_sum.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer"] + [f"T{k}" for k in range(usage_sum.shape[1])])
                for l in range(usage_sum.shape[0]):
                    w.writerow([f"L{l}"] + usage_sum[l].tolist())

            # usage_top1
            with open(os.path.join(save_dir, f"reins_usage_{idx:02d}_top1.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer"] + [f"T{k}" for k in range(usage_top1.shape[1])])
                for l in range(usage_top1.shape[0]):
                    w.writerow([f"L{l}"] + usage_top1[l].tolist())

            # call_cnt, attn_sum, nan_cnt
            with open(os.path.join(save_dir, f"reins_usage_{idx:02d}_debug.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "call_cnt", "attn_sum", "nan_cnt"])
                for l in range(m._call_cnt.shape[0]):
                    w.writerow([f"L{l}",
                                float(m._call_cnt[l].item()),
                                float(m._attn_sum[l].item()),
                                float(m._nan_cnt[l].item())])

            # 也导出一个易读 CSV（每个模块一份，每层一行）
            import csv
            with open(os.path.join(save_dir, f"reins_usage_{idx:02d}_sum.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer_token"] + [f"T{k}" for k in range(usage_sum.shape[1])])
                for l in range(usage_sum.shape[0]):
                    w.writerow([f"L{l}"] + usage_sum[l].tolist())
            with open(os.path.join(save_dir, f"reins_usage_{idx:02d}_top1.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer_token"] + [f"T{k}" for k in range(usage_top1.shape[1])])
                for l in range(usage_top1.shape[0]):
                    w.writerow([f"L{l}"] + usage_top1[l].tolist())

            # 统计“被使用过的 token 数量”（排除静默 token 0）
            used = (usage_sum[:, 1:] > 0).sum(axis=1).tolist()
            with open(os.path.join(save_dir, f"reins_usage_{idx:02d}_summary.json"), "w") as f:
                json.dump({
                    "num_layers": int(usage_sum.shape[0]),
                    "token_length": int(usage_sum.shape[1]),
                    "used_token_count_per_layer_sum": used,  # 加权口径
                }, f, indent=2)
            idx += 1




# =============== 工具函数 ===============

def denorm_image(t, mean=None, std=None):
    """
    t: Tensor[C,H,W] or np.ndarray[C,H,W], 值域可能在标准化空间
    返回: np.uint8 的 RGB 图像 (H,W,3)，范围 [0,255]
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    assert t.ndim == 3 and t.shape[0] in (1, 3), f"expect CHW, got {t.shape}"

    # 若是灰度单通道，转成 3 通道显示
    if t.shape[0] == 1:
        t = np.repeat(t, 3, axis=0)

    # 反归一化
    if (mean is not None) and (std is not None):
        mean = np.array(mean).reshape(-1, 1, 1)
        std = np.array(std).reshape(-1, 1, 1)
        t = t * std + mean

    # 常见两种输入：[-1,1] 或 [0,1]
    t_min, t_max = t.min(), t.max()
    if t_min < 0.0 - 1e-3 or t_max > 1.0 + 1e-3:
        # 大概率是 [-1,1] 区间
        t = (t + 1.0) / 2.0

    t = np.clip(t, 0.0, 1.0)
    img = (t.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return img


def overlay_masks(rgb, disc_mask, cup_mask, alpha_disc=0.35, alpha_cup=0.55,
                  colors=((88, 255, 51), (0, 0, 240)), draw_contour=False):
    """
    rgb: np.uint8 (H,W,3)
    disc_mask, cup_mask: bool 型 (H,W)
    colors: (disc_color, cup_color)
    draw_contour: True 时仅画轮廓（更清楚）
    """
    h, w, _ = rgb.shape
    out = rgb.copy()

    def _paint(mask, color, alpha):
        if draw_contour:
            # 仅描边：用简单腐蚀-异或做边界
            from scipy.ndimage import binary_erosion
            edge = mask ^ binary_erosion(mask, iterations=1)
            overlay = out.copy()
            overlay[edge] = color
            return (overlay * alpha + out * (1 - alpha)).astype(np.uint8)
        else:
            overlay = out.copy()
            overlay[mask] = (np.array(color, dtype=np.uint8))
            return (overlay * alpha + out * (1 - alpha)).astype(np.uint8)

    # 先画盘，再画杯（杯颜色覆盖）
    out = _paint(disc_mask, colors[0], alpha_disc)
    out = _paint(cup_mask, colors[1], alpha_cup)
    return out


def save_vis_sample(x_tensor, prob_tensor, save_path, thr=0.5,
                    mean=None, std=None, enforce_cup_in_disc=True,
                    draw_contour=False):
    """
    x_tensor: Tensor[C,H,W]
    prob_tensor: Tensor[2,H,W], sigmoid 后概率
    """
    rgb = denorm_image(x_tensor, mean=mean, std=std)

    prob = prob_tensor.detach().cpu().numpy()
    assert prob.shape[0] == 2, f"expect 2 channels (disc,cup), got {prob.shape}"

    disc_mask = (prob[0] > thr)
    cup_mask = (prob[1] > thr)

    if enforce_cup_in_disc:
        cup_mask = np.logical_and(cup_mask, disc_mask)

    vis = overlay_masks(rgb, disc_mask, cup_mask, draw_contour=draw_contour)
    Image.fromarray(vis).save(save_path)


# =============== 主循环（替换你原来的 if vis_count < max_vis: ... 片段） ===============

def inference_riga(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    print("\nTesting and Saving the results...")
    print("--" * 15)
        # === 在这里插入：确认 ReINS 统计是否已开启 ===
    from fundus.segment_anything.modeling.modules.reins import Reins
    for m in model.modules():
        if isinstance(m, Reins):
            print("[DEBUG] ReINS logging:", m.log_usage)   # 期望 True
            break
    val_disc_dice_list = []
    val_cup_dice_list = []

    # 如果你有归一化的均值/方差，填上；没有就都置 None
    IMG_MEAN = getattr(args, "img_mean", None)  # 例如 [0.485, 0.456, 0.406]
    IMG_STD = getattr(args, "img_std", None)  # 例如 [0.229, 0.224, 0.225]

    max_vis = getattr(args, "max_vis", 20)
    thr = getattr(args, "vis_thr", 0.5)
    draw_edge = getattr(args, "vis_draw_contour", False)

    save_dir = test_save_path if test_save_path is not None else "./vis_results"
    os.makedirs(save_dir, exist_ok=True)

    vis_count = 0
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
            x, y = data['data'], data['seg']  # x: (B,C,H,W), y: (B,1,H,W) with {0,1,2}
            x = torch.from_numpy(x).to(torch.float32).cuda(non_blocking=True)
            y = torch.from_numpy(y).to(torch.float32)

            seg_logit = model(x, False, args.img_size)  # 期望 seg_logit['masks'] -> (B,2,H,W)
            seg_prob = torch.sigmoid(seg_logit['masks'].detach().cpu())

            # ---- 保存可视化 ----
            bsz = x.shape[0]
            for i in range(bsz):
                if vis_count >= max_vis:
                    break
                # 单张图：x[i], seg_prob[i] (2,H,W)
                save_path = os.path.join(save_dir, f'sample_{vis_count:03d}.png')
                save_vis_sample(
                    x_tensor=x[i].detach().cpu(),
                    prob_tensor=seg_prob[i],
                    save_path=save_path,
                    thr=thr,
                    mean=IMG_MEAN, std=IMG_STD,
                    enforce_cup_in_disc=True,
                    draw_contour=draw_edge
                )
                with torch.enable_grad():
                    cam_map = grad_cam_from_last_feature(
                        model_sam=model,
                        image_tensor=x[i].unsqueeze(0),
                        img_size=args.img_size,        # ★ 传你的验证分辨率
                        mask_index=0,                  # 需要的话可换成你最终选中的候选掩膜 idx
                        logits_key="masks"             # 若你 forward 返回的是 'low_res_masks'，这里就写 "low_res_masks"
                    )
                import cv2
                img_np = denorm_image(x[i].cpu(), mean=IMG_MEAN, std=IMG_STD)  # 原图
                heat = cv2.applyColorMap((cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                overlay = (0.5 * img_np + 0.5 * heat).clip(0, 255).astype(np.uint8)

                cam_save_path = os.path.join(save_dir, f'sample_{vis_count:03d}_cam.png')
                Image.fromarray(overlay).save(cam_save_path)
                print(f"图片已保存到: {os.path.abspath(save_path)}")
                vis_count += 1

            # ---- 计算 Dice（与你原逻辑一致）----
            val_disc_dice_list.append(
                get_hard_dice(seg_prob[:, 0], (y[:, 0] > 0).cpu().float())
            )
            val_cup_dice_list.append(
                get_hard_dice(seg_prob[:, 1], (y[:, 0] == 2).cpu().float())
            )

    mean_val_disc_dice = np.mean(val_disc_dice_list)
    mean_val_cup_dice = np.mean(val_cup_dice_list)
    print(f"Disc Dice: {mean_val_disc_dice:.4f} | Cup Dice: {mean_val_cup_dice:.4f}")

    # def inference_riga(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    #     print("\nTesting and Saving the results...")
    #     print("--" * 15)
    #     val_disc_dice_list = list()
    #     val_cup_dice_list = list()
    #     with torch.no_grad():
    #         max_vis = 10  # 你想保存的最大图片数
    #         vis_count = 0
    #         for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
    #             x, y = data['data'], data['seg']

    #             x = torch.from_numpy(x).to(dtype=torch.float32)
    #             y = torch.from_numpy(y).to(dtype=torch.float32)

    #             x = x.cuda()
    #             seg_logit = model(x, False, args.img_size)
    #             seg_output = torch.sigmoid(seg_logit['masks'].detach().cpu())

    # #             if vis_count < max_vis:
    # #                 image_np = x[0].cpu().numpy()
    # #                 mask_np = (seg_output[0, 0].numpy() > 0.5).astype(np.float32)

    # #                 os.makedirs('vis_results', exist_ok=True)
    # #                 save_path = f'vis_results/sample_{batch}.png'
    # #                 save_vis(image_np, mask_np, save_path)

    # #                 vis_count += 1
    #             # if vis_count < max_vis:
    #             #     image_np = x[0].cpu().numpy()
    #             #     mask_np = (seg_output[0, 0].numpy() > 0.5).astype(np.float32)
    #             #     # 保存到脚本同级目录
    #             #     save_path = f'./sample_{vis_count}.png'
    #             #     save_vis(image_np, mask_np, save_path)
    #             #     print(f"图片已保存到: {os.path.abspath(save_path)}")  # 打印完整绝对路径
    #             #     vis_count += 1
    #             if vis_count < max_vis:
    #                 image_np = x[0].cpu().numpy()
    #                 mask_np = (seg_output[0, 0].numpy() > 0.5).astype(np.float32)

    #                 # 使用传入的 test_save_path，如果没有传就用默认目录
    #                 save_dir = test_save_path if test_save_path is not None else "./vis_results"
    #                 os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    #                 save_path = os.path.join(save_dir, f'sample_{vis_count}.png')
    #                 save_vis(image_np, mask_np, save_path)

    #                 print(f"图片已保存到: {os.path.abspath(save_path)}")
    #                 vis_count += 1

    #             val_disc_dice_list.append(get_hard_dice(seg_output[:, 0].cpu(), (y[:, 0] > 0).cpu() * 1.0))
    #             val_cup_dice_list.append(get_hard_dice(seg_output[:, 1].cpu(), (y[:, 0] == 2).cpu() * 1.0))

    #     mean_val_disc_dice = np.mean(val_disc_dice_list)
    #     mean_val_cup_dice = np.mean(val_cup_dice_list)

    logging.info('Epoch{}  Val disc dice: {}; Cup dice: {}'.format(epoch, mean_val_disc_dice, mean_val_cup_dice))

    with open(snapshot_path + '/' + 'test_' + args.Source_Dataset + '_to' + '.txt', 'a', encoding='utf-8') as f:
        f.write('Epoch ' + str(epoch) + ' Test Metrics:\n')
        f.write(str('Val disc dice: {}; Cup dice: {}'.format(mean_val_disc_dice, mean_val_cup_dice)) + '\n')  # Dice

    return mean_val_disc_dice, mean_val_cup_dice


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--root_path', type=str,
                        default='./data/datasets/RIGAPlus', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='RIGA', help='experiment_name')
    parser.add_argument('--Source_Dataset', type=str, default='BinRushed',
                        help='BinRushed/Magrabia')
    parser.add_argument('--Target_Dataset', nargs='+', type=str,
                        default=['MESSIDOR_Base1', 'MESSIDOR_Base2', 'MESSIDOR_Base3'],
                        help='MESSIDOR_Base1/MESSIDOR_Base2/MESSIDOR_Base3')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--output', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='./pretrained/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')

    parser.add_argument('--snapshot', type=str, default='./snapshot/DAPSAM-BinRushed.pth', help='model snapshot')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'FUNDUS': {
            'Dataset': args.dataset,
            'num_classes': args.num_classes,
        }
    }
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
    net = sam.cuda()
    if args.snapshot is not None:
        weights = torch.load(args.snapshot)
        net.load_state_dict(weights)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # to rest target domain
    target_name = args.Target_Dataset
    valid_loader = []
    for t_n in target_name:
        target_csv = [os.path.join(args.root_path, t_n + '.csv')]
        ts_img_list, ts_label_list = convert_labeled_list(target_csv, r=1)

        ts_dataset = RIGA_labeled_set(args.root_path, ts_img_list, ts_label_list)
        valid_loader.append(torch.utils.data.DataLoader(ts_dataset,
                                                        batch_size=1,
                                                        num_workers=0,
                                                        shuffle=False,
                                                        pin_memory=True,
                                                        collate_fn=collate_fn_ts))

    # ===== 在 main() 里，调用 inference_riga 前加 =====
    _toggle_reins_logging(net, flag=True, reset=True)
    mean_val_disc_dice, mean_val_cup_dice = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
                                                           test_loader=valid_loader[0], model=net,
                                                           test_save_path="/root/autodl-tmp/dapsam_rein_heat/save_img_1")
    # ===== 在 inference_riga 调用结束后加 =====
    _collect_reins_usage(net, save_dir=os.path.join(args.output, "token_usage"))
    _toggle_reins_logging(net, flag=False)  # 可选，关闭统计

#     mean_val_disc_dice1, mean_val_cup_dice1 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[1], model=net,
#                                                              test_save_path=None)

#     mean_val_disc_dice2, mean_val_cup_dice2 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[2], model=net,
#                                                              test_save_path=None)
#     mean_val_disc_dice3, mean_val_cup_dice3 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[3], model=net,
#                                                              test_save_path=None)
#     mean_val_disc_dice4, mean_val_cup_dice4 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[4], model=net,
#                                                              test_save_path=None)
#     mean_val_disc_dice5, mean_val_cup_dice5 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[5], model=net,
#                                                              test_save_path=None)
#     mean_val_disc_dice6, mean_val_cup_dice6 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[6], model=net,
#                                                              test_save_path=None)
#     mean_val_disc_dice7, mean_val_cup_dice7 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[7], model=net,
#                                                              test_save_path=None)
#     mean_val_disc_dice8, mean_val_cup_dice8 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
#                                                              test_loader=valid_loader[8], model=net,
#                                                              test_save_path=None)
