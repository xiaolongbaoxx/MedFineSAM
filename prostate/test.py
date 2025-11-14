import os
import sys

import cv2
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from datasets.prostate.PROSTATE_dataloader import PROSTATE_dataset
from datasets.prostate.convert_csv_to_list import convert_labeled_list
from datasets.prostate.transform import collate_fn_wo_transform

from segment_anything.utils.metrics import calculate_metrics
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import logging

def _denorm_image(t, mean=None, std=None):
    """
    t: Tensor[C,H,W] or np.ndarray[C,H,W]，返回 uint8 RGB (H,W,3) [0,255]
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    assert t.ndim == 3 and t.shape[0] in (1, 3), f"expect CHW, got {t.shape}"

    if t.shape[0] == 1:
        t = np.repeat(t, 3, axis=0)

    if (mean is not None) and (std is not None):
        mean = np.array(mean).reshape(-1,1,1)
        std  = np.array(std).reshape(-1,1,1)
        t = t * std + mean
    else:
        t_min, t_max = t.min(), t.max()
        if t_min < -1.1 or t_max > 1.1:
            t = (t + 1.0) / 2.0

    t = np.clip(t, 0.0, 1.0)
    return (t.transpose(1,2,0) * 255.0).round().astype(np.uint8)

def _overlay_red(rgb_u8, mask_bool, alpha=0.5, color=(255,0,0)):
    """
    rgb_u8: np.uint8 (H,W,3)
    mask_bool: (H,W) bool
    """
    out = rgb_u8.copy()
    if mask_bool.any():
        overlay = out.copy()
        overlay[mask_bool] = color
        out = (overlay * alpha + out * (1 - alpha)).astype(np.uint8)
    return out

def _save_vis_single(x_tensor, prob_tensor, save_path, thr=0.5, mean=None, std=None, alpha=0.5):
    """
    x_tensor: Tensor[C,H,W]
    prob_tensor: Tensor[1,H,W] or Tensor[H,W]
    """
    rgb = _denorm_image(x_tensor, mean=mean, std=std)
    if prob_tensor.ndim == 3:
        prob = prob_tensor[0].detach().cpu().numpy()
    else:
        prob = prob_tensor.detach().cpu().numpy()
    mask = (prob > thr)
    vis = _overlay_red(rgb, mask, alpha=alpha, color=(255,0,0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(vis).save(save_path)


# ---------- 主函数（单分类 + 红色可视化） ----------
def inference(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    print("\nTesting and Saving the results...")
    print("--" * 15)
    metrics_y = [[], []]  # [Dice_list, ASD_list]
    metric_dict = ['Dice', 'ASD']

    # 可视化配置
    save_dir = test_save_path if test_save_path is not None else os.path.join(snapshot_path, "vis_results")
    max_vis = getattr(args, "max_vis", 20)
    thr = getattr(args, "thr", 0.5)
    IMG_MEAN = getattr(args, "img_mean", None)  # 例如 [0.485,0.456,0.406]
    IMG_STD  = getattr(args, "img_std",  None)  # 例如 [0.229,0.224,0.225]
    alpha = getattr(args, "vis_alpha", 0.5)

    vis_count = 0
    last_name = None
    seg_output3D = None
    y3D = None

    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
            x, y, path = data['data'], data['mask'], data['name']  # 假定每次一个样本；若为批，可按需拆分
            current_name = path
            if last_name is None:
                last_name = path

            x = torch.from_numpy(x).to(dtype=torch.float32).cuda()
            y = torch.from_numpy(y).to(dtype=torch.float32)

            # 前向：期望输出 (B,1,H,W) 或 (B,C,H,W)（单类时 C=1）
            seg_logit = model(x, False, args.img_size)
            logits = seg_logit['masks'] if isinstance(seg_logit, dict) else seg_logit
            seg_output = torch.sigmoid(logits.detach().cpu())  # (B,1,H,W)

            B = x.shape[0]
            for i in range(B):
                if vis_count >= max_vis:
                    break
                save_path = os.path.join(save_dir, f"{current_name}_slice{batch:04d}_{i}.png") \
                            if isinstance(current_name, str) else os.path.join(save_dir, f"sample_{vis_count:04d}.png")
                _save_vis_single(
                    x_tensor=x[i].detach().cpu(),
                    prob_tensor=seg_output[i, 0] if seg_output.ndim == 4 else seg_output[i],
                    save_path=save_path,
                    thr=thr,
                    mean=IMG_MEAN, std=IMG_STD,
                    alpha=alpha
                )
                vis_count += 1

            if current_name != last_name:
                metrics = calculate_metrics(seg_output3D, y3D)
                for i in range(len(metrics)):
                    metrics_y[i].append(metrics[i])
                del seg_output3D
                del y3D
                seg_output3D = None
                y3D = None

            try:
                seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_output3D = seg_output.unsqueeze(2)
                y3D = y.unsqueeze(2)

            last_name = current_name

    metrics = calculate_metrics(seg_output3D, y3D)
    for i in range(len(metrics)):
        metrics_y[i].append(metrics[i])

    test_metrics_y = np.mean(metrics_y, axis=1)
    print_test_metric = {metric_dict[i]: test_metrics_y[i] for i in range(len(test_metrics_y))}

    os.makedirs(snapshot_path, exist_ok=True)
    with open(os.path.join(snapshot_path, 'test_' + args.Source_Dataset + '_to' + '.txt'),
              'a', encoding='utf-8') as f:
        f.write('Epoch ' + str(epoch) + ' Test Metrics:\n')
        f.write(str(print_test_metric) + '\n')

    logging.info("Test Metrics: " + str(print_test_metric))
    return test_metrics_y


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
    parser.add_argument('--root_path', type=str,
                        default='E:\\data\\datasets\\prostate', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='PROSTATE', help='experiment_name')
    parser.add_argument('--Source_Dataset', type=str, default='RUNMC',
                        help='BIDMC/BMC/HK/I2CVB/RUNMC/UCL')
    parser.add_argument('--Target_Dataset', nargs='+', type=str, default=['BIDMC', 'BMC', 'HK', 'I2CVB', 'UCL'],
                        help='BIDMC/BMC/HK/I2CVB/RUNMC/UCL')
    parser.add_argument('--num_classes', type=int, default=1)

    parser.add_argument('--output', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=384, help='Input image size of the network')

    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='./pretrained/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--snapshot', type=str, default='./snapshot/epoch_final.pth',
                        help='model snapshot')

    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

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
        'PROSTATE': {
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

    log_folder = os.path.join(args.output, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    target_names = args.Target_Dataset
    result_dict = {}

    for t_name in target_names:
        logging.info(f"\n=== Testing on target domain: {t_name} ===")


        target_csv = [t_name + '.csv']
        ts_img_list, ts_label_list = convert_labeled_list(args.root_path, target_csv)

        target_valid_dataset = PROSTATE_dataset(args.root_path, ts_img_list, ts_label_list,
                                                args.img_size, img_normalize=True)
        valid_loader = DataLoader(dataset=target_valid_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  collate_fn=collate_fn_wo_transform,
                                  num_workers=0)

        result_list = inference(args=args, epoch='Test', snapshot_path=log_folder,
                                test_loader=valid_loader, model=net, test_save_path=test_save_path)

        result_dict[t_name] = result_list

    with open(os.path.join(log_folder, 'domainwise_results.csv'), 'w') as f:
        f.write('Domain,Dice,ASD\n')
        for domain, metrics in result_dict.items():
            f.write(f'{domain},{metrics[0]:.4f},{metrics[1]:.4f}\n')

    logging.info("\n=== Summary of All Domains ===")
    for domain, metrics in result_dict.items():
        logging.info(f"[{domain}] Dice: {metrics[0]:.4f} | ASD: {metrics[1]:.4f}")
    print("\n=== Final Summary of All Domains ===")
    for domain, metrics in result_dict.items():
        print(f"[{domain}] Dice: {metrics[0]:.4f}, ASD: {metrics[1]:.4f}")
