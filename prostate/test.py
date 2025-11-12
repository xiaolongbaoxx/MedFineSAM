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
from cam_utils import (
    grad_cam_from_last_feature,
    denorm_image,
    overlay_cam_on_image,   # 如果你只想用 save_cam_overlay 也可以
    save_cam_overlay
)

def inference(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    print("\nTesting and Saving the results...")
    print("--" * 15)
    metrics_y = [[], []]
    metric_dict = ['Dice', 'ASD']

    # === 可视化保存根目录 ===
    cam_root = test_save_path if test_save_path is not None else args.cam_save_dir
    os.makedirs(cam_root, exist_ok=True)

    last_name = None
    # 每个体（按 name/路径分组）计数，最多 20 张
    per_volume_count = 0

    # 可选：图像反归一化参数
    IMG_MEAN = args.img_mean if hasattr(args, "img_mean") else None
    IMG_STD  = args.img_std  if hasattr(args, "img_std")  else None

    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
            x, y, path = data['data'], data['mask'], data['name']   # path 是当前切片所属体的标识
            current_name = path

            if last_name is None:
                last_name = current_name
                per_volume_count = 0  # 新体开始计数

            # —— 标准推理（和你原来一致）——
            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)
            x = x.cuda()
            seg_logit = model(x, False, args.img_size)
            seg_output = torch.sigmoid(seg_logit['masks'].detach().cpu())

            # —— 当切换到新的 3D 体时，先汇总上一个体的指标 —— 
            if current_name != last_name:
                metrics = calculate_metrics(seg_output3D, y3D)
                for i in range(len(metrics)):
                    metrics_y[i].append(metrics[i])
                # 释放并重置
                del seg_output3D
                del y3D
                per_volume_count = 0  # 新体重新计数

            # —— 累积 3D 概率与标签体用于体素级评估（和你原来一致）——
            try:
                seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_output3D = seg_output.unsqueeze(2)
                y3D = y.unsqueeze(2)

            # ========= 每体最多 20 张切片做 CAM 可视化 =========
            if per_volume_count < int(args.cam_per_volume):
                # 只对当前 batch 的这张切片做 CAM
                # 注意：CAM 需要梯度，因此单独再 forward 一次（不走 no_grad）
                with torch.enable_grad():
                    _ = model.eval()  # eval 行为但允许梯度
                    cam_map = grad_cam_from_last_feature(
                        model_sam=model,
                        image_tensor=x,                 # x: [1,C,H,W]
                        img_size=args.img_size,
                        mask_index=0,                   # 前列腺是单通道，0 就够
                        logits_key="masks",
                        use_relu=True
                    )

                # 叠加热力图并保存
                rgb_uint8 = denorm_image(x[0].detach().cpu(), mean=IMG_MEAN, std=IMG_STD)
                overlay = overlay_cam_on_image(rgb_uint8, cam_map, alpha=float(args.cam_alpha))

                # 组织保存路径：每个 3D 体一个子目录
                vol_dir = os.path.join(cam_root, str(current_name))
                os.makedirs(vol_dir, exist_ok=True)
                save_path = os.path.join(vol_dir, f"slice_{per_volume_count:03d}_cam.png")
                Image.fromarray(overlay).save(save_path)

                per_volume_count += 1  # 这体可视化计数+1

            # —— 更新“上一体名称” —— 
            last_name = current_name

    # 收尾：别忘了把最后一个 3D 体的指标也算上
    metrics = calculate_metrics(seg_output3D, y3D)
    for i in range(len(metrics)):
        metrics_y[i].append(metrics[i])

    test_metrics_y = np.mean(metrics_y, axis=1)
    print_test_metric = {metric_dict[i]: test_metrics_y[i] for i in range(len(test_metrics_y))}

    with open(snapshot_path + '/' + 'test_' + args.Source_Dataset + '_to'+ '.txt', 'a', encoding='utf-8') as f:
        f.write('Epoch '+str(epoch)+' Test Metrics:\n')
        f.write(str(print_test_metric) + '\n')

    logging.info("Test Metrics: "+str(print_test_metric))
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
    parser.add_argument('--cam_save_dir', type=str, default='./vis_prostate_cam',
                        help='Where to save CAM overlays for prostate MRI')
    parser.add_argument('--cam_per_volume', type=int, default=20,
                        help='Max number of slices to visualize per 3D volume')
    parser.add_argument('--cam_alpha', type=float, default=0.5,
                        help='Heatmap overlay alpha')
    parser.add_argument('--img_mean', nargs='+', type=float, default=None,
                        help='Optional mean for denorm (e.g., 0.5 0.5 0.5 or single 0.5)')
    parser.add_argument('--img_std', nargs='+', type=float, default=None,
                        help='Optional std for denorm (e.g., 0.5 0.5 0.5 or single 0.5)')


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

    #to rest target domain
    target_name = args.Target_Dataset
    target_csv = []
    for t_n in target_name:
        target_csv.append(t_n + '.csv')
    ts_img_list, ts_label_list = convert_labeled_list(args.root_path, target_csv)

    target_valid_dataset = PROSTATE_dataset(args.root_path, ts_img_list, ts_label_list,
                                            args.img_size, img_normalize=True)
    valid_loader = DataLoader(dataset=target_valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              collate_fn=collate_fn_wo_transform,
                              num_workers=0)


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

    result_list = inference(args=args, epoch='Test', snapshot_path=log_folder, test_loader=valid_loader,
                            model=net, test_save_path=args.cam_save_dir )

