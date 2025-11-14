import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform


def get_train_transform(patch_size=(384, 384)):
    tr_transforms = []
    # tr_transforms.append(RandomCropTransform(crop_size=256, margins=(0, 0, 0), data_key="image", label_key="mask"))
    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
            data_key="data", label_key="mask")
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="data"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="data"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="data"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="data"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="data"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="data"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="data"))

    tr_transforms.append(MirrorTransform(axes=(0, 1), data_key="data", label_key="mask"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


# def collate_fn_w_transform(batch):
#     image, label, name = zip(*batch)
#     image = np.stack(image, 0)
#     label = np.stack(label, 0)
#     name = np.stack(name, 0)
#     data_dict = {'data': image, 'mask': label, 'name': name}
#     tr_transforms = get_train_transform()
#     data_dict = {
#         'data': image,   # 图像数据
#         'seg': label      # 分割标签，必须有这个字段！
#     }
#     print("images", image[0].shape)  # 每张图的 shape
#
#     print("data", data_dict['data'].shape, "seg", data_dict['seg'].shape)
#
#     data_dict = tr_transforms(**data_dict)
#     data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)
#     return data_dict

def collate_fn_w_transform(batch):
    images, labels, names = zip(*batch)
    images = list(images)  # 转成可赋值的 list
    new_images = []
    for i, img in enumerate(images):
        # print(f"before stack img[{i}].shape = {img.shape}")
        # 如果是 (1, H, W, 3) 错误格式，修正为 (3, H, W)
        if img.ndim == 4:
            # (1, H, W, 3) -> (3, H, W)
            img = np.squeeze(img, axis=0)
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 3 and img.shape[-1] == 3:
            # (H, W, 3) -> (3, H, W)
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            # (H, W) -> (1, H, W)
            img = np.expand_dims(img, axis=0)
        # 如果已经是 (1, H, W) 或 (3, H, W)，无须操作
        new_images.append(img)

    images = np.stack(new_images, 0)  # (B, C, H, W)
    labels = np.stack(labels, 0)
    # print("stacked images shape:", images.shape)  # 应该 (B, 1, H, W) or (B, 3, H, W)

    data_dict = {'data': images, 'seg': labels}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    # 如确实需要3通道输入，用 repeat 转成 (B, 3, H, W)
    if data_dict['data'].shape[1] == 1:
        data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)
    # print("final data shape:", data_dict['data'].shape, "seg:", data_dict['seg'].shape)
    data_dict['mask'] = data_dict['seg']  # 保持兼容
    return data_dict



# def collate_fn_wo_transform(batch):
#     image, label, name = zip(*batch)
#     image = np.stack(image, 0)
#     label = np.stack(label, 0)
#     name = np.stack(name, 0)
#     data_dict = {'data': image, 'mask': label, 'name': name}
#     data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)
#     print("collate_fn_wo_transform: image batch shape:", image.shape)  # 确认维度
#     return data_dict
def collate_fn_wo_transform(batch):
    images, labels, names = zip(*batch)
    new_images = []
    for idx, img in enumerate(images):
        # print(f"[collate] 原始 img[{idx}].shape = {img.shape}")
        # (1, H, W, 3) -> (H, W, 3)
        if img.ndim == 4 and img.shape[0] == 1 and img.shape[-1] == 3:
            img = np.squeeze(img, axis=0)          # (H, W, 3)
            # print(f"  squeeze(0) -> {img.shape}")
            img = np.transpose(img, (2, 0, 1))     # (3, H, W)
            # print(f"  transpose(2,0,1) -> {img.shape}")
        elif img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))     # (3, H, W)
            # print(f"  transpose(2,0,1) -> {img.shape}")
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)      # (1, H, W)
            # print(f"  expand_dims(0) -> {img.shape}")
            img = np.repeat(img, 3, axis=0)        # (3, H, W)
            # print(f"  repeat(3,axis=0) -> {img.shape}")
        elif img.ndim == 3 and img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)        # (3, H, W)
            # print(f"  repeat(3,axis=0) -> {img.shape}")
        elif img.ndim == 3 and img.shape[0] == 3:
            pass
            # print(f"  已经是 (3,H,W)，无需处理")
        else:
            # print(f"  [ERROR] Unexpected shape: {img.shape}")
            raise RuntimeError(f"Unsupported image shape: {img.shape}")
        # print(f"[collate] 修正后 img[{idx}].shape = {img.shape}")
        new_images.append(img)
    images = np.stack(new_images, 0)  # (B, 3, H, W)
    labels = np.stack(labels, 0)
    names = np.stack(names, 0)
    # print(f"[collate] 最终 batch image shape: {images.shape}, label shape: {labels.shape}")
    data_dict = {'data': images, 'mask': labels, 'name': names}
    return data_dict

