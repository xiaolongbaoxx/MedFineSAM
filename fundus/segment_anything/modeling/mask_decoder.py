# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        confidence_score: torch.Tensor,  # 新增参数
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            confidence_score=confidence_score  # 添加
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,  # [B, C, H, W]
            image_pe: torch.Tensor,  # [B, C, H, W]
            sparse_prompt_embeddings: torch.Tensor,  # [B, N_prompt, D]
            dense_prompt_embeddings: torch.Tensor,  # [B, C, H, W]
            confidence_score: torch.Tensor  # [B, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. Adapted to support multi-image batch (no repeat_interleave)."""

        B, C, H, W = image_embeddings.shape

        # === 1. 构造 tokens: [B, (1+num_masks+num_prompts), D]
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)  # [1+num_masks, D]
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 1+num_masks, D]

        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)  # [B, T, D]

        # === 2. 动态融合 image_feature 和 dense_prompt → src
        alpha = confidence_score.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        src = alpha * dense_prompt_embeddings + (1 - alpha) * image_embeddings  # [B, C, H, W]
        pos_src = image_pe  # [B, C, H, W]

        # === 3. Transformer
        hs, src = self.transformer(src, pos_src, tokens)  # all are [B, ...]

        iou_token_out = hs[:, 0, :]  # [B, D]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]  # [B, num_masks, D]

        # === 4. Mask prediction
        src = src.transpose(1, 2).view(B, C, H, W)
        upscaled_embedding = self.output_upscaling(src)  # [B, C', H', W']

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))  # [B, C']
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_masks, C']

        B, num_masks, C_ = hyper_in.shape
        _, _, H_up, W_up = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(B, C_, H_up * W_up))  # [B, num_masks, H*W]
        masks = masks.view(B, num_masks, H_up, W_up)  # [B, num_masks, H, W]

        # === 5. IOU head
        iou_pred = self.iou_prediction_head(iou_token_out)  # [B, num_masks]

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
