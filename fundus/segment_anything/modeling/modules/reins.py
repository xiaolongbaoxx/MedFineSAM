import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor


class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        token_length: int = 100,
        use_softmax: bool = True,
        scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.token_length = token_length
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.create_model()
        self.log_usage = False
        
        self.register_buffer("_usage_sum", torch.zeros(self.num_layers, self.token_length), persistent=False) 
        self.register_buffer("_usage_top1", torch.zeros(self.num_layers, self.token_length), persistent=False) 
        self._eps = 1e-12
        self.register_buffer("_call_cnt", torch.zeros(self.num_layers), persistent=False)
        self.register_buffer("_attn_sum", torch.zeros(self.num_layers), persistent=False)
        self.register_buffer("_nan_cnt", torch.zeros(self.num_layers), persistent=False)



    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def enable_usage_logging(self, flag: bool = True):
        self.log_usage = flag

    @torch.no_grad()
    def reset_usage(self):
        self._usage_sum.zero_()
        self._usage_top1.zero_()

    @torch.no_grad()
    def get_usage(self):
        return (self._usage_sum.detach().cpu().numpy(),
                self._usage_top1.detach().cpu().numpy())

    @torch.no_grad()
    def _tally_usage(self, layer_idx: int, attn: torch.Tensor):
        sum_over_NB = attn.sum(dim=(0, 1))  # [M]
        self._usage_sum[layer_idx, :].add_(sum_over_NB)

        top1 = attn.argmax(dim=-1)  # [N, B]
        hist = torch.bincount(top1.view(-1), minlength=attn.shape[-1]).float()  # [M]
        self._usage_top1[layer_idx, :].add_(hist)

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        while tokens.dim() > 2:
            tokens = tokens.squeeze(0)

        if tokens.dim() != 2:
            raise ValueError(f"[Reins] tokens shape invalid: {tokens.shape}, expected [M, C]")

        attn = torch.einsum("nbc,mc->nbm", feats, tokens)


        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)
        if self.log_usage:
            if attn.dim() != 3:
                pass  
            else:
                if attn.shape[-1] != self.token_length:
                    print(f"[REINS] token_length mismatch: attn={attn.shape}, M={self.token_length}")
                else:
                    if attn.shape[0] < attn.shape[1]:
                        attn_nb = attn           # [N,B,M]
                    else:
                        attn_nb = attn.transpose(0, 1)  # [B,N,M] -> [N,B,M]

                    attn_nb = torch.nan_to_num(attn_nb, 0.0)

                    dev = self._usage_sum.device
                    sum_over_nb = attn_nb.sum(dim=(0, 1)).to(dev)                # [M]
                    self._usage_sum[int(layers), :].add_(sum_over_nb)
                    top1 = attn_nb.argmax(dim=-1)                                # [N,B]
                    hist = torch.bincount(top1.reshape(-1).to(torch.int64),
                                          minlength=self.token_length).float().to(dev)   # [M]
                    self._usage_top1[int(layers), :].add_(hist)

                    if not hasattr(self, "_dbg_once"):
                        print(f"[REINS] layer={int(layers)} attn_sum={float(attn_nb.sum().item()):.3f}")
                        self._dbg_once = True

        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f



class LoRAReins(Reins):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:

            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
