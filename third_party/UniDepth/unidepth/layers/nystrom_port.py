"""Pure-PyTorch port of xformers 0.0.24 `xformers.components.attention.NystromAttention`
(facebookresearch/xformers, tag v0.0.24, files components/attention/{nystrom,core,utils}.py).

Newer xformers releases removed `xformers.components`, and the UniDepthV2 checkpoints were trained with
this block, so the original logic is reproduced here without the xformers dependency.
Only the dense tensor code paths are ported (no sparse / triton / blocksparse specialisations); for
dense inputs those paths compute the same result.
"""
import math
from typing import Optional

import torch
import torch.nn as nn


def _bool_mask_to_additive(mask: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    mask_ = torch.zeros_like(mask, dtype=dtype)
    mask_[~mask] = float("-inf")
    return mask_


def _reshape_key_padding_mask(key_padding_mask: torch.Tensor, batched_dim: int) -> torch.Tensor:
    batch_size, src_len = key_padding_mask.size()
    num_heads = batched_dim // batch_size
    return (key_padding_mask.view(batch_size, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(batch_size * num_heads, 1, src_len))


def scaled_query_key_softmax(q: torch.Tensor, k: torch.Tensor, att_mask: Optional[torch.Tensor]) -> torch.Tensor:
    q = q / math.sqrt(k.size(-1))
    att = q @ k.transpose(-2, -1)
    if att_mask is not None:
        if att_mask.dtype == torch.bool:
            if att_mask.ndim == 2:
                att_mask = att_mask.unsqueeze(0).expand(att.shape[0], -1, -1)
            att[~att_mask] = float("-inf")
        else:
            if att_mask.ndim == 3 and att_mask.shape[0] != att.shape[0] and att.shape[0] % att_mask.shape[0] == 0:
                att_mask = att_mask.repeat([att.shape[0] // att_mask.shape[0], 1, 1])
            att = att + att_mask
    return torch.softmax(att, dim=att.ndim - 1)


def scaled_dot_product_attention(q, k, v, att_mask: Optional[torch.Tensor], dropout: Optional[nn.Module] = None):
    att = scaled_query_key_softmax(q, k, att_mask=att_mask)
    if dropout is not None:
        att = dropout(att)
    return att @ v


def iterative_pinv(softmax_mat: torch.Tensor, n_iter=6, pinverse_original_init=False):
    """Moore-Penrose inverse via the iterative method of Razavi et al. 2014 (as in xformers)."""
    i = torch.eye(softmax_mat.size(-1), device=softmax_mat.device, dtype=softmax_mat.dtype)
    k = softmax_mat
    if pinverse_original_init:
        v = 1 / torch.max(torch.sum(k, dim=-2)) * k.transpose(-1, -2)
    else:
        v = 1 / torch.max(torch.sum(k, dim=-2), dim=-1).values[:, None, None] * k.transpose(-1, -2)
    for _ in range(n_iter):
        kv = torch.matmul(k, v)
        v = torch.matmul(0.25 * v, 13 * i - torch.matmul(kv, 15 * i - torch.matmul(kv, 7 * i - kv)))
    return v


class AvgPool(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        head_dim = x.shape[2]
        segments = seq_len // self.n
        assert segments > 0, "num_landmarks should be smaller than the sequence length"
        if seq_len % self.n == 0:
            return x.reshape(-1, self.n, segments, head_dim).mean(dim=-2)
        n_round = self.n - seq_len % self.n
        x_avg_round = x[:, : n_round * segments, :].reshape(-1, n_round, segments, head_dim).mean(dim=-2)
        x_avg_off = x[:, n_round * segments:, :].reshape(-1, self.n - n_round, segments + 1, head_dim).mean(dim=-2)
        return torch.cat((x_avg_round, x_avg_off), dim=-2)


class NystromAttention(nn.Module):
    """Nystrom attention (Xiong et al. 2021), ported from xformers 0.0.24."""

    def __init__(self, dropout: float, num_heads: int, num_landmarks: int = 64,
                 landmark_pooling: Optional[nn.Module] = None, causal: bool = False,
                 use_razavi_pinverse: bool = True, pinverse_original_init: bool = False,
                 inv_iterations: int = 6, v_skip_connection: Optional[nn.Module] = None,
                 conv_kernel_size: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.use_razavi_pinverse = use_razavi_pinverse
        self.pinverse_original_init = pinverse_original_init
        self.inv_iterations = inv_iterations
        self.attn_drop = nn.Dropout(dropout)
        self.skip_connection = v_skip_connection
        self.causal = causal
        if self.skip_connection is None and conv_kernel_size is not None:
            self.skip_connection = nn.Conv2d(in_channels=num_heads, out_channels=num_heads,
                                             kernel_size=(conv_kernel_size, 1), padding=(conv_kernel_size // 2, 0),
                                             bias=False, groups=num_heads)
        self.landmark_pooling = landmark_pooling if landmark_pooling is not None else AvgPool(n=num_landmarks)
        self.causal_mask_1 = self.causal_mask_2 = self.causal_mask_3 = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        batched_dim = k.size(0)
        seq_len = k.size(-2)
        tt = {"dtype": q.dtype, "device": q.device}

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = _bool_mask_to_additive(key_padding_mask)
            if key_padding_mask.ndim == 2:
                key_padding_mask = _reshape_key_padding_mask(key_padding_mask, batched_dim)
            is_masked = torch.isinf(-key_padding_mask)
            _mask = torch.where(is_masked, torch.zeros_like(key_padding_mask), torch.ones_like(key_padding_mask))
            _mask = _mask.transpose(2, 1)
            assert _mask.shape == (batched_dim, q.shape[1], 1)
            q = q * _mask
            k = k * _mask
            assert key_padding_mask.size() == (batched_dim, 1, seq_len)

        if self.num_landmarks >= seq_len:
            mask = None
            if self.causal:
                mask = self._triu_mask(batched_dim, seq_len, seq_len, **tt)
            if key_padding_mask is not None:
                mask = key_padding_mask if mask is None else mask + key_padding_mask
            x = scaled_dot_product_attention(q=q, k=k, v=v, att_mask=mask)
        else:
            q_landmarks = self.landmark_pooling(q)
            k_landmarks = self.landmark_pooling(k)
            if self.causal and (self.causal_mask_1 is None
                                or (batched_dim, seq_len, self.num_landmarks) != self.causal_mask_1.size()):
                self.causal_mask_1 = self._triu_mask(batched_dim, seq_len, self.num_landmarks, **tt)
                self.causal_mask_2 = self._triu_mask(batched_dim, self.num_landmarks, self.num_landmarks, **tt)
                self.causal_mask_3 = self._triu_mask(batched_dim, self.num_landmarks, seq_len, **tt)
            mask_3 = self.causal_mask_3
            if key_padding_mask is not None:
                mask_3 = key_padding_mask if mask_3 is None else mask_3 + key_padding_mask
            kernel_1 = scaled_query_key_softmax(q=q, k=k_landmarks, att_mask=None)
            kernel_2 = scaled_query_key_softmax(q=q_landmarks, k=k_landmarks, att_mask=None)
            kernel_3 = scaled_dot_product_attention(q=q_landmarks, k=k, v=v, att_mask=mask_3)
            kernel_2_inv = (iterative_pinv(kernel_2, self.inv_iterations, self.pinverse_original_init)
                            if self.use_razavi_pinverse else torch.linalg.pinv(kernel_2))
            x = torch.matmul(torch.matmul(kernel_1, kernel_2_inv), kernel_3)

        if self.skip_connection:
            v_conv = self.skip_connection(v.reshape(-1, self.num_heads, v.size(-2), v.size(-1)))
            x += v_conv.reshape(-1, v_conv.size(-2), v_conv.size(-1))
        x = self.attn_drop(x)
        return x

    def _triu_mask(self, dim_1: int, dim_2: int, dim_3: int, **kwargs) -> torch.Tensor:
        return torch.triu(torch.ones(dim_2, dim_3, dtype=kwargs["dtype"], device=kwargs["device"]) * float("-inf"),
                          diagonal=1).expand(dim_1, -1, -1)
