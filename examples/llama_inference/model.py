# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mp_utils
import torch
from torch import nn
from torch.nn import functional as F

from xformers.ops import RMSNorm, fmha, rope_padded
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)


@dataclass
class ModelArgs:
    dim: int = 512

    n_layers: int = 8

    n_heads: int = 8
    n_kv_heads: Optional[int] = None

    vocab_size: int = -1

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256
    """
    Enforces that the SwiGLU hidden layer size is a multiple
    of large power of 2.
    """

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    """
    Positional encoding parameter; increase to 1e6 to run
    Code Llama models with long contexts.
    """


LayerCache = Tuple[torch.Tensor, torch.Tensor]


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()
        mp_size = mp_utils.get_world_size()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads // mp_size
        self.n_local_kv_heads = n_kv_heads // mp_size

        self.wqkv = nn.Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
        )
        self._register_load_state_dict_pre_hook(self.load_hook)

    # This adapter makes sure we can load vanilla
    # Llama checkpoints where wq, wk, and wv are
    # not fused in a single parameter
    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
        position_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x.shape is (sum(seq_lens), dim)
        #
        # Since we support heterogenous sequence
        # lengths, the hidden states are all
        # concatenated together along the usual
        # sequence dimension. The attention below
        # finds out where sequences start & end
        # using the provided attention bias.

        xqkv = self.wqkv(x)
        xq = xqkv[:, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, 1)

        output_shape = xq.shape
        heads_per_group = self.n_local_heads // self.n_local_kv_heads
        xq = xq.view(
            1, xq.shape[0], self.n_local_kv_heads, heads_per_group, self.head_dim
        )
        xk = xk.view(1, xk.shape[0], self.n_local_kv_heads, 1, self.head_dim)
        xv = xv.view(1, xv.shape[0], self.n_local_kv_heads, 1, self.head_dim)
        cache_k, cache_v = cache

        # rope_padded: 用于在计算 Query, Key, Value 时应用 RoPE 旋转位置编码。这会同时更新缓存中的 Key 和 Value
        xq = rope_padded(
            xq=xq,
            xk=xk,
            xv=xv,
            cache_k=cache_k,
            cache_v=cache_v,
            attn_bias=attn_bias,
            theta=self.rope_theta,
        )

        # rope_padded() updated the caches, so we
        # call attention directly
        output = fmha.memory_efficient_attention_forward(
            xq, cache_k, cache_v, attn_bias
        )
        output = output.reshape(output_shape)
        # ？？？？？？？？？？？？？？？？？？？？？？ 应该放在哪里
        if position_index is not None:
            output = output[position_index]
        output = self.wo(output)
        mp_utils.all_reduce(output)

        # if position_index is not None:
        #     output = output[position_index]

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        mp_size = mp_utils.get_world_size()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.w13 = nn.Linear(
            dim,
            2 * hidden_dim // mp_size,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim // mp_size,
            dim,
            bias=False,
        )
        self._register_load_state_dict_pre_hook(self.load_hook)

    # This adapter makes sure we can load vanilla
    # Llama checkpoints where w1 and w3 are not
    # fused in a single parameter
    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "w1.weight" in state_dict:
            w1 = state_dict.pop(prefix + "w1.weight")
            w3 = state_dict.pop(prefix + "w3.weight")
            state_dict[prefix + "w13.weight"] = torch.cat([w1, w3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        output = self.w2(F.silu(x1) * x3)
        mp_utils.all_reduce(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        # print("In TransformerBlock")
        # print(f"args.dim is {args.dim}, args.n_heads is {args.n_heads}, head_dim is {head_dim}")
        # args.dim is 4096, args.n_heads is 32, head_dim is 128

        mp_size = mp_utils.get_world_size()
        assert args.n_heads % n_kv_heads == 0
        assert args.n_heads % mp_size == 0
        assert n_kv_heads % mp_size == 0

        self.is_last_layer = layer_index + 1 == args.n_layers

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:
        position_index = None
        position_index_tmp = None
        if self.is_last_layer and attn_bias.q_seqinfo.max_seqlen > 1:
            position_index = attn_bias.q_seqinfo.seqstart[1:] - 1
            position_index_tmp = attn_bias.q_seqinfo.seqstart[1:] - 2
            # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx, position_index is ", position_index)
            # position_index is  tensor([8], device='cuda:0', dtype=torch.int32)

        h = self.attention.forward(
            self.attention_norm(x),
            cache,
            attn_bias,
            # position_index=position_index_tmp,
            position_index=position_index,
        )
        # print("h is ", h)
        # print("h shape is ", h.shape)
        if position_index is not None:
            # torch.set_printoptions(profile="full")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("x = ", x)
            print("x shape= ", x.shape)
            print("x[position_index] = ", x[position_index])
            print("x[position_index] shape = ", x[position_index].shape)
            print("x[position_index - 1] = ", x[position_index_tmp])
            # torch.set_printoptions(profile="default")
            
            x = x[position_index]
            # x = x[position_index_tmp]
            # h = h[position_index]           
            
            # torch.set_printoptions(profile="full")
            print("h = ", h)
            # torch.set_printoptions(profile="default")
        # print("h is ", h)
        # print("x is ", x)
        h = h + x
        # print("h = ", h)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        mp_size = mp_utils.get_world_size()
        assert args.dim % mp_size == 0
        assert args.vocab_size > 0
        assert args.vocab_size % mp_size == 0

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim // mp_size,
        )

        self.layers = nn.ModuleList()
        for layer_index in range(args.n_layers):
            self.layers.append(TransformerBlock(args, layer_index))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size // mp_size,
            bias=False,
        )

    @torch.no_grad()
    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias: AttnBias,
        cache: list[LayerCache],
    ) -> torch.Tensor:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ in forward_with_attn_bias")
        h_parallel = self.tok_embeddings(token_values)
        # print("token_values is ", token_values)
        # print("token_values shape is ", token_values.shape)
        # print("h_parallel is ", h_parallel)
        # print("h_parallel shape is ", h_parallel.shape)       # [tokens_length, 4096], 前者在 prefill 是 9, decode 时是 1
        h = mp_utils.all_gather(h_parallel)
        # print("h is ", h)
        # print("h shape is ", h.shape)                         # 与 h_parallel 形状相同

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], attn_bias)
            # print("h is ", h)
            # print("h shape is ", h.shape)                         # 与 h_parallel 形状相同. prefill 时特殊，前 31 层都是 [9, 4096], 最后一层的输出是 [1, 4096]

        logits_parallel = self.output(self.norm(h))
        logits = mp_utils.all_gather(logits_parallel)
        return logits.float()

    def forward(
        self,
        token_values: torch.Tensor,
        token_lengths: torch.Tensor,
        start_pos: torch.Tensor,
        cache: list[LayerCache],
        kv_padding: int,
    ) -> torch.Tensor:
        attn_bias = AttnBias.from_seqlens(
            q_seqlen=token_lengths.tolist(),
            kv_seqlen=(start_pos + token_lengths).tolist(),
            kv_padding=kv_padding,
        )
        return self.forward_with_attn_bias(token_values, attn_bias, cache)


def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[LayerCache]:
    """
    Allocate a cache to be used with the Transformer module.

    Args:
        args (ModelArgs): the model configuration.
        length (int): per layer cache size.
            It is usually budgeted as ``max_batch * max_seq``
        device (torch.device, optional): the device on which
            the cache should be allocated.
        n_layers (int, optional): the number of layers to
            allocate a cache for (defaults to the model
            settings).
        dtype (torch.dtype, optional): the dtype to use for
            cache entries (defaults to the default dtype).

    Returns:
        The cache object to pass to ``Tranformer.forward``.
    """

    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = args.n_heads
    n_local_kv_heads = n_kv_heads // mp_utils.get_world_size()

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_local_kv_heads, 1, head_dim)
    heads_per_group = args.n_heads // n_kv_heads
    expansion = (-1, -1, -1, heads_per_group, -1)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: list[LayerCache], length: int) -> list[LayerCache]:
    """
    Take a prefix view of a larger cache.

    The original cache object remains of identical size and valid
    after the shrinked alias has been used. This function is useful
    when a cache was allocated for a larger batch size than what is
    necessary.

    Args:
        cache: the cache to take a view in.
        length (int): the desired length

    Returns:
        A view in the input cache object.
    """

    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length

    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]
