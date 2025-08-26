# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for GPTQ."""
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import direct_register_custom_op


def fused_tilelang_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Optional[str] = "silu",
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type == scalar_types.float4_e2m1f
    num_bits = 4

    # model config:
    # hiden_size = 2880
    # num_experts = 32
    # intermediate_size = 1472

    # hidden_states.shape: [16384, 2880]
    # w1.shape: [32, 2944, 1440]
    # w2.shape: [32, 2880, 736]
    # bias1.shape: [32, 2944]
    # bias2.shape: [32, 2880]
    # w1_scale.shape: [32, 2944, 90]
    # w2_scale.shape: [32, 2880, 46]

    # topk_ids.shape: [16384, 4]

    # more details: https://github.com/vllm-project/vllm/blob/8a3cd90af534c39425ebfdfd295eea0a4582d541/vllm/model_executor/layers/quantization/mxfp4.py#L278-L301

    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[
        0], "Number of tokens mismatch"
    assert hidden_states.shape[1] // 2 == w1.shape[2], ("Hidden size mismatch")
    assert hidden_states.shape[1] == w2.shape[1], ("Hidden size mismatch w2")
    assert w1.shape[1] == w2.shape[2] * 4, ("shape mismatch w1 and w2")
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert topk_weights.dtype == torch.float32

    M, K = hidden_states.shape
    E, N, _ = w1.size()
    topk = topk_ids.shape[1]

    if global_num_experts == -1:
        global_num_experts = E
    # TODO: tune this number later
    block_M = 256
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, block_M, global_num_experts,
                             expert_map)

    intermediate_cache2 = torch.empty(
        (M * topk, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        M * topk * max(2 * N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[:M * topk * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[:M * topk * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    fast_dequant = True
    with_bias_1 = bias1 is not None
    # TODO: tune these configs further for specific models later
    block_N = 128
    block_K = 128
    num_stages = 2
    threads = 256
    split = 1
    gemm_kernel_1 = tilelang_gemm_bf16_mxfp4_hopper(
        M,
        N,
        K,
        "bfloat16",
        "bfloat16",
        "float32",
        num_bits=num_bits,
        scale_size=32,  # mxfp4 block size
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        split=split,
        fast_dequant=fast_dequant,
        with_bias=with_bias_1)
    intermediate_cache1 = gemm_kernel_2(
        hidden_states,
        w1,
        bias1,
        w1_scale,
        intermediate_cache1,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        top_k=topk,
        mul_topk_weights=apply_router_weight_on_input)

    if activation == "silu":
        torch.ops._C.silu_and_mul(intermediate_cache2,
                                  intermediate_cache1.view(-1, 2 * N))
    elif activation == "swigluoai":
        # alpha = 1.702, limit = 7.0
        torch.ops._C.swigluoai_and_mul(intermediate_cache2,
                                       intermediate_cache1.view(-1, 2 * N))
    else:
        raise ValueError(f"Unsupported activation: {activation}. "
                         "Only silu and swigluoai activations are supported.")

    with_bias_2 = bias2 is not None
    gemm_kernel_2 = tilelang_gemm_bf16_mxfp4_hopper(
        M * topk,  # M
        K,  # N
        N,  # K
        "bfloat16",
        "bfloat16",
        "float32",
        num_bits=num_bits,
        scale_size=32,  # mxfp4 block size
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        split=split,
        fast_dequant=fast_dequant,
        with_bias=with_bias_2)
    intermediate_cache1 = gemm_kernel_2(
        intermediate_cache2,
        w2,
        bias2,
        w2_scale,
        intermediate_cache3,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        top_k=1,
        mul_topk_weights=not apply_router_weight_on_input).view(-1, topk, K)

    output = torch.empty_like(hidden_states)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1,
                     out=output)


def fused_tilelang_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Optional[str] = "silu",
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_tilelang_moe",
    op_func=fused_tilelang_moe,
    mutates_args=[],
    fake_impl=fused_tilelang_moe_fake,
)
