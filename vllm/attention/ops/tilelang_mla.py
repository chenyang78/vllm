# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The tilelang kernel mla_decode_tilelang is based on the example at
# https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/example_mla_decode_paged.py

from typing import Optional, Tuple

import torch
from tilelang import tvm as tvm

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# TODO: TileLang supports FlashMLA for AMD MI300X.
# Allow it later.
if current_platform.is_cuda():
    try:
        import tilelang
        import tilelang.language as T
        _tilelang_mla_AVAILABLE = True
    except ImportError:
        _tilelang_mla_AVAILABLE = False
else:
    _tilelang_mla_AVAILABLE = False


def is_tilelang_mla_supported() -> Tuple[bool, Optional[str]]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    if not current_platform.is_cuda():
        return False, "TileLangMLA is only supported on CUDA devices."
    if current_platform.get_device_capability()[0] != 9:
        return False, "TileLangMLA is only supported on Hopper devices."
    if not _tilelang_mla_AVAILABLE:
        return False, "Failed to import tilelang. Please install it by "\
            "following the instructions provided at "\
            "https://github.com/tile-ai/tilelang."

    return True, "Unknown"


@tilelang.jit(out_idx=[8], verbose=True)
def mla_decode_tilelang(h_q, h_kv, max_seqlen_pad, dv, dpe, block_N, block_H,
                        scale, num_split, block_size, dtype, accum_dtype):
    batch = tvm.te.var("batch")

    kv_group_num = h_q // h_kv
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert h_kv == 1, "h_kv must be 1"
    assert block_size >= block_N and block_size % block_N == 0, (
        "block_size must be larger than block_N and a multiple of block_N")

    @T.macro
    def flash_mla_kernel(
            Q: T.Tensor([batch, h_q, dv], dtype),
            Q_pe: T.Tensor([batch, h_q, dpe], dtype),
            KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
            K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
            BLOCK_TABLE: T.Tensor([batch, max_seqlen_pad // block_size
                                   ], "int32"),
            CACHE_SEQLENS: T.Tensor([batch], "int32"),
            Output: T.Tensor([batch, h_q, dv], dtype),
    ):
        with T.Kernel(batch, h_q // min(block_H, kv_group_num),
                      threads=256) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H)
            T.use_swizzle(10)
            T.annotate_layout({
                O_shared:
                tilelang.layout.make_swizzled_layout(O_shared),
                S_shared:
                tilelang.layout.make_swizzled_layout(S_shared),
            })

            T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :],
                   Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :],
                   Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(CACHE_SEQLENS[bx], block_N)
            for kr in T.Pipelined(loop_range, num_stages=2):
                k = loop_range - 1 - kr
                kv_start = BLOCK_TABLE[bx, (k * block_N) //
                                       block_size] * block_size + (
                                           k * block_N) % block_size
                T.copy(KV[kv_start:kv_start + block_N, cur_kv_head, :],
                       KV_shared)
                T.copy(K_pe[kv_start:kv_start + block_N, cur_kv_head, :],
                       K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared,
                       KV_shared,
                       acc_s,
                       transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared,
                       K_pe_shared,
                       acc_s,
                       transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                if kr == 0:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= CACHE_SEQLENS[bx],
                            -T.infinity(accum_dtype), acc_s[i, j])
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                             scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale -
                                         scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared,
                       KV_shared,
                       acc_o,
                       policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared,
                   Output[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])

    @T.macro
    def flash_mla_split_kv_kernel(
            Q: T.Tensor([batch, h_q, dv], dtype),
            Q_pe: T.Tensor([batch, h_q, dpe], dtype),
            KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
            K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
            BLOCK_TABLE: T.Tensor([batch, max_seqlen_pad // block_size],
                                  "int32"),
            CACHE_SEQLENS: T.Tensor([batch], "int32"),
            glse: T.Tensor([batch, h_q, num_split], dtype),
            Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
    ):
        with T.Kernel(batch,
                      h_q // min(block_H, kv_group_num),
                      num_split,
                      threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H)
            T.use_swizzle(10)
            T.annotate_layout({
                O_shared:
                tilelang.layout.make_swizzled_layout(O_shared),
                S_shared:
                tilelang.layout.make_swizzled_layout(S_shared),
            })

            T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :],
                   Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :],
                   Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            total_blocks = T.ceildiv(CACHE_SEQLENS[bx], block_N)
            blocks_per_split = T.floordiv(total_blocks, num_split)
            remaining_blocks = T.floormod(total_blocks, num_split)
            loop_range = (blocks_per_split +
                          T.if_then_else(bz < remaining_blocks, 1, 0))
            start = (blocks_per_split * bz +
                     T.min(bz, remaining_blocks)) * block_N

            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = BLOCK_TABLE[bx, (start + k * block_N) //
                                       block_size] * block_size + (
                                           k * block_N) % block_size
                T.copy(KV[kv_start:kv_start + block_N, cur_kv_head, :],
                       KV_shared)
                T.copy(K_pe[kv_start:kv_start + block_N, cur_kv_head, :],
                       K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared,
                       KV_shared,
                       acc_s,
                       transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared,
                       K_pe_shared,
                       acc_s,
                       transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        start + k * block_N + j >= CACHE_SEQLENS[bx],
                        -T.infinity(accum_dtype), acc_s[i, j])
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                             scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale -
                                         scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast,
                       KV_shared,
                       acc_o,
                       policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum,
                   glse[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz])
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output_partial[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H,
                               bz, :])

    @T.macro
    def combine(
            glse: T.Tensor([batch, h_q, num_split], dtype),
            Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
            Output: T.Tensor([batch, h_q, dv], dtype),
    ):
        with T.Kernel(h_q, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dv], dtype)
            o_accum_local = T.alloc_fragment([dv], accum_dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_local([1], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            T.annotate_layout({
                lse_logsum_local:
                T.Fragment(lse_logsum_local.shape,
                           forward_thread_fn=lambda i: i),
            })

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local[0] = T.max(lse_max_local[0], glse[bz, by, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, by, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] -
                                              lse_max_local[0])
            lse_logsum_local[0] = T.log2(
                lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dv):
                    po_local[i] = Output_partial[bz, by, k, i]
                lse_local_split[0] = glse[bz, by, k]
                scale_local[0] = T.exp2(lse_local_split[0] -
                                        lse_logsum_local[0])
                for i in T.Parallel(dv):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dv):
                Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def main_split(
            Q: T.Tensor([batch, h_q, dv], dtype),
            Q_pe: T.Tensor([batch, h_q, dpe], dtype),
            KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
            K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
            block_table: T.Tensor([batch, max_seqlen_pad // block_size
                                   ], "int32"),
            cache_seqlens: T.Tensor([batch], "int32"),
            glse: T.Tensor([batch, h_q, num_split], dtype),
            Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
            Output: T.Tensor([batch, h_q, dv], dtype),
    ):
        flash_mla_split_kv_kernel(Q, Q_pe, KV, K_pe, block_table,
                                  cache_seqlens, glse, Output_partial)
        combine(glse, Output_partial, Output)

    @T.prim_func
    def main_no_split(
            Q: T.Tensor([batch, h_q, dv], dtype),
            Q_pe: T.Tensor([batch, h_q, dpe], dtype),
            KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
            K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
            block_table: T.Tensor([batch, max_seqlen_pad // block_size
                                   ], "int32"),
            cache_seqlens: T.Tensor([batch], "int32"),
            glse: T.Tensor([batch, h_q, num_split], dtype),
            Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
            Output: T.Tensor([batch, h_q, dv], dtype),
    ):
        flash_mla_kernel(Q, Q_pe, KV, K_pe, block_table, cache_seqlens, Output)

    if num_split > 1:
        return main_split
    else:
        return main_no_split


def tilelang_mla_decode_with_kv_cache(
    num_heads_q: int,
    num_heads_kv: int,
    max_seqlen_pad: int,
    v_head_dim: int,
    qk_rope_head_dim: int,
    BLOCK_N: int,
    BLOCK_H: int,
    scale: float,
    num_kv_splits: int,
    page_block_size: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype = "float",
):
    #scale = (1.0 / (dv + dpe))**0.5 * 1.44269504  # log2(e)
    return mla_decode_tilelang(num_heads_q, num_heads_kv, max_seqlen_pad,
                               v_head_dim, qk_rope_head_dim, BLOCK_N, BLOCK_H,
                               scale, num_kv_splits, page_block_size, dtype,
                               accum_dtype)
