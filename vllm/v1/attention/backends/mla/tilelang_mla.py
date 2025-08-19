# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import Optional

import torch
from tilelang.jit import JITKernel

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.ops.tilelang_mla import tilelang_mla_decode_with_kv_cache
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.attention.backends.utils import (get_per_layer_parameters,
                                              infer_global_hyperparameters)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class TileLangMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "TILELANG_MLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["TileLangMLAMetadata"]:
        return TileLangMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["TileLangMLAMetadataBuilder"]:
        return TileLangMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TileLangMLAImpl"]:
        return TileLangMLAImpl


@dataclass
class TileLangMLADecodeMetadata(MLACommonDecodeMetadata):
    mla_decoder_kernel: JITKernel
    num_kv_splits: int


@dataclass
class TileLangMLAMetadata(MLACommonMetadata[TileLangMLADecodeMetadata]):
    pass


class TileLangMLAMetadataBuilder(MLACommonMetadataBuilder[TileLangMLAMetadata]
                                 ):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         TileLangMLAMetadata)
        num_heads_kv = self.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, MLACommonImpl))
        scale = global_hyperparameters.sm_scale

        # TODO: The TileLang MLA decode kernel is currently configured
        # for fp16 and bf16 on H100. Running it with float exceeds the shared
        # memory capacity of H100.We will need to tune the implementation
        # for other GPU architectures and data types in the future.
        dtype_to_str = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        dtype = self.model_config.dtype
        dtype_str = dtype_to_str.get(dtype)
        if dtype_str is None:
            raise NotImplementedError(
                f"TileLangMLAImpl does not support {dtype} yet.")

        # heuristics from TileLang's example
        self.num_kv_splits = 4
        BLOCK_N = 64
        BLOCK_H = min(64, self.num_heads // num_heads_kv)

        max_seqlen = self.model_config.max_model_len
        max_seqlen_pad = math.ceil(max_seqlen / 256) * 256
        qk_rope_head_dim = self.mla_dims.qk_rope_head_dim
        kv_lora_rank = self.mla_dims.kv_lora_rank
        block_size = self.kv_cache_spec.block_size

        self.mla_decoder_kernel = tilelang_mla_decode_with_kv_cache(
            self.num_heads, num_heads_kv, max_seqlen_pad, kv_lora_rank,
            qk_rope_head_dim, BLOCK_N, BLOCK_H, scale, self.num_kv_splits,
            block_size, dtype_str)

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens: torch.Tensor) -> TileLangMLADecodeMetadata:
        return TileLangMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens,
            mla_decoder_kernel=self.mla_decoder_kernel,
            num_kv_splits=self.num_kv_splits,
        )


class TileLangMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        # TODO: TileLang has a high-performance FlashMLA implementation for
        # AMD MI300X. Let's add it later.
        if current_platform.is_rocm():
            raise NotImplementedError(
                "TileLangMLAImpl does not support ROCM yet.")

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TileLangMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TileLangMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TileLangMLA V1 with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 TileLang MLA not yet supported")

        q = torch.cat([q_nope, q_pe], dim=-1)  # [B, h_q, dv + dpe]

        B = q_nope.shape[0]
        dtype = q_nope.dtype
        device = q_nope.device
        output_partial = torch.empty(B,
                                     self.num_heads,
                                     attn_metadata.decode.num_kv_splits,
                                     self.kv_lora_rank,
                                     dtype=dtype,
                                     device=device)
        glse = torch.empty(B,
                           self.num_heads,
                           attn_metadata.decode.num_kv_splits,
                           dtype=dtype,
                           device=device)

        head_dim = kv_c_and_k_pe_cache.size(-1)  # nope + rope

        # Run TileLang FlashMLA
        mla_decoder = attn_metadata.decode.mla_decoder_kernel
        output = mla_decoder(
            q, kv_c_and_k_pe_cache.view(-1, self.num_kv_heads, head_dim),
            attn_metadata.decode.block_table, attn_metadata.decode.seq_lens,
            glse, output_partial)
        return self._v_up_proj(output)
