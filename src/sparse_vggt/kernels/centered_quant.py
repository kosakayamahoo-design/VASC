"""Per-block INT8 quantization with fused global-key centering."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_per_block_centered_kernel(
    input_ptr,
    mean_ptr,
    output_ptr,
    scale_ptr,
    length,
    stride_ib,
    stride_ih,
    stride_in,
    stride_mb,
    stride_mh,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    head_dim: tl.constexpr,
    block_tokens: tl.constexpr,
    subtract_mean: tl.constexpr,
    emulate_fp16_source: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    token = block * block_tokens + tl.arange(0, block_tokens)
    dim = tl.arange(0, head_dim)
    valid = token[:, None] < length
    input_offsets = (
        batch * stride_ib
        + head * stride_ih
        + token[:, None] * stride_in
        + dim[None, :]
    )
    value = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)
    if emulate_fp16_source:
        value = value.to(tl.float16)
    if subtract_mean:
        mean = tl.load(
            mean_ptr + batch * stride_mb + head * stride_mh + dim
        )
        # Match the materialized FP16 subtraction before promoting to FP32.
        centered_dtype = tl.float16 if emulate_fp16_source else input_ptr.dtype.element_ty
        value = (value.to(tl.float32) - mean.to(tl.float32)).to(centered_dtype)
    value = value.to(tl.float32)
    scale = tl.max(tl.abs(value)) / 127.0 + 1.0e-7
    quantized = value / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    quantized = quantized.to(tl.int8)
    output_offsets = (
        batch * stride_ob
        + head * stride_oh
        + token[:, None] * stride_on
        + dim[None, :]
    )
    tl.store(output_ptr + output_offsets, quantized, mask=valid)
    tl.store(scale_ptr + batch * stride_sb + head * stride_sh + block, scale)


def _launch_quant(
    source: torch.Tensor,
    mean: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
    block_tokens: int,
    *,
    subtract_mean: bool,
    emulate_fp16_source: bool,
) -> None:
    batch, heads, length, head_dim = source.shape
    grid = ((length + block_tokens - 1) // block_tokens, heads, batch)
    _quant_per_block_centered_kernel[grid](
        source,
        mean,
        output,
        scale,
        length,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        mean.stride(0),
        mean.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scale.stride(0),
        scale.stride(1),
        head_dim=head_dim,
        block_tokens=block_tokens,
        subtract_mean=subtract_mean,
        emulate_fp16_source=emulate_fp16_source,
        num_warps=4,
    )


def per_block_int8_fused_centered(
    query: torch.Tensor,
    key: torch.Tensor,
    key_mean: torch.Tensor,
    *,
    query_block: int = 128,
    key_block: int = 64,
    center_key: bool = True,
    emulate_fp16_query_source: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize Q and globally centered K without materializing centered K."""
    if not query.is_cuda or not key.is_cuda or not key_mean.is_cuda:
        raise ValueError("fused centered quantization requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or key_mean.ndim != 4:
        raise ValueError("Q, K, and K mean must be four-dimensional")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Q and K head layouts must match")
    if key_mean.shape != key.shape[:2] + (1, key.shape[-1]):
        raise ValueError("K mean must have shape [B, H, 1, D]")
    if key.dtype != key_mean.dtype:
        raise ValueError("K and K mean must share one dtype")
    if emulate_fp16_query_source:
        if not query.is_floating_point():
            raise ValueError("source-fused Q quantization requires floating-point Q")
        if key.dtype != torch.float16:
            raise ValueError("source-fused Q quantization requires materialized FP16 K")
    elif query.dtype != key.dtype:
        raise ValueError("Q and K must share one dtype")
    query = query.contiguous()
    key = key.contiguous()
    key_mean = key_mean.contiguous()
    q_int8 = torch.empty_like(query, dtype=torch.int8)
    k_int8 = torch.empty_like(key, dtype=torch.int8)
    q_scale = torch.empty(
        query.shape[0],
        query.shape[1],
        (query.shape[2] + query_block - 1) // query_block,
        1,
        device=query.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        key.shape[0],
        key.shape[1],
        (key.shape[2] + key_block - 1) // key_block,
        1,
        device=key.device,
        dtype=torch.float32,
    )
    _launch_quant(
        query,
        key_mean,
        q_int8,
        q_scale,
        query_block,
        subtract_mean=False,
        emulate_fp16_source=emulate_fp16_query_source,
    )
    _launch_quant(
        key,
        key_mean,
        k_int8,
        k_scale,
        key_block,
        subtract_mean=center_key,
        emulate_fp16_source=False,
    )
    return q_int8, q_scale, k_int8, k_scale
