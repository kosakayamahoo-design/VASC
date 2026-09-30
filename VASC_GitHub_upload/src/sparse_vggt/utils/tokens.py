import torch
def get_patch_tokens(x, N, P, S):
    """Get patch tokens from the input tensor.

    Args:
        x (torch.Tensor): input tensor of shape (..., N * P, C)

    Returns:
        patch (torch.Tensor): patch tokens of shape (..., N * (P - S), C)

    where
        N: number of frames
        P: number of patch tokens + special tokens
        S: number of special tokens
    """
    assert x.shape[-2] == N * P

    batch_dims = x.shape[:-2]
    channels = x.shape[-1]

    x = x.view(batch_dims + (N, P, channels))
    patch = x[..., S:, :]  # (..., N, P - S, C)
    patch = patch.reshape(batch_dims + (N * (P - S), channels))
    return patch.contiguous()


def get_special_tokens(x, N, P, S):
    """Get special tokens from the input tensor.

    Args:
        x (torch.Tensor): input tensor of shape (..., N * P, C)

    Returns:
        special (torch.Tensor): special tokens of shape (..., N * S, C)
    """
    assert x.shape[-2] == N * P

    batch_dims = x.shape[:-2]
    channels = x.shape[-1]

    x = x.view(batch_dims + (N, P, channels))
    special = x[..., :S, :]  # (..., N, S, C)
    special = special.reshape(batch_dims + (N * S, channels))
    if special.numel() == 0:
        return None
    return special


def reorder_to_patch_then_special(x, N, P, S):
    """Reorder frame-major tokens into [all patch tokens, all special tokens].

    Args:
        x (torch.Tensor): input tensor of shape (..., N * P, C)

    Returns:
        torch.Tensor: reordered tensor of shape (..., N * P, C)
    """
    patch = get_patch_tokens(x, N, P, S)
    special = get_special_tokens(x, N, P, S)
    if special is None:
        return patch
    return torch.cat([patch, special], dim=-2).contiguous()


def reorder_to_patch_then_special_with_patch(x, N, P, S):
    """Return the reordered tensor and reuse its initial patch extraction."""
    patch = get_patch_tokens(x, N, P, S)
    special = get_special_tokens(x, N, P, S)
    if special is None:
        return patch, patch
    reordered = torch.cat([patch, special], dim=-2).contiguous()
    return patch, reordered


def pack_patch_then_special_direct_with_patch(x, N, P, S):
    """Pack the final layout once and expose its zero-copy patch prefix."""
    assert x.shape[-2] == N * P

    batch_dims = x.shape[:-2]
    channels = x.shape[-1]
    patch_tokens = P - S
    if S == 0:
        reordered = x.contiguous()
        return reordered, reordered

    source = x.view(batch_dims + (N, P, channels))
    reordered = torch.empty(
        x.shape,
        dtype=x.dtype,
        device=x.device,
    )
    patch = reordered[..., : N * patch_tokens, :]
    patch.view(batch_dims + (N, patch_tokens, channels)).copy_(
        source[..., S:, :]
    )
    reordered[..., N * patch_tokens :, :].view(
        batch_dims + (N, S, channels)
    ).copy_(source[..., :S, :])
    return patch, reordered


def restore_to_frame_major(x, N, P, S):
    """Restore [all patch tokens, all special tokens] back to frame-major layout.

    Args:
        x (torch.Tensor): input tensor of shape (..., N * P, C)

    Returns:
        torch.Tensor: restored tensor of shape (..., N * P, C)
    """
    assert x.shape[-2] == N * P

    batch_dims = x.shape[:-2]
    channels = x.shape[-1]
    patch_tokens = P - S

    patch = x[..., : N * patch_tokens, :]
    special = x[..., N * patch_tokens :, :]

    patch = patch.view(batch_dims + (N, patch_tokens, channels))
    if S > 0:
        special = special.view(batch_dims + (N, S, channels))
        restored = torch.cat([special, patch], dim=-2)
    else:
        restored = patch
    return restored.reshape(batch_dims + (N * P, channels)).contiguous()


def combine_patch_and_special_frame_major(patch, special, N, P, S):
    """Combine per-frame patch and flattened special tokens in one copy."""
    assert patch.shape[-3] == N
    assert patch.shape[-2] == P - S

    batch_dims = patch.shape[:-3]
    channels = patch.shape[-1]
    if S > 0:
        assert special is not None
        assert special.shape == batch_dims + (N * S, channels)
        special = special.view(batch_dims + (N, S, channels))
        restored = torch.cat([special, patch], dim=-2)
    else:
        assert special is None or special.numel() == 0
        restored = patch
    return restored.reshape(batch_dims + (N * P, channels)).contiguous()
