from __future__ import annotations

from functools import partial
from types import MethodType

import torch


def _slice_frame_tokens(
    aggregated_tokens_list: list[torch.Tensor | None],
    start: int,
    end: int,
) -> list[torch.Tensor | None]:
    return [
        None if tokens is None else tokens[:, start:end]
        for tokens in aggregated_tokens_list
    ]


def _run_dpt_head_in_frame_chunks(
    head,
    aggregated_tokens_list: list[torch.Tensor | None],
    images: torch.Tensor,
    patch_start_idx: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_count = images.shape[1]
    outputs: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    for start in range(0, frame_count, chunk_size):
        end = min(start + chunk_size, frame_count)
        output, confidence = head(
            _slice_frame_tokens(aggregated_tokens_list, start, end),
            images=images[:, start:end],
            patch_start_idx=patch_start_idx,
        )
        outputs.append(output)
        confidences.append(confidence)
    return torch.cat(outputs, dim=1), torch.cat(confidences, dim=1)


def _chunked_vggt_forward(
    self,
    images: torch.Tensor,
    query_points: torch.Tensor | None = None,
    *,
    chunk_size: int,
):
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if query_points is not None and query_points.ndim == 2:
        query_points = query_points.unsqueeze(0)

    aggregated_tokens_list, patch_start_idx = self.aggregator(images)
    predictions = {}
    with torch.cuda.amp.autocast(enabled=False):
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

        if self.depth_head is not None:
            depth, depth_conf = _run_dpt_head_in_frame_chunks(
                self.depth_head,
                aggregated_tokens_list,
                images,
                patch_start_idx,
                chunk_size,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            points, points_conf = _run_dpt_head_in_frame_chunks(
                self.point_head,
                aggregated_tokens_list,
                images,
                patch_start_idx,
                chunk_size,
            )
            predictions["world_points"] = points
            predictions["world_points_conf"] = points_conf

    if self.track_head is not None and query_points is not None:
        track_list, visibility, confidence = self.track_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
            query_points=query_points,
        )
        predictions["track"] = track_list[-1]
        predictions["vis"] = visibility
        predictions["conf"] = confidence

    if not self.training:
        predictions["images"] = images
    return predictions


def install_frame_chunked_dpt_heads(model, chunk_size: int):
    """Install an inference-only, frame-chunked VGGT head forward path."""
    if chunk_size <= 0:
        return model
    if model.training:
        raise ValueError("frame-chunked DPT heads are inference-only")
    forward = partial(_chunked_vggt_forward, chunk_size=chunk_size)
    model.forward = MethodType(forward, model)
    model._dpt_frame_chunk_size = chunk_size
    return model
