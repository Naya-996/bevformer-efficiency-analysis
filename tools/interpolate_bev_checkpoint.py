#!/usr/bin/env python3
"""Interpolate BEVFormer BEV embeddings in a trusted local checkpoint.

This utility intentionally loads checkpoints with ``weights_only=False`` so
that legacy OpenMMLab metadata is preserved. Only use it with checkpoints from
a trusted local source.
"""

import argparse
import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


BEV_EMBED_KEY = 'pts_bbox_head.bev_embedding.weight'
ROW_EMBED_KEY = 'pts_bbox_head.positional_encoding.row_embed.weight'
COL_EMBED_KEY = 'pts_bbox_head.positional_encoding.col_embed.weight'

BEV_EMBED_DIM = 256
POSITION_EMBED_DIM = 128
METADATA_KEY = 'bev_checkpoint_interpolations'


class CheckpointConversionError(RuntimeError):
    """Raised when a checkpoint cannot be converted safely."""


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('size must be a positive integer')
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Interpolate BEVFormer BEV checkpoint embeddings.')
    parser.add_argument('input_checkpoint', type=Path)
    parser.add_argument('output_checkpoint', type=Path)
    parser.add_argument(
        '--src-size', type=positive_int, default=200,
        help='source square BEV size (default: 200)')
    parser.add_argument(
        '--dst-size', type=positive_int, required=True,
        help='destination square BEV size')
    parser.add_argument(
        '--mode', choices=('bilinear', 'bicubic'), default='bicubic',
        help='2D interpolation mode for BEV queries (default: bicubic)')
    parser.add_argument(
        '--force', action='store_true',
        help='replace an existing output checkpoint atomically')
    return parser.parse_args()


def load_trusted_checkpoint(path: Path):
    """Load a trusted local checkpoint on CPU across PyTorch versions."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # PyTorch versions predating the weights_only argument.
        return torch.load(path, map_location='cpu')
    except Exception as exc:
        raise CheckpointConversionError(
            f'failed to load trusted checkpoint {path}: {exc}') from exc


def expected_shapes(src_size: int) -> Dict[str, Tuple[int, int]]:
    """Return the exact shapes expected from the BEVFormer base model."""
    return {
        BEV_EMBED_KEY: (src_size * src_size, BEV_EMBED_DIM),
        ROW_EMBED_KEY: (src_size, POSITION_EMBED_DIM),
        COL_EMBED_KEY: (src_size, POSITION_EMBED_DIM),
    }


def validate_checkpoint(checkpoint, src_size: int) -> MutableMapping:
    """Validate checkpoint structure and all size-dependent tensors."""
    if not isinstance(checkpoint, MutableMapping):
        raise CheckpointConversionError(
            'checkpoint must be a mapping containing a state_dict')

    state_dict = checkpoint.get('state_dict')
    if not isinstance(state_dict, MutableMapping):
        raise CheckpointConversionError(
            'checkpoint["state_dict"] must be a mutable mapping')

    for key, shape in expected_shapes(src_size).items():
        if key not in state_dict:
            raise CheckpointConversionError(
                f'missing required state_dict key: {key}')
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise CheckpointConversionError(
                f'{key} must be a torch.Tensor, got {type(tensor).__name__}')
        if tuple(tensor.shape) != shape:
            raise CheckpointConversionError(
                f'{key} has shape {tuple(tensor.shape)}, expected {shape} '
                f'for --src-size {src_size}')
        if not tensor.is_floating_point():
            raise CheckpointConversionError(
                f'{key} must have a floating-point dtype, got {tensor.dtype}')

    meta = checkpoint.get('meta')
    if meta is None:
        checkpoint['meta'] = {}
    elif not isinstance(meta, MutableMapping):
        raise CheckpointConversionError(
            'checkpoint["meta"] must be a mutable mapping when present')

    return state_dict


def interpolation_input(tensor: torch.Tensor) -> torch.Tensor:
    """Use a CPU interpolation dtype supported by PyTorch."""
    if tensor.dtype in (torch.float32, torch.float64):
        return tensor
    return tensor.to(dtype=torch.float32)


def interpolate_bev_embedding(
        tensor: torch.Tensor, src_size: int, dst_size: int,
        mode: str) -> torch.Tensor:
    """Interpolate flattened H-W-C BEV queries in two dimensions."""
    original_dtype = tensor.dtype
    grid = interpolation_input(tensor).reshape(
        src_size, src_size, BEV_EMBED_DIM)
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        grid, size=(dst_size, dst_size), mode=mode, align_corners=False)
    resized = resized.squeeze(0).permute(1, 2, 0).contiguous()
    return resized.reshape(
        dst_size * dst_size, BEV_EMBED_DIM).to(dtype=original_dtype)


def interpolate_axis_embedding(
        tensor: torch.Tensor, dst_size: int) -> torch.Tensor:
    """Linearly interpolate a row or column embedding in one dimension."""
    original_dtype = tensor.dtype
    sequence = interpolation_input(tensor).transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(
        sequence, size=dst_size, mode='linear', align_corners=False)
    return resized.squeeze(0).transpose(0, 1).contiguous().to(
        dtype=original_dtype)


def add_conversion_metadata(checkpoint: MutableMapping, input_path: Path,
                            src_size: int, dst_size: int, mode: str) -> None:
    """Append conversion provenance without changing existing metadata."""
    meta = checkpoint['meta']
    history = meta.get(METADATA_KEY)
    if history is None:
        history = []
        meta[METADATA_KEY] = history
    elif not isinstance(history, list):
        raise CheckpointConversionError(
            f'checkpoint meta field {METADATA_KEY!r} must be a list')

    history.append({
        'source_checkpoint': str(input_path.resolve()),
        'src_size': src_size,
        'dst_size': dst_size,
        'mode': mode,
        'positional_embedding_mode': 'linear',
    })


def save_checkpoint_atomic(checkpoint, output_path: Path) -> None:
    """Save beside the destination and atomically install the result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f'.{output_path.name}.', suffix='.tmp',
        dir=str(output_path.parent))
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def convert_checkpoint(input_path: Path, output_path: Path, src_size: int,
                       dst_size: int, mode: str, force: bool) -> None:
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()

    if not input_path.is_file():
        raise CheckpointConversionError(
            f'input checkpoint is not a regular file: {input_path}')
    if input_path.resolve() == output_path.resolve():
        raise CheckpointConversionError(
            'input and output checkpoint paths must be different')
    if output_path.exists():
        if output_path.is_dir():
            raise CheckpointConversionError(
                f'output path is a directory: {output_path}')
        if not force:
            raise CheckpointConversionError(
                f'output checkpoint already exists: {output_path}; '
                'pass --force to replace it')

    checkpoint = load_trusted_checkpoint(input_path)
    state_dict = validate_checkpoint(checkpoint, src_size)

    state_dict[BEV_EMBED_KEY] = interpolate_bev_embedding(
        state_dict[BEV_EMBED_KEY], src_size, dst_size, mode)
    state_dict[ROW_EMBED_KEY] = interpolate_axis_embedding(
        state_dict[ROW_EMBED_KEY], dst_size)
    state_dict[COL_EMBED_KEY] = interpolate_axis_embedding(
        state_dict[COL_EMBED_KEY], dst_size)
    add_conversion_metadata(
        checkpoint, input_path, src_size, dst_size, mode)

    save_checkpoint_atomic(checkpoint, output_path)
    print(f'Saved interpolated checkpoint: {output_path}')
    print(f'  BEV size: {src_size}x{src_size} -> '
          f'{dst_size}x{dst_size}')
    print(f'  BEV mode: {mode}; positional mode: linear')


def main() -> None:
    args = parse_args()
    try:
        convert_checkpoint(
            args.input_checkpoint,
            args.output_checkpoint,
            args.src_size,
            args.dst_size,
            args.mode,
            args.force,
        )
    except CheckpointConversionError as exc:
        raise SystemExit(f'error: {exc}') from exc


if __name__ == '__main__':
    main()
