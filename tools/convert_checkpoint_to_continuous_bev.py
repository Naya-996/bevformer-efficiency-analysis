#!/usr/bin/env python3
"""Fit continuous BEV query/position parameters to one discrete checkpoint grid.

The conversion is deliberately labelled lossy: the generated checkpoint requires
multi-resolution adaptation and must not be reported as an evaluated method result.
"""

import argparse
import json
from pathlib import Path

import torch

from projects.mmdet3d_plugin.bevformer.modules.resolution_continuous import (
    ContinuousBEVQueryGenerator)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--height', type=int, default=200)
    parser.add_argument('--width', type=int, default=200)
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0))
    parser.add_argument('--num-bands', type=int, default=16)
    parser.add_argument('--hidden-dims', type=int, default=256)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def find_key(state, suffix):
    matches = [key for key in state if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f'expected one key ending in {suffix!r}, found {matches}')
    return matches[0]


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.input, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    query_key = find_key(state, 'pts_bbox_head.bev_embedding.weight')
    row_key = find_key(state, 'pts_bbox_head.positional_encoding.row_embed.weight')
    col_key = find_key(state, 'pts_bbox_head.positional_encoding.col_embed.weight')
    query_target = state[query_key].float()
    row = state[row_key].float()
    col = state[col_key].float()
    if query_target.shape[0] != args.height * args.width:
        raise ValueError('source query count does not match --height/--width')
    if row.shape[0] < args.height or col.shape[0] < args.width:
        raise ValueError('source positional tables are smaller than source grid')
    position_target = torch.cat((
        col[:args.width].unsqueeze(0).expand(args.height, -1, -1),
        row[:args.height].unsqueeze(1).expand(-1, args.width, -1)), dim=-1)
    position_target = position_target.reshape(args.height * args.width, -1)
    generator = ContinuousBEVQueryGenerator(
        embed_dims=query_target.shape[1], num_bands=args.num_bands,
        hidden_dims=args.hidden_dims, pc_range=args.pc_range)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr)
    features = generator.fourier_features(
        args.height, args.width, args.pc_range, generator.content.device)
    for step in range(args.steps):
        indices = torch.randint(0, query_target.shape[0], (args.batch_size,))
        selected = features[indices]
        predicted_query = generator.content + generator.query_mlp(selected)
        predicted_position = generator.position_mlp(selected)
        loss_query = torch.nn.functional.mse_loss(
            predicted_query, query_target[indices])
        loss_position = torch.nn.functional.mse_loss(
            predicted_position, position_target[indices])
        loss = loss_query + loss_position
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    prefix = query_key[:-len('bev_embedding.weight')] + 'continuous_query_generator.'
    for key, value in generator.state_dict().items():
        state[prefix + key] = value
    checkpoint['state_dict'] = state
    checkpoint.setdefault('meta', {})['continuous_bev_conversion'] = {
        'status': 'WARM_START_NOT_EVALUATED',
        'lossy': True,
        'source_grid': [args.height, args.width],
        'steps': args.steps,
        'seed': args.seed,
        'final_query_mse': float(loss_query.item()),
        'final_position_mse': float(loss_position.item()),
        'new_parameter_prefix': prefix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(json.dumps(checkpoint['meta']['continuous_bev_conversion'], indent=2))


if __name__ == '__main__':
    main()

