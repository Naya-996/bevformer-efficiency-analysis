#!/usr/bin/env python3
"""Render existing RC-BEV predictions on the previously GT-selected keyframes."""
import gc
import hashlib
import json
import re
from pathlib import Path

from visualize_base_lite import (PROJECT_ROOT, NuScenes, atomic_json,
                                 load_results, render_scene)


def main():
    root = PROJECT_ROOT
    original = json.loads((root / 'experiments/lite/visualization_manifest.json').read_text())
    selections = original['selections']
    tokens = {item['sample_token'] for item in selections}
    sources = [('Base-200', root / original['base_results'])]
    log_root = root / 'experiments/rc_bev/runs/rc_eval_seen_s0_epoch24_20260920'
    for resolution in (100, 150, 200):
        log = (log_root / f'eval_{resolution}.log').read_text()
        match = re.search(r'Results writes to (.*results_nusc.json)', log)
        if not match:
            raise RuntimeError(f'No prediction path for {resolution}')
        sources.append((f'RC-BEV-{resolution}', root / match.group(1).strip()))
    records = {}
    provenance = []
    for label, path in sources:
        print('Loading', label, path, flush=True)
        document, results = load_results(path)
        if not tokens.issubset(results):
            raise RuntimeError(f'Missing selected samples in {path}')
        records[label] = {token: results[token] for token in tokens}
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        provenance.append(dict(label=label, path=str(path.relative_to(root)),
                               sha256=digest.hexdigest(), samples=len(results)))
        del document, results
        gc.collect()
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(root / 'data/nuscenes'), verbose=False)
    output_dir = root / 'docs/assets/rc_bev_visualizations'
    for item in selections:
        token = item['sample_token']
        output = output_dir / f"{item['selection_reason']}_{token}.png"
        render_scene(nusc, item, None, None, output, original['score_threshold'],
                     original['max_distance_m'],
                     prediction_rows=[(label, records[label][token]) for label, _ in sources])
        item['output'] = str(output.relative_to(root))
        print('Saved', output, flush=True)
    atomic_json(output_dir / 'manifest.json', dict(
        selection_uses_predictions=False,
        selection_source='experiments/lite/visualization_manifest.json',
        score_threshold=original['score_threshold'], max_distance_m=original['max_distance_m'],
        rows=['GT'] + [label for label, _ in sources], sources=provenance, selections=selections,
        note='Base uses official weights; RC rows use one seed-0 epoch-24 checkpoint.'))


if __name__ == '__main__':
    main()
