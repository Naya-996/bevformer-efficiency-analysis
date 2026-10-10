#!/usr/bin/env python3
"""Summarize only completed training/evaluation/profile triples from a queue."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from run_rc_bev_followup import ROOT, save
from run_rc_bev_training_followup import experiments


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    state = json.loads((directory / 'status.json').read_text())
    rows = []
    hashes = {}
    # The existing Seed 0 belongs to the same RC multi-resolution family.
    evidence = ROOT / 'experiments/rc_bev/evidence/seed0_epoch24'
    for resolution in (100, 125, 150, 175, 200):
        metrics = json.loads((evidence / 'metrics' / f'metrics_{resolution}.json').read_text())
        profile = json.loads((evidence / 'profiles' / f'profile_{resolution}.json').read_text())
        rows.append(dict(family='rc_multires', seed=0, resolution=resolution,
                         nds=metrics['nd_score'], map=metrics['mean_ap'],
                         p50_ms=profile['latency_ms']['p50'], p95_ms=profile['latency_ms']['p95'],
                         fps=profile['throughput']['fps']))
    for spec in experiments():
        trained = state['jobs'].get(spec['id'] + '_train', {})
        if trained.get('status') != 'COMPLETE':
            continue
        checkpoint = Path(trained['artifact'])
        hash_file = checkpoint.parent.parent / 'checkpoint_sha256.json'
        if not hash_file.exists():
            digest = hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            save(hash_file, dict(path=str(checkpoint), sha256=digest.hexdigest(),
                                 bytes=checkpoint.stat().st_size))
        hashes[spec['id']] = json.loads(hash_file.read_text())
        for resolution in spec['resolutions']:
            evaluation = state['jobs'].get(f'{spec["id"]}_eval_{resolution}', {})
            profiling = state['jobs'].get(f'{spec["id"]}_profile_{resolution}', {})
            if evaluation.get('status') != 'COMPLETE' or profiling.get('status') != 'COMPLETE':
                continue
            metrics = json.loads(Path(evaluation['artifact']).read_text())
            profile = json.loads(Path(profiling['artifact']).read_text())
            assert len(profile['latency_ms']['per_iteration']) == 200
            rows.append(dict(family=spec['id'].rsplit('_s', 1)[0], seed=spec['seed'],
                             resolution=resolution, nds=metrics['nd_score'], map=metrics['mean_ap'],
                             p50_ms=profile['latency_ms']['p50'], p95_ms=profile['latency_ms']['p95'],
                             fps=profile['throughput']['fps']))
    with (directory / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['family'], row['resolution']), []).append(row)
    aggregate = []
    for (family, resolution), samples in sorted(groups.items()):
        seeds = sorted({sample['seed'] for sample in samples})
        record = dict(family=family, resolution=resolution, seeds=seeds, n=len(samples))
        if len(seeds) == 3 and len(samples) == 3:
            record.update(status='THREE_SEEDS_COMPLETE')
            for metric in ('nds', 'map'):
                values = [sample[metric] for sample in samples]
                record[metric + '_mean'] = statistics.mean(values)
                record[metric + '_sample_std'] = statistics.stdev(values)
        else:
            record['status'] = 'INCOMPLETE_NO_MULTI_SEED_CLAIM'
        aggregate.append(record)
    save(directory / 'statistics.json', dict(groups=aggregate, checkpoints=hashes,
         note='Accuracy standard deviations use independent seeds, not validation frames.'))


if __name__ == '__main__':
    main()
