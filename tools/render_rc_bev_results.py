#!/usr/bin/env python3
"""Validate raw RC-BEV evidence and render CSV/LaTeX from one truth source."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/rc_bev/results.json'
OUTPUT_DIR = RESULTS.parent / 'generated'


def load(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def evidence(record, key):
    value = record.get(key)
    if not value:
        raise ValueError(f'{record["id"]}: COMPLETE requires {key}')
    path = ROOT / value
    if not path.is_file():
        raise FileNotFoundError(f'{record["id"]}: missing {path}')
    return path, load(path)


def normalized_records(payload):
    output = []
    for original in payload['records']:
        record = dict(original)
        status = record.get('status')
        if status not in ('NOT RUN', 'BLOCKED', 'COMPLETE'):
            raise ValueError(f'{record.get("id")}: invalid status {status!r}')
        forbidden = {'nds', 'map', 'latency_mean_ms', 'latency_p95_ms', 'fps'}
        if status != 'COMPLETE' and forbidden.intersection(record):
            raise ValueError(f'{record["id"]}: unrun record contains result values')
        if status == 'COMPLETE':
            metrics_path, metrics = evidence(record, 'metrics_path')
            profile_path, profile = evidence(record, 'profile_path')
            samples = profile.get('latency_ms', {}).get('per_iteration')
            settings = profile.get('settings', {})
            if not isinstance(samples, list) or len(samples) != settings.get('measured_iterations'):
                raise ValueError(f'{record["id"]}: incomplete latency samples')
            if settings.get('warmup_iterations', 0) < 20 or settings.get('measured_iterations', 0) < 200:
                raise ValueError(f'{record["id"]}: profile is below the 20+200 protocol')
            if profile.get('latency_ms', {}).get('p99') is None:
                raise ValueError(f'{record["id"]}: profile has no P99 latency')
            end_to_end = profile.get('end_to_end_latency_ms', {})
            if not isinstance(end_to_end.get('per_iteration'), list) \
                    or len(end_to_end['per_iteration']) != settings['measured_iterations']:
                raise ValueError(f'{record["id"]}: incomplete end-to-end samples')
            record.update({
                'nds': metrics['nd_score'], 'map': metrics['mean_ap'],
                'latency_mean_ms': profile['latency_ms']['avg'],
                'latency_p50_ms': profile['latency_ms']['p50'],
                'latency_p95_ms': profile['latency_ms']['p95'],
                'latency_p99_ms': profile['latency_ms'].get('p99'),
                'fps': profile['throughput']['fps'],
                'metrics_sha256': digest(metrics_path),
                'profile_sha256': digest(profile_path),
            })
        output.append(record)
    return output


def main():
    payload = load(RESULTS)
    records = normalized_records(payload)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ('id', 'family', 'seed', 'resolution', 'status', 'nds', 'map',
              'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms',
              'latency_p99_ms', 'fps', 'metrics_path', 'profile_path')
    with (RESULTS.parent / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    complete = [record for record in records if record['status'] == 'COMPLETE']
    latex = [
        '% Auto-generated. Do not enter measurements by hand.',
        '\\begin{tabular}{lrrrr}',
        '\\toprule',
        'Method & Resolution & NDS & mAP & Latency (ms) \\\\',
        '\\midrule',
    ]
    for record in complete:
        latex.append('{} & {} & {:.4f} & {:.4f} & {:.2f} \\\\'.format(
            record['id'].replace('_', '\\_'), record.get('resolution', '--'),
            record['nds'], record['map'], record['latency_mean_ms']))
    if not complete:
        latex.append('\\multicolumn{5}{c}{No verified method results yet} \\\\')
    latex += ['\\bottomrule', '\\end{tabular}', '']
    (OUTPUT_DIR / 'main_results.tex').write_text('\n'.join(latex), encoding='utf-8')
    ablations = [record for record in complete if record['family'].startswith('ablation_')]
    ablation_latex = [
        '% Auto-generated from verified raw evidence.',
        '\\begin{tabular}{lrrr}', '\\toprule',
        'Ablation & NDS & mAP & Latency (ms) \\\\', '\\midrule']
    for record in ablations:
        ablation_latex.append('{} & {:.4f} & {:.4f} & {:.2f} \\\\'.format(
            record['id'].replace('_', '\\_'), record['nds'], record['map'],
            record['latency_mean_ms']))
    if not ablations:
        ablation_latex.append('\\multicolumn{4}{c}{No verified ablations yet} \\\\')
    ablation_latex += ['\\bottomrule', '\\end{tabular}', '']
    (OUTPUT_DIR / 'ablations.tex').write_text(
        '\n'.join(ablation_latex), encoding='utf-8')

    def empty_or_label(ax, message):
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    if complete:
        ax.scatter([item['latency_mean_ms'] for item in complete],
                   [item['nds'] for item in complete])
        for item in complete:
            ax.annotate(item['id'], (item['latency_mean_ms'], item['nds']), fontsize=7)
        ax.set(xlabel='Mean latency (ms)', ylabel='NDS', title='Accuracy-latency observations')
    else:
        empty_or_label(ax, 'NOT RUN: no verified accuracy/latency pairs')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'accuracy_latency_pareto.png', dpi=180)
    plt.close(fig)

    resolution_rows = [item for item in complete if item.get('resolution') is not None]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    if resolution_rows:
        resolution_rows.sort(key=lambda item: item['resolution'])
        ax.plot([item['resolution'] for item in resolution_rows],
                [item['nds'] for item in resolution_rows], marker='o')
        ax.set(xlabel='BEV resolution', ylabel='NDS', title='Resolution-NDS observations')
    else:
        empty_or_label(ax, 'NOT RUN: no verified resolution sweep')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'resolution_nds_curve.png', dpi=180)
    plt.close(fig)

    dynamic = next((item for item in complete if item['family'] == 'budget_adaptive'), None)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    if dynamic and dynamic.get('controller_trace_path'):
        trace_path = ROOT / dynamic['controller_trace_path']
        trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        ax.step(range(len(trace)), [item['resolution'] for item in trace], where='post')
        ax.set(xlabel='Frame', ylabel='BEV resolution', title='Adaptive resolution timeline')
    else:
        empty_or_label(ax, 'NOT RUN: no verified controller trace')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dynamic_resolution_timeline.png', dpi=180)
    plt.close(fig)

    artifact_paths = [RESULTS, RESULTS.parent / 'results.csv',
                      OUTPUT_DIR / 'main_results.tex', OUTPUT_DIR / 'ablations.tex',
                      OUTPUT_DIR / 'accuracy_latency_pareto.png',
                      OUTPUT_DIR / 'resolution_nds_curve.png',
                      OUTPUT_DIR / 'dynamic_resolution_timeline.png']
    manifest = {
        'schema_version': 1,
        'complete_result_count': len(complete),
        'not_run_count': sum(record['status'] == 'NOT RUN' for record in records),
        'artifacts': [
            {'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
             'bytes': path.stat().st_size} for path in artifact_paths],
    }
    with (RESULTS.parent / 'artifact_manifest.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps({
        'records': len(records), 'complete': len(complete),
        'not_run': manifest['not_run_count']}, indent=2))


if __name__ == '__main__':
    main()
