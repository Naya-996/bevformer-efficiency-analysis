#!/usr/bin/env python3
"""Audit the local evidence chain without rewriting historical manifests."""

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit'
METRIC_KEYS = ('nd_score', 'mean_ap')
TP_KEYS = ('trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err')


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def rel(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def git(*args):
    return subprocess.run(
        ('git',) + args, cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE).stdout.strip()


def close(a, b, tolerance=1e-12):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) \
        and abs(a - b) <= tolerance


def discover_artifacts():
    patterns = (
        'experiments/**/eval_manifest.json', 'experiments/**/*profile*.json',
        'experiments/**/*summary*.json', 'test/**/metrics_summary.json',
        'logs/*.log', 'work_dirs/*.json', 'ckpts/**/*.pth',
        'work_dirs/**/*.pth')
    found = set()
    for pattern in patterns:
        found.update(path for path in ROOT.glob(pattern) if path.is_file())
    for manifest_path in ROOT.glob('experiments/**/eval_manifest.json'):
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError):
            continue
        for section in ('config', 'checkpoint'):
            value = manifest.get(section, {}).get('path')
            if value and (ROOT / value).is_file():
                found.add(ROOT / value)
    return sorted(found)


def audit_eval_manifests():
    records = []
    missing = []
    for path in sorted(ROOT.glob('experiments/**/eval_manifest.json')):
        item = {'path': rel(path), 'parseable': True, 'references': {}}
        try:
            manifest = load_json(path)
        except (OSError, ValueError) as exc:
            item.update(parseable=False, classification='completely_missing', error=str(exc))
            records.append(item)
            continue
        for name in ('config', 'checkpoint'):
            declared = manifest.get(name, {})
            target = ROOT / declared.get('path', '')
            entry = {
                'path': declared.get('path'), 'exists': target.is_file(),
                'declared_sha256': declared.get('sha256'),
            }
            if target.is_file():
                entry['actual_sha256'] = sha256(target)
                entry['hash_matches'] = entry['declared_sha256'] == entry['actual_sha256']
            else:
                entry['actual_sha256'] = None
                entry['hash_matches'] = False
                missing.append({'owner': rel(path), 'kind': name, 'path': declared.get('path')})
            item['references'][name] = entry
        metrics_value = manifest.get('metrics_path')
        metrics_path = ROOT / metrics_value if metrics_value else None
        metrics_exists = bool(metrics_path and metrics_path.is_file())
        item['references']['metrics'] = {'path': metrics_value, 'exists': metrics_exists}
        if not metrics_exists:
            missing.append({'owner': rel(path), 'kind': 'metrics', 'path': metrics_value})
        refs = item['references']
        if not all(value['exists'] for value in refs.values()):
            item['classification'] = 'completely_missing'
        elif not refs['config']['hash_matches'] or not refs['checkpoint']['hash_matches']:
            item['classification'] = 'hash_mismatch'
        else:
            item['classification'] = 'directly_verifiable'
        records.append(item)
    return records, missing


def audit_profiles():
    records = []
    for path in sorted(ROOT.glob('experiments/**/*profile*.json')):
        try:
            data = load_json(path)
        except (OSError, ValueError) as exc:
            records.append({'path': rel(path), 'parseable': False, 'error': str(exc)})
            continue
        settings = data.get('settings', {})
        latency = data.get('latency_ms', {})
        software = data.get('software')
        device = data.get('device')
        checks = {
            'per_iteration_latency': isinstance(latency.get('per_iteration'), list),
            'warmup_iterations': isinstance(settings.get('warmup_iterations'), int),
            'measured_iterations': isinstance(settings.get('measured_iterations'), int),
            'gpu_identity': isinstance(device, dict) and bool(device.get('name')),
            'software_versions': isinstance(software, dict) and bool(software),
            'timing_boundary': bool(settings.get('measurement_scope')),
            'cuda_synchronization_declared': 'cuda_synchronize_before_and_after' in settings,
        }
        measured = settings.get('measured_iterations')
        samples = latency.get('per_iteration')
        records.append({
            'path': rel(path), 'parseable': True, 'checks': checks,
            'recorded_samples': len(samples) if isinstance(samples, list) else None,
            'sample_count_matches': isinstance(samples, list) and measured == len(samples),
            'classification': 'directly_verifiable' if all(checks.values()) else 'summary_only',
        })
    return records


def audit_aggregates():
    checks = []
    summary_path = ROOT / 'experiments/summary.json'
    if summary_path.is_file():
        summary = load_json(summary_path)
        for experiment in summary.get('experiments', []):
            metrics_path = ROOT / experiment.get('metrics_path', '')
            profile_path = ROOT / experiment.get('profile_path', '')
            entry = {'owner': rel(summary_path), 'id': experiment.get('id'), 'checks': {}}
            if metrics_path.is_file():
                metrics = load_json(metrics_path)
                entry['checks']['nd_score'] = close(experiment.get('nds'), metrics.get('nd_score'))
                entry['checks']['mean_ap'] = close(experiment.get('mean_ap'), metrics.get('mean_ap'))
                for key in TP_KEYS:
                    entry['checks'][key] = close(experiment.get(key), metrics.get('tp_errors', {}).get(key))
            else:
                entry['checks']['metrics_source_exists'] = False
            if profile_path.is_file():
                profile = load_json(profile_path)
                entry['checks']['latency_avg_ms'] = close(
                    experiment.get('latency_avg_ms'), profile.get('latency_ms', {}).get('avg'))
                entry['checks']['latency_p50_ms'] = close(
                    experiment.get('latency_p50_ms'), profile.get('latency_ms', {}).get('p50'))
                entry['checks']['latency_p95_ms'] = close(
                    experiment.get('latency_p95_ms'), profile.get('latency_ms', {}).get('p95'))
                entry['checks']['fps'] = close(experiment.get('fps'), profile.get('throughput', {}).get('fps'))
            else:
                entry['checks']['profile_source_exists'] = False
            entry['reproducible'] = all(entry['checks'].values())
            checks.append(entry)
    extended_path = ROOT / 'experiments/bev150_extended/summary.json'
    if extended_path.is_file():
        summary = load_json(extended_path)
        for epoch in summary.get('epochs', []):
            metrics_path = ROOT / epoch.get('metrics_path', '')
            entry = {'owner': rel(extended_path), 'id': 'epoch_%s' % epoch.get('epoch'), 'checks': {}}
            if metrics_path.is_file():
                raw = load_json(metrics_path)
                entry['checks']['nd_score'] = close(epoch['metrics'].get('nd_score'), raw.get('nd_score'))
                entry['checks']['mean_ap'] = close(epoch['metrics'].get('mean_ap'), raw.get('mean_ap'))
                for key in TP_KEYS:
                    entry['checks'][key] = close(
                        epoch['metrics'].get('tp_errors', {}).get(key), raw.get('tp_errors', {}).get(key))
            else:
                entry['checks']['metrics_source_exists'] = False
            entry['reproducible'] = all(entry['checks'].values())
            checks.append(entry)
    return checks


def render_report(payload):
    manifests = payload['eval_manifests']
    counts = payload['classification_counts']
    lines = [
        '# Evidence Integrity Report', '',
        'Generated from the local repository by `tools/audit_evidence_integrity.py`.',
        'The audit is read-only with respect to historical evidence and does not repair declarations.', '',
        '## Scope and status', '',
        f"- Git commit: `{payload['git']['commit']}`",
        f"- Worktree dirty during audit: `{str(payload['git']['dirty']).lower()}`",
        f"- Evaluation manifests: {len(manifests)}",
        f"- Directly verifiable: {counts.get('directly_verifiable', 0)}",
        f"- Hash mismatch: {counts.get('hash_mismatch', 0)}",
        f"- Summary only: {counts.get('summary_only', 0)}",
        f"- Completely missing: {counts.get('completely_missing', 0)}", '',
        '## Evaluation evidence', '',
        '| Manifest | Classification | Config | Checkpoint | Metrics |',
        '|---|---|---:|---:|---:|',
    ]
    for item in manifests:
        refs = item.get('references', {})
        def state(name):
            value = refs.get(name, {})
            if not value.get('exists'):
                return 'missing'
            if name in ('config', 'checkpoint') and not value.get('hash_matches'):
                return 'HASH MISMATCH'
            return 'verified'
        lines.append('| `{}` | {} | {} | {} | {} |'.format(
            item['path'], item['classification'], state('config'), state('checkpoint'), state('metrics')))
    lines.extend(['', '## Profiling evidence', '',
        'A profile is directly verifiable here only when it stores per-iteration samples, warmup and measured counts, GPU identity, software versions, timing boundary, and synchronization policy.', ''])
    for profile in payload['profiles']:
        failed = [key for key, value in profile.get('checks', {}).items() if not value]
        lines.append('- `{}`: **{}**; samples={}; missing fields={}.'.format(
            profile['path'], profile.get('classification', 'unparseable'),
            profile.get('recorded_samples'), ', '.join(failed) if failed else 'none'))
    lines.extend(['', '## Aggregate reproducibility', ''])
    good = sum(item['reproducible'] for item in payload['aggregate_checks'])
    lines.append(f'- {good}/{len(payload["aggregate_checks"])} summary rows reproduce their stored raw metric/profile values exactly within `1e-12`.')
    lines.extend(['', '## Interpretation', '',
        '- `directly_verifiable`: referenced config, checkpoint and raw metrics exist and declared hashes match.',
        '- `hash_mismatch`: artifacts exist, but at least one declared digest differs; the historical declaration is retained.',
        '- `summary_only`: an aggregate/profile exists without all raw evidence required for independent recomputation.',
        '- `completely_missing`: a declared path is absent or a manifest cannot be parsed.',
        '- Existing fixed-grid measurements are historical observations. They are not evidence for the new resolution-continuous method.',
        '- No new method accuracy, latency, memory or controller result has been run; those fields remain `NOT RUN`.', ''])
    return '\n'.join(lines)


def main():
    AUDIT.mkdir(exist_ok=True)
    manifests, missing = audit_eval_manifests()
    profiles = audit_profiles()
    aggregates = audit_aggregates()
    artifacts = []
    for path in discover_artifacts():
        stat = path.stat()
        artifacts.append({
            'path': rel(path), 'kind': path.suffix.lstrip('.') or 'file',
            'bytes': stat.st_size, 'sha256': sha256(path),
        })
    counts = {}
    for item in manifests:
        key = item['classification']
        counts[key] = counts.get(key, 0) + 1
    for profile in profiles:
        if profile.get('classification') == 'summary_only':
            counts['summary_only'] = counts.get('summary_only', 0) + 1
    payload = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'git': {
            'commit': git('rev-parse', 'HEAD'),
            # Generated audit outputs are excluded to avoid a self-referential
            # dirty flag; all source and experiment evidence remains included.
            'dirty': bool(git('status', '--porcelain', '--', '.', ':(exclude)audit')),
            'dirty_check_excludes': ['audit/'],
        },
        'classification_counts': counts,
        'eval_manifests': manifests,
        'profiles': profiles,
        'aggregate_checks': aggregates,
        'missing_artifacts': missing,
        'artifact_count': len(artifacts),
    }
    with (AUDIT / 'evidence_integrity.json').open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with (AUDIT / 'artifact_hashes.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('path', 'kind', 'bytes', 'sha256'))
        writer.writeheader()
        writer.writerows(artifacts)
    (AUDIT / 'evidence_integrity_report.md').write_text(render_report(payload), encoding='utf-8')
    missing_lines = ['# Missing Artifacts', '']
    if missing:
        missing_lines += [
            '- `{path}` ({kind}), referenced by `{owner}`.'.format(**item) for item in missing]
    else:
        missing_lines.append('No paths referenced by evaluation manifests are missing.')
    missing_lines += ['', 'This statement does not imply that unrecorded raw predictions, energy traces, FLOPs, or method results exist.', '']
    (AUDIT / 'missing_artifacts.md').write_text('\n'.join(missing_lines), encoding='utf-8')
    print(json.dumps({
        'manifests': len(manifests), 'profiles': len(profiles),
        'artifacts': len(artifacts), 'classifications': counts,
        'missing': len(missing), 'aggregate_rows': len(aggregates),
        'aggregate_rows_reproduced': sum(item['reproducible'] for item in aggregates),
    }, indent=2))


if __name__ == '__main__':
    main()
