#!/usr/bin/env python3
"""Archive completed evaluation/profile pairs and refresh verified result tables."""
import argparse
import json
import shutil
from pathlib import Path
import subprocess
import sys

from run_rc_bev_followup import ROOT, save


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    state = json.loads((run_dir / 'status.json').read_text())
    truth = ROOT / 'experiments/rc_bev/results.json'
    payload = json.loads(truth.read_text())
    records = {row['id']: row for row in payload['records']}
    archive = ROOT / 'experiments/rc_bev/evidence' / run_dir.name
    pairs = [(f'rc_eval_{r}', f'unseen_{r}', f'profile_unseen_{r}', r,
              'continuous_fixed_eval_unseen', 'FP32') for r in (140, 160, 180)]
    pairs += [('rc_dynamic', 'dynamic_physical', 'profile_dynamic_physical', None,
               'budget_adaptive', 'FP32'),
              ('rc_dynamic_reset', 'dynamic_reset_on_change', 'profile_dynamic_reset_on_change',
               None, 'ablation_history_reset_closed_loop', 'FP32')]
    pairs += [(f'rc_eval_fp16_{r}', f'eval_fp16_{r}', f'profile_fp16_{r}', r,
               'continuous_fixed_eval_fp16', 'FP16') for r in (100, 125, 150, 175, 200)]
    for identifier, evaluation, profile, resolution, family, precision in pairs:
        if any(state['jobs'].get(key, {}).get('status') != 'COMPLETE'
               for key in (evaluation, profile)):
            continue
        metrics_path = Path(state['jobs'][evaluation]['artifact'])
        profile_path = Path(state['jobs'][profile]['artifact'])
        # This validation prevents publishing a partial or malformed latency run.
        performance = json.loads(profile_path.read_text())
        assert performance['settings']['measured_iterations'] == 200
        assert len(performance['latency_ms']['per_iteration']) == 200
        output = archive / identifier
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_path, output / 'metrics.json')
        shutil.copy2(profile_path, output / 'profile.json')
        row = dict(id=identifier, family=family, seed=0, status='COMPLETE',
                   precision=precision, checkpoint_sha256=state['checkpoint_sha256'],
                   metrics_path=str((output / 'metrics.json').relative_to(ROOT)),
                   profile_path=str((output / 'profile.json').relative_to(ROOT)))
        if resolution is not None:
            row['resolution'] = resolution
        else:
            trace = run_dir / evaluation / 'trace.jsonl'
            trace_rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
            if len(trace_rows) != 6019:
                raise RuntimeError(f'{trace}: expected 6019 full-validation records')
            shutil.copy2(trace, output / 'trace.jsonl')
            row['controller_trace_path'] = str((output / 'trace.jsonl').relative_to(ROOT))
            row['resolutions'] = [100, 150, 200]
            row['note'] = ('Closed-loop policies may choose different resolution sequences. '
                           'This is not a fixed-switch-schedule causal migration ablation.')
        records[identifier] = row
    payload['records'] = list(records.values())
    save(truth, payload)
    subprocess.run([sys.executable, 'tools/render_rc_bev_results.py'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
