#!/usr/bin/env python3
"""Serial, restartable evaluation queue for the trained RC-BEV checkpoint.

Every child owns a log and explicit output directory. Failed jobs are recorded
and do not hide later results. Re-running retries only unfinished jobs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'projects/configs/bevformer_rc/'
CHECKPOINT = ('experiments/rc_bev/runs/'
              'rc_multires_s0_dual_retry_20260911T2225CST/work_dir/epoch_24.pth')
EXPECTED_HASH = '741e6b937fadc5083ca833ee2b1ed4a3339bcf4714a967124704f0baa6a71e73'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def plan(run_dir):
    jobs = []

    def add(kind, name, config, options):
        output = run_dir / name
        if kind == 'eval':
            command = [sys.executable, 'tools/test.py', CONFIG + config, CHECKPOINT,
                       '--eval', 'bbox', '--seed', '0', '--eval-options',
                       'jsonfile_prefix=' + str(output / 'predictions')]
            artifact = output / 'predictions/pts_bbox/metrics_summary.json'
        else:
            artifact = output / 'profile.json'
            command = [sys.executable, 'tools/profile_bevformer.py', CONFIG + config,
                       CHECKPOINT, '--warmup', '20', '--iters', '200',
                       '--workers', '2', '--output', str(artifact)]
        command += ['--cfg-options', 'data.workers_per_gpu=2'] + options
        jobs.append(dict(id=name, kind=kind, command=command,
                         artifact=str(artifact), directory=str(output)))

    for resolution in (140, 160, 180):
        options = [f'model.pts_bbox_head.test_bev_shape=({resolution},{resolution})']
        add('eval', f'unseen_{resolution}', 'rc_bev_eval_unseen.py', options)
        add('profile', f'profile_unseen_{resolution}', 'rc_bev_eval_unseen.py', options)
    for migration in ('physical', 'reset_on_change'):
        name = 'dynamic_' + migration
        options = ['model.pts_bbox_head.prev_bev_migration=' + migration,
                   'model.controller_trace_path=' + str(run_dir / name / 'trace.jsonl')]
        add('eval', name, 'rc_bev_dynamic.py', options)
        # Profiling uses its own trace, never appends to the full-validation trace.
        add('profile', 'profile_' + name, 'rc_bev_dynamic.py', [options[0]])
    for resolution in (100, 125, 150, 175, 200):
        options = [f'model.pts_bbox_head.test_bev_shape=({resolution},{resolution})']
        add('profile', f'profile_fp16_{resolution}', 'rc_bev_multires_fp16.py', options)
        add('eval', f'eval_fp16_{resolution}', 'rc_bev_multires_fp16.py', options)
    return jobs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = plan(run_dir)
    save(run_dir / 'plan.json', jobs)
    if args.plan_only:
        print(json.dumps(jobs, indent=2))
        return
    digest = hashlib.sha256()
    with (ROOT / CHECKPOINT).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_HASH:
        raise RuntimeError('Checkpoint hash mismatch')
    env = os.environ.copy()
    for key in ('http_proxy', 'https_proxy', 'all_proxy',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_HOME='/usr/local/cuda',
               PYTHONPATH=str(ROOT / 'mmdetection3d') + ':' + str(ROOT),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    state_path = run_dir / 'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'created_at': now(), 'checkpoint_sha256': digest.hexdigest(),
        'physical_gpu': args.gpu, 'jobs': {}}
    state.update(status='RUNNING', pid=os.getpid(), updated_at=now())
    save(state_path, state)
    for job in jobs:
        old = state['jobs'].get(job['id'], {})
        if old.get('status') == 'COMPLETE' and Path(job['artifact']).is_file():
            continue
        directory = Path(job['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        attempt = int(old.get('attempt', 0)) + 1
        log_path = directory / f'attempt_{attempt}.log'
        trace = directory / 'trace.jsonl'
        if trace.exists():
            trace.rename(directory / f'trace_before_attempt_{attempt}.jsonl')
        record = dict(status='RUNNING', started_at=now(), attempt=attempt,
                      log=str(log_path), command=job['command'])
        state['jobs'][job['id']] = record
        state.update(current_job=job['id'], updated_at=now())
        save(state_path, state)
        print(now(), 'START', job['id'], flush=True)
        with log_path.open('w') as log:
            child = subprocess.Popen(job['command'], env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            save(state_path, state)
            code = child.wait()
        record.update(returncode=code, finished_at=now())
        try:
            if code != 0:
                raise RuntimeError(f'process exit {code}')
            evidence = json.loads(Path(job['artifact']).read_text())
            if job['kind'] == 'eval':
                record.update(nds=evidence['nd_score'], map=evidence['mean_ap'])
            else:
                assert len(evidence['latency_ms']['per_iteration']) == 200
                record.update(latency=evidence['latency_ms']['avg'],
                              fps=evidence['throughput']['fps'])
            record.update(status='COMPLETE', artifact=job['artifact'])
        except Exception as exc:
            record.update(status='FAILED', error=str(exc))
        state['updated_at'] = now()
        save(state_path, state)
        print(now(), record['status'], job['id'], flush=True)
    failed = [key for key, value in state['jobs'].items() if value['status'] != 'COMPLETE']
    state.update(status='FAILED' if failed else 'COMPLETE', current_job=None,
                 finished_at=now(), failed_jobs=failed)
    save(state_path, state)
    sys.exit(bool(failed))


if __name__ == '__main__':
    main()
