#!/usr/bin/env python3
"""Matched-budget training queue; waits for the short evaluation queue first.

Training is serial two-GPU DDP (global batch 2). Each final checkpoint is
evaluated and profiled on GPU1 before the next training starts. A failure stops
the queue for inspection. Reinvocation resumes from the last epoch checkpoint.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_rc_bev_followup import ROOT, now, save


def experiments():
    base = 'projects/configs/bevformer/bevformer_base.py'
    discrete150 = 'projects/configs/bevformer_ablation/bev150.py'
    continuous = 'projects/configs/bevformer_rc/'
    specs = [
        ('base200_s0', base, 0, [200], False),
        ('rc_fixed200_s0', continuous + 'rc_bev_fixed200.py', 0, [200], True),
        ('rc_multires_s1', continuous + 'rc_bev_multires.py', 1, [100, 125, 150, 175, 200], True),
        ('rc_multires_s2', continuous + 'rc_bev_multires.py', 2, [100, 125, 150, 175, 200], True),
    ]
    specs += [(f'base200_s{s}', base, s, [200], False) for s in (1, 2)]
    specs += [(f'discrete150_s{s}', discrete150, s, [150], False) for s in (0, 1, 2)]
    specs += [(f'rc_fixed150_s{s}', continuous + 'rc_bev_fixed150.py', s, [150], True)
              for s in (0, 1, 2)]
    return [dict(id=i, config=c, seed=s, resolutions=r, continuous=q)
            for i, c, s, r, q in specs]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--after', required=True, help='Short queue status.json')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    specs = experiments()
    save(run_dir / 'plan.json', dict(epochs=24, physical_training_gpus=[0, 1],
         global_batch_size=2, learning_rate=0.0002, evaluation_gpu=1,
         initialization='ckpts/r101_dcn_fcos3d_pretrain.pth', experiments=specs))
    if args.plan_only:
        return
    state_path = run_dir / 'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': {}}
    state.update(status='WAITING_EVALUATION', pid=os.getpid(), updated_at=now())
    save(state_path, state)
    dependency = Path(args.after).resolve()
    while True:
        if dependency.exists():
            upstream = json.loads(dependency.read_text())
            if upstream['status'] in ('COMPLETE', 'FAILED'):
                state['evaluation_dependency_status'] = upstream['status']
                break
        time.sleep(30)
    env = os.environ.copy()
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        env.pop(key, None)
    env.update(CUDA_HOME='/usr/local/cuda', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(ROOT / 'mmdetection3d') + ':' + str(ROOT),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    # Publish only successful metric/profile pairs even if an unrelated FP16 job failed.
    try:
        subprocess.run([sys.executable, 'tools/collect_rc_bev_followup.py',
                        str(dependency.parent)], env=env, cwd=ROOT, check=True)
    except Exception as exc:
        state.update(status='FAILED', error='Evidence collection: ' + str(exc), updated_at=now())
        save(state_path, state)
        raise

    def wait_for_gpus():
        while True:
            busy = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if not busy:
                return
            state.update(status='WAITING_GPU', updated_at=now())
            save(state_path, state)
            time.sleep(30)

    def execute(key, command, gpu, artifact):
        previous = state['jobs'].get(key, {})
        if previous.get('status') == 'COMPLETE' and artifact.exists():
            return
        wait_for_gpus()
        if shutil.disk_usage(run_dir).free < 30 * 1024**3:
            raise RuntimeError('Less than 30 GiB disk space; refusing to start another job')
        attempt = int(previous.get('attempt', 0)) + 1
        log_path = run_dir / f'{key}_attempt_{attempt}.log'
        record = dict(status='RUNNING', started_at=now(), attempt=attempt,
                      command=command, gpu=gpu, log=str(log_path))
        state['jobs'][key] = record
        state.update(status='RUNNING', current_job=key, updated_at=now())
        save(state_path, state)
        print(now(), 'START', key, flush=True)
        with log_path.open('w') as log:
            child = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                     stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            save(state_path, state)
            code = child.wait()
        record.update(returncode=code, finished_at=now())
        if code != 0 or not artifact.is_file():
            record['status'] = 'FAILED'
            save(state_path, state)
            raise RuntimeError(f'{key} failed; see {log_path}')
        record.update(status='COMPLETE', artifact=str(artifact))
        save(state_path, state)
        print(now(), 'COMPLETE', key, flush=True)

    try:
        for spec in specs:
            directory = run_dir / spec['id']
            work_dir = directory / 'work_dir'
            work_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = work_dir / 'epoch_24.pth'
            train = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                     '--nproc_per_node=2', 'tools/train.py', spec['config'],
                     '--launcher', 'pytorch', '--no-validate', '--deterministic',
                     '--seed', str(spec['seed']), '--work-dir', str(work_dir)]
            latest = work_dir / 'latest.pth'
            if latest.is_file():
                train += ['--resume-from', str(latest)]
            options = ['dist_params.backend=nccl', 'data.samples_per_gpu=1',
                       'data.workers_per_gpu=4', 'optimizer.lr=0.0002', 'runner.max_epochs=24']
            if spec['continuous']:
                options += [f'model.pts_bbox_head.resolution_seed={spec["seed"]}']
            train += ['--cfg-options'] + options
            execute(spec['id'] + '_train', train, '0,1', checkpoint)
            for resolution in spec['resolutions']:
                output = directory / f'eval_{resolution}'
                output.mkdir(parents=True, exist_ok=True)
                options = ['data.workers_per_gpu=2']
                if spec['continuous']:
                    options += [f'model.pts_bbox_head.test_bev_shape=({resolution},{resolution})']
                command = [sys.executable, 'tools/test.py', spec['config'], str(checkpoint),
                           '--eval', 'bbox', '--seed', str(spec['seed']), '--eval-options',
                           'jsonfile_prefix=' + str(output / 'predictions'), '--cfg-options'] + options
                execute(f'{spec["id"]}_eval_{resolution}', command, '1',
                        output / 'predictions/pts_bbox/metrics_summary.json')
                command = [sys.executable, 'tools/profile_bevformer.py', spec['config'],
                           str(checkpoint), '--warmup', '20', '--iters', '200',
                           '--output', str(output / 'profile.json'), '--cfg-options'] + options
                execute(f'{spec["id"]}_profile_{resolution}', command, '1', output / 'profile.json')
            subprocess.run([sys.executable, 'tools/collect_rc_bev_training.py', str(run_dir)],
                           cwd=ROOT, env=env, check=True)
    except Exception as exc:
        state.update(status='FAILED', error=str(exc), updated_at=now())
        save(state_path, state)
        raise
    state.update(status='COMPLETE', current_job=None, finished_at=now())
    save(state_path, state)


if __name__ == '__main__':
    main()
