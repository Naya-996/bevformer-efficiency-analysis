import json
from pathlib import Path

from mmcv.runner import get_dist_info
from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class ResolutionTraceHook(Hook):
    """Append the active training grid and migration cost as JSON Lines."""

    def __init__(self, filename='resolution_trace.jsonl', interval=1):
        self.filename = filename
        self.interval = int(interval)

    def after_train_iter(self, runner):
        rank, _ = get_dist_info()
        if rank != 0 or not self.every_n_iters(runner, self.interval):
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.pts_bbox_head
        transformer = head.transformer
        record = {
            'epoch': int(runner.epoch),
            'iteration': int(runner.iter),
            'inner_iteration': int(runner.inner_iter),
            'bev_shape': list(head.last_bev_shape),
            'prev_bev_resize_ms': float(
                getattr(transformer, 'last_prev_bev_resize_ms', 0.0)),
        }
        output = Path(runner.work_dir) / self.filename
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
