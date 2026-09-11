#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv5090py39/bin/python"
CONFIG="$ROOT/projects/configs/bevformer_ablation/bev150_fulltrain_24ep.py"
WORK_DIR="$ROOT/work_dirs/bev150_fulltrain_24ep"
EXPERIMENT_DIR="$ROOT/experiments/bev150_fulltrain_24ep"
LOG_DIR="$ROOT/logs"
TRAIN_EXIT="$EXPERIMENT_DIR/training_exit_code.txt"
STATUS_LOG="$LOG_DIR/bev150_fulltrain_postprocess.log"

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/mmdetection3d:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
status() { echo "[$(timestamp)] $*" | tee -a "$STATUS_LOG"; }

status "WAIT_TRAINING_START exit_file=$TRAIN_EXIT"
while [[ ! -f "$TRAIN_EXIT" ]]; do
    sleep 300
done

training_rc="$(tr -d '[:space:]' < "$TRAIN_EXIT")"
if [[ "$training_rc" != "0" ]]; then
    status "ABORT training_rc=$training_rc"
    exit 1
fi
if [[ ! -f "$WORK_DIR/epoch_24.pth" ]]; then
    status "ABORT reason=missing_epoch_24"
    exit 1
fi
status "TRAINING_CONFIRMED_COMPLETE"

for epoch in 6 12 18 24; do
    checkpoint="$WORK_DIR/epoch_${epoch}.pth"
    record="$EXPERIMENT_DIR/epoch_${epoch}_eval.json"
    if [[ ! -f "$checkpoint" ]]; then
        status "EVAL_ABORT epoch=$epoch reason=missing_checkpoint"
        exit 1
    fi
    if "$PYTHON" tools/manage_extended_experiment.py check \
        --record "$record" --checkpoint "$checkpoint" --epoch "$epoch" \
        >/dev/null 2>&1; then
        status "EVAL_SKIP epoch=$epoch record=$record"
        continue
    fi

    started_ns="$(date +%s%N)"
    eval_log="$LOG_DIR/bev150_fulltrain_epoch_${epoch}_eval.log"
    status "EVAL_START epoch=$epoch checkpoint=$checkpoint"
    set +e
    "$PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
        --eval bbox --seed 0 --cfg-options data.workers_per_gpu=2 \
        2>&1 | tee -a "$eval_log"
    eval_rc=${PIPESTATUS[0]}
    set -e
    if (( eval_rc != 0 )); then
        status "EVAL_FAILED epoch=$epoch rc=$eval_rc"
        exit "$eval_rc"
    fi
    "$PYTHON" tools/manage_extended_experiment.py record \
        --results-root "$ROOT/test/bev150_fulltrain_24ep" \
        --started-ns "$started_ns" --checkpoint "$checkpoint" \
        --epoch "$epoch" --output "$record"
    status "EVAL_COMPLETE epoch=$epoch record=$record"
done

selection_summary="$EXPERIMENT_DIR/selection_summary.json"
best_checkpoint="$($PYTHON tools/manage_extended_experiment.py summarize \
    --records-dir "$EXPERIMENT_DIR" --output "$selection_summary" | tail -n 1)"
status "SELECT_BEST checkpoint=$best_checkpoint rule=maximum_NDS_then_mAP"

profile="$EXPERIMENT_DIR/profile.json"
profile_log="$LOG_DIR/bev150_fulltrain_profile.log"
status "PROFILE_START checkpoint=$best_checkpoint"
set +e
"$PYTHON" tools/profile_bevformer.py "$CONFIG" "$ROOT/$best_checkpoint" \
    --warmup 20 --iters 200 --workers 2 --output "$profile" \
    2>&1 | tee -a "$profile_log"
profile_rc=${PIPESTATUS[0]}
set -e
if (( profile_rc != 0 )); then
    status "PROFILE_FAILED rc=$profile_rc"
    exit "$profile_rc"
fi

touch "$EXPERIMENT_DIR/COMPLETE"
status "POSTPROCESS_COMPLETE selection=$selection_summary profile=$profile"
