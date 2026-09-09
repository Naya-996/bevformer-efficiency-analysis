#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$(command -v "$PYTHON_BIN" 2>/dev/null || printf '%s' "$PYTHON_BIN")"
elif [[ -x "$ROOT/.venv5090py39/bin/python" ]]; then
    PYTHON="$ROOT/.venv5090py39/bin/python"
else
    PYTHON="$(command -v python3)"
fi
CUDA_DEVICE="${CUDA_DEVICE:-1}"
CONFIG="$ROOT/projects/configs/bevformer_ablation/bev150_finetune_6ep.py"
PREPARED="$ROOT/ckpts/ablation/bev150_epoch2_resume_lr5e-6.pth"
WORK_DIR="$ROOT/work_dirs/bev150_finetune_6ep"
EXPERIMENT_DIR="$ROOT/experiments/bev150_extended"
LOG_DIR="$ROOT/logs"
TRAIN_LOG="$LOG_DIR/bev150_finetune_6ep.log"
STATUS_LOG="$LOG_DIR/bev150_extended_pipeline_status.log"
LOCK_FILE="$LOG_DIR/.bevformer_gpu1_extended.lock"

mkdir -p "$WORK_DIR" "$EXPERIMENT_DIR" "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another BEV-150 extended pipeline holds $LOCK_FILE" >&2
    exit 75
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/mmdetection3d:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
status() { echo "[$(timestamp)] $*" | tee -a "$STATUS_LOG"; }

for required in "$PYTHON" "$CONFIG" "$PREPARED"; do
    if [[ ! -e "$required" ]]; then
        status "FAILED missing=$required"
        exit 1
    fi
done

checkpoint_epoch() {
    "$PYTHON" - "$1" <<'PY'
import sys, torch
try:
    value = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
except TypeError:
    value = torch.load(sys.argv[1], map_location='cpu')
epoch = value.get('meta', {}).get('epoch')
if not isinstance(epoch, int):
    raise SystemExit('checkpoint meta.epoch is missing')
print(epoch)
PY
}

resume_checkpoint="$PREPARED"
if [[ -e "$WORK_DIR/latest.pth" ]]; then
    resume_checkpoint="$WORK_DIR/latest.pth"
fi
resume_epoch="$(checkpoint_epoch "$resume_checkpoint")"

if (( resume_epoch < 6 )); then
    status "TRAIN_START cuda_device=$CUDA_DEVICE resume_epoch=$resume_epoch checkpoint=$resume_checkpoint"
    set +e
    "$PYTHON" tools/train.py "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --resume-from "$resume_checkpoint" \
        --gpus 1 --seed 0 --no-validate \
        2>&1 | tee -a "$TRAIN_LOG"
    train_rc=${PIPESTATUS[0]}
    set -e
    if (( train_rc != 0 )); then
        status "TRAIN_FAILED rc=$train_rc"
        exit "$train_rc"
    fi
    status "TRAIN_COMPLETE"
else
    status "TRAIN_SKIP checkpoint_epoch=$resume_epoch"
fi

for epoch in 3 4 5 6; do
    checkpoint="$WORK_DIR/epoch_${epoch}.pth"
    record="$EXPERIMENT_DIR/epoch_${epoch}_eval.json"
    if [[ ! -f "$checkpoint" ]]; then
        status "EVAL_FAILED epoch=$epoch reason=missing_checkpoint"
        exit 1
    fi
    if "$PYTHON" tools/manage_extended_experiment.py check \
        --record "$record" --checkpoint "$checkpoint" --epoch "$epoch" \
        >/dev/null 2>&1; then
        status "EVAL_SKIP epoch=$epoch valid_record=$record"
        continue
    fi

    started_ns="$(date +%s%N)"
    eval_log="$LOG_DIR/bev150_extended_epoch_${epoch}_eval.log"
    status "EVAL_START epoch=$epoch checkpoint=$checkpoint"
    set +e
    "$PYTHON" tools/test.py "$CONFIG" "$checkpoint" \
        --eval bbox --cfg-options data.workers_per_gpu=2 \
        2>&1 | tee -a "$eval_log"
    eval_rc=${PIPESTATUS[0]}
    set -e
    if (( eval_rc != 0 )); then
        status "EVAL_FAILED epoch=$epoch rc=$eval_rc"
        exit "$eval_rc"
    fi
    "$PYTHON" tools/manage_extended_experiment.py record \
        --results-root "$ROOT/test/bev150_finetune_6ep" \
        --started-ns "$started_ns" --checkpoint "$checkpoint" \
        --epoch "$epoch" --output "$record"
    status "EVAL_COMPLETE epoch=$epoch record=$record"
done

summary="$EXPERIMENT_DIR/summary.json"
best_checkpoint="$($PYTHON tools/manage_extended_experiment.py summarize \
    --records-dir "$EXPERIMENT_DIR" --output "$summary" | tail -n 1)"
status "SELECT_BEST checkpoint=$best_checkpoint"

base_profile="$EXPERIMENT_DIR/paired_base_profile_gpu1.json"
previous_profile="$EXPERIMENT_DIR/paired_2ep_profile_gpu1.json"
best_profile="$EXPERIMENT_DIR/paired_best_profile_gpu1.json"

profile_one() {
    local name="$1" config="$2" checkpoint="$3" output="$4"
    local profile_log="$LOG_DIR/bev150_extended_${name}_profile.log"
    local profile_rc
    status "PROFILE_START name=$name checkpoint=$checkpoint"
    set +e
    "$PYTHON" tools/profile_bevformer.py "$config" "$checkpoint" \
        --warmup 20 --iters 200 --workers 2 --output "$output" \
        2>&1 | tee -a "$profile_log"
    profile_rc=${PIPESTATUS[0]}
    set -e
    if (( profile_rc != 0 )); then
        status "PROFILE_FAILED name=$name rc=$profile_rc"
        return "$profile_rc"
    fi
    status "PROFILE_COMPLETE name=$name output=$output"
}

profile_one base \
    "$ROOT/projects/configs/bevformer/bevformer_base.py" \
    "$ROOT/ckpts/bevformer_r101_dcn_24ep.pth" "$base_profile"
profile_one two_epoch \
    "$ROOT/projects/configs/bevformer_ablation/bev150_finetune.py" \
    "$ROOT/work_dirs/bev150_finetune_2ep/epoch_2.pth" "$previous_profile"
profile_one selected "$CONFIG" "$best_checkpoint" "$best_profile"

"$PYTHON" tools/render_extended_report.py \
    --summary "$summary" --profile "$best_profile" \
    --baseline-profile "$base_profile" --previous-profile "$previous_profile" \
    --output "$EXPERIMENT_DIR/result_report.md"
touch "$EXPERIMENT_DIR/COMPLETE"
status "PIPELINE_COMPLETE summary=$summary profile=$best_profile"
