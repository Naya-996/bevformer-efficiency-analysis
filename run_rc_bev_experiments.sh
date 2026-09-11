#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv5090py39/bin/python"
ACTION="${1:-help}"
CHECKPOINT="${2:-}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')}"
RUN_DIR="$ROOT/experiments/rc_bev/runs/$RUN_ID"

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/mmdetection3d:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

usage() {
  echo "Usage: $0 {audit|cpu-test|gpu-component-smoke|train-base200|train-fixed150|train-multires|train-multires-fp16|eval-seen|eval-unseen|eval-dynamic|profile-seen|profile-seen-fp16|profile-small|profile-tiny} [checkpoint]"
}

require_checkpoint() {
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint does not exist: $CHECKPOINT" >&2
    exit 2
  fi
}

case "$ACTION" in
  audit)
    cd "$ROOT"
    "$PYTHON" tools/audit_evidence_integrity.py
    ;;
  cpu-test)
    cd "$ROOT"
    "$PYTHON" -m pytest -q tests/test_resolution_continuous.py tests/test_rc_bev_config_contract.py
    ;;
  gpu-component-smoke)
    cd "$ROOT"
    "$PYTHON" tools/smoke_test_rc_bev_components.py \
      --output "$ROOT/experiments/rc_bev/gpu_component_smoke.json"
    ;;
  train-base200|train-fixed150|train-multires|train-multires-fp16)
    mkdir -p "$RUN_DIR"
    config="$ROOT/projects/configs/bevformer_rc/rc_bev_fixed150.py"
    [[ "$ACTION" == "train-base200" ]] && config="$ROOT/projects/configs/bevformer/bevformer_base.py"
    [[ "$ACTION" == "train-multires" ]] && config="$ROOT/projects/configs/bevformer_rc/rc_bev_multires.py"
    [[ "$ACTION" == "train-multires-fp16" ]] && config="$ROOT/projects/configs/bevformer_rc/rc_bev_multires_fp16.py"
    cd "$ROOT"
    "$PYTHON" tools/train.py "$config" --gpus 1 --seed "$SEED" \
      --work-dir "$RUN_DIR/work_dir" 2>&1 | tee "$RUN_DIR/train.log"
    ;;
  eval-seen|eval-unseen|profile-seen|profile-seen-fp16)
    require_checkpoint
    mkdir -p "$RUN_DIR"
    config="$ROOT/projects/configs/bevformer_rc/rc_bev_eval_seen.py"
    resolutions=(100 125 150 175 200)
    if [[ "$ACTION" == "eval-unseen" ]]; then
      config="$ROOT/projects/configs/bevformer_rc/rc_bev_eval_unseen.py"
      resolutions=(140 160 180)
    fi
    [[ "$ACTION" == "profile-seen-fp16" ]] && config="$ROOT/projects/configs/bevformer_rc/rc_bev_multires_fp16.py"
    cd "$ROOT"
    for resolution in "${resolutions[@]}"; do
      if [[ "$ACTION" == "profile-seen" || "$ACTION" == "profile-seen-fp16" ]]; then
        "$PYTHON" tools/profile_bevformer.py "$config" "$CHECKPOINT" \
          --warmup 20 --iters 200 --workers 2 \
          --output "$RUN_DIR/profile_${resolution}.json" \
          --cfg-options model.pts_bbox_head.test_bev_shape="($resolution,$resolution)" \
          2>&1 | tee "$RUN_DIR/profile_${resolution}.log"
      else
        "$PYTHON" tools/test.py "$config" "$CHECKPOINT" --eval bbox --seed "$SEED" \
          --cfg-options model.pts_bbox_head.test_bev_shape="($resolution,$resolution)" \
          data.workers_per_gpu=2 2>&1 | tee "$RUN_DIR/eval_${resolution}.log"
      fi
    done
    ;;
  profile-small|profile-tiny)
    require_checkpoint
    mkdir -p "$RUN_DIR"
    config="$ROOT/projects/configs/bevformer/bevformer_small.py"
    [[ "$ACTION" == "profile-tiny" ]] && config="$ROOT/projects/configs/bevformer/bevformer_tiny.py"
    cd "$ROOT"
    "$PYTHON" tools/profile_bevformer.py "$config" "$CHECKPOINT" \
      --warmup 20 --iters 200 --workers 2 --output "$RUN_DIR/${ACTION}.json" \
      2>&1 | tee "$RUN_DIR/${ACTION}.log"
    ;;
  eval-dynamic)
    require_checkpoint
    mkdir -p "$RUN_DIR"
    cd "$ROOT"
    "$PYTHON" tools/test.py projects/configs/bevformer_rc/rc_bev_dynamic.py \
      "$CHECKPOINT" --eval bbox --seed "$SEED" \
      --cfg-options data.workers_per_gpu=2 \
      model.controller_trace_path="$RUN_DIR/controller_trace.jsonl" \
      2>&1 | tee "$RUN_DIR/eval_dynamic.log"
    ;;
  *)
    usage
    [[ "$ACTION" == "help" ]] || exit 2
    ;;
esac
