#!/bin/bash
# RoboTwin eval entry point for the DreamZero multi-agent (bimanual)
# WebSocket client. The dreamzero VLA itself does NOT load here — start
# it separately with dreamzero/scripts/inference/serve_bimanual.sh.
#
# Usage:
#   DREAMZERO_HOST=ws-server-host DREAMZERO_PORT=5001 \
#     bash eval.sh beat_block_hammer demo_clean checkpoint-10 0 0

policy_name=DreamZero
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}

DREAMZERO_HOST=${DREAMZERO_HOST:-127.0.0.1}
DREAMZERO_PORT=${DREAMZERO_PORT:-5001}
TEST_NUM=${TEST_NUM:-100}
MAX_FIRST_STEP_DELTA=${MAX_FIRST_STEP_DELTA:-}
ACTION_TRACE_DIR=${ACTION_TRACE_DIR:-}
ACTION_TRACE_INTERVAL=${ACTION_TRACE_INTERVAL:-1}

extra_args=()
if [[ -n "${MAX_FIRST_STEP_DELTA}" ]]; then
    extra_args+=(--max_first_step_delta "${MAX_FIRST_STEP_DELTA}")
fi
if [[ -n "${ACTION_TRACE_DIR}" ]]; then
    extra_args+=(--action_trace_dir "${ACTION_TRACE_DIR}")
    extra_args+=(--action_trace_interval "${ACTION_TRACE_INTERVAL}")
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mserver:          ws://${DREAMZERO_HOST}:${DREAMZERO_PORT}\033[0m"

cd ../.. # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 \
python -u script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --server_host ${DREAMZERO_HOST} \
    --server_port ${DREAMZERO_PORT} \
    --test_num ${TEST_NUM} \
    "${extra_args[@]}" \
    --need_plan False \
    --expert_check False \
    --instruction_type unseen
