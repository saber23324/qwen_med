#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 RL — GRPO on Phase-3 SFT checkpoint (CLAUDE.md §9).
#
# Layout (2 GPU constraint, only 6 + 7 available):
#   GPU 6: vLLM rollout server  (also runs MedSAM2 in the scheduler process —
#          PHASE3_MEDSAM2_DEVICE=cuda:0 inside the server's CUDA namespace)
#   GPU 7: GRPO trainer
#
# Run in TWO terminals (or use tmux):
#   Terminal A:  bash Qwen3_VL/train_phase3_rl.sh server
#   Terminal B:  bash Qwen3_VL/train_phase3_rl.sh train
#
# Stage-1 warmup defaults are baked in (§9.7):
#   • PHASE3_SHAPING_SCALE=2.0   (2× shaping)
#   • PHASE3_GATE_SOFT=1         (invalid trajectories get 0.1× not 0×)
#   • β_kl = 0.04                (high KL anchor to SFT)
#   • lr   = 5e-7                (LoRA, conservative)
#   • num_generations = 8        (group size K)
#   • temperature = 0.9          (mid of 0.8–1.0 band in §9.1)
#
# To advance to Stage 2 (main): set PHASE3_SHAPING_SCALE=1.0,
#   PHASE3_GATE_SOFT=0, --beta 0.015 in the train block below.
# Stage 3 (polish):              PHASE3_SHAPING_SCALE=0.0 except keep bbox_iou
#   weight in --reward_weights, --beta 0.01.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODE=${1:-help}

# ── Paths (mirror train_phase3.sh / infer_phase3.py) ──────────────────────────
BASE_MODEL=/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3_18-22/v2-20260422-230040/Qwen3-VL-8B-Instruct
# SFT_ADAPTER=/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3_18-22/v2-20260422-230040/checkpoint-4050
DATA_ROOT=/BDSZ6/private/user/yxd/data/M3D/data_18-22/train
SFT_TRAIN=/BDSZ6/private/user/yxd/data/qwen/agent_phase3_18-22/agent_train.jsonl
SFT_VAL=/BDSZ6/private/user/yxd/data/qwen/agent_phase3_18-22/agent_val.jsonl

GRPO_DATA_DIR=/BDSZ6/private/user/yxd/data/qwen/agent_phase3_18-22
GRPO_TRAIN=$GRPO_DATA_DIR/grpo_train.jsonl
GRPO_VAL=$GRPO_DATA_DIR/grpo_val.jsonl

OUTPUT_DIR=/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3_rl_stage1
RENDER_ROOT=/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3_rl_stage1/renders

MEDSAM2_CKPT=/home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt
MEDSAM2_CFG=/home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml

PLUGIN_REWARD=/home/yxd/medagent/Qwen3_VL/grpo_plugin.py
PLUGIN_ROLLOUT=/home/yxd/medagent/Qwen3_VL/phase3_rl_rollout.py

# ── Shared rollout-side env (read by phase3_rl_rollout + grpo_plugin) ─────────
export PHASE3_MEDSAM2_CKPT=$MEDSAM2_CKPT
export PHASE3_MEDSAM2_CFG=$MEDSAM2_CFG
export PHASE3_MEDSAM2_DEVICE=cuda:0       # cuda:0 inside the rollout server's CVD
export PHASE3_RENDER_ROOT=$RENDER_ROOT
export PHASE3_MAX_TURNS=45
export PHASE3_EFFICIENCY_CAP=45
export PHASE3_SHAPING_SCALE=2.0           # Stage 1 — 2× shaping
export PHASE3_GATE_SOFT=1                 # Stage 1 — soft gate
export QWENVL_BBOX_FORMAT=new

mkdir -p "$OUTPUT_DIR" "$RENDER_ROOT" "$GRPO_DATA_DIR"

case "$MODE" in
# ──────────────────────────────────────────────────────────────────────────────
data)
  # One-time: build GRPO JSONL from the SFT JSONL.
  python3 /home/yxd/medagent/Qwen3_VL/convert_to_grpo_dataset.py \
      --sft_jsonl  "$SFT_TRAIN" \
      --data_root  "$DATA_ROOT" \
      --output     "$GRPO_TRAIN"
  python3 /home/yxd/medagent/Qwen3_VL/convert_to_grpo_dataset.py \
      --sft_jsonl  "$SFT_VAL" \
      --data_root  "$DATA_ROOT" \
      --output     "$GRPO_VAL"
  ;;

# ──────────────────────────────────────────────────────────────────────────────
server)
  # vLLM rollout server with the multi-turn scheduler + MedSAM2 in-process.
  # The rollout scheduler imports phase3_rl_rollout.py via --external_plugins,
  # which registers `phase3_nav` in `swift.rollout.multi_turn.multi_turns`.
  CUDA_VISIBLE_DEVICES=6 \
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
  swift rollout \
      --model "$BASE_MODEL" \
      --host 127.0.0.1 \
      --port 8000 \
      --multi_turn_scheduler phase3_nav \
      --max_turns "$PHASE3_MAX_TURNS" \
      --external_plugins "$PLUGIN_ROLLOUT" "$PLUGIN_REWARD"
  ;;

# ──────────────────────────────────────────────────────────────────────────────
train)
  # GRPO trainer. Uses the rollout server in `server` mode for sampling.
  # All eight diagnostic ORMs have weight=0.0 — they are logged per-step but
  # do not contribute to the gradient. Only `phase3_combined` does.
  CUDA_VISIBLE_DEVICES=7 \
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
  IMAGE_MAX_TOKEN_NUM=1024 \
  NPROC_PER_NODE=1 \
  MASTER_PORT=25504 \
  swift rlhf \
      --rlhf_type grpo \
      --model "$BASE_MODEL" \
      --tuner_type lora \
      --lora_rank 128 \
      --lora_alpha 256 \
      --target_modules all-linear \
      --freeze_vit true \
      --freeze_aligner true \
      --torch_dtype bfloat16 \
      --attn_impl flash_attn \
      --agent_template hermes \
      --dataset      "$GRPO_TRAIN" \
      --val_dataset  "$GRPO_VAL" \
      --external_plugins "$PLUGIN_REWARD" "$PLUGIN_ROLLOUT" \
      --reward_funcs phase3_combined phase3_dice phase3_key_dice \
                     phase3_bbox_iou phase3_dice_gain phase3_coverage \
                     phase3_nav_style phase3_efficiency phase3_format_gate \
      --reward_weights 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
      --use_vllm true \
      --vllm_mode server \
      --vllm_server_host 127.0.0.1 \
      --vllm_server_port 8000 \
      --multi_turn_scheduler phase3_nav \
      --max_turns "$PHASE3_MAX_TURNS" \
      --vllm_server_pass_dataset true \
      --num_generations 8 \
      --per_device_train_batch_size 1\
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 8 \
      --learning_rate 5e-7 \
      --beta 0.04 \
      --temperature 0.9 \
      --max_length 16384 \
      --max_completion_length 2048 \
      --num_train_epochs 4 \
      --warmup_ratio 0.05 \
      --max_grad_norm 0.5 \
      --eval_steps 100 \
      --save_steps 100 \
      --save_total_limit 5 \
      --logging_steps 1 \
      --num_iterations 1 \
      --async_generate false \
      --log_completions true \
      --output_dir "$OUTPUT_DIR" \
      --gradient_checkpointing true \
      --vit_gradient_checkpointing false \
      --dataset_num_proc 4 \
      --dataloader_num_workers 4 \
      --report_to tensorboard
  ;;

# ──────────────────────────────────────────────────────────────────────────────
help|*)
  cat <<'EOF'
Usage: bash Qwen3_VL/train_phase3_rl.sh {data|server|train}

  data    One-time: build grpo_{train,val}.jsonl from the SFT JSONL.
  server  Launch the vLLM rollout server with the phase3_nav scheduler
          (uses CUDA_VISIBLE_DEVICES=6, MedSAM2 on cuda:0 in that namespace).
  train   Launch the GRPO trainer (uses CUDA_VISIBLE_DEVICES=7).

Run `data` once. Then `server` and `train` in two separate terminals.
EOF
  ;;
esac
