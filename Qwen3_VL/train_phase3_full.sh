#!/bin/bash
# Phase 3 agent training: navigation (scroll + get_slice) + bbox + MedSAM2 + point refine.
# Tools: get_slice, scroll, add_bbox, run_medsam2, add_point, finish_3d_segmentation
# agent_template=hermes for Qwen3-VL parallel tool calling.

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
QWENVL_BBOX_FORMAT='new' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
MASTER_PORT=25503 \
swift sft \
    --model /BDSZ6/private/user/yxd/models/Qwen3-VL-8B-Instruct \
    --dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase3/agent_train.jsonl' \
    --val_dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase3/agent_val.jsonl' \
    --agent_template hermes \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner false \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 16384 \
    --output_dir '/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3' \
    --warmup_ratio 0.1 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4