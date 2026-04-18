#!/bin/bash
# Phase 2 agent training: bbox annotation + MedSAM2 3D segmentation + point refinement
# Tools: add_bbox, run_medsam2, add_point, finish_3d_segmentation
# agent_template=hermes for Qwen3-VL parallel tool calling
# GPUs: 4,5,6,7  (4× GPU)

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
QWENVL_BBOX_FORMAT='new' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=3,7 \
MASTER_PORT=25502 \
swift sft \
    --model /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_train.jsonl' \
    --val_dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_val.jsonl' \
    --agent_template hermes \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 12288 \
    --output_dir '/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase2' \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
