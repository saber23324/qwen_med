# 2 * 21GiB
 #   'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \ #   'swift/VideoChatGPT:Generic#2000' \
# export CUDA_VISIBLE_DEVICES=4,5

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
QWENVL_BBOX_FORMAT='new'\
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=4,5 \
MASTER_PORT=25501 \
swift sft \
    --model /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --dataset '/BDSZ6/private/user/yxd/data/qwen/data_4_one/mri_grounding_train.jsonl' \
    --val_dataset '/BDSZ6/private/user/yxd/data/qwen/data_4_one/mri_grounding_val.jsonl' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 100 \
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
    --gradient_accumulation_steps 2 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 4096 \
    --output_dir '/BDSZ6/private/user/yxd/dtos_output/qwen/4one' \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4