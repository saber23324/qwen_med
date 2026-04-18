# 限制 PyTorch 底层计算使用的 CPU 线程数（假设你希望限制在较少的核心上，可以根据服务器总核心数调整该数字）
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=3,7 \
QWENVL_BBOX_FORMAT='new' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
swift sft \
    --model /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --tuner_type lora \
    --dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_train.jsonl' \
    --val_dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_val.jsonl' \
    --load_from_cache_file true \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --save_only_model true \
    --packing true \
    --output_dir '/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase2' \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4