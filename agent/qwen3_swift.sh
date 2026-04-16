CUDA_VISIBLE_DEVICES=3,4 \
MASTER_PORT=20501 \
swift  sft \
    --model /BDSZ6/private/user/yxd/models/Qwen3-VL-8B-Instruct \
    --tuner_type lora \
    --dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase6-13/agent_train.jsonl' \
    --val_dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase6-13/agent_val.jsonl' \
    --load_from_cache_file true \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 8192 \
    --save_only_model true \
    --packing true \
    --use_liger_kernel false \
    --output_dir '/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase6-13' \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16