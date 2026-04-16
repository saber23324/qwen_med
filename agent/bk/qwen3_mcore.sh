# full: 2 * 70GiB 0.61s/it
# lora: 2 * 14GiB 0.45s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=5,6 \
MASTER_PORT=22501 \
megatron sft \
    --model /BDSZ6/private/user/yxd/models/MedMO-8B-Next \
    --save_safetensors true \
    --merge_lora false \
    --dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase1/agent_train.jsonl' \
    --val_dataset '/BDSZ6/private/user/yxd/data/qwen/agent_phase1/agent_val.jsonl' \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 50 \
    --output_dir '/BDSZ6/private/user/yxd/dtos_output/qwen/agent_lesion' \
    --save_steps 500 \
    --max_length 8192 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --save_total_limit 2 \
    --model_type 'qwen3_vl'\
    --model_name swift-robot
