MAX_PIXELS=28501 \
CUDA_VISIBLE_DEVICES=5 \
swift infer \
    --adapters /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3/v0-20260419-225116/checkpoint-200\
    --attn_impl flash_attn \
    --stream true \
    --load_data_args true \
    --temperature 0 \
    --max_new_tokens 2048