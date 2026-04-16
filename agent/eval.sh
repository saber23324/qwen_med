MAX_PIXELS=28501 \
CUDA_VISIBLE_DEVICES=7 \
swift infer \
    --adapters /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase6-13/v0-20260415-230620/checkpoint-380\
    --attn_impl flash_attn \
    --stream true \
    --load_data_args true \
    --temperature 0 \
    --max_new_tokens 2048