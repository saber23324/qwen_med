# megatron export
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=3,5 \
megatron export \
    --model /BDSZ6/private/user/yxd/models/MedMO-8B-Next \
    --tensor_model_parallel_size 2 \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /BDSZ6/private/user/yxd/models/Qwen3-Med \
    --model_type 'qwen3_vl'\
    --test_convert_precision false

# swift export
# CUDA_VISIBLE_DEVICES=0 \
# swift export \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --to_mcore true \
#     --torch_dtype bfloat16 \
#     --output_dir Qwen2.5-7B-Instruct-mcore \
#     --test_convert_precision true