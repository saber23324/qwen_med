PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift infer \
    --adapters /BDSZ6/private/user/yxd/dtos_output/qwen/data_18-22/v0-20260329-162115/checkpoint-46000 \
    --stream true \
    --max_new_tokens 2048 \
    --load_data_args true \
    --result_path  /BDSZ6/private/user/yxd/data/qwen/eval_18-22/result.jsonl

CUDA_VISIBLE_DEVICES=5 \
swift eval \
    --model /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --adapters  /BDSZ6/private/user/yxd/dtos_output/qwen/v0-20260329-025317/checkpoint-17000 \
    --eval_backend VLMEvalKit \
    --infer_backend vllm  \
    --eval_dataset /BDSZ6/private/user/yxd/data/qwen/data_6/mri_grounding_val.jsonl 
    # --eval_dataset_args '{"general_qa": {"local_path": "/BDSZ6/private/user/yxd/data/qwen/data_6/mri_grounding_val.jsonl", "subset_list": ["content"]}}'

# ValueError: Unrecognized model in /BDSZ6/private/user/yxd/dtos_output/qwen/v0-20260329-025317/checkpoint-17000. Should have a `model_type` key in its config.json.
    --dataset '/BDSZ6/private/user/yxd/data/qwen/data_6/mri_grounding_train.jsonl' \
