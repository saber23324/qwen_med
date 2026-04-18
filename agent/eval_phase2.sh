# 不可用 用eval.sh 之后 visualize
CUDA_VISIBLE_DEVICES=0,4 \
python Qwen3_VL/infer_phase2.py \
    --model  /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --ckpt   /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase2/v3-20260417-100443/checkpoint-300 \
    --val_jsonl /BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_val.jsonl \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --output_dir /tmp/phase2_eval \
    --device cuda:0

conda run -n qwen3 python Qwen3_VL/infer_phase2.py \
    --model  /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --ckpt   /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase2/v3-20260417-100443/checkpoint-300 \
    --val_jsonl /BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_val.jsonl \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --output_dir /tmp/phase2_eval \
    --device cuda:1