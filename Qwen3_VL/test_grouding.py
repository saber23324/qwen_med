import os
os.environ["MAX_PIXELS"] = "1003520"
os.environ["QWENVL_BBOX_FORMAT"]='new'
# QWENVL_BBOX_FORMAT='new'
from swift import get_processor, get_template

ckpt = '/BDSZ6/private/user/yxd/dtos_output/qwen/one/v5-20260330-131546/checkpoint-4000'
# processor = get_processor('/BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct')
processor = get_processor(ckpt)
    
template = get_template(processor)
data = {"messages": [{"role": "system", "content": "You are a medical imaging assistant specializing in MRI analysis."}, {"role": "user", "content": "<image>This image is slice 16 out of 100 from an MRI volume. Identify and localize <ref-object> regions in this slice."}, {"role": "assistant", "content": "<bbox>"}], "images": ["/BDSZ6/private/user/yxd/data/M3D/data_6/train/JPEGImages/0006_case_00000_image/00015.jpg"], "objects": {"ref": ["Renal organ involved in filtering metabolic waste from the bloodstream."], "bbox": [[297, 342, 305, 355]]}}

# data = {"messages": [{"role": "system", "content": "You are a medical imaging assistant specializing in MRI analysis."}, {"role": "user", "content": "<image>This image is slice 26 out of 100 from an MRI volume. Identify and localize <ref-object> regions in this slice."}, {"role": "assistant", "content": "<bbox>"}], "images": ["/BDSZ6/private/user/yxd/data/M3D/data_6/train/JPEGImages/0006_case_00191_image/00025.jpg"], "objects": {"ref": ["Organ responsible for removing waste products and excess fluids from the body."], "bbox": [[327, 318, 344, 342]]}}
template.set_mode('train')
encoded = template.encode(data, return_template_inputs=True)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
print(f'images: {encoded["template_inputs"].images}')