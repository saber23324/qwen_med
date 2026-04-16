import os
os.environ["MAX_PIXELS"] = "1003520"
from swift import get_processor, get_template

processor = get_processor('/BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct')
template = get_template(processor)
data = {...}
template.set_mode('train')
encoded = template.encode(data, return_template_inputs=True)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
print(f'images: {encoded["template_inputs"].images}')