"""
Custom bbox-weighted loss for per-frame MRI grounding with Qwen3-VL.

Tokens between <|box_start|> and <|box_end|> receive BBOX_WEIGHT × normal
CE weight.  A smooth-L1 regression component is added on predicted vs. GT
bbox coordinate values.

Registration
------------
Add to Qwen3_VL/loss/mapping.py:
    from Qwen3_VL.custom_loss import MRIBboxLoss   # adjust import path
    loss_map['mri_bbox'] = MRIBboxLoss

Or pass directly via --external_plugins and register at the bottom of this
file (see last line).

Training launch example
-----------------------
QWENVL_BBOX_FORMAT=new \\
swift sft \\
  --model        Qwen/Qwen3-VL-7B-Instruct \\
  --dataset      /path/to/mri_video_train.jsonl \\
  --val_dataset  /path/to/mri_video_val.jsonl \\
  --train_type   lora \\
  --output_dir   output/qwen3vl_mri_bbox \\
  --num_train_epochs 3 \\
  --per_device_train_batch_size 1 \\
  --gradient_accumulation_steps 16 \\
  --learning_rate 1e-4 \\
  --max_pixels   401408 \\
  --loss_type    mri_bbox \\
  --enable_channel_loss true

Channel Loss notes
------------------
The dataset already contains a "channel" field per entry:
  - "grounding" : positive entries (frames with visible lesions + bboxes)
  - "negative"  : no-lesion entries (all-None expressions)

With --enable_channel_loss true, ms-swift computes the loss separately for
each channel and averages across channels, preventing the ~99.7% "grounding"
majority from dominating the ~0.3% "negative" minority.
"""

import re
from typing import Optional

import torch
import torch.nn.functional as F

from swift.loss import BaseLoss
from swift.loss.mapping import loss_map

# ---------------------------------------------------------------------------
# Configurable weights
# ---------------------------------------------------------------------------

BBOX_WEIGHT = 3.0    # CE weight multiplier for tokens inside bbox spans
L1_WEIGHT   = 0.05   # weight for the smooth-L1 regression component
L1_BETA     = 0.1    # smooth-L1 beta (transition point)

_COORD_RE = re.compile(r'\b(\d{1,4})\b')


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_bbox_mask(shift_labels: torch.Tensor,
                      start_id: int, end_id: int) -> torch.Tensor:
    """
    Return bool tensor (B, L-1): True for tokens strictly *inside* a bbox
    span (between <|box_start|> and <|box_end|>). Marker tokens excluded.
    """
    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    B, L = shift_labels.shape
    for b in range(B):
        inside = False
        for t in range(L):
            tok = shift_labels[b, t].item()
            if tok == start_id:
                inside = True
            elif tok == end_id:
                inside = False
            elif inside and tok != -100:
                mask[b, t] = True
    return mask


def _parse_coords(text: str) -> Optional[list]:
    """
    Extract [x1, y1, x2, y2] from decoded bbox span text such as
    '(221,423),(569,886)'. Returns None if fewer than 4 values ≤ 1000.
    """
    nums = [int(m) for m in _COORD_RE.findall(text) if int(m) <= 1000]
    return nums[:4] if len(nums) >= 4 else None


def _bbox_l1_loss(shift_logits: torch.Tensor,
                   shift_labels: torch.Tensor,
                   start_id: Optional[int],
                   end_id:   Optional[int],
                   tokenizer) -> torch.Tensor:
    """
    Smooth-L1 on predicted vs. GT bbox coordinates.

    For each bbox span in the label sequence, decodes the GT and predicted
    (argmax) token subsequences, parses (x1, y1, x2, y2), and computes
    normalised smooth-L1 (thousandths → divide by 1000).

    Returns 0.0 when no bbox spans are found or tokenizer is unavailable.
    """
    device = shift_logits.device
    if start_id is None or tokenizer is None:
        return torch.tensor(0.0, device=device)

    pred_ids = shift_logits.argmax(dim=-1)    # (B, L-1)
    gt_coords, pred_coords = [], []

    for b in range(shift_labels.size(0)):
        seq_gt   = shift_labels[b].tolist()
        seq_pred = pred_ids[b].tolist()
        i = 0
        while i < len(seq_gt):
            if seq_gt[i] == start_id:
                gt_span, pred_span = [], []
                j = i + 1
                while j < len(seq_gt) and seq_gt[j] != end_id:
                    if seq_gt[j] != -100:
                        gt_span.append(seq_gt[j])
                        pred_span.append(seq_pred[j])
                    j += 1
                try:
                    gt_text   = tokenizer.decode(gt_span,   skip_special_tokens=False)
                    pred_text = tokenizer.decode(pred_span, skip_special_tokens=False)
                    c_gt   = _parse_coords(gt_text)
                    c_pred = _parse_coords(pred_text)
                    if c_gt and c_pred:
                        gt_coords.append(c_gt)
                        pred_coords.append(c_pred)
                except Exception:
                    pass
                i = j + 1
            else:
                i += 1

    if not gt_coords:
        return torch.tensor(0.0, device=device)

    gt_t   = torch.tensor(gt_coords,   dtype=torch.float, device=device) / 1000.0
    pred_t = torch.tensor(pred_coords, dtype=torch.float, device=device) / 1000.0
    return F.smooth_l1_loss(pred_t, gt_t, beta=L1_BETA)


# ---------------------------------------------------------------------------
# Loss class
# ---------------------------------------------------------------------------

class MRIBboxLoss(BaseLoss):
    """
    Weighted CE + smooth-L1 regression for per-frame MRI bbox grounding.

    BaseLoss.__init__(args, trainer) is called by the framework, so
    self.trainer and self.args are available here.
    """

    # Class-level cache — resolved once across all training steps
    _box_start_id: Optional[int] = None
    _box_end_id:   Optional[int] = None
    _tokenizer                   = None
    _init_done: bool             = False

    def _init_tokens(self) -> None:
        """Lazily resolve <|box_start|> / <|box_end|> token IDs."""
        if self._init_done:
            return
        MRIBboxLoss._init_done = True
        model = self.trainer.model
        for obj in [model, getattr(model, 'model', None), getattr(model, 'base_model', None)]:
            if obj is None:
                continue
            for attr in ('tokenizer', 'processor'):
                tok = getattr(obj, attr, None)
                if tok is None:
                    continue
                try:
                    ids = tok.convert_tokens_to_ids(['<|box_start|>', '<|box_end|>'])
                    unk = getattr(tok, 'unk_token_id', None)
                    if ids[0] is not None and ids[0] != unk:
                        MRIBboxLoss._box_start_id = ids[0]
                        MRIBboxLoss._box_end_id   = ids[1]
                        MRIBboxLoss._tokenizer    = tok
                        return
                except Exception:
                    pass

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        from swift.trainers import per_token_loss_func

        self._init_tokens()

        # per_token_loss_func returns (B, L-1) with 0 at masked positions,
        # consistent with CustomCrossEntropyLoss in causal_lm.py
        token_loss   = per_token_loss_func(outputs, labels)   # (B, L-1)
        shift_labels = labels[..., 1:].contiguous()            # (B, L-1)

        # Upweight tokens inside <|box_start|> … <|box_end|> spans
        weights = torch.ones_like(token_loss)
        if self._box_start_id is not None:
            bbox_mask          = _build_bbox_mask(shift_labels, self._box_start_id, self._box_end_id)
            weights[bbox_mask] = weights[bbox_mask] * BBOX_WEIGHT

        weighted_loss = token_loss * weights

        if num_items_in_batch is None:
            num_items_in_batch = (shift_labels != -100).sum()

        ce_loss = weighted_loss.sum() / num_items_in_batch

        # Smooth-L1 regression on decoded bbox coordinate values
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        l1_loss = _bbox_l1_loss(
            shift_logits, shift_labels,
            self._box_start_id, self._box_end_id, self._tokenizer,
        )

        return ce_loss + L1_WEIGHT * l1_loss


# ---------------------------------------------------------------------------
# Register in loss_map (add this entry to Qwen3_VL/loss/mapping.py as well)
# ---------------------------------------------------------------------------

loss_map['mri_bbox'] = MRIBboxLoss
