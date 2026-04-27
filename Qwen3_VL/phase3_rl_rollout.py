"""
Phase 3 multi-turn GRPO rollout scheduler.

Runs the trained Qwen3-VL agent's full tool-using trajectory inside a
ms-swift `MultiTurnScheduler`, executing real MedSAM2 calls and computing
every reward component from the executed trajectory + GT.

Registered as `phase3_nav` in `swift.rollout.multi_turn.multi_turns` when
this module is imported (via `--external_plugins` in the launcher; or via
`SWIFT_EXTERNAL_ROLLOUT_PLUGIN=Qwen3_VL/phase3_rl_rollout.py` for the
rollout server).

Per-rollout flow
----------------
1. The dataset row is converted to a `RolloutInferRequest` with `data_dict`
   carrying GT references (vid, anno_id, sampled_z, lesion_ordinals, …).
2. `Phase3NavScheduler.run()` builds a `RewardTrackingExecutor` (subclass of
   `NavAgentExecutor`) with MedSAM2 + the case's GT masks, then drives the
   conversation:
     • call `infer_engine.infer_async` → assistant turn (CoT + tool_calls)
     • parse tool_calls, run executor sequentially
     • append `tool_call` / `tool_response` messages and any new `<image>`
       paths to `infer_request.images`
     • repeat until `finish_3d_segmentation`, max turns, or hard format error.
3. Compute reward components from executor state + per-call traces, return
   them via `RolloutOutput.rollout_infos` for the reward plugin to consume.

Hardware
--------
Per CLAUDE.md §9.9: vLLM (this scheduler is co-located with the engine) on
one GPU, MedSAM2 on a second. Set `MEDSAM2_DEVICE` (default `cuda:1`).

Caches
------
MedSAM2 predictor and the per-data-root `(mask_dict, meta_expressions)` pair
are cached at process scope keyed by file path so they survive across
trajectories within a server.
"""

import json
import logging
import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# ── ms-swift imports ────────────────────────────────────────────────────────
from swift.infer_engine.protocol import (ChatCompletionResponse,
                                          ChatCompletionResponseChoice,
                                          RolloutInferRequest, RolloutOutput)
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.utils import remove_response

_log = logging.getLogger('phase3_rl_rollout')
if not _log.handlers:
    _log.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[phase3_nav] %(message)s'))
    _log.addHandler(_h)


def _load_paths_to_pil(paths: List[str]) -> List[Image.Image]:
    """Open render paths into PIL.Image objects up-front.

    Bypasses ms-swift's `_check_path` → base64 fallback, which can mis-classify
    a momentarily-missing PNG path as base64 and produce
    `BytesIO(garbage) → UnidentifiedImageError`. Loading here fails loudly with
    the actual filename instead.
    """
    out: List[Image.Image] = []
    for p in paths:
        if not isinstance(p, str):
            out.append(p)
            continue
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if not os.path.isfile(p):
            _log.error('render path missing on disk: %r', p)
            raise FileNotFoundError(p)
        try:
            img = Image.open(p)
            img.load()  # force read so we can close the FD
            if img.mode != 'RGB':
                img = img.convert('RGB')
            out.append(img)
        except Exception as e:
            _log.error('failed to open render %r: %s', p, e)
            raise
    return out

# ── Local imports — pull NavAgentExecutor + helpers from infer_phase3.py ────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_phase3 import (TOOLS, load_volume, parse_tool_calls,
                           split_cot_and_calls, SAMPLED_Z_RE)
from phase3_reward_lib import (RewardTrackingExecutor,
                                compute_reward_components,
                                collect_gate_violations)

# ── Constants ───────────────────────────────────────────────────────────────
MAX_AGENT_TURNS = int(os.environ.get('PHASE3_MAX_TURNS', '45'))   # §9.1 cap
MEDSAM2_DEVICE  = os.environ.get('PHASE3_MEDSAM2_DEVICE', 'cuda:1')
RENDER_ROOT     = os.environ.get('PHASE3_RENDER_ROOT', '/tmp/phase3_rl_renders')


# ────────────────────────────────────────────────────────────────────────────
# Caches — process-scope, thread-safe
# ────────────────────────────────────────────────────────────────────────────

_predictor_lock = threading.Lock()
_predictor = None  # type: Optional[Any]

_mask_lock = threading.Lock()
_mask_cache: Dict[str, Tuple[Dict, Dict]] = {}  # data_root -> (mask_dict, meta)


def _get_predictor(ckpt: str, cfg: str):
    global _predictor
    with _predictor_lock:
        if _predictor is not None:
            return _predictor
        import torch
        from sam2.build_sam import build_sam2_video_predictor_npz
        cfg_path = cfg
        if os.path.isabs(cfg_path) and not cfg_path.startswith('//'):
            cfg_path = '//' + cfg_path
        torch.set_float32_matmul_precision('high')
        device = torch.device(MEDSAM2_DEVICE if torch.cuda.is_available() else 'cpu')
        _predictor = build_sam2_video_predictor_npz(
            cfg_path, ckpt, device=device, apply_postprocessing=False
        )
        return _predictor


def _get_mask_meta(data_root: str) -> Tuple[Dict, Dict]:
    with _mask_lock:
        if data_root in _mask_cache:
            return _mask_cache[data_root]
        with open(os.path.join(data_root, 'mask_dict.pkl'), 'rb') as f:
            mask_dict = pickle.load(f)
        with open(os.path.join(data_root, 'meta_expressions.json')) as f:
            meta = json.load(f)['videos']
        _mask_cache[data_root] = (mask_dict, meta)
        return mask_dict, meta


# ────────────────────────────────────────────────────────────────────────────
# Phase 3 multi-turn scheduler
# ────────────────────────────────────────────────────────────────────────────

class Phase3NavScheduler(MultiTurnScheduler):
    """Drives a Qwen3-VL Phase-3 trajectory with live MedSAM2 execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.medsam2_ckpt = os.environ.get('PHASE3_MEDSAM2_CKPT')
        self.medsam2_cfg  = os.environ.get('PHASE3_MEDSAM2_CFG')
        if not self.medsam2_ckpt or not self.medsam2_cfg:
            raise RuntimeError(
                'PHASE3_MEDSAM2_CKPT and PHASE3_MEDSAM2_CFG must be set for the rollout server')
        os.makedirs(RENDER_ROOT, exist_ok=True)

    # ── Build a per-trajectory executor from data_dict ──────────────────────
    def _build_executor(self, data: Dict[str, Any]) -> RewardTrackingExecutor:
        import torch
        vid = data['vid']
        anno_id = str(data['anno_id'])
        data_root = data['data_root']
        sampled = list(data['sampled_z'])

        mask_dict, meta = _get_mask_meta(data_root)
        masks = mask_dict[anno_id]
        vd = meta[vid]
        frames = sorted(vd['frames'])
        img_w, img_h = vd['width'], vd['height']

        jpeg_dir = os.path.join(data_root, 'JPEGImages', vid)

        predictor = _get_predictor(self.medsam2_ckpt, self.medsam2_cfg)
        device = torch.device(MEDSAM2_DEVICE if torch.cuda.is_available() else 'cpu')
        img_tensor, orig_H, orig_W = load_volume(jpeg_dir, device)

        case_render_dir = os.path.join(RENDER_ROOT, vid, anno_id)
        os.makedirs(case_render_dir, exist_ok=True)

        return RewardTrackingExecutor(
            predictor=predictor, device=device,
            jpeg_dir=jpeg_dir, img_tensor=img_tensor,
            orig_H=orig_H, orig_W=orig_W,
            frames=frames, masks=masks, sampled=sampled,
            render_dir=case_render_dir, vid=vid,
            img_w=img_w, img_h=img_h,
        )

    # ── Run a single rollout ────────────────────────────────────────────────
    async def run(self, infer_request: 'RolloutInferRequest', request_config,
                  **kwargs) -> 'RolloutOutput':
        data = infer_request.data_dict or {}
        sampled_z = list(data.get('sampled_z') or [])
        # Fallback: parse from the user content if the dataset lacks the column.
        if not sampled_z:
            user_text = infer_request.messages[0].get('content', '') or ''
            m = SAMPLED_Z_RE.search(user_text)
            if not m:
                raise RuntimeError('phase3_nav: sampled_z not in data_dict and not parsable from user message')
            sampled_z = [int(x.strip()) for x in m.group(1).split(',')]

        lesion_ords = list(data.get('lesion_ordinals') or [])
        non_lesion_ords = list(data.get('non_lesion_ordinals') or
                                [i for i in range(len(sampled_z)) if i not in set(lesion_ords)])

        # Build executor (loads MedSAM2 + GT masks + volume)
        try:
            executor = self._build_executor(data)
        except Exception as e:
            # Fatal data error → return a zero-reward, gated trajectory so GRPO can skip.
            return self._abort_output(infer_request, reason=f'build_executor_failed: {e}')

        # Tools must match the SFT trajectories.
        infer_request.tools = TOOLS
        # Phase 3 Turn 1 has NO images.
        infer_request.images = []

        parsed_calls: List[Tuple[str, Dict[str, Any]]] = []
        n_tool_calls = 0
        max_turns = self.max_turns or MAX_AGENT_TURNS
        finished_cleanly = False

        last_response: Optional[ChatCompletionResponse] = None

        for turn in range(1, max_turns + 1):
            if turn == 1:
                remove_response(infer_request.messages)

            _log.info('turn=%d msgs=%d images=%d', turn,
                      len(infer_request.messages), len(infer_request.images))
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                infer_request, request_config, **kwargs)
            last_response = response
            choice: 'ChatCompletionResponseChoice' = response.choices[0]
            asst_text = choice.message.content or ''

            # Per Agent-support.md (hermes agent_template): the assistant
            # message must carry CoT ONLY — `<tool_call>` blocks belong in
            # separate `role: tool_call` messages whose `content` is a clean
            # JSON string `{"name": ..., "arguments": ...}`. Appending the
            # full assistant text duplicates the tool-call markers in the
            # rendered prompt and triggers `_parse_tool_call` errors on the
            # next encode if any block is malformed.
            cot, _raw_calls = split_cot_and_calls(asst_text)
            tool_calls = parse_tool_calls(asst_text)

            if cot:
                infer_request.messages.append({'role': 'assistant', 'content': cot})

            # Hit length cap or no parseable tool_calls → stop.
            if choice.finish_reason == 'length' or not tool_calls:
                break

            # Execute parseable tool calls sequentially. Each gets a clean
            # JSON `tool_call` message + a paired `tool_response` message.
            for tc in tool_calls:
                name = tc.get('name', '')
                args = tc.get('arguments', {}) or {}
                tc_payload = json.dumps({'name': name, 'arguments': args},
                                         ensure_ascii=False)
                infer_request.messages.append({'role': 'tool_call', 'content': tc_payload})
                parsed_calls.append((name, args))
                n_tool_calls += 1

                try:
                    if name == 'get_slice':
                        resp_json = executor.exec_get_slice(int(args['z_index']))
                    elif name == 'scroll':
                        resp_json = executor.exec_scroll(int(args['delta']))
                    elif name == 'add_bbox':
                        resp_json = executor.exec_add_bbox(
                            int(args['z_index']), list(args['bbox']))
                    elif name == 'run_medsam2':
                        resp_json = executor.exec_run_medsam2(
                            int(args['key_z']), list(args['bbox']))
                    elif name == 'add_point':
                        resp_json = executor.exec_add_point(
                            int(args['z_index']),
                            list(args['points']),
                            list(args['labels']))
                    elif name == 'finish_3d_segmentation':
                        executor.finish_called = True
                        finished_cleanly = True
                        resp_json = json.dumps({'status': 'terminated'})
                    else:
                        resp_json = json.dumps({'error': f'unknown tool: {name}'})
                except Exception as e:
                    resp_json = json.dumps({
                        'error': f'tool exec failed: {type(e).__name__}: {e}'})

                infer_request.messages.append({'role': 'tool_response', 'content': resp_json})

                if finished_cleanly:
                    break

            # The executor appends render paths to its own image list. Convert
            # paths → PIL.Image up-front to bypass ms-swift's `_check_path`
            # base64-fallback which can produce BytesIO(garbage).
            infer_request.images = _load_paths_to_pil(executor.images)

            if finished_cleanly:
                break

            # Hard cap on tool calls — saves wallclock when policy goes feral.
            if n_tool_calls >= max_turns:
                break

        gate_violations = collect_gate_violations(parsed_calls, sampled_z)
        components = compute_reward_components(
            executor,
            lesion_ordinals=lesion_ords,
            non_lesion_ordinals=non_lesion_ords,
            n_tool_calls=n_tool_calls,
            gate_violations=gate_violations,
        )

        # Free the volume tensor and any per-case GPU state.
        try:
            del executor.img_tensor
        except Exception:
            pass

        return RolloutOutput(
            response=last_response,
            messages=infer_request.messages,
            response_token_ids=[],   # text-only completion; trainer re-tokenises
            response_loss_mask=[],
            rollout_infos={**components,
                           'finished': finished_cleanly,
                           'parsed_call_count': n_tool_calls},
            rollout_logprobs=[],
        )

    # ── Build a zero-reward output for catastrophic failure ────────────────
    def _abort_output(self, infer_request: 'RolloutInferRequest', *, reason: str) -> 'RolloutOutput':
        return RolloutOutput(
            response=None,
            messages=infer_request.messages,
            response_token_ids=[],
            response_loss_mask=[],
            rollout_infos={
                'r_dice': 0.0, 'r_key_dice': 0.0,
                'r_bbox_iou': 0.0, 'r_dice_gain': 0.0,
                'r_coverage': 0.0, 'r_nav_style': 0.0,
                'r_efficiency': 0.0,
                'gate': 0,
                'gate_violations': [reason],
                'num_turns': 0,
                'n_add_bbox': 0, 'n_add_point': 0,
                'n_run_medsam2': 0, 'n_finish': 0,
                'finished': False,
                'parsed_call_count': 0,
            },
            rollout_logprobs=[],
        )


# ────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────

multi_turns['phase3_nav'] = Phase3NavScheduler
