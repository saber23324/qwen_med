# Phase 3 — Trajectory Generation → SFT Cold-Start → RL

End-to-end specification of the Phase 3 navigation-driven 3D MRI segmentation
agent. This doc covers the four stages the project is built around, with
pointers to the concrete files that implement each one.

```
M3D volumes + GT masks
        │
        ▼
┌──────────────────────────────────┐
│ 1. Trajectory generation         │   convert_to_agent_trajectory_phase3.py
│    (teacher fabricates a 40-turn │   → SFT JSONL (~260 train, ~26 val)
│     reference dialog per case)   │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│ 2. Stage 1 — SFT cold-start       │   train_phase3.sh
│    LoRA fine-tune on teacher      │   → checkpoint-N (val Dice ≥ 0.75)
│    trajectories                   │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│ 3. Stage 2 — RL                   │   train_phase3_rl.sh
│    GRPO with multi-turn rollouts  │   → final RL checkpoint
│    over real MedSAM2              │
└──────────────────────────────────┘
        ▲
        │
┌───────┴──────────────────────────┐
│ 4. Reward design (§9 of CLAUDE.md)│   grpo_plugin.py + phase3_reward_lib.py
│    outcome + shaping + gate       │
└──────────────────────────────────┘
```

---

## 1. Trajectory Generation

**File:** `Qwen3_VL/convert_to_agent_trajectory_phase3.py`
**Output:** `agent_train.jsonl` / `agent_val.jsonl`
**Tool inventory:** `get_slice`, `scroll`, `add_bbox`, `run_medsam2`,
`add_point`, `finish_3d_segmentation` (full schema at line 83 of that file;
prose contract in `Qwen3_VL/PHASE3_TOOLS.md`).

### What the teacher produces

For each (volume, lesion-Z range) pair the teacher fabricates a fully
worked-out reference dialog that mimics how a competent radiologist would
operate the agent. Phase 3 trajectories average ~40 tool calls and consist of
three deterministic phases:

| Phase | Purpose | Dominant tool | Exit |
|---|---|---|---|
| **A — Exploration** | locate key slice | `get_slice(mid)` then `scroll ±1 / ±2` zigzag | key slice declared in CoT |
| **B — Annotation** | bbox every lesion ordinal | `scroll` interleaved with `add_bbox`; `get_slice` only for jumps ≥ 3 | `unvisited_ordinals` empty |
| **C — Mask review** | inspect each mask, refine poor ones | `get_slice(Z[0])` → `scroll(+1)` × 9 sweep, `add_point` on Dice < 0.70 | `finish_3d_segmentation` |

10 Z-indices are sampled per case: `np.linspace(z_min, z_max, 10, dtype=int)`.
The agent's segmentation scope is exactly these 10 ordinals — the
chain-of-thought refers to them by ordinal index 0..9, with the absolute Z
echoed in every navigation tool_response.

### State invariants enforced at generation time

(`validate_entry` in the generator filters violators):

1. `i_cur ∈ {0..9}` once initialized; undefined before first `get_slice`.
2. `Z_list` immutable after Turn 1.
3. Every `<image>` token in the response stream matches one entry in
   `images`, in order.
4. `add_bbox` / `add_point` with `z_index != Z_list[i_cur]` are rejected.
5. `run_medsam2` fires exactly once, after Phase B completes.
6. `finish_3d_segmentation` is the last tool call and unique.

### Real MedSAM2 in the loop

The generator drives MedSAM2 to compute the masks the teacher reports back
in `tool_response` (so the trajectory contains *real* per-slice Dice values,
not fabricated ones). Two implementation traps the code gets right:

- `apply_postprocessing=False` on `build_sam2_video_predictor_npz` —
  the default `fill_holes_in_mask_scores` CUDA kernel triggers an illegal
  memory access on this hardware.
- Absolute Hydra config paths need a `//` prefix:
  `if os.path.isabs(cfg_path) and not cfg_path.startswith('//'): cfg_path = '//' + cfg_path`.

Token-budget choices: input slices are 512×512 (~333 vision tokens each);
all teacher-generated render overlays (bbox, mask, refined-mask) are
256×256 (~84 tokens each). Trajectory mean length ≈ 40 tool calls fits
inside `max_length=16384`.

### JSONL row schema

Top-level keys: `tools`, `messages`, `images` — exactly the SFT format ms-swift
expects with `--agent_template hermes`. The `images` list contains paths in
the same order that `<image>` tokens appear across all messages.

```jsonl
{
  "tools": "[...]",
  "messages": [
    {"role": "user",          "content": "Task: …\nSampled Z = [40, 43, …, 70]"},
    {"role": "assistant",     "content": "I'll start from the middle slice …"},
    {"role": "tool_call",     "content": "{\"name\": \"get_slice\", \"arguments\": {\"z_index\": 55}}"},
    {"role": "tool_response", "content": "{\"z_index\": 55, \"ordinal\": 5, \"slice_image\": \"<image>\", …}"},
    ...
  ],
  "images": ["volumes/case_001/slice_055.png", ...]
}
```

> Important format detail (per `Agent-support.md`): the assistant message
> carries CoT *only*. Every tool call is a separate `{"role": "tool_call",
> "content": "<JSON string>"}` message; the JSON string must be parseable by
> `json.loads`. The agent_template renders these into the
> `<tool_call>...</tool_call>` XML markers Qwen3-VL expects.

---

## 2. Stage 1 — SFT Cold-Start

**Launcher:** `Qwen3_VL/train_phase3.sh`
**Goal:** raise the policy from "zero-shot" to "format-correct &
competent" so the RL stage has a well-anchored reference policy and a
non-trivial KL prior.

### Configuration that matters

| Setting | Value | Why |
|---|---|---|
| `--agent_template hermes` | hermes | wraps `tool_call` JSON into `<tool_call>...</tool_call>` for Qwen3-VL |
| `--max_length` | 16384 | Phase 3 trajectories are ~2× Phase 2; needs the headroom |
| `--tuner_type lora`, `--lora_rank 8`, `--lora_alpha 32` | LoRA only | keeps the base model intact for RL warmstart |
| `--target_modules all-linear` | all linear | matches Phase 2 SFT |
| `--freeze_vit true --freeze_aligner true` | freeze vision | only LoRA on the LM is trained |
| `--packing true --padding_free true --attn_impl flash_attn` | throughput | needed to fit 40-turn samples in batch |
| `--learning_rate 1e-4`, 10 epochs, ZeRO-2 | LoRA SFT defaults | matches Phase 2 |

### Loss scope (identical across all phases)

| Content | In loss? |
|---|---|
| Assistant CoT (phase transitions, reviews) | yes |
| Every `tool_call` message (incl. navigation) | yes |
| `finish_3d_segmentation` | yes |
| `tool_response` (renders, mask batches, nav reads) | no |
| `user` (task, Z list) | no |

### Cold-start success criteria

RL should **not** be launched until SFT meets both:

- val Dice ≥ 0.75
- format-violation rate ≤ 2%

Below these floors the KL-to-SFT regularizer in the RL stage is anchoring
to a policy that doesn't know the task, so GRPO will either thrash (high
β_kl) or drift into reward hacking (low β_kl).

---

## 3. Stage 2 — RL

**Launcher:** `Qwen3_VL/train_phase3_rl.sh` (three modes: `data`, `server`,
`train`).
**Algorithm:** GRPO (Group Relative Policy Optimization).
**Why GRPO and not PPO/DPO** (CLAUDE.md §9.1):

1. Reward is verifiable (Dice vs GT) — no learned reward model needed.
2. Memory budget is tight (8B LM + LoRA + MedSAM2 + KV cache on 2 GPUs);
   GRPO drops the critic.
3. Group-mean advantage on a final, dense outcome reward sidesteps the
   credit-assignment problem PPO hits at 40+ turns with vision observations.

### File layout

| File | Role |
|---|---|
| `Qwen3_VL/grpo_plugin.py` | ms-swift `external_plugins` module. Registers nine ORMs (one combined + eight diagnostic). |
| `Qwen3_VL/phase3_reward_lib.py` | vLLM-free core: `RewardTrackingExecutor` (extends `NavAgentExecutor`), `compute_reward_components`, `collect_gate_violations`. |
| `Qwen3_VL/phase3_rl_rollout.py` | `Phase3NavScheduler` — ms-swift `MultiTurnScheduler` subclass. Drives the 45-turn agent loop with live MedSAM2 inside the rollout server, returns `RolloutOutput.rollout_infos` with every reward component pre-computed. Registered as `phase3_nav` in `swift.rollout.multi_turn.multi_turns`. |
| `Qwen3_VL/convert_to_grpo_dataset.py` | Strips assistant/tool turns from the SFT JSONL. Outputs RL rows with `messages` (user only), empty `images`, plus side-channel GT (`vid`, `caption`, `anno_id`, `data_root`, `sampled_z`, `lesion_ordinals`, `non_lesion_ordinals`, `oracle_key_z`). |
| `Qwen3_VL/train_phase3_rl.sh` | Three-mode launcher: `data` builds the GRPO JSONL; `server` runs vLLM + scheduler on GPU 6; `train` runs the GRPO trainer on GPU 7. |

### How a single rollout flows

1. Trainer ships a prompt + GT side-channel to the rollout server
   (`--vllm_server_pass_dataset true` forwards every dataset column into
   `RolloutInferRequest.data_dict`).
2. `Phase3NavScheduler.run()` lazy-builds a `RewardTrackingExecutor` for the
   case (loads the volume, GT masks, and the cached MedSAM2 predictor).
3. Loop, up to `PHASE3_MAX_TURNS=45`:
   - call `infer_engine.infer_async(infer_request, ...)` → assistant turn
   - parse `<tool_call>` blocks → execute via the executor → append a
     **clean JSON** `tool_call` message + a `tool_response` message + any
     newly rendered images (loaded as `PIL.Image` to bypass ms-swift's
     `_check_path` base64 fallback).
   - stop on `finish_3d_segmentation`, length cap, or no parseable tool call.
4. Compute every reward component from executor state + per-call traces;
   stuff them into `RolloutOutput.rollout_infos`.
5. Trainer-side, `Phase3CombinedReward` reads `rollout_infos` and produces
   the scalar policy reward; the eight diagnostic ORMs each surface one
   component (with `--reward_weights` set to `0.0` for the diagnostics so
   they appear in TensorBoard but do not contribute to the gradient).

### Hardware layout (2-GPU constraint)

| GPU | Process |
|---|---|
| 6 | vLLM rollout server + `Phase3NavScheduler` + MedSAM2 (`PHASE3_MEDSAM2_DEVICE=cuda:0` inside the server's CVD namespace) |
| 7 | GRPO trainer |

The scheduler runs **inside** the rollout server process — that's why
`phase3_rl_rollout.py` is loaded server-side via `--external_plugins`,
while `grpo_plugin.py` is loaded trainer-side.

> The CLAUDE.md §9.9 ideal is a 3-GPU layout (training + rollout + dedicated
> MedSAM2). With two GPUs MedSAM2 has to time-share with vLLM on GPU 6 —
> vLLM is mostly memory-bound, so MedSAM2's bursts (~1–3 min per
> trajectory's `run_medsam2`) fit alongside it on a 24 GB card.

### Curriculum (§9.7 of CLAUDE.md)

| Stage | Steps | Shaping scale | Gate | β_kl | Goal |
|---|---|---|---|---|---|
| **1 — Warmup** | ~1k | `PHASE3_SHAPING_SCALE=2.0` | `PHASE3_GATE_SOFT=1` (0.1× on invalid) | `--beta 0.04` | stabilize on-policy at near-SFT format |
| **2 — Main** | 5k–10k | `1.0` | hard (`0`) | `--beta 0.015` | outcome-dominant training |
| **3 — Polish** (optional) | 1k–2k | `0.0` except keep `R_bbox_iou` weight | hard | `--beta 0.01` | test whether policy stands on pure outcome |

Stage transitions are launcher edits — the env vars / `--beta` switches above
plus updating `--reward_weights` for stage 3.

### Hyperparameters baked into stage 1

```
group size K        = 8 rollouts per prompt   (--num_generations 8)
sampling temp       = 0.9                      (--temperature 0.9)
LoRA learning rate  = 5e-7                     (--learning_rate 5e-7)
rollout cap         = 45 tool calls            (PHASE3_MAX_TURNS=45)
max_length          = 16384                    (Phase 3 trajectory length)
gradient clip       = 0.5                      (--max_grad_norm 0.5)
shaping scale       = 2.0                      (PHASE3_SHAPING_SCALE)
gate                = soft (0.1× on invalid)   (PHASE3_GATE_SOFT=1)
β_kl                = 0.04                     (--beta 0.04)
```

---

## 4. Reward Design

The full §9 of `CLAUDE.md` is canonical — this is a tighter restatement keyed
to the actual code in `grpo_plugin.py` and `phase3_reward_lib.py`.

### The formula (§9.6)

```
R(trajectory) = gate · [
      1.00 · (0.7 · R_dice + 0.3 · R_key_dice)        # outcome
    + s · 0.20 · R_bbox_iou                            # process shaping
    + s · 0.15 · R_dice_gain                           # process shaping
    + s · 0.10 · R_coverage                            # process shaping
    + s · 0.05 · R_nav_style                           # process shaping
    + 1.00 · R_efficiency                              # soft cap
]
```

`s = SHAPING_SCALE` (2.0 in stage 1, 1.0 in stage 2, 0.0 in stage 3 except
`R_bbox_iou`). `gate ∈ {0, 1}` (or `0.1` if `PHASE3_GATE_SOFT=1`).

Expected scale: teacher-quality trajectory ≈ 0.85, strong rollout 0.90–0.95,
malformed 0.

### Outcome layer (§9.3) — non-hackable

| Term | Value |
|---|---|
| `R_dice` | mean per-slice Dice over the 10 sampled Z, **after refinement** (post-`add_point`, pre-`finish_3d_segmentation`) |
| `R_key_dice` | Dice on the ordinal **the model committed in `run_medsam2`** (not a re-declaration). Without this the policy can pick any key_z and `R_dice` averages it away. |

`R_outcome = 0.7 · R_dice + 0.3 · R_key_dice`.

The plan explicitly **rejects** `R_dice_delta_vs_SFT`: GRPO's intra-group
relative advantage already does that math. Optional `R_surface` (boundary
Dice or `1 − normalized_HD95`) can be added if HD95 is a primary eval metric;
it occasionally fights `R_dice` on small lesions.

### Process shaping layer (§9.4) — designed against specific failure modes

| Term | β | What it credits | Cap / threat addressed |
|---|---|---|---|
| `R_bbox_iou` | 0.20 | mean IoU over `add_bbox` calls vs GT bbox | densest, safest signal across Phase B |
| `R_dice_gain` | 0.15 | sum of clipped per-step (`max(0, min(0.3, dice_after − dice_before))`) over `add_point` calls | the 0.3 clip blocks the "deliberate bad bbox to farm gains" hack |
| `R_coverage` | 0.10 | `recall(annotated ∩ lesion_ords) − 0.5 · count(annotated ∩ non_lesion_ords)` | FP penalty blocks "annotate every ordinal" collapse |
| `R_nav_style` | 0.05 | per-call: scroll Δ ≤ 2 = +0.01; get_slice Δ ≥ 3 = +0.01; scroll that clamps = −0.02; get_slice re-read with unchanged overlay = −0.05 | rewards correct tool-choice, not movement frequency |
| `R_efficiency` | 1.00 | `−0.01 · max(0, n_tool_calls − 45)` | soft cap; per-step penalty would discourage refinements |

### Format gate (§9.5) — hard

`gate = 1` iff **all** of:

- Every `add_bbox` / `add_point` satisfies `z_index == Z_list[i_cur]`.
- `run_medsam2` fires exactly once.
- `key_z ∈ Z_list`.
- `finish_3d_segmentation` is the last tool call and unique.
- No `scroll` before the first `get_slice`.
- Every tool-call content parses as valid JSON.

Otherwise `gate = 0`. GRPO tolerates this because advantage is computed
**within** group: as long as some group members are valid, there's a learning
signal. If entire groups go invalid early in training, `PHASE3_GATE_SOFT=1`
softens to `0.1×`.

### Reward-hacking threat model (§9.8)

| Threat | Defense |
|---|---|
| Collapse to "annotate every ordinal" | `R_coverage` FP term + `R_bbox_iou` zero on non-lesion slices |
| Key-slice laundering (post-hoc pick) | `R_key_dice` uses `key_z` committed in `run_medsam2`, not a re-declaration |
| Deliberate bad bbox to farm `R_dice_gain` | per-step clip on `R_dice_gain` + `R_bbox_iou` rewards good initial bboxes — terms pull opposite directions |
| Trajectory-length inflation | `R_nav_style` rewards tool-choice correctness, not frequency; `R_efficiency` caps total length |
| Schema-valid but degenerate (scroll back & forth) | no shaping credit for movement without annotation or Dice gain; monitor "tool calls per annotation" |

### Logging & diagnostics

`Phase3CombinedReward` is the policy reward (weight `1.0`). The eight
diagnostic ORMs (`phase3_dice`, `phase3_key_dice`, `phase3_bbox_iou`,
`phase3_dice_gain`, `phase3_coverage`, `phase3_nav_style`,
`phase3_efficiency`, `phase3_format_gate`) are weight `0.0` — they appear in
TensorBoard as `rewards/phase3_<name>/{mean,std}` per logging step but do not
contribute to the gradient. `rollout_infos` also carries:

- `num_turns`, `n_add_bbox`, `n_add_point`, `n_run_medsam2`, `n_finish`
- `boundary_clamp_count`, `revisit_no_change_count`
- `key_z_chosen`, `bbox_iou_at_run`
- `gate_violations` (list of strings)

The §9 diagnostic to watch: **`R_dice` trending up while `R_coverage` stays
near 1 and `R_nav_style` does not regress.** Any divergence between primary
and secondary metrics is the early signature of reward hacking — back off the
shaping weights or strengthen the relevant gate.

---

## Quick reference

```bash
# 0. Activate env (vLLM ≥ 0.10.2 required for the rollout-importance-sampling
#    correction):
conda activate qwen3
pip install 'vllm>=0.10.2' -U

# 1. Build SFT trajectories (one-time, run inside qwen3 env):
python3 Qwen3_VL/convert_to_agent_trajectory_phase3.py \
    --data_root /BDSZ6/.../M3D/data_18-22/train \
    --output_dir /BDSZ6/.../agent_phase3_18-22 \
    --ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --device cuda:4

# 2. Stage 1 — SFT cold-start
bash Qwen3_VL/train_phase3.sh
# (validate: val Dice ≥ 0.75, format-violation rate ≤ 2%)

# 3. Build GRPO dataset (strips trajectories down to user prompt + GT side-channel)
bash Qwen3_VL/train_phase3_rl.sh data

# 4. Stage 2 — GRPO (two terminals)
# Terminal A:
bash Qwen3_VL/train_phase3_rl.sh server
# Terminal B:
bash Qwen3_VL/train_phase3_rl.sh train
```
