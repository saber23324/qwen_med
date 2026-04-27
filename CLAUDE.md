# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

---

## 1. Project Overview

A 3D MRI lesion segmentation **agent** built on **Qwen3-VL-8B-Instruct** with **ms-swift** tool calling and **MedSAM2** as the segmentation backbone. The model observes MRI slices one at a time, reasons about anatomy through chain-of-thought, annotates bounding boxes, drives MedSAM2 to propagate a 3D mask, and self-corrects poorly-segmented slices with point prompts. All training uses LoRA in the conda `base` / `qwen3` environment.

**The agent does not segment directly.** It plans, annotates, and supervises MedSAM2. Its intelligence is in the decision loop — which slice to read, where to draw the bbox, which slice to seed propagation from, which masks to refine — not in pixel-level prediction.

**Phase 1, Phase 2 and Phase 3 SFT are now completed. Your next task is to finalize and implement the Phase 3 RL Plan.**
---

## 2. Overall Objectives

### 2.1 Final system

A tool-using agent that, given an MRI volume and a lesion Z-range, produces a 3D segmentation whose quality matches or exceeds a human-driven MedSAM2 session, while following a read-before-write, scroll-based navigation protocol that mirrors how a radiologist reviews a DICOM series.

### 2.2 Concrete quality targets (val set, n=26–33)

| Metric | Phase 2 baseline | Phase 3 SFT target | Phase 3 RL target |
|---|---|---|---|
| 3D Dice (mean per-slice on sampled Z) | 0.826 | ≥ 0.82 | ≥ 0.87 |
| HD95 (px) | 9.50 | ≤ 10 | ≤ 8 |
| Format-violation rate | n/a | ≤ 2% | ≤ 0.5% |
| Mean tool calls / trajectory | ~11 | ~40 | ~40 (stable) |

Phase 3 RL is expected to deliver the Dice lift by correcting the two main Phase 3 SFT failure modes: suboptimal key-slice choice and conservative refinement (agent skips `add_point` on genuinely poor slices because the teacher data under-represents it).

### 2.3 Training strategy

Three phases of SFT, then outcome-driven RL on top of the Phase 3 SFT checkpoint. Each phase strictly extends the previous JSONL format — only the `tools` field grows and the turn structure evolves. Loss computation scope is identical across phases: assistant CoT and tool_calls contribute to loss, tool_responses and user turns do not.

---

## 3. Phased Roadmap

| Phase | Goal | Training | Status | Output artifact |
|---|---|---|---|---|
| **1** | 2D bbox per slice, global spatial CoT | LoRA SFT | Done | Bbox-annotating model |
| **2** | bbox → MedSAM2 mask → point refinement | LoRA SFT | Done | 3D-segmenting model (Dice 0.826) |
| **3 SFT** | Scroll-based navigation over 10 sampled Z; on-demand slice reads | LoRA SFT | Done (data pipeline, training in progress) | Navigation-driven segmenting model |
| **3 RL** | Outcome-driven polish of the Phase 3 SFT policy | GRPO on verifiable Dice | **Planned — spec in §8** | Final model |

Tool evolution:

```
Phase 1   add_bbox  finish_bbox_annotation
Phase 2   add_bbox  run_medsam2  add_point  finish_3d_segmentation
Phase 3   add_bbox  run_medsam2  add_point  finish_3d_segmentation  +  scroll  get_slice
```

---

## 4. Repository Structure

```
medagent/
├── CLAUDE.md                                  # This file
├── Agent-support.md                           # ms-swift agent framework notes
├── Complete Multimodal GRPO Experiment Workflow.md # Important!!
├── GRPO-code-Training.md # Important!!
├── GRPO.md # Important!!
├── MedSAM2/                                   # MedSAM2 submodule
│   ├── checkpoints/MedSAM2/MedSAM2_latest.pt
│   └── sam2/configs/sam2.1_hiera_t512.yaml    # 512×512 Hydra config
├── agent/
│   ├── qwen3_mcore.sh                         # Phase 2 training (no DeepSpeed)
│   └── eval.sh                                # swift infer evaluation
└── Qwen3_VL/
    ├── train.sh / train_phase2.sh / train_phase3.sh
    ├── infer.py / infer_phase2.py / infer_phase3.py
    ├── visualize_phase2.py / visualize_phase3.py
    ├── custom_loss.py                         # MRIBboxLoss: weighted CE + Smooth-L1
    ├── convert_to_agent_trajectory.py         # Phase 1 trajectory generator
    ├── convert_to_agent_trajectory_phase2.py  # Phase 2 trajectory generator
    ├── convert_to_agent_trajectory_phase3.py  # Phase 3 trajectory generator
    ├── medsam2_phase2.py                      # Standalone MedSAM2 inference
    ├── PHASE2_WORKFLOW.md                     # Full Phase 2 reference
    ├── PHASE3_TOOLS.md                        # Full Phase 3 tool spec
    └── loss/ custom/                          # Loss and dataset extensions for ms-swift
```

---

## 5. Key Paths and Environment

```
GPU constraint         CUDA_VISIBLE_DEVICES=6,7  (only GPUs 6 and 7 available)
Conda env              qwen3   (training, ms-swift, MedSAM2 inference)
                       qwen3  (legacy Phase 2 dataset generation)

Base model             /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct
M3D data               /BDSZ6/private/user/yxd/data/M3D/data_6-13/train
Phase 2 dataset        /BDSZ6/private/user/yxd/data/qwen/agent_phase2/
Training output        /BDSZ6/private/user/yxd/dtos_output/qwen/
MedSAM2 checkpoint     /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt
MedSAM2 config         /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml
```

---

## 6. Common Commands

### Training

```bash
bash Qwen3_VL/train.sh            # Phase 1 LoRA (GPU 4,5, DeepSpeed Zero-2)
bash Qwen3_VL/train_phase2.sh     # Phase 2 LoRA (GPU 4,5,6,7, DeepSpeed Zero-2)
bash Qwen3_VL/train_phase3.sh     # Phase 3 LoRA (GPU 3,7, max_length=16384)
bash agent/qwen3_mcore.sh         # Phase 2 alternative (GPU 3,7, no DeepSpeed)
```

### Phase 3 dataset generation (real MedSAM2)

```bash
conda run -n qwen3 python3 Qwen3_VL/convert_to_agent_trajectory_phase3.py \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --output_dir /BDSZ6/private/user/yxd/data/qwen/agent_phase3 \
    --ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --device cuda:4
```

### Inference and evaluation

```bash
# Phase 3 agent-loop inference
conda run -n qwen3 python3 Qwen3_VL/infer_phase3.py \
    --model  /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --ckpt   /path/to/checkpoint \
    --val_jsonl /BDSZ6/private/user/yxd/data/qwen/agent_phase3/agent_val.jsonl \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --output_dir /tmp/phase3_eval --device cuda:4

# Swift batch infer + visualize
bash agent/eval.sh
conda run -n qwen3 python3 Qwen3_VL/visualize_phase3.py \
    --infer_jsonl /path/to/infer_result/*.jsonl --output_dir /tmp/phase3_vis
```

---

## 7. Phase 1 and Phase 2 — Core Summary

Phases 1 and 2 are **done and frozen**. Only the key facts needed for Phase 3 work are kept here. For the full specs see commit history or the PHASE2_WORKFLOW.md doc.

### 7.1 Phase 1 — bbox annotation per slice

**Goal.** Given 10 Z-indexed slices sampled via `np.linspace(z_min, z_max, 10)` from the lesion range, emit one `add_bbox(z_index, [x1,y1,x2,y2])` per lesion-containing slice, then `finish_bbox_annotation`. Turn 2 is a mandatory **global spatial-analysis CoT** listing lesion Z range, key (largest-area) slice, morphology trend, and explicit no-lesion declarations. Turn 3 emits all bbox calls in parallel.

**Custom loss.** `MRIBboxLoss` = weighted CE (tokens inside `<|box_start|>…<|box_end|>` upweighted 3×) + 0.05 × Smooth-L1 on decoded coordinates in thousandths-normalized space. Enable with `--loss_type mri_bbox`.

**Training.** `--agent_template hermes --loss_scale default --max_length 8192`.

### 7.2 Phase 2 — MedSAM2 propagation + point refinement

**Goal.** Extend Phase 1 with a one-shot `run_medsam2(key_z, bbox)` call that seeds MedSAM2 on the key slice and propagates forward/backward through all 10 sampled slices. Slices with reported Dice < 0.70 trigger `add_point(z_index, points, labels)` refinement.

**MedSAM2 propagation protocol** (three separate init/propagate/reset cycles to avoid state contamination):

1. **Phase A** — seed bbox on `key_z`, extract key-slice mask, reset.
2. **Phase B** — init with key_mask, `propagate_in_video(reverse=False)` from `key_z`, reset.
3. **Phase C** — init with key_mask, `propagate_in_video(reverse=True)` from `key_z`, reset.

**Two non-negotiable implementation details** (both burned into carry-over code):

- `apply_postprocessing=False` on `build_sam2_video_predictor_npz`. The default `fill_holes_in_mask_scores` CUDA kernel triggers illegal memory access on this hardware.
- Absolute Hydra config paths need a `//` prefix: `if os.path.isabs(cfg_path) and not cfg_path.startswith('//'): cfg_path = '//' + cfg_path`.

**Token budget.** Render images (bbox overlays, mask overlays, refinement renders) are saved at **256×256** (~84 vision tokens each). Input slices remain 512×512 (~333 tokens). Training runs at `--max_length 12288`.

**Results (checkpoint-300, val n=33).**

| Metric | Mean ± Std | Min / Max |
|---|---|---|
| Dice | 0.826 ± 0.127 | 0.303 / 0.953 |
| Precision | 0.849 ± 0.151 | 0.186 / 0.991 |
| Recall | 0.826 ± 0.110 | 0.530 / 0.978 |
| HD95 (px) | 9.50 ± 12.91 | 1.0 / 48.6 |

---

## 8. Phase 3 — Full Spec

### 8.1 Goal

Remove the Turn-1 bird's-eye view. The agent receives only the task description and the 10-element sampled `Z_list`; slices load on demand via **`scroll`** (relative step) and **`get_slice`** (absolute jump). The segmentation scope is the **same 10 sampled Z** as Phase 2 — only the perceptual workflow changes. The agent must build its spatial model of the volume from sequential single-slice reads and accumulated CoT, mimicking DICOM scrolling.

### 8.2 Tool Inventory

Six tools total. The two new ones are fully specified below; the other four keep their Phase 2 semantics. Full machine-readable schema: `Qwen3_VL/convert_to_agent_trajectory_phase3.py` line 83. Full reference: `Qwen3_VL/PHASE3_TOOLS.md`.

| # | Tool | Role |
|---|---|---|
| 1 | `get_slice(z_index)` | Absolute-jump read. Sets `i_cur` to the given ordinal. |
| 2 | `scroll(delta)` | Relative move, `delta ∈ ±9 \ {0}`. Out-of-bounds clamps, not errors. |
| 3 | `add_bbox(z_index, bbox)` | Requires `z_index == Z_list[i_cur]` (read-before-write). |
| 4 | `run_medsam2(key_z, bbox)` | Fires exactly once, after all lesion ordinals annotated. |
| 5 | `add_point(z_index, points, labels)` | Point refinement. Same read-before-write rule. |
| 6 | `finish_3d_segmentation()` | Terminal. Last tool call, unique per trajectory. |

### 8.3 Navigation State

| Concept | Definition |
|---|---|
| `Z_list` | 10 Z indices from `np.linspace(z_min, z_max, 10, dtype=int)`. Echoed in Turn 1, immutable thereafter. |
| Ordinal `i ∈ {0..9}` | Position within `Z_list`. All navigation is expressed over ordinals internally. |
| `i_cur` | Environment pointer to the "on-screen" slice. Undefined before first `get_slice`; mutated by every navigation call. |
| Per-slice overlay | `raw` / `bbox` / `mask` / `refined_mask`. Reads bake the most recent overlay onto the returned image. |

### 8.4 Navigation Tool-Response Schema

Both `get_slice` and `scroll` return this shape (one `<image>` token per call):

```json
{
  "z_index": 55,
  "ordinal": 5,
  "slice_image": "<image>",
  "sampled_z_list": [40, 43, 46, 50, 53, 56, 60, 63, 66, 70],
  "overlays": {
    "has_bbox": true,  "bbox": [148, 112, 172, 138],
    "has_mask": true,  "mask_dice": 0.81,
    "has_refined_mask": false
  },
  "boundary": {"at_start": false, "at_end": false, "clamped": false},
  "history": {
    "visited_ordinals":   [5, 4, 6, 3],
    "annotated_ordinals": [5, 4, 6],
    "unvisited_ordinals": [0, 1, 2, 7, 8, 9]
  }
}
```

The `history` block is the model's **only memory** of prior navigation. Spatial-reasoning CoT must cite it explicitly rather than re-derive state from context.

### 8.5 Phase A / B / C Workflow

Every Phase 3 trajectory follows this three-phase structure. The SFT generator enforces it; the RL policy is initialized from it.

| Phase | Purpose | Dominant tool | Exit condition |
|---|---|---|---|
| **A — Exploration** | Locate the key slice, build initial spatial model | `get_slice` (1–2), then `scroll ±1 / ±2` | Key slice announced in CoT |
| **B — Annotation** | Bbox every lesion-containing ordinal | `scroll` interleaved with `add_bbox`; `get_slice` only for jumps ≥ 3 | `unvisited_ordinals` empty |
| **C — Mask review** | Inspect each mask, refine poor ones | Sequential `scroll +1` sweep from ordinal 0; `get_slice` for flagged non-adjacent slices | All Dice ≥ 0.70 or agent decides done |

Transitions: A→B on first `add_bbox`; B→C on the single `run_medsam2`; C→end on `finish_3d_segmentation`.

### 8.6 Decision Heuristics (learnable preferences)

1. **Start middle, widen outward.** First `get_slice` targets ordinal 5; probes fan out with `scroll` `-1, +1, -2, +2`.
2. **`scroll` for adjacency, `get_slice` for jumps.** `|delta| ≤ 2` → `scroll`; `|delta| ≥ 3` → `get_slice`.
3. **No revisit without state change.** A visited ordinal is re-read only after `run_medsam2` or `add_point` changed its overlay.
4. **Online key-slice choice.** Unlike Phase 2's offline `argmax(areas)`, the model compares bbox sizes across visited slices in CoT and commits to `key_z` before `run_medsam2`.
5. **Phase C default sweep.** `get_slice(Z_list[0])` → `scroll(+1)` × 9, broken by `get_slice` only for non-adjacent poor slices.

### 8.7 State Invariants (environment-enforced)

1. `i_cur ∈ {0..9}` once initialized; undefined before first `get_slice`.
2. `Z_list` immutable after Turn 1.
3. Every `<image>` token in the response stream matches an entry in the `images` list, in order.
4. `add_bbox` / `add_point` with `z_index != Z_list[i_cur]` are rejected.
5. `run_medsam2` fires exactly once, after Phase B completes.
6. `finish_3d_segmentation` is the last tool call and unique.

Violations during inference return error tool_responses; violations in training data are filtered by `validate_entry` in the generator.

### 8.8 Dataset

260 train / 26 val trajectories. Average calls per sample: `get_slice` ≈ 9, `scroll` ≈ 16, `add_bbox` ≈ 9.4, `run_medsam2` = 1, `add_point` ≈ 1.6–2.4, `finish_3d_segmentation` = 1. Trajectory mean length ~40 tool calls. Training runs at `--max_length 16384 --agent_template hermes`.

---

## 9. Phase 3 RL Plan

RL is a post-SFT polish stage, not a replacement. It is triggered **only after** Phase 3 SFT reaches val Dice ≥ 0.75 and format-violation rate ≤ 2% — the KL-to-SFT regularizer only works if the SFT policy is actually competent.

### 9.1 Algorithm: GRPO

GRPO (Group Relative Policy Optimization) is selected over PPO / DPO for three project-specific reasons:
** you can read the following: to use GRPO, you need to know the following: **./Complete Multimodal GRPO Experiment** **./GRPO-code-Training.md** **./GRPO.md** **
1. **Reward is verifiable.** Dice vs. GT is deterministic and cheap. No learned value/reward model needed.
2. **Memory budget.** GPUs 0 + 7 already hold Qwen3-VL-8B (LoRA), MedSAM2, and KV caches. GRPO drops the critic.
3. **Long multi-image trajectories.** Group-mean advantage on the final reward sidesteps the credit-assignment problem PPO hits at 40+ turns with vision observations.

**Hyperparameters.**

```
Group size K          = 8 rollouts per prompt
Sampling temperature  = 0.8–1.0
KL to SFT policy      β_kl = 0.01–0.04 (stage-dependent, see §9.5)
LoRA learning rate    = 5e-7  (lower than SFT)
Rollout cap           = 45 tool calls per trajectory (trajectory terminated if exceeded)
```

DPO / step-DPO is a fallback if rollout cost proves prohibitive — can be constructed offline from teacher trajectories vs. current-model rollouts — but it is a warm-up, not a replacement for outcome-driven GRPO.

### 9.2 Reward Structure — three layers

```
R(trajectory) = gate · [ α · R_outcome  +  Σ β_i · R_shape_i ]
```

| Layer | Role | Weight | Hackable? |
|---|---|---|---|
| Outcome | Dominant objective (3D Dice) | α = 1.0 | No — GT-verified |
| Process shaping | Dense credit-assignment for sub-goals | β_i ≈ 0.05–0.2 | Yes — designed against specific failure modes |
| Format gates | Trajectory validity | Multiplicative {0, 1} | N/A — hard binary |

The core discipline: **every shaping term must be a closed-form function of the trajectory and GT.** If it can't be written as such, it doesn't go in.

### 9.3 Outcome Layer

Two components, combined 0.7 / 0.3:

**`R_dice`** — mean per-slice Dice on the 10 sampled Z, computed after all refinements (post-`add_point`, right before `finish_3d_segmentation`):

```
R_dice = (1/10) · Σ_i Dice(pred_mask_i, gt_mask_i)
```

**`R_key_dice`** — Dice on the ordinal the model itself declared as `key_z`. Binds online key-slice CoT to outcome; without it, the agent can pick any key_z and mean Dice averages it away. The `key_z` used is the one committed in the `run_medsam2` call, **not** any post-hoc re-declaration:

```
R_key_dice = Dice(pred_mask[key_ord_chosen], gt_mask[key_ord_chosen])
```

```
R_outcome = 0.7 · R_dice + 0.3 · R_key_dice
```

**Do not use** `R_dice_delta_vs_SFT`. GRPO's intra-group relative advantage is the correct version of that intuition.

**Optional** `R_surface` (boundary Dice or `1 − normalized_HD95`). Add only if HD95 is a first-class evaluation metric — it occasionally fights `R_dice` on small lesions.

### 9.4 Process Shaping Layer

Five terms. Weights summed into `R_shape` via `Σ β_i · R_shape_i`.

**`R_bbox_iou` (β = 0.20)** — per-call bbox quality, the densest and safest signal in the plan. Gives per-step credit across Phase B, where nothing else does:

```
R_bbox_iou_t = IoU(b_t, gt_bbox(z_t))        for each add_bbox call t
R_bbox_iou   = mean_t R_bbox_iou_t
```

**`R_dice_gain` (β = 0.15)** — reward refinement by Dice improvement, not by count. Each `add_point` term clipped at 0.3 so the agent can't game it by starting from a deliberately bad mask. `max(0, ·)` avoids double-penalizing failed refinements (outcome layer already handles that):

```
R_dice_gain = Σ_t max(0, min(0.3,  dice_after_t − dice_before_t))
```

**`R_coverage` (β = 0.10)** — lesion-ordinal recall with a false-positive penalty. The FP term prevents the collapse-mode "annotate every ordinal":

```
R_coverage = |annotated ∩ lesion_ordinals| / |lesion_ordinals|
             − 0.5 · |annotated ∩ non_lesion_ordinals|
```

**`R_nav_style` (β = 0.05)** — small structural reward for the two-tool division of labor. Kept small because it's a style prior, not the objective. Scheduled strong→weak (§9.5):

```
For each navigation call, Δ = |target_ord − i_cur_before|:
  scroll, Δ ≤ 2            → +0.01
  get_slice, Δ ≥ 3         → +0.01
  scroll that would clamp  → −0.02
  get_slice re-read, overlay unchanged since last read → −0.05
```

**`R_efficiency`** — soft cap, not per-step penalty. A per-call negative reward would encourage skipping refinements.

```
R_efficiency = −0.01 · max(0, n_tool_calls − 45)
```

### 9.5 Format Gates (hard)

Multiplicative gate on the total reward. `gate = 1` iff **all** of:

- Every `add_bbox` / `add_point` satisfies `z_index == Z_list[i_cur]`.
- `run_medsam2` fires exactly once, after all lesion ordinals annotated.
- `key_z ∈ Z_list`.
- `finish_3d_segmentation` is the last tool call and unique.
- No `scroll` before the first `get_slice`.
- Every tool call parses as valid JSON against the declared schema.

Otherwise `gate = 0`. GRPO tolerates this because advantage is computed within-group — as long as some group members are valid, there is a learning signal. If entire groups go invalid early in training, soften to `gate = 0.1` temporarily.

**Do not** put "no unnecessary revisit" or "no boundary clamp" in the gate — those are stylistic and belong in `R_nav_style`.

### 9.6 Combined Reward Formula

```
R(trajectory) = gate · [
      1.00 · (0.7 · R_dice + 0.3 · R_key_dice)
    + 0.20 · R_bbox_iou
    + 0.15 · R_dice_gain
    + 0.10 · R_coverage
    + 0.05 · R_nav_style
    + 1.00 · R_efficiency
]
```

Expected scale: teacher-quality trajectory ≈ 0.85, strong rollout 0.90–0.95, malformed 0.

**Log every component separately.** The primary diagnostic: `R_dice` trending up while `R_coverage` stays near 1 and `R_nav_style` does not regress. Any divergence is an early reward-hacking warning.

### 9.7 Curriculum — three stages

| Stage | Steps | Shaping weights | KL to SFT | Goal |
|---|---|---|---|---|
| 1 — Warmup | ~1k | **2×** baseline | β_kl = 0.04 | Stabilize on-policy at near-SFT format |
| 2 — Main | 5k–10k | Baseline (as in §9.6) | β_kl = 0.01–0.02 | Outcome-dominant training |
| 3 — Polish (optional) | 1k–2k | Keep only `R_bbox_iou`, zero the rest | β_kl = 0.01 | Test whether policy stands on pure outcome |

If Stage 3 degrades Dice, roll back — the shaping was not fully redundant.

### 9.8 Reward Hacking — known threats and fixes

Empirically observed failure modes in agentic VLM RL and the term that neutralizes each:

| Threat | Fix |
|---|---|
| Collapse to "annotate every ordinal" | `λ_fp = 0.5` term in `R_coverage`; `R_bbox_iou = 0` on non-lesion slices |
| Key-slice laundering (post-hoc pick) | `R_key_dice` uses the `key_z` committed in `run_medsam2`, not a re-declaration |
| Deliberate bad bbox to farm `R_dice_gain` | `R_dice_gain` clipped per-step; `R_bbox_iou` rewards good initial bboxes — two terms pull opposite directions |
| Trajectory-length inflation | `R_nav_style` rewards correct tool choice per move, not frequency; `R_efficiency` caps total length |
| Schema-valid but degenerate trajectories (scroll back and forth) | No shaping credit for movement without annotation or Dice gain; monitor "tool calls per annotation" |

### 9.9 Infrastructure Notes

**Rollout cost is dominated by MedSAM2.** With group size 8 and one `run_medsam2` per trajectory, one group ≈ 1–3 min of MedSAM2 wall-clock. Recommended layout: policy forward + sampling on GPU 0, MedSAM2 serialized on GPU 7. Do **not** run MedSAM2 concurrently on both GPUs during training — the optimizer step will deadlock.

**Reward caching.** All reward components are deterministic functions of `(trajectory, GT)`. Hash the tool-call sequence keyed to case ID and cache — early training produces far more duplicate trajectories than you'd expect.

**Vision-token pressure.** `max_length=16384` is already tight for 40-turn Phase 3 trajectories. RL tends to produce longer trajectories than SFT. Monitor truncation rate; if > 5%, either bump `max_length` or cap rollout tool-calls at 35. Truncation silently corrupts reward computation because format gates will fail on incomplete trajectories.

**Warm start.** Required: Phase 3 SFT checkpoint with val Dice ≥ 0.75 and format-violation rate ≤ 2%. RL cannot substitute for adequate SFT.

### 9.10 RL-time Evaluation

Logged every 100–200 optimizer steps on the 26-sample val set.

- **Primary:** mean 3D Dice, key-slice Dice, HD95.
- **Secondary:** format-violation rate, mean tool calls, mean `add_point` count, efficiency-cap hit rate, `R_bbox_iou` distribution.
- **Diagnostic:** per-phase time budget (A/B/C), boundary-clamp rate, revisit-without-state-change rate.

Expected trajectory: primary metrics improve; diagnostic metrics stay near SFT values. Divergence in diagnostics while primary improves is the signature of reward hacking — revisit shaping weights.

---

## 10. Loss Computation Scope (all phases)

| Content | Contributes to loss? |
|---|---|
| Assistant CoT (phase transitions, reviews) | Yes |
| Every `tool_call` (including navigation) | Yes |
| `finish_3d_segmentation` | Yes |
| `tool_response` (all renders, mask batches, nav reads) | No |
| `user` (task instruction, Z_list) | No |

RL uses the same generative-only scope — the reward is applied to the full trajectory but the gradient flows only through the policy's own outputs.

---

## 11. JSONL Format (all phases)

Top-level fields: `tools`, `messages`, `images`. The `images` list contains paths in the exact order that `<image>` tokens appear across all messages (input slices → nav reads → bbox renders → mask batch → refinement renders).

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

---

## 12. Cross-references

- **Phase 3 tool contract (authoritative):** `Qwen3_VL/PHASE3_TOOLS.md`
- **Phase 3 tool schema (code):** `Qwen3_VL/convert_to_agent_trajectory_phase3.py:83`
- **Phase 3 inference engine:** `Qwen3_VL/infer_phase3.py` → `NavAgentExecutor`
- **Phase 3 visualizer:** `Qwen3_VL/visualize_phase3.py`
- **Phase 2 full workflow:** `Qwen3_VL/PHASE2_WORKFLOW.md`
- **ms-swift agent framework notes:** `Agent-support.md`
