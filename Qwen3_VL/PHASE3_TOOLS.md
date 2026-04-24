# Phase 3 Tool Reference

Phase 3 registers **six tools** with the ms-swift `hermes` agent template. Two are new in Phase 3 (the navigation pair), four are carried over from Phase 2 with unchanged semantics. This document describes what each tool does, when the agent is expected to call it, the parameter contract, the environment-returned `tool_response`, and the invariants the training trajectories enforce.

Authoritative schema: `Qwen3_VL/convert_to_agent_trajectory_phase3.py` — TOOLS list at line 83.

---

## 1. Tool inventory

| # | Name | New in Phase 3 | Role |
|---|---|---|---|
| 1 | `get_slice` | yes | Absolute-jump navigation: read one sampled Z slice |
| 2 | `scroll`    | yes | Relative-step navigation: move the view pointer by `±delta` |
| 3 | `add_bbox`  | no  | Record a 2D bounding box on the slice currently being viewed |
| 4 | `run_medsam2` | no | Propagate a mask through the volume from one key slice (MedSAM2) |
| 5 | `add_point` | no  | Refine one slice's mask with foreground / background point prompts |
| 6 | `finish_3d_segmentation` | no | Terminate the trajectory |

Phase 1 registered only tools 3 + 6, Phase 2 registered 3–6, Phase 3 adds 1–2 on top so the agent loads slices on demand instead of receiving all ten upfront.

---

## 2. Shared concepts

| Concept | Definition |
|---|---|
| `Z_list` | Ten Z indices produced by `np.linspace(z_min, z_max, 10, dtype=int)` and echoed verbatim in the opening user turn. Immutable for the whole trajectory. |
| Ordinal `i ∈ {0..9}` | Index of a slice within `Z_list`; `Z_list[i]` is its Z coordinate. All navigation is internally in ordinal space. |
| Pointer `i_cur` | The "currently visible" ordinal. Undefined before the first `get_slice`; mutated by `get_slice` / `scroll`. |
| Overlay state | Per ordinal, one of `raw / bbox / mask / refined_mask`. Navigation reads bake the most recent overlay onto the slice image returned to the model. |
| Scope | Navigation and annotation are restricted to the 10 sampled Z. Non-sampled Z are unreachable. |

---

## 3. Tool specifications

### 3.1 `get_slice(z_index)`

**Purpose.** Random-access slice read. Used for initial exploration (e.g. land on the middle ordinal first), jumping ≥ 3 ordinals, or re-inspecting a slice flagged by the mask review.

**Parameters.**
- `z_index: int` — must be a member of the sampled Z list.

**Effect.** Sets `i_cur = Z_list.index(z_index)`; appends `i_cur` to the visited history; returns the rendered view.

**Validation.**
- `z_index` not in `Z_list` → error tool_response with the sampled list echoed back; `i_cur` unchanged.
- Re-reading the current slice is permitted (e.g. after `run_medsam2` or `add_point` changed its overlay).

### 3.2 `scroll(delta)`

**Purpose.** Relative navigation. Used for neighbour inspection (`±1`), short hops (`±2`), and the sequential mask-review sweep after MedSAM2.

**Parameters.**
- `delta: int` — step in sampled-list ordering. Allowed range `±9`, excluding `0`.

**Effect.** Sets `i_cur = clamp(i_cur + delta, 0, 9)`; appends to visited; returns the rendered view. Out-of-bounds scrolls are **clamped, not rejected**; the response's `boundary.clamped` field is set to `true` so the model learns to recognise boundaries through observation rather than error handling.

**Validation.**
- `delta == 0` is rejected.
- `scroll` before any `get_slice` is rejected — the pointer is undefined. The environment returns `{"error": "pointer not initialized — call get_slice first"}`.

### 3.3 `add_bbox(z_index, bbox)`

**Purpose.** Record the 2D bounding box of the lesion on the slice the agent is currently viewing.

**Parameters.**
- `z_index: int` — must equal `Z_list[i_cur]` (read-before-write).
- `bbox: [x1, y1, x2, y2]` — four integers in original-image pixel coordinates, top-left / bottom-right convention.

**Effect.** Stores the bbox for ordinal `i_cur`; marks the ordinal annotated; updates its overlay state to `bbox`. Subsequent navigation reads of this ordinal return the slice with the box drawn on.

**Validation.** Rejected unless `z_index == Z_list[i_cur]`. This is the central mechanism that forces the model to read before writing.

### 3.4 `run_medsam2(key_z, bbox)`

**Purpose.** Single-shot 3D propagation. MedSAM2 seeds a mask on the `key_z` slice from `bbox`, then propagates forward and backward through the entire volume.

**Parameters.**
- `key_z: int` — the chosen key slice (typically the largest cross-section).
- `bbox: [x1, y1, x2, y2]` — the bbox on `key_z`. In well-formed trajectories this equals the `add_bbox` previously recorded for that ordinal.

**Effect.** Environment runs the three-phase init/propagate/reset cycle from Phase 2 (`Qwen3_VL/PHASE2_WORKFLOW.md`). All ten sampled ordinals receive a `mask` overlay; each one's Dice-vs-GT is reported in the tool_response. `i_cur` is unchanged.

**Validation.**
- Fires **exactly once** per trajectory.
- Must be preceded by `add_bbox` on every lesion-containing ordinal.
- `apply_postprocessing=False` on the predictor — the default CUDA post-processing kernel crashes on this hardware.

### 3.5 `add_point(z_index, points, labels)`

**Purpose.** Correct a poor-quality slice mask using foreground / background clicks. Triggered after `run_medsam2` for any ordinal whose reported Dice is < 0.70.

**Parameters.**
- `z_index: int` — must equal `Z_list[i_cur]` (same read-before-write rule as `add_bbox`).
- `points: [[x, y], ...]` — pixel coordinates.
- `labels: [int, ...]` — one label per point, `1` = foreground, `0` = background. Length must match `points`.

**Effect.** Re-runs MedSAM2 on the target slice with the given point prompts; stores the refined mask; sets that ordinal's overlay to `refined_mask`.

**Validation.** Rejected unless `z_index == Z_list[i_cur]`.

### 3.6 `finish_3d_segmentation()`

**Purpose.** Terminator. Signals "segmentation and all refinements are complete".

**Parameters.** None.

**Effect.** The agent loop stops after this call.

**Validation.** Must be the last tool_call in every trajectory (unique).

---

## 4. Navigation `tool_response` contract (both `get_slice` and `scroll`)

Both navigation tools return the same JSON schema, packed into a single tool_response that contains one `<image>` token:

```json
{
  "z_index": 55,
  "ordinal": 5,
  "slice_image": "<image>",
  "sampled_z_list": [40, 43, 46, 50, 53, 56, 60, 63, 66, 70],
  "overlays": {
    "has_bbox": true,
    "bbox": [148, 112, 172, 138],
    "has_mask": true,
    "mask_dice": 0.81,
    "has_refined_mask": false
  },
  "boundary": {
    "at_start": false,
    "at_end": false,
    "clamped": false
  },
  "history": {
    "visited_ordinals":   [5, 4, 6, 3],
    "annotated_ordinals": [5, 4, 6],
    "unvisited_ordinals": [0, 1, 2, 7, 8, 9]
  }
}
```

The rendered `slice_image` is the raw 512-pixel slice with the most recent overlay baked in (bbox stroke, translucent mask fill, point markers). All renders are downsampled to **256×256** before serialisation to stay within the vision-token budget (~84 tokens/image, a Phase 2 convention).

The `history` block is the model's **only** memory of where it has been and what is annotated; spatial-reasoning CoT is expected to cite it explicitly rather than re-derive state.

The other four tools return short JSON tool_responses:
- `add_bbox` → `{"z_index", "bbox_image": "<image>", "iou_with_gt"}`
- `run_medsam2` → `{"status": "propagation_complete", "key_z", "slices": [{"z_index", "ordinal", "mask_image": "<image>", "dice_with_gt"}, ...]}` — one `<image>` token per sampled slice
- `add_point` → `{"z_index", "mask_image": "<image>", "dice_with_gt"}`
- `finish_3d_segmentation` → terminal, no response

---

## 5. Phase A / B / C: when each tool is called

Every Phase 3 trajectory follows the same three-phase navigation plan. The trajectory generator emits this structure and the model learns it through SFT.

| Phase | Purpose | Dominant tool | Typical calls | Exit condition |
|---|---|---|---|---|
| **A — Exploration** | Locate the key slice; build the initial spatial model | `get_slice` (1–2 calls), then `scroll ±1 / ±2` | 3–5 reads | Key slice announced in CoT |
| **B — Annotation** | Cover every lesion slice with a bbox | `scroll` interleaved with `add_bbox`; `get_slice` only for jumps ≥ 3 | `unvisited_ordinals` empty → 10 reads, up to 10 bboxes | All lesion ordinals annotated |
| **C — Mask review** | After `run_medsam2`, inspect per-slice masks and refine poor ones | Sequential `scroll +1` sweep from ordinal 0, plus `get_slice` for flagged slices | 2–10 reads, 0–4 `add_point` | All Dice ≥ 0.70 or agent deems done |

Transitions:
- A → B: triggered by the first `add_bbox` in the trajectory.
- B → C: the single `run_medsam2` tool_call.
- C → end: `finish_3d_segmentation`.

Representative dataset statistics (260 train / 26 val trajectories):

| Tool | Avg calls / sample |
|---|---|
| `get_slice` | 9.0 |
| `scroll` | 16.0 |
| `add_bbox` | 9.4 |
| `run_medsam2` | 1.0 |
| `add_point` | 1.6–2.4 |
| `finish_3d_segmentation` | 1.0 |

---

## 6. Decision heuristics the trajectories encode

These are not hard rules but preferences that the supervised data rewards:

1. **Start middle, widen outward.** Phase A's first `get_slice` targets ordinal 5; probes then fan out with `scroll` `-2, +1, +2, +1`, covering `{3,4,5,6,7}` with zero revisits.
2. **`scroll` for adjacency, `get_slice` for jumps.** `|delta| ≤ 2` → `scroll`; `|delta| ≥ 3` → `get_slice`. This encodes the two-tool division of labor.
3. **Never revisit unless state changed.** A slice already in `visited_ordinals` is revisited only after `run_medsam2` or `add_point` updated its overlay.
4. **Online key-slice choice.** Unlike Phase 2's offline `argmax(areas)`, the model compares bbox sizes across visited slices in CoT and commits to `key_z` before `run_medsam2`.
5. **Phase C default sweep.** `get_slice(Z_list[0])` → `scroll(+1)` × 9. The sweep is broken only by `get_slice` to re-visit a non-adjacent poor slice.

---

## 7. State invariants (environment-enforced)

The environment validates these across the whole trajectory:

1. `i_cur ∈ {0..9}` once initialised; undefined before the first `get_slice`.
2. `Z_list` is immutable after Turn 1.
3. Every `<image>` token in the tool_response stream corresponds to exactly one of: navigation read, `add_bbox` render, `run_medsam2` mask batch, `add_point` refinement render. The ordered sequence of `<image>` tokens in `messages` matches the `images` list entry-by-entry.
4. `add_bbox` / `add_point` without matching `i_cur` are rejected.
5. `run_medsam2` fires exactly once, after Phase B completes.
6. `finish_3d_segmentation` is the last tool_call and unique per trajectory.

Violations during inference trigger the error tool_responses listed above; violations in training data would be dataset bugs (filtered by `validate_entry` in the generator).

---

## 8. Loss computation scope

Same rule as Phase 1/2:

| Content | Contributes to loss? |
|---|---|
| Assistant CoT (phase transitions, reviews) | yes |
| Every `tool_call` (including navigation) | yes |
| `finish_3d_segmentation` | yes |
| `tool_response` (navigation reads, renders, mask batches) | no |
| `user` (opening instruction + sampled Z list) | no |

Only generative outputs are supervised; the environment's observations are context, not targets.

---

## 9. Cross-references

- Tool schema source of truth: `Qwen3_VL/convert_to_agent_trajectory_phase3.py:83`
- Inference-side implementation: `Qwen3_VL/infer_phase3.py` → `NavAgentExecutor`
- Oracle-pipeline visualiser: `Qwen3_VL/visualize_phase3.py`
- Training script: `Qwen3_VL/train_phase3.sh` (uses `--agent_template hermes`, `--max_length 16384`)
- Phase 2 workflow (carried-over tool semantics): `Qwen3_VL/PHASE2_WORKFLOW.md`
- Project overview and phased roadmap: `CLAUDE.md`
