import argparse
import ast
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault('VIDEO_MAX_TOKEN_NUM', '128')
os.environ.setdefault('FPS_MAX_FRAMES', '16')
os.environ.setdefault('QWENVL_BBOX_FORMAT', 'new')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5,6')

import torch
from PIL import Image
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

DEFAULT_MODEL = '/BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct'
DEFAULT_JSONL = str(Path(__file__).with_name('mri_grounding_val.jsonl'))
BOX_START_PATTERN = re.compile(
    r'<\|box_start\|>\s*\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)\s*,\s*\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)\s*<\|box_end\|>'
)
BBOX_2D_PATTERN = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]'
)
PLAIN_BOX_PATTERN = re.compile(
    r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]'
)


@dataclass
class SampleResult:
    index: int
    image: str
    gt_boxes: List[List[float]]
    pred_boxes: List[List[float]]
    response: str
    matched_ious: List[float]
    matched_gt: int
    matched_pred: int
    parse_error: Optional[str] = None
    latency_sec: Optional[float] = None


@dataclass
class MatchSummary:
    matched_pairs: int
    tp: int
    fp: int
    fn: int
    matched_ious: List[float]


@dataclass
class MetricsSummary:
    num_samples: int
    total_gt: int
    total_pred: int
    matched_pairs: int
    mean_iou_matched: float
    mean_iou_per_gt: float
    precision_at_50: float
    recall_at_50: float
    f1_at_50: float
    ap50: float
    ap75: float
    map_50_95: float
    avg_latency_sec: Optional[float]
    parse_failures: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Qwen3-VL grounding inference on a JSONL file and report IoU / AP metrics.'
    )
    parser.add_argument('--jsonl', default=DEFAULT_JSONL, help='Validation jsonl path.')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Base model path.')
    parser.add_argument('--ckpt', required=True, help='Adapter / checkpoint path.')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of samples per infer batch.')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap on number of jsonl rows.')
    parser.add_argument('--max-tokens', type=int, default=256, help='Generation max_tokens.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature.')
    parser.add_argument('--device', default=None, help='Optional CUDA_VISIBLE_DEVICES override.')
    parser.add_argument(
        '--coord-mode',
        choices=['auto', 'real', 'norm1', 'norm1000'],
        default='norm1000',
        help='How to scale predicted bbox coordinates back to image space.',
    )
    parser.add_argument('--save-preds', default=None, help='Optional path to save per-sample predictions JSONL.')
    parser.add_argument('--print-every', type=int, default=50, help='Progress interval.')
    return parser.parse_args()


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def chunked(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def get_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def sanitize_box(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def maybe_scale_box(box: Sequence[float], width: int, height: int, mode: str) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    chosen_mode = mode
    if mode == 'auto':
        max_coord = max(abs(v) for v in (x1, y1, x2, y2))
        if max_coord <= 1.5:
            chosen_mode = 'norm1'
        else:
            chosen_mode = 'real'
    if chosen_mode == 'norm1':
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    if chosen_mode == 'norm1000':
        return [x1 * 0.512, y1 * 0.512, x2 * 0.512, y2 * 0.512]
    return [x1, y1, x2, y2]


def try_parse_json_boxes(text: str) -> Optional[List[List[float]]]:
    start = text.find('[')
    end = text.rfind(']')
    if start < 0 or end <= start:
        return None
    snippet = text[start:end + 1]
    candidates = []
    for loader in (json.loads, ast.literal_eval):
        try:
            candidates.append(loader(snippet))
        except Exception:
            continue
    for parsed in candidates:
        if not isinstance(parsed, list):
            continue
        boxes: List[List[float]] = []
        valid = True
        for item in parsed:
            if isinstance(item, dict):
                raw_box = item.get('bbox_2d') or item.get('bbox') or item.get('box')
            else:
                raw_box = item
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                valid = False
                break
            boxes.append([float(v) for v in raw_box])
        if valid:
            return boxes
    return None


def parse_pred_boxes(text: str, width: int, height: int, coord_mode: str) -> Tuple[List[List[float]], Optional[str]]:
    boxes: List[List[float]] = []
    parsed_explicitly = False

    json_boxes = try_parse_json_boxes(text)
    if json_boxes is not None:
        boxes = json_boxes
        parsed_explicitly = True
    else:
        for match in BOX_START_PATTERN.findall(text):
            boxes.append([float(v) for v in match])
        if boxes:
            parsed_explicitly = True
        if not boxes:
            for match in BBOX_2D_PATTERN.findall(text):
                boxes.append([float(v) for v in match])
        if boxes:
            parsed_explicitly = True
        if not boxes:
            stripped = text.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                plain_matches = PLAIN_BOX_PATTERN.findall(text)
                if plain_matches:
                    if len(plain_matches) == 1 and stripped.count('[') == 1:
                        boxes.append([float(v) for v in plain_matches[0]])
                    else:
                        for match in plain_matches:
                            boxes.append([float(v) for v in match])
                    parsed_explicitly = True

    scaled_boxes: List[List[float]] = []
    for box in boxes:
        candidate = sanitize_box(maybe_scale_box(box, width, height, coord_mode), width, height)
        if candidate is not None:
            scaled_boxes.append(candidate)

    stripped = text.strip()
    parse_error = None
    if not scaled_boxes and not parsed_explicitly and 'no match' not in text.lower() and 'none' not in text.lower() and stripped not in {'[]', '[ ]'}:
        parse_error = 'no_bbox_parsed'
    return scaled_boxes, parse_error


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match(gt_boxes: Sequence[Sequence[float]], pred_boxes: Sequence[Sequence[float]]) -> MatchSummary:
    candidates: List[Tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            candidates.append((bbox_iou(gt_box, pred_box), gt_idx, pred_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_gt = set()
    used_pred = set()
    matched_ious: List[float] = []
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matched_ious.append(iou)

    matched_pairs = len(matched_ious)
    return MatchSummary(
        matched_pairs=matched_pairs,
        tp=0,
        fp=len(pred_boxes),
        fn=len(gt_boxes),
        matched_ious=matched_ious,
    )


def compute_ap_from_ious(matched_ious: Sequence[float], total_gt: int, total_pred: int, threshold: float) -> float:
    if total_gt == 0:
        return 1.0 if total_pred == 0 else 0.0
    hits = sorted((1 if iou >= threshold else 0) for iou in matched_ious)
    hits.extend([0] * max(0, total_pred - len(matched_ious)))
    hits = list(reversed(hits))
    if not hits:
        return 0.0

    tp = 0
    fp = 0
    precisions: List[float] = []
    recalls: List[float] = []
    for hit in hits:
        if hit:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / total_gt)

    ap = 0.0
    prev_recall = 0.0
    precision_envelope = 0.0
    for precision, recall in sorted(zip(precisions, recalls), key=lambda item: item[1]):
        precision_envelope = max(precision_envelope, precision)
        ap += precision_envelope * max(0.0, recall - prev_recall)
        prev_recall = recall
    return ap


def summarize_metrics(results: Sequence[SampleResult]) -> MetricsSummary:
    total_gt = sum(len(r.gt_boxes) for r in results)
    total_pred = sum(len(r.pred_boxes) for r in results)
    all_ious = [iou for r in results for iou in r.matched_ious]
    matched_pairs = len(all_ious)
    parse_failures = sum(1 for r in results if r.parse_error is not None)

    tp50 = sum(sum(1 for iou in r.matched_ious if iou >= 0.5) for r in results)
    precision50 = tp50 / total_pred if total_pred else 0.0
    recall50 = tp50 / total_gt if total_gt else 0.0
    f1_50 = 2 * precision50 * recall50 / (precision50 + recall50) if (precision50 + recall50) else 0.0

    thresholds = [0.5 + 0.05 * i for i in range(10)]
    ap_by_threshold = {
        thresh: compute_ap_from_ious(all_ious, total_gt, total_pred, thresh) for thresh in thresholds
    }
    latencies = [r.latency_sec for r in results if r.latency_sec is not None]

    return MetricsSummary(
        num_samples=len(results),
        total_gt=total_gt,
        total_pred=total_pred,
        matched_pairs=matched_pairs,
        mean_iou_matched=sum(all_ious) / matched_pairs if matched_pairs else 0.0,
        mean_iou_per_gt=sum(all_ious) / total_gt if total_gt else 0.0,
        precision_at_50=precision50,
        recall_at_50=recall50,
        f1_at_50=f1_50,
        ap50=ap_by_threshold[0.5],
        ap75=ap_by_threshold[0.75],
        map_50_95=sum(ap_by_threshold.values()) / len(ap_by_threshold),
        avg_latency_sec=sum(latencies) / len(latencies) if latencies else None,
        parse_failures=parse_failures,
    )


def build_request(sample: Dict[str, Any]) -> InferRequest:
    return InferRequest(
        messages=sample['messages'],
        images=sample.get('images'),
        objects=sample.get('objects'),
    )


def infer_dataset(
    engine: TransformersEngine,
    samples: Sequence[Dict[str, Any]],
    batch_size: int,
    request_config: RequestConfig,
    coord_mode: str,
    print_every: int,
) -> List[SampleResult]:
    results: List[SampleResult] = []
    processed = 0

    for batch in chunked(samples, batch_size):
        requests = [build_request(sample) for sample in batch]
        batch_start = time.perf_counter()
        responses = engine.infer(requests, request_config=request_config)
        batch_latency = time.perf_counter() - batch_start

        for sample, response in zip(batch, responses):
            image_path = sample['images'][0]
            width, height = get_image_size(image_path)
            response_text = response.choices[0].message.content or ''
            pred_boxes, parse_error = parse_pred_boxes(response_text, width, height, coord_mode)
            gt_boxes = [
                sanitize_box(box, width, height)
                for box in sample.get('objects', {}).get('bbox', [])
            ]
            gt_boxes = [box for box in gt_boxes if box is not None]
            match = greedy_match(gt_boxes, pred_boxes)
            results.append(
                SampleResult(
                    index=processed,
                    image=image_path,
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    response=response_text,
                    matched_ious=match.matched_ious,
                    matched_gt=len(gt_boxes),
                    matched_pred=len(pred_boxes),
                    parse_error=parse_error,
                    latency_sec=batch_latency / max(1, len(batch)),
                )
            )
            processed += 1
            if print_every > 0 and processed % print_every == 0:
                print(f'[Progress] processed={processed}/{len(samples)}')

    return results


def maybe_write_predictions(path: Optional[str], results: Sequence[SampleResult]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for result in results:
            record = {
                'index': result.index,
                'image': result.image,
                'gt_boxes': result.gt_boxes,
                'pred_boxes': result.pred_boxes,
                'matched_ious': result.matched_ious,
                'parse_error': result.parse_error,
                'response': result.response,
                'latency_sec': result.latency_sec,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def print_summary(metrics: MetricsSummary) -> None:
    print('\n[Dataset]')
    print(f'  samples:           {metrics.num_samples}')
    print(f'  total_gt_boxes:    {metrics.total_gt}')
    print(f'  total_pred_boxes:  {metrics.total_pred}')
    print(f'  matched_pairs:     {metrics.matched_pairs}')
    print(f'  parse_failures:    {metrics.parse_failures}')
    if metrics.avg_latency_sec is not None:
        print(f'  avg_latency_sec:   {metrics.avg_latency_sec:.4f}')

    print('\n[IoU]')
    print(f'  mean_iou_matched:  {metrics.mean_iou_matched:.4f}')
    print(f'  mean_iou_per_gt:   {metrics.mean_iou_per_gt:.4f}')

    print('\n[Detection @ IoU=0.50]')
    print(f'  precision:         {metrics.precision_at_50:.4f}')
    print(f'  recall:            {metrics.recall_at_50:.4f}')
    print(f'  f1:                {metrics.f1_at_50:.4f}')

    print('\n[AP]')
    print(f'  AP50:              {metrics.ap50:.4f}')
    print(f'  AP75:              {metrics.ap75:.4f}')
    print(f'  mAP50-95:          {metrics.map_50_95:.4f}')


def main() -> None:
    args = parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    jsonl_path = os.path.abspath(args.jsonl)
    samples = load_jsonl(jsonl_path, max_samples=args.max_samples)
    if not samples:
        raise ValueError(f'No samples found in {jsonl_path}')

    print(f'[Config] model={args.model}')
    print(f'[Config] ckpt={args.ckpt}')
    print(f'[Config] jsonl={jsonl_path}')
    print(f'[Config] samples={len(samples)} batch_size={args.batch_size} coord_mode={args.coord_mode}')

    request_config = RequestConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    engine = TransformersEngine(args.model, adapters=[args.ckpt])

    model_baseline_bytes = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        model_baseline_bytes = torch.cuda.memory_allocated()

    start = time.perf_counter()
    results = infer_dataset(
        engine=engine,
        samples=samples,
        batch_size=args.batch_size,
        request_config=request_config,
        coord_mode=args.coord_mode,
        print_every=args.print_every,
    )
    total_elapsed = time.perf_counter() - start

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        if model_baseline_bytes is not None:
            input_vram_gb = (peak_vram_bytes - model_baseline_bytes) / (1024 ** 3)
            print(
                f'\n[Peak VRAM (input only)] {input_vram_gb:.2f} GB '
                f'(peak={peak_vram_bytes / 1024 ** 3:.2f} GB, model={model_baseline_bytes / 1024 ** 3:.2f} GB)'
            )

    metrics = summarize_metrics(results)
    print_summary(metrics)
    print(f'\n[Wall Time] {total_elapsed:.2f}s total, {total_elapsed / len(results):.4f}s/sample')
    maybe_write_predictions(args.save_preds, results)
    if args.save_preds:
        print(f'[Saved] {args.save_preds}')


if __name__ == '__main__':
    main()
