import json
import ast
import re
import numpy as np
from PIL import Image
import os

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def parse_response_to_boxes(response_str):
    if not response_str or not isinstance(response_str, str): return []
    boxes = []
    pattern = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
    matches = re.findall(pattern, response_str)
    
    if matches:
        for match in matches:
            boxes.append([float(x) for x in match])
        return boxes
    return []

def scale_boxes_dynamic(boxes, img_width, img_height, is_yxyx=False):
    """
    根据真实的图片宽高动态反归一化。
    :param is_yxyx: 如果模型输出是 [ymin, xmin, ymax, xmax] 则设为 True
    """
    scaled_boxes = []
    for box in boxes:
        if is_yxyx:
            # 模型输出是 [y1, x1, y2, x2]
            y1, x1, y2, x2 = box
            real_x1 = (x1 / 1000.0) * img_width
            real_y1 = (y1 / 1000.0) * img_height
            real_x2 = (x2 / 1000.0) * img_width
            real_y2 = (y2 / 1000.0) * img_height
        else:
            # 模型输出是 [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            # , y1, x2, y2 = box
            real_x1 = (x1 / 1000.0) * img_width
            real_y1 = (y1 / 1000.0) * img_height
            real_x2 = (x2 / 1000.0) * img_width
            real_y2 = (y2 / 1000.0) * img_height
            
        scaled_boxes.append([real_x1, real_y1, real_x2, real_y2])
    return scaled_boxes

def evaluate(jsonl_path, iou_threshold=0.3, is_yxyx=False):
    total_gt, total_pred, total_tp = 0, 0, 0
    all_ious = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
                
            data = json.loads(line)
            gt_boxes = data.get('gt_boxes', [])
            response_str = data.get('response', '')
            img_path = data.get('image', '')
            
            # --- 核心改动：动态读取图片宽高 ---
            if not os.path.exists(img_path):
                print(f"找不到图片: {img_path}，将跳过此行")
                continue
                
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            # ----------------------------------

            pred_boxes_normalized = parse_response_to_boxes(response_str)
            
            # 动态缩放坐标
            pred_boxes = scale_boxes_dynamic(
                pred_boxes_normalized, 
                img_width, 
                img_height, 
                is_yxyx=is_yxyx  # 控制坐标反转
            )
            
            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)
            
            matched_gt = set()
            for pred_box in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, gt_box in enumerate(gt_boxes):
                    if g_idx in matched_gt: continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                if best_iou >= iou_threshold:
                    matched_gt.add(best_gt_idx)
                    total_tp += 1
                    all_ious.append(best_iou)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    avg_iou = np.mean(all_ious) if all_ious else 0.0

    print("="*45)
    print(f"评估完成 (IoU 阈值: {iou_threshold}, 是否 YXYX: {is_yxyx})")
    print(f"总真实框: {total_gt} | 总预测框: {total_pred} | 匹配数(TP): {total_tp}")
    print(f"精确率: {precision:.4f} | 召回率: {recall:.4f} | F1: {f1_score:.4f}")
    print(f"平均匹配 IoU: {avg_iou:.4f}")
    print("="*45)

if __name__ == "__main__":
    jsonl_file_path = "/home/yxd/OPEN-DTOS-LMM/Qwen3_VL/infer_preds2.jsonl" 
    
    # 建议你先运行这个，如果结果还是差，就把 is_yxyx 改为 True 再试一次！
    evaluate(jsonl_file_path, iou_threshold=0.005, is_yxyx=False)