import json
import re
import numpy as np

def parse_response_to_boxes(response_str):
    """正则解析预测框 (保留之前的逻辑)"""
    if not response_str or not isinstance(response_str, str): return []
    boxes = []
    pattern = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
    matches = re.findall(pattern, response_str)
    if matches:
        for match in matches:
            boxes.append([float(x) for x in match])
        return boxes
    return []

def scale_boxes(boxes, scale_factor=0.512):
    """按固定系数放缩 (复现你当前的环境)"""
    return [[coord * scale_factor for coord in box] for box in boxes]

def get_center(box):
    """计算边界框的中心点 (cx, cy)"""
    # 假设格式是 [x1, y1, x2, y2]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return cx, cy

def analyze_offsets(jsonl_path, scale_factor=0.512):
    delta_x_list = []
    delta_y_list = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
                
            data = json.loads(line)
            gt_boxes = data.get('gt_boxes', [])
            response_str = data.get('response', '')
            
            # 解析并放缩预测框
            pred_boxes_normalized = parse_response_to_boxes(response_str)
            pred_boxes = scale_boxes(pred_boxes_normalized, scale_factor)
            
            if not gt_boxes or not pred_boxes:
                continue

            # 对每个预测框，寻找中心点最近的 GT 框
            for pred_box in pred_boxes:
                pred_cx, pred_cy = get_center(pred_box)
                
                min_dist = float('inf')
                closest_gt_cx, closest_gt_cy = 0, 0
                
                for gt_box in gt_boxes:
                    gt_cx, gt_cy = get_center(gt_box)
                    # 计算欧氏距离
                    dist = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_gt_cx = gt_cx
                        closest_gt_cy = gt_cy
                
                # 计算差值: 预测坐标 - 真实坐标 (正数说明预测偏右/偏下)
                dx = pred_cx - closest_gt_cx
                dy = pred_cy - closest_gt_cy
                
                delta_x_list.append(dx)
                delta_y_list.append(dy)

    if not delta_x_list:
        print("没有可对比的框！")
        return

    # 将结果转为 numpy 数组以便统计
    dx_arr = np.array(delta_x_list)
    dy_arr = np.array(delta_y_list)

    print("="*50)
    print("像素偏移统计 (预测框中心 - 最近真实框中心)")
    print("="*50)
    print(f"总计对比了 {len(dx_arr)} 个预测框对")
    print("-" * 50)
    print(f"【X 轴偏移 (左右方向)】")
    print(f"平均偏移量 (Mean): {np.mean(dx_arr):.2f} 像素")
    print(f"中位数偏移 (Median): {np.median(dx_arr):.2f} 像素")
    print(f"最大偏差: {np.max(dx_arr):.2f}, 最小偏差: {np.min(dx_arr):.2f}")
    print("-" * 50)
    print(f"【Y 轴偏移 (上下方向)】")
    print(f"平均偏移量 (Mean): {np.mean(dy_arr):.2f} 像素")
    print(f"中位数偏移 (Median): {np.median(dy_arr):.2f} 像素")
    print(f"最大偏差: {np.max(dy_arr):.2f}, 最小偏差: {np.min(dy_arr):.2f}")
    print("="*50)

if __name__ == "__main__":
    jsonl_file_path = "/home/yxd/OPEN-DTOS-LMM/Qwen3_VL/infer_preds2.jsonl" 
    
    # 我们先用你之前的 0.512 来看看错位到底有多严重
    analyze_offsets(jsonl_file_path, scale_factor=0.512)