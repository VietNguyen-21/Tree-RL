"""
utils/metrics.py
----------------
Các hàm tính IoU, recall, và mAP dùng trong Tree-RL.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Tính IoU giữa 2 bounding box [x1, y1, x2, y2].
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def compute_max_iou(window: np.ndarray, gt_boxes: np.ndarray) -> Tuple[float, int]:
    """
    Tính IoU lớn nhất giữa window và tập ground-truth boxes.

    Returns:
        max_iou: giá trị IoU lớn nhất
        best_idx: index của gt box tương ứng
    """
    if len(gt_boxes) == 0:
        return 0.0, -1
    ious = [compute_iou(window, gt) for gt in gt_boxes]
    best_idx = int(np.argmax(ious))
    return ious[best_idx], best_idx


def compute_reward(
    current_window: np.ndarray,
    next_window: np.ndarray,
    gt_boxes: np.ndarray,
    hit_flags: np.ndarray,
    iou_threshold: float = 0.5,
    first_hit_bonus: float = 5.0,
) -> Tuple[float, np.ndarray]:
    """
    Tính reward theo công thức bài báo (Equation 2):

      r(s,a) = +5   nếu bất kỳ gt nào được hit lần đầu (IoU > 0.5)
             = max_i sign(IoU(w', gi) - IoU(w, gi))   otherwise

    Args:
        current_window: cửa sổ trước khi action [x1,y1,x2,y2]
        next_window: cửa sổ sau action
        gt_boxes: ground-truth boxes [N, 4]
        hit_flags: array [N] = 1 nếu gt i đã được hit trước đó, else -1
        iou_threshold: 0.5
        first_hit_bonus: 5.0

    Returns:
        reward: float
        new_hit_flags: cập nhật hit_flags
    """
    new_hit_flags = hit_flags.copy()

    # Kiểm tra first-time hit
    for i, gt in enumerate(gt_boxes):
        iou_next = compute_iou(next_window, gt)
        if hit_flags[i] < 1 and iou_next >= iou_threshold:
            new_hit_flags[i] = 1
            return first_hit_bonus, new_hit_flags

    # Reward thông thường: max over all gt của sign(IoU_next - IoU_curr)
    max_sign = -1.0
    for gt in gt_boxes:
        iou_curr = compute_iou(current_window, gt)
        iou_next = compute_iou(next_window, gt)
        diff = iou_next - iou_curr
        if diff > 0:
            max_sign = 1.0
        # Nếu bất kỳ gt nào cải thiện thì reward = +1
        if max_sign > 0:
            break

    return max_sign, new_hit_flags


def compute_recall(
    proposals_per_image: List[np.ndarray],
    gt_per_image: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """
    Tính recall: tỷ lệ gt objects được cover bởi ít nhất 1 proposal
    với IoU >= iou_threshold.

    Args:
        proposals_per_image: list of [K, 4] arrays
        gt_per_image: list of [N, 4] arrays
        iou_threshold: float

    Returns:
        recall: float trong [0, 1]
    """
    total_gt = 0
    total_hit = 0

    for proposals, gt_boxes in zip(proposals_per_image, gt_per_image):
        if len(gt_boxes) == 0:
            continue
        for gt in gt_boxes:
            total_gt += 1
            # Check nếu có proposal nào cover gt này
            for prop in proposals:
                if compute_iou(prop, gt) >= iou_threshold:
                    total_hit += 1
                    break

    if total_gt == 0:
        return 0.0
    return total_hit / total_gt


def compute_recall_curve(
    proposals_per_image: List[np.ndarray],
    gt_per_image: List[np.ndarray],
    iou_thresholds: List[float] = None,
    size_threshold: int = 2000,
) -> Dict:
    """
    Tính recall tại nhiều IoU threshold cho large/small/all objects.
    Large objects: diện tích > size_threshold pixels (bài báo dùng 2000).

    Returns dict với keys: 'all', 'large', 'small', mỗi key là dict
    {iou_threshold: recall}
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    results = {"all": {}, "large": {}, "small": {}}

    for iou_thr in iou_thresholds:
        counts = {"all": [0, 0], "large": [0, 0], "small": [0, 0]}  # [hit, total]

        for proposals, gt_boxes in zip(proposals_per_image, gt_per_image):
            for gt in gt_boxes:
                area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                size_key = "large" if area > size_threshold else "small"

                hit = any(compute_iou(p, gt) >= iou_thr for p in proposals)
                hit_val = 1 if hit else 0

                counts["all"][0] += hit_val
                counts["all"][1] += 1
                counts[size_key][0] += hit_val
                counts[size_key][1] += 1

        for key in results:
            total = counts[key][1]
            results[key][iou_thr] = counts[key][0] / total if total > 0 else 0.0

    return results
