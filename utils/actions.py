"""
utils/actions.py
----------------
Định nghĩa 13 hành động của Tree-RL theo đúng bài báo:
  - 5 scaling actions: thu nhỏ cửa sổ về 5 vị trí khác nhau, mỗi sub-window
    có diện tích = 0.55 × diện tích cửa sổ hiện tại
  - 8 local translation actions: dịch chuyển cửa sổ 0.25 × kích thước hiện tại
    theo 8 hướng (trái, phải, trên, dưới, co ngang trái, co ngang phải,
    co dọc trên, co dọc dưới)

Mỗi action nhận [x1, y1, x2, y2] (chuẩn hóa 0-1) và trả về window mới.
"""

import numpy as np
import torch
from typing import List, Tuple


# Index các action
ACTION_NAMES = [
    # Scaling (0-4): 5 sub-windows theo bài báo Figure 2
    "scale_top_left",
    "scale_top_right",
    "scale_bottom_left",
    "scale_bottom_right",
    "scale_center",
    # Translation (5-12): 8 hướng theo bài báo Figure 2
    "trans_left",
    "trans_right",
    "trans_up",
    "trans_down",
    "trans_wider",       # dãn ngang (rộng hơn)
    "trans_narrower",    # co ngang (hẹp hơn)
    "trans_taller",      # dãn dọc (cao hơn)
    "trans_shorter",     # co dọc (thấp hơn)
]

NUM_ACTIONS = 13
NUM_SCALING = 5
NUM_TRANSLATION = 8

SCALING_ACTION_IDS    = list(range(0, 5))
TRANSLATION_ACTION_IDS = list(range(5, 13))


def apply_action(
    window: np.ndarray,
    action_id: int,
    scaling_ratio: float = 0.55,
    translation_ratio: float = 0.25,
    img_w: int = 1,
    img_h: int = 1,
) -> np.ndarray:
    """
    Áp dụng một action lên cửa sổ hiện tại.

    Args:
        window: [x1, y1, x2, y2] tọa độ pixel (không chuẩn hóa)
        action_id: 0-12
        scaling_ratio: 0.55 (bài báo)
        translation_ratio: 0.25 (bài báo)
        img_w, img_h: kích thước ảnh để clamp

    Returns:
        window mới [x1, y1, x2, y2] đã clamp trong ảnh
    """
    x1, y1, x2, y2 = window
    w = x2 - x1
    h = y2 - y1

    if action_id < NUM_SCALING:
        # ---- Scaling actions ----
        # Sub-window có kích thước sqrt(0.55) ≈ 0.742 lần cạnh gốc
        # để diện tích = 0.55 × diện tích cũ
        sw = w * scaling_ratio          # chiều rộng sub-window
        sh = h * scaling_ratio          # chiều cao sub-window
        half_dw = (w - sw) / 2
        half_dh = (h - sh) / 2

        if action_id == 0:   # top-left
            nx1, ny1 = x1, y1
        elif action_id == 1: # top-right
            nx1, ny1 = x1 + (w - sw), y1
        elif action_id == 2: # bottom-left
            nx1, ny1 = x1, y1 + (h - sh)
        elif action_id == 3: # bottom-right
            nx1, ny1 = x1 + (w - sw), y1 + (h - sh)
        else:                # center
            nx1, ny1 = x1 + half_dw, y1 + half_dh

        nx2 = nx1 + sw
        ny2 = ny1 + sh

    else:
        # ---- Translation actions ----
        dx = w * translation_ratio
        dy = h * translation_ratio

        action_id -= NUM_SCALING   # offset về 0-7

        if action_id == 0:   # left
            nx1, ny1, nx2, ny2 = x1 - dx, y1, x2 - dx, y2
        elif action_id == 1: # right
            nx1, ny1, nx2, ny2 = x1 + dx, y1, x2 + dx, y2
        elif action_id == 2: # up
            nx1, ny1, nx2, ny2 = x1, y1 - dy, x2, y2 - dy
        elif action_id == 3: # down
            nx1, ny1, nx2, ny2 = x1, y1 + dy, x2, y2 + dy
        elif action_id == 4: # wider (dãn ngang)
            nx1, ny1, nx2, ny2 = x1 - dx, y1, x2 + dx, y2
        elif action_id == 5: # narrower (co ngang)
            nx1, ny1, nx2, ny2 = x1 + dx, y1, x2 - dx, y2
        elif action_id == 6: # taller (dãn dọc)
            nx1, ny1, nx2, ny2 = x1, y1 - dy, x2, y2 + dy
        else:                # shorter (co dọc)
            nx1, ny1, nx2, ny2 = x1, y1 + dy, x2, y2 - dy

    # Clamp trong ảnh và đảm bảo window hợp lệ (w,h > 0)
    nx1 = max(0, min(nx1, img_w - 2))
    ny1 = max(0, min(ny1, img_h - 2))
    nx2 = max(nx1 + 1, min(nx2, img_w))
    ny2 = max(ny1 + 1, min(ny2, img_h))

    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def encode_action_history(
    action_history: List[int],
    history_len: int = 50,
    num_actions: int = 13,
) -> np.ndarray:
    """
    Mã hóa lịch sử hành động thành binary vector.
    Mỗi hành động là one-hot 13-d → tổng history_len × 13 = 650-d.
    Theo bài báo: "50 past actions are encoded in the state".

    Args:
        action_history: list action_id đã thực hiện (mới nhất ở cuối)
        history_len: 50
        num_actions: 13

    Returns:
        numpy array shape [history_len × num_actions] = [650]
    """
    encoding = np.zeros(history_len * num_actions, dtype=np.float32)
    # Lấy history_len actions gần nhất
    recent = action_history[-history_len:]
    for i, action_id in enumerate(recent):
        encoding[i * num_actions + action_id] = 1.0
    return encoding
