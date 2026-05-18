"""
models/agent.py
---------------
Tree-RL Agent: kết hợp tất cả thành phần:
  - MDP state builder
  - Epsilon-greedy với biến thể tree-search (chọn từ 2 best)
  - Tree-structured search khi testing
  - Reward computation
"""

import numpy as np
import torch
import torch.nn as nn
import random
from typing import List, Tuple, Dict, Optional
from collections import deque

from models.feature_extractor import VGG16FeatureExtractor
from models.q_network import QNetwork, hard_update
from utils.actions import (
    apply_action, encode_action_history,
    SCALING_ACTION_IDS, TRANSLATION_ACTION_IDS, NUM_ACTIONS
)
from utils.metrics import compute_reward, compute_iou, compute_max_iou


class TreeRLAgent:
    """
    Tree-RL Agent theo bài báo NIPS 2016.

    Luồng hoạt động:
      Training:
        - Với mỗi ảnh, chạy 1 episode (tối đa 50 bước)
        - Mỗi bước: build state → epsilon-greedy → apply action → reward → store
        - Replay memory → Q-learning update

      Testing:
        - Tree search: mỗi bước chọn 2 actions (scaling + translation)
        - Đệ quy từ root (toàn ảnh) → tạo tree proposals

    Args:
        feature_extractor: VGG16FeatureExtractor
        q_network: QNetwork (online)
        target_network: QNetwork (target, updated periodically)
        config: dict hyperparameters
        device: torch.device
    """

    def __init__(
        self,
        feature_extractor: VGG16FeatureExtractor,
        q_network: QNetwork,
        target_network: QNetwork,
        config: Dict,
        device: torch.device,
    ):
        self.feat_ext     = feature_extractor
        self.q_net        = q_network
        self.target_net   = target_network
        self.device       = device

        # Config
        train_cfg = config["training"]
        act_cfg   = config["actions"]
        rew_cfg   = config["reward"]
        model_cfg = config["model"]

        self.max_steps       = train_cfg["max_steps_per_episode"]
        self.history_len     = model_cfg["action_history_len"]
        self.scaling_ratio   = act_cfg["scaling_ratio"]
        self.trans_ratio     = act_cfg["translation_ratio"]
        self.iou_threshold   = rew_cfg["iou_threshold"]
        self.first_hit_bonus = rew_cfg["first_hit_bonus"]

    # ------------------------------------------------------------------ #
    #  State Builder                                                       #
    # ------------------------------------------------------------------ #

    def build_state(
        self,
        global_feat: torch.Tensor,    # [4096]
        roi_feat: torch.Tensor,       # [4096]
        action_history: List[int],
    ) -> np.ndarray:
        """
        Ghép state vector theo bài báo:
          state = [roi_feat (4096) | global_feat (4096) | action_history (650)]
               = 8842-d

        Returns:
            state: numpy array [8842]
        """
        history_vec = encode_action_history(
            action_history,
            history_len=self.history_len,
            num_actions=NUM_ACTIONS,
        )
        state = np.concatenate([
            roi_feat.cpu().numpy(),      # 4096
            global_feat.cpu().numpy(),   # 4096
            history_vec,                 # 650
        ])
        return state.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Epsilon-Greedy (Training)                                           #
    # ------------------------------------------------------------------ #

    def select_action_eps_greedy(
        self,
        state: np.ndarray,
        epsilon: float,
    ) -> int:
        """
        Epsilon-greedy với biến thể của Tree-RL:
          - Xác suất ε: chọn ngẫu nhiên 1 trong 13 actions
          - Xác suất 1-ε: chọn ngẫu nhiên trong 2 best actions
                         (best_scaling, best_translation)
        Theo bài báo: "selects a random action from the two best actions
        in the two action groups with probability 1−ε"
        """
        if random.random() < epsilon:
            # Exploration: random trong 13 actions
            return random.randint(0, NUM_ACTIONS - 1)
        else:
            # Exploitation: random trong 2 best actions (tree-consistent)
            state_t = torch.tensor(state, device=self.device)
            best_s, best_t = self.q_net.get_two_best_actions(state_t)
            return random.choice([best_s, best_t])

    # ------------------------------------------------------------------ #
    #  Episode Runner (Training)                                           #
    # ------------------------------------------------------------------ #

    def run_episode(
        self,
        image: torch.Tensor,              # [1, 3, H, W]
        gt_boxes: np.ndarray,             # [N, 4]
        epsilon: float,
        feature_map: Optional[torch.Tensor] = None,
    ) -> List[Tuple]:
        """
        Chạy 1 training episode cho 1 ảnh.
        Bắt đầu từ toàn bộ ảnh, thực hiện tối đa max_steps bước.

        Returns:
            transitions: list of (state, action, reward, next_state, done)
        """
        img_h, img_w = image.shape[2], image.shape[3]

        # Pre-compute feature map nếu chưa có
        if feature_map is None:
            with torch.no_grad():
                feature_map = self.feat_ext.get_conv_feature_map(image)

        # Khởi đầu: cửa sổ = toàn bộ ảnh
        current_window = np.array([0, 0, img_w, img_h], dtype=np.float32)
        action_history: List[int] = []

        # Hit flags: theo dõi gt nào đã được hit
        hit_flags = np.full(len(gt_boxes), -1.0, dtype=np.float32)

        # Global feature (không đổi trong suốt episode)
        global_box_t = torch.tensor(
            [[0, 0, img_w, img_h]], dtype=torch.float32, device=self.device
        )
        spatial_scale = 1.0 / 16.0
        with torch.no_grad():
            global_feat = self.feat_ext.extract_roi_feature(
                feature_map, global_box_t, spatial_scale
            ).squeeze(0)

        transitions = []

        for step in range(self.max_steps):
            # Feature cửa sổ hiện tại
            win_t = torch.from_numpy(np.array([current_window], dtype=np.float32)).to(self.device)

            with torch.no_grad():
                roi_feat = self.feat_ext.extract_roi_feature(
                    feature_map, win_t, spatial_scale
                ).squeeze(0)

            # Build state
            state = self.build_state(global_feat, roi_feat, action_history)

            # Chọn action
            action_id = self.select_action_eps_greedy(state, epsilon)

            # Apply action
            next_window = apply_action(
                current_window, action_id,
                scaling_ratio=self.scaling_ratio,
                translation_ratio=self.trans_ratio,
                img_w=img_w, img_h=img_h,
            )

            # Tính reward
            reward, hit_flags = compute_reward(
                current_window, next_window, gt_boxes, hit_flags,
                iou_threshold=self.iou_threshold,
                first_hit_bonus=self.first_hit_bonus,
            )

            # Next state
            action_history.append(action_id)
            next_win_t = torch.tensor(
                [next_window], dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                next_roi_feat = self.feat_ext.extract_roi_feature(
                    feature_map, next_win_t, spatial_scale
                ).squeeze(0)
            next_state = self.build_state(global_feat, next_roi_feat, action_history)

            done = (step == self.max_steps - 1)
            transitions.append((state, action_id, reward, next_state, float(done)))

            # Move to next window
            current_window = next_window

        return transitions

    # ------------------------------------------------------------------ #
    #  Tree Search (Testing)                                               #
    # ------------------------------------------------------------------ #

    def tree_search(
        self,
        image: torch.Tensor,           # [1, 3, H, W]
        num_levels: int = 5,
        feature_map: Optional[torch.Tensor] = None,
    ) -> List[np.ndarray]:
        """
        Tree-structured search để tạo proposals.

        Mỗi node chọn 2 actions (best scaling + best translation),
        tạo 2 nhánh con, đệ quy đến num_levels tầng.

        Level 1: 1 node (toàn ảnh) → 1 proposal
        Level 2: 2 nodes → 2 proposals
        ...
        Level k: 2^(k-1) nodes → 2^(k-1) proposals
        Tổng: 2^num_levels - 1 proposals

        Proposals được sắp xếp theo thứ tự BFS (level by level).

        Returns:
            proposals: list of [x1,y1,x2,y2] arrays
        """
        img_h, img_w = image.shape[2], image.shape[3]
        spatial_scale = 1.0 / 16.0

        if feature_map is None:
            with torch.no_grad():
                feature_map = self.feat_ext.get_conv_feature_map(image)

        # Global feature
        global_box_t = torch.tensor(
            [[0, 0, img_w, img_h]], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            global_feat = self.feat_ext.extract_roi_feature(
                feature_map, global_box_t, spatial_scale
            ).squeeze(0)

        proposals = []

        # BFS queue: (window, action_history, level)
        init_window = np.array([0, 0, img_w, img_h], dtype=np.float32)
        queue = deque()
        queue.append((init_window, [], 1))

        while queue:
            window, action_history, level = queue.popleft()

            # Lưu proposal (từ level 1, bao gồm cả toàn ảnh)
            proposals.append(window.copy())

            if level >= num_levels:
                continue

            # Tính state cho node này
            win_t = torch.tensor(
                [window], dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                roi_feat = self.feat_ext.extract_roi_feature(
                    feature_map, win_t, spatial_scale
                ).squeeze(0)

            state = self.build_state(global_feat, roi_feat, action_history)
            state_t = torch.tensor(state, device=self.device)

            # Chọn 2 best actions
            with torch.no_grad():
                best_scaling, best_translation = self.q_net.get_two_best_actions(state_t)

            # Tạo 2 nhánh con
            for action_id in [best_scaling, best_translation]:
                next_window = apply_action(
                    window, action_id,
                    scaling_ratio=self.scaling_ratio,
                    translation_ratio=self.trans_ratio,
                    img_w=img_w, img_h=img_h,
                )
                next_history = action_history + [action_id]
                queue.append((next_window, next_history, level + 1))

        return proposals
