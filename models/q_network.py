"""
models/q_network.py
-------------------
Deep Q-Network cho Tree-RL.

Kiến trúc theo bài báo (Figure 4):
  Input: [4096-d ROI feat] + [4096-d image feat] + [650-d action history]
       = 8842-d
  → Linear(8842, 1024) + ReLU
  → Linear(1024, 1024) + ReLU
  → Linear(1024, 1024) + ReLU
  → Linear(1024, 13)   (Q-values cho 13 actions)

Bài báo: "MLP predicts the estimated values of the 13 actions"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import copy


class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron để ước lượng Q(s, a) cho 13 actions.

    Args:
        state_dim: kích thước state vector (4096 + 4096 + 650 = 8842)
        hidden_dim: số neuron mỗi hidden layer (1024 theo bài báo)
        num_actions: 13 (5 scaling + 8 translation)
    """

    def __init__(
        self,
        state_dim: int = 8842,
        hidden_dim: int = 1024,
        num_actions: int = 13,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.hidden_dim  = hidden_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization cho các Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] hoặc [state_dim]
        Returns:
            q_values: [B, num_actions] hoặc [num_actions]
        """
        return self.net(state)

    def get_best_scaling_action(self, state: torch.Tensor) -> int:
        """
        Trả về action_id tốt nhất trong nhóm Scaling (0-4).
        Dùng trong tree search.
        """
        with torch.no_grad():
            q_vals = self.forward(state.unsqueeze(0)).squeeze(0)
            # Chỉ xét scaling actions (0-4)
            scaling_q = q_vals[:5]
            return int(scaling_q.argmax().item())

    def get_best_translation_action(self, state: torch.Tensor) -> int:
        """
        Trả về action_id tốt nhất trong nhóm Translation (5-12).
        Dùng trong tree search.
        """
        with torch.no_grad():
            q_vals = self.forward(state.unsqueeze(0)).squeeze(0)
            # Chỉ xét translation actions (5-12)
            trans_q = q_vals[5:]
            return int(trans_q.argmax().item()) + 5

    def get_two_best_actions(self, state: torch.Tensor):
        """
        Trả về (best_scaling_id, best_translation_id).
        Đây là core của tree search — chọn 2 actions ở mỗi bước.
        """
        with torch.no_grad():
            q_vals = self.forward(state.unsqueeze(0)).squeeze(0)
            best_scaling     = int(q_vals[:5].argmax().item())
            best_translation = int(q_vals[5:].argmax().item()) + 5
            return best_scaling, best_translation

    def get_greedy_action(self, state: torch.Tensor) -> int:
        """
        Chọn action tốt nhất trong toàn bộ 13 actions.
        Dùng trong epsilon-greedy training (exploitation branch).
        """
        with torch.no_grad():
            q_vals = self.forward(state.unsqueeze(0)).squeeze(0)
            return int(q_vals.argmax().item())


def build_target_network(online_net: QNetwork) -> QNetwork:
    """
    Tạo target network (deep copy) từ online network.
    Target network được update định kỳ (hard update).
    """
    target_net = copy.deepcopy(online_net)
    for param in target_net.parameters():
        param.requires_grad = False
    return target_net


def soft_update(
    target_net: QNetwork,
    online_net: QNetwork,
    tau: float = 0.005,
) -> None:
    """Soft update: θ_target ← τ·θ_online + (1-τ)·θ_target"""
    for t_param, o_param in zip(target_net.parameters(), online_net.parameters()):
        t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)


def hard_update(target_net: QNetwork, online_net: QNetwork) -> None:
    """Hard update: copy toàn bộ weights từ online → target."""
    target_net.load_state_dict(online_net.state_dict())
