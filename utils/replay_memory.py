# """
# utils/replay_memory.py
# ----------------------
# Experience Replay Buffer cho Deep Q-Learning.
# Capacity = 800,000 transitions theo bài báo.
# """

# import random
# import numpy as np
# from collections import deque, namedtuple
# from typing import Tuple, List


# Transition = namedtuple(
#     "Transition",
#     ["state", "action", "reward", "next_state", "done"]
# )


# class ReplayMemory:
#     """
#     Circular buffer lưu trữ transitions (s, a, r, s', done).
#     Sample ngẫu nhiên mini-batch để phá vỡ correlation giữa các samples.

#     Args:
#         capacity: số transitions tối đa (800,000 theo bài báo)
#     """

#     def __init__(self, capacity: int = 800_000):
#         self.capacity = capacity
#         self.memory: deque = deque(maxlen=capacity)

#     def push(
#         self,
#         state: np.ndarray,
#         action: int,
#         reward: float,
#         next_state: np.ndarray,
#         done: bool,
#     ) -> None:
#         """Lưu một transition vào buffer."""
#         self.memory.append(
#             Transition(
#                 state=state.astype(np.float32),
#                 action=int(action),
#                 reward=float(reward),
#                 next_state=next_state.astype(np.float32),
#                 done=bool(done),
#             )
#         )

#     def sample(self, batch_size: int = 64) -> Tuple:
#         """
#         Sample ngẫu nhiên batch_size transitions.

#         Returns:
#             states, actions, rewards, next_states, dones — mỗi cái là
#             numpy array shape [batch_size, ...]
#         """
#         batch = random.sample(self.memory, batch_size)

#         states      = np.stack([t.state      for t in batch])
#         actions     = np.array([t.action     for t in batch], dtype=np.int64)
#         rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
#         next_states = np.stack([t.next_state for t in batch])
#         dones       = np.array([t.done       for t in batch], dtype=np.float32)

#         return states, actions, rewards, next_states, dones

#     def __len__(self) -> int:
#         return len(self.memory)

#     @property
#     def is_ready(self) -> bool:
#         """True nếu đã có đủ samples để bắt đầu training."""
#         return len(self) >= 64


"""
utils/replay_memory.py
----------------------
Lưu state dạng float16 để tiết kiệm RAM 2×.
800K × 2 × 8842 × 2 bytes ≈ 28GB → vừa đủ 32GB RAM.
"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import Tuple

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


class ReplayMemory:
    def __init__(self, capacity: int = 800_000):
        self.capacity = capacity
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Lưu float16 thay vì float32 → tiết kiệm RAM 2×
        self.memory.append(
            Transition(
                state=state.astype(np.float16),      # ← float16
                action=int(action),
                reward=float(reward),
                next_state=next_state.astype(np.float16),  # ← float16
                done=bool(done),
            )
        )

    def sample(self, batch_size: int = 64) -> Tuple:
        batch = random.sample(self.memory, batch_size)
        # Convert về float32 khi sample để train
        states      = np.stack([t.state      for t in batch]).astype(np.float32)
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)