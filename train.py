"""
train.py
--------
Training script cho Tree-RL.
Reproduce đúng bài báo NIPS 2016:
  - 25 epochs trên VOC 2007+2012 trainval (~16K ảnh)
  - ε-greedy: 1.0 → 0.1 trong 10 epoch đầu, giữ 0.1 sau đó
  - Replay memory 800K transitions, mini-batch 64
  - Discount γ = 0.9
  - Target network hard-update mỗi 1000 steps

Chạy:
    python train.py --config configs/default.yaml
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from data.voc_dataset import VOCDataset, voc_collate_fn
from models.feature_extractor import VGG16FeatureExtractor
from models.q_network import QNetwork, build_target_network, hard_update
from models.agent import TreeRLAgent
from utils.replay_memory import ReplayMemory


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_epsilon(epoch: int, cfg: dict) -> float:
    """
    Tính epsilon theo lịch của bài báo:
      - Giảm tuyến tính từ eps_start → eps_end trong eps_decay_epochs đầu
      - Giữ nguyên eps_end sau đó
    """
    eps_start       = cfg["training"]["eps_start"]
    eps_end         = cfg["training"]["eps_end"]
    decay_epochs    = cfg["training"]["eps_decay_epochs"]

    if epoch >= decay_epochs:
        return eps_end
    # Tuyến tính
    return eps_start - (eps_start - eps_end) * (epoch / decay_epochs)


def load_feature_cache(cache_dir: str, year: str, split: str, img_id: str, device):
    """Load pre-computed feature map từ cache."""
    path = os.path.join(cache_dir, f"VOC{year}", split, f"{img_id}.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu")
    feat_map = data["feature_map"].float().unsqueeze(0).to(device)  # [1,512,H',W']
    return feat_map


def compute_loss(
    batch,
    q_net: QNetwork,
    target_net: QNetwork,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Q-learning loss (Huber loss) cho 1 mini-batch.
    Theo công thức bài báo (Equation 3):
      θ_{i+1} = θ_i + α(r + γ max_{a'} Q(s',a';θ_i) - Q(s,a;θ_i)) ∇Q
    """
    states, actions, rewards, next_states, dones = batch

    states      = torch.tensor(states,      dtype=torch.float32, device=device)
    actions     = torch.tensor(actions,     dtype=torch.int64,   device=device)
    rewards     = torch.tensor(rewards,     dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones       = torch.tensor(dones,       dtype=torch.float32, device=device)

    # Q(s, a) — online network
    q_values = q_net(states)                      # [B, 13]
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

    # max Q(s', a') — target network (no grad)
    with torch.no_grad():
        q_next = target_net(next_states).max(dim=1)[0]  # [B]
        td_target = rewards + gamma * q_next * (1 - dones)

    loss = nn.HuberLoss()(q_sa, td_target)
    return loss


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path checkpoint để resume")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["system"]["seed"])

    device = torch.device(
        cfg["system"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Thư mục output ────────────────────────────────────────────────
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["training"]["log_dir"])

    # ── Dataset ───────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = VOCDataset(
        root=cfg["data"]["root"],
        years=cfg["data"]["years"],
        split=cfg["data"]["split"],
    )
    # Không dùng DataLoader thông thường vì mỗi ảnh là 1 episode độc lập
    # Shuffle thủ công mỗi epoch

    # ── Models ────────────────────────────────────────────────────────
    print("Initializing models...")
    model_cfg = cfg["model"]

    feat_extractor = VGG16FeatureExtractor(
        roi_pool_size=model_cfg["roi_pool_size"]
    ).to(device)
    feat_extractor.eval()  # Luôn eval mode (không train CNN)

    # State dim: 4096 (roi) + 4096 (global) + 650 (history) = 8842
    state_dim = 4096 + 4096 + model_cfg["action_history_len"] * model_cfg["num_actions"]

    q_net = QNetwork(
        state_dim=state_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_actions=model_cfg["num_actions"],
    ).to(device)

    target_net = build_target_network(q_net)
    target_net.eval()

    # ── Agent & Memory ────────────────────────────────────────────────
    agent = TreeRLAgent(feat_extractor, q_net, target_net, cfg, device)

    memory = ReplayMemory(capacity=cfg["training"]["replay_capacity"])

    optimizer = optim.Adam(q_net.parameters(), lr=cfg["training"]["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["system"]["amp"])

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming từ {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        q_net.load_state_dict(ckpt["q_net"])
        target_net.load_state_dict(ckpt["target_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resume từ epoch {start_epoch}")

    # ── Training Loop ─────────────────────────────────────────────────
    gamma            = cfg["training"]["gamma"]
    batch_size       = cfg["training"]["batch_size"]
    target_upd_freq  = cfg["training"]["target_update_freq"]
    replay_start     = cfg["training"]["replay_start"]
    total_epochs     = cfg["training"]["epochs"]
    feature_cache    = cfg["data"]["feature_cache_dir"]

    print(f"\n{'='*60}")
    print(f"Bắt đầu training Tree-RL")
    print(f"Dataset: {len(dataset)} ảnh")
    print(f"Epochs: {total_epochs}, Max steps/episode: {cfg['training']['max_steps_per_episode']}")
    print(f"Replay capacity: {cfg['training']['replay_capacity']:,}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, total_epochs):
        epsilon = get_epsilon(epoch, cfg)
        print(f"\n[Epoch {epoch+1}/{total_epochs}] ε = {epsilon:.3f}, "
              f"Memory = {len(memory):,}")

        # Shuffle dataset indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        epoch_losses = []
        epoch_rewards = []
        q_net.train()

        pbar = tqdm(indices, desc=f"Epoch {epoch+1}", unit="img")
        for img_idx in pbar:
            sample = dataset[img_idx]

            if len(sample["gt_boxes"]) == 0:
                continue  # Bỏ qua ảnh không có gt box

            # Load ảnh lên device
            image = sample["image"].unsqueeze(0).to(device)  # [1,3,H,W]
            gt_boxes = sample["gt_boxes"]                     # [N,4]

            # Load pre-computed feature map nếu có
            feat_map = load_feature_cache(
                feature_cache, sample["year"], cfg["data"]["split"], sample["img_id"], device
            )

            # Chạy 1 episode
            transitions = agent.run_episode(
                image, gt_boxes, epsilon, feature_map=feat_map
            )

            # Lưu transitions vào replay memory
            ep_reward = 0
            for (s, a, r, s_, done) in transitions:
                memory.push(s, a, r, s_, done)
                ep_reward += r
            epoch_rewards.append(ep_reward)

            # Q-learning update
            if len(memory) >= replay_start and len(memory) >= batch_size:
                batch = memory.sample(batch_size)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=cfg["system"]["amp"]):
                    loss = compute_loss(batch, q_net, target_net, gamma, device)

                scaler.scale(loss).backward()
                # Gradient clipping để ổn định training
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss.item())
                global_step += 1

                # Hard update target network
                if global_step % target_upd_freq == 0:
                    hard_update(target_net, q_net)

                # Logging
                if global_step % 100 == 0:
                    avg_loss = np.mean(epoch_losses[-100:])
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/epsilon", epsilon, global_step)
                    writer.add_scalar("train/memory_size", len(memory), global_step)

            pbar.set_postfix({
                "loss": f"{np.mean(epoch_losses[-20:]):.4f}" if epoch_losses else "N/A",
                "reward": f"{np.mean(epoch_rewards[-20:]):.2f}" if epoch_rewards else "N/A",
                "mem": len(memory),
            })

        # Epoch summary
        avg_loss   = np.mean(epoch_losses)   if epoch_losses   else 0
        avg_reward = np.mean(epoch_rewards)  if epoch_rewards  else 0
        print(f"  Avg Loss: {avg_loss:.4f} | Avg Episode Reward: {avg_reward:.2f}")
        writer.add_scalar("epoch/avg_loss",   avg_loss,   epoch)
        writer.add_scalar("epoch/avg_reward", avg_reward, epoch)
        writer.add_scalar("epoch/epsilon",    epsilon,    epoch)

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "q_net":       q_net.state_dict(),
                "target_net":  target_net.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "epsilon":     epsilon,
                "config":      cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    writer.close()
    print("\n=== Training hoàn thành! ===")


if __name__ == "__main__":
    train()
