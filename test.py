"""
test.py
-------
Evaluation script: tính recall rate của Tree-RL trên PASCAL VOC 2007 test set.
Reproduce Table 1 và Table 2 của bài báo.

Chạy:
    python test.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/epoch_025.pth \
        --levels 5 6       # 31 và 63 proposals
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import yaml

from data.voc_dataset import VOCDataset
from models.feature_extractor import VGG16FeatureExtractor
from models.q_network import QNetwork
from models.agent import TreeRLAgent
from utils.metrics import compute_recall_curve


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_feature_cache(cache_dir, year, split, img_id, device):
    path = os.path.join(cache_dir, f"VOC{year}", split, f"{img_id}.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu")
    return data["feature_map"].float().unsqueeze(0).to(device)


def evaluate(
    agent: TreeRLAgent,
    dataset: VOCDataset,
    num_levels: int,
    feature_cache_dir: str,
    device: torch.device,
    iou_thresholds: list,
    max_images: int = None,
) -> dict:
    """
    Chạy tree search trên dataset và tính recall.

    Returns:
        results: dict với cấu trúc {size_key: {iou_thr: recall}}
    """
    all_proposals = []
    all_gt_boxes  = []

    n = len(dataset) if max_images is None else min(max_images, len(dataset))

    for idx in tqdm(range(n), desc=f"Tree search (levels={num_levels})"):
        sample = dataset[idx]
        if len(sample["gt_boxes"]) == 0:
            continue

        image = sample["image"].unsqueeze(0).to(device)
        gt_boxes = sample["gt_boxes"]

        feat_map = load_feature_cache(
            feature_cache_dir,
            sample["year"],
            dataset.split,
            sample["img_id"],
            device,
        )

        with torch.no_grad():
            proposals = agent.tree_search(
                image,
                num_levels=num_levels,
                feature_map=feat_map,
            )

        all_proposals.append(proposals)
        all_gt_boxes.append(gt_boxes)

    results = compute_recall_curve(
        all_proposals, all_gt_boxes,
        iou_thresholds=iou_thresholds,
    )
    return results


def print_recall_table(results: dict, num_proposals: int, levels: int):
    """In bảng recall theo format bài báo (Table 1 & 2)."""
    print(f"\n{'='*65}")
    print(f"Tree-RL Recall | Levels={levels}, Proposals={num_proposals}")
    print(f"{'='*65}")
    print(f"{'Category':<12} {'IoU=0.5':>10} {'IoU=0.6':>10} {'IoU=0.7':>10} {'IoU=0.8':>10}")
    print(f"{'-'*65}")
    for size_key in ["all", "large", "small"]:
        row = results[size_key]
        label = {"all": "All", "large": "Large", "small": "Small"}[size_key]
        vals = [row.get(t, 0) for t in [0.5, 0.6, 0.7, 0.8]]
        print(f"{label:<12} " + " ".join(f"{v*100:>10.1f}" for v in vals))
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--levels",     nargs="+", type=int, default=[5, 6])
    parser.add_argument("--max_images", type=int, default=None,
                        help="Giới hạn số ảnh để test nhanh")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    print("Loading test dataset (VOC2007 test)...")
    dataset = VOCDataset(
        root=cfg["data"]["root"],
        years=[cfg["data"]["test_year"]],
        split=cfg["data"]["test_split"],
    )

    # Models
    model_cfg = cfg["model"]
    feat_extractor = VGG16FeatureExtractor(
        roi_pool_size=model_cfg["roi_pool_size"]
    ).to(device)
    feat_extractor.eval()

    state_dim = 4096 + 4096 + model_cfg["action_history_len"] * model_cfg["num_actions"]
    q_net = QNetwork(
        state_dim=state_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_actions=model_cfg["num_actions"],
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    q_net.load_state_dict(ckpt["q_net"])
    q_net.eval()
    print(f"  (Epoch {ckpt['epoch']+1}, step {ckpt.get('global_step', '?')})")

    # Tạo dummy target net (không dùng trong test)
    from models.q_network import build_target_network
    target_net = build_target_network(q_net)

    agent = TreeRLAgent(feat_extractor, q_net, target_net, cfg, device)

    iou_thresholds = cfg["testing"]["iou_thresholds"]
    feature_cache  = cfg["data"]["feature_cache_dir"]

    # Evaluate tại từng level
    for levels in args.levels:
        num_proposals = 2**levels - 1  # Tổng proposals: 2^L - 1
        results = evaluate(
            agent, dataset, levels, feature_cache, device,
            iou_thresholds, max_images=args.max_images,
        )
        print_recall_table(results, num_proposals, levels)


if __name__ == "__main__":
    main()
