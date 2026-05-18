# Tree-RL: Tree-Structured Reinforcement Learning for Sequential Object Localization

PyTorch reproduction của bài báo NIPS 2016:
> Zequn Jie et al. "Tree-Structured Reinforcement Learning for Sequential Object Localization"

---

## Cấu trúc project

```
tree_rl/
├── configs/
│   └── default.yaml          # Tất cả hyperparameters
├── data/
│   └── voc_dataset.py        # PASCAL VOC dataloader
├── models/
│   ├── feature_extractor.py  # VGG-16 + ROI Pooling
│   ├── q_network.py          # Deep Q-Network (MLP)
│   └── agent.py              # Tree-RL Agent (MDP + Tree Search)
├── utils/
│   ├── replay_memory.py      # Experience replay buffer
│   ├── actions.py            # 13 actions (5 scaling + 8 translation)
│   └── metrics.py            # IoU, recall, mAP helpers
├── scripts/
│   ├── download_voc.sh       # Download PASCAL VOC 2007+2012
│   └── extract_features.py   # Pre-compute conv5_3 feature maps
├── train.py                  # Training script
├── test.py                   # Evaluation / recall computation
└── requirements.txt
```

---

## Cài đặt

```bash
pip install -r requirements.txt

# Download PASCAL VOC
bash scripts/download_voc.sh

# Pre-compute VGG-16 conv5_3 feature maps (tiết kiệm thời gian train)
python scripts/extract_features.py --data_root ./VOCdevkit --split trainval --year 2007
python scripts/extract_features.py --data_root ./VOCdevkit --split trainval --year 2012
```

---

## Training

```bash
python train.py --config configs/default.yaml
```

## Evaluation (recall)

```bash
python test.py --config configs/default.yaml \
               --checkpoint checkpoints/best.pth \
               --levels 5   # 31 proposals
```

---

## Chi tiết implement theo bài báo

| Thành phần | Bài báo | Implement |
|---|---|---|
| Backbone | VGG-16 pretrained ImageNet | `torchvision.models.vgg16` |
| Feature | fc6 (4096-d) qua ROI Pooling trên conv5_3 | `torchvision.ops.roi_pool` |
| State | [ROI feat, img feat, action history 650-d] | concat → 8842-d |
| Q-Network | MLP 3 lớp 1024-d → 13 outputs | `QNetwork` |
| Actions | 5 scaling + 8 translation = 13 | `ActionSpace` |
| Reward | ±1 IoU improvement, +5 first-hit | `compute_reward()` |
| Tree search | Bifurcation mỗi bước: best scaling + best translation | `tree_search()` |
| Training | ε-greedy (1→0.1, 10 epoch), replay 800K, batch 64 | `Trainer` |
| Dataset | VOC 2007+2012 trainval (~16K ảnh) | `VOCDataset` |
