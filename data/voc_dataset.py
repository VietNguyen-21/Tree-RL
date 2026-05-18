"""
data/voc_dataset.py
-------------------
PASCAL VOC 2007 + 2012 Dataset loader cho Tree-RL.
Hỗ trợ load ảnh và ground-truth bounding boxes.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    """
    Dataset cho PASCAL VOC.
    Tree-RL dùng class-agnostic proposals nên chỉ cần bounding boxes,
    không cần nhãn class khi train agent.

    Args:
        root: thư mục VOCdevkit
        years: ["2007", "2012"]
        split: "trainval" hoặc "test"
        transforms: torchvision transforms áp dụng lên ảnh
    """

    def __init__(
        self,
        root: str,
        years: List[str] = ["2007", "2012"],
        split: str = "trainval",
        transforms=None,
    ):
        self.root = root
        self.years = years
        self.split = split
        self.transforms = transforms or self._default_transforms()

        # Thu thập tất cả (year, image_id)
        self.samples: List[Tuple[str, str]] = []
        for year in years:
            split_file = os.path.join(
                root, f"VOC{year}", "ImageSets", "Main", f"{split}.txt"
            )
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Không tìm thấy: {split_file}")
            with open(split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
            self.samples.extend([(year, img_id) for img_id in ids])

        print(f"[VOCDataset] {split}: {len(self.samples)} ảnh "
              f"từ năm {years}")

    def _default_transforms(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        year, img_id = self.samples[idx]

        # Load ảnh
        img_path = os.path.join(
            self.root, f"VOC{year}", "JPEGImages", f"{img_id}.jpg"
        )
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # Load annotations
        ann_path = os.path.join(
            self.root, f"VOC{year}", "Annotations", f"{img_id}.xml"
        )
        gt_boxes, gt_labels = self._parse_annotation(ann_path, img_w, img_h)

        # Apply transforms
        img_tensor = self.transforms(img)

        return {
            "image":     img_tensor,           # [3, H, W]
            "img_id":    img_id,
            "year":      year,
            "img_w":     img_w,
            "img_h":     img_h,
            "gt_boxes":  gt_boxes,             # [N, 4] pixel coords
            "gt_labels": gt_labels,            # [N] class indices
        }

    def _parse_annotation(
        self, ann_path: str, img_w: int, img_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse XML annotation file của VOC."""
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            # Bỏ qua difficult objects (tùy chọn, bài báo không đề cập)
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue

            cls_name = obj.find("name").text.strip().lower()
            if cls_name not in CLASS_TO_IDX:
                continue

            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text) - 1  # VOC dùng 1-indexed
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1

            # Clamp
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[cls_name])

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes  = np.array(boxes,  dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        return boxes, labels


def voc_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate: gt_boxes có số lượng khác nhau giữa ảnh,
    nên không stack được — giữ dạng list.
    """
    images   = torch.stack([b["image"]   for b in batch])
    img_ids  = [b["img_id"]  for b in batch]
    years    = [b["year"]    for b in batch]
    img_ws   = [b["img_w"]   for b in batch]
    img_hs   = [b["img_h"]   for b in batch]
    gt_boxes = [b["gt_boxes"]  for b in batch]
    gt_labels= [b["gt_labels"] for b in batch]

    return {
        "images":    images,
        "img_ids":   img_ids,
        "years":     years,
        "img_ws":    img_ws,
        "img_hs":    img_hs,
        "gt_boxes":  gt_boxes,
        "gt_labels": gt_labels,
    }
