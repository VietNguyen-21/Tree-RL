"""
scripts/extract_features.py
----------------------------
Pre-compute VGG-16 conv5_3 feature maps cho toàn bộ dataset
và lưu cache vào disk để tránh tính lại mỗi epoch.

Theo bài báo: "all the feature vectors are computed on top of
pre-computed feature maps of the layer conv5_3"

Dùng:
    python scripts/extract_features.py \
        --data_root ./VOCdevkit \
        --split trainval \
        --year 2007 \
        --output_dir ./feature_cache \
        --device cuda
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

from models.feature_extractor import VGG16FeatureExtractor


def get_image_ids(data_root: str, year: str, split: str):
    split_file = os.path.join(
        data_root, f"VOC{year}", "ImageSets", "Main", f"{split}.txt"
    )
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default="./VOCdevkit")
    parser.add_argument("--split",       default="trainval")
    parser.add_argument("--year",        default="2007")
    parser.add_argument("--output_dir",  default="./feature_cache")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tạo thư mục output
    out_dir = os.path.join(args.output_dir, f"VOC{args.year}", args.split)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    extractor = VGG16FeatureExtractor(roi_pool_size=7).to(device)
    extractor.eval()

    # Image transforms (giống với training)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Lấy danh sách ảnh
    img_ids = get_image_ids(args.data_root, args.year, args.split)
    print(f"Tổng số ảnh cần extract: {len(img_ids)}")

    skipped = 0
    for img_id in tqdm(img_ids, desc=f"Extracting VOC{args.year} {args.split}"):
        out_path = os.path.join(out_dir, f"{img_id}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue

        # Load ảnh
        img_path = os.path.join(
            args.data_root, f"VOC{args.year}", "JPEGImages", f"{img_id}.jpg"
        )
        if not os.path.exists(img_path):
            print(f"Warning: Không tìm thấy {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img_t = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

        # Tính conv5_3 feature map
        with torch.no_grad():
            feat_map = extractor.get_conv_feature_map(img_t)  # [1,512,H/16,W/16]

        # Lưu: chuyển về CPU float16 để tiết kiệm dung lượng
        feat_map_cpu = feat_map.squeeze(0).half().cpu()  # [512, H/16, W/16]

        torch.save({
            "feature_map": feat_map_cpu,
            "img_w": img_w,
            "img_h": img_h,
        }, out_path)

    total = len(img_ids) - skipped
    print(f"\nHoàn thành! Đã extract {total} ảnh, bỏ qua {skipped} (đã có cache).")
    print(f"Cache lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
