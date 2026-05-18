"""
models/feature_extractor.py
---------------------------
VGG-16 Feature Extractor theo bài báo:
  - Dùng conv5_3 feature maps (pre-computed, lưu cache để train nhanh)
  - ROI Pooling → fc6 (4096-d) làm feature vector
  - Feature của ảnh toàn cục + feature của cửa sổ hiện tại

Theo bài báo:
  "features of both the current window and the whole image are extracted
   using a VGG-16 layer CNN model pre-trained on ImageNet. We use the
   feature vector of layer fc6."
  "all the feature vectors are computed on top of pre-computed feature
   maps of the layer conv5_3 after using ROI Pooling"
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
from typing import Optional, Tuple
import numpy as np


class VGG16FeatureExtractor(nn.Module):
    """
    Trích xuất feature từ VGG-16 pretrained.
    Gồm 2 phần:
      1. conv_layers: conv1_1 → conv5_3 (dùng để tạo feature maps)
      2. fc_layers: adaptive pool → fc6 → ReLU → Dropout

    Trong training, conv_layers được freeze (không train),
    chỉ dùng như feature extractor thuần túy.
    """

    def __init__(self, roi_pool_size: int = 7):
        super().__init__()
        self.roi_pool_size = roi_pool_size

        # Load VGG-16 pretrained
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Conv layers: features[0:30] là conv1_1 → conv5_3
        # VGG-16 features: indices 0-30
        #   0-4:   conv1 block (conv1_1, relu, conv1_2, relu, maxpool)
        #   5-9:   conv2 block
        #   10-16: conv3 block
        #   17-23: conv4 block
        #   24-30: conv5 block (conv5_3 output tại index 30)
        self.conv_layers = nn.Sequential(*list(vgg.features.children())[:30])

        # FC layers: fc6 (4096) với ReLU và Dropout
        # Tương ứng với vgg.classifier[0:3] (fc6, relu, dropout)
        self.fc6 = nn.Sequential(
            nn.Linear(512 * roi_pool_size * roi_pool_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        # Khởi tạo fc6 từ VGG-16 pretrained weights
        fc6_weight = vgg.classifier[0].weight.data
        fc6_bias   = vgg.classifier[0].bias.data

        # VGG-16 fc6 nhận input 25088-d (512×7×7)
        # Nếu roi_pool_size=7, không cần reshape
        if roi_pool_size == 7:
            self.fc6[0].weight.data = fc6_weight
            self.fc6[0].bias.data   = fc6_bias
        # Nếu roi_pool_size khác, init ngẫu nhiên (hiếm gặp)

        # Freeze conv layers — bài báo không fine-tune CNN
        for param in self.conv_layers.parameters():
            param.requires_grad = False

    def get_conv_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        """
        Tính conv5_3 feature map cho toàn bộ ảnh.
        Được gọi một lần và cache lại để tính feature nhanh cho nhiều windows.

        Args:
            image: [1, 3, H, W]
        Returns:
            feature_map: [1, 512, H/16, W/16]
        """
        with torch.no_grad():
            return self.conv_layers(image)

    def extract_roi_feature(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        spatial_scale: float,
    ) -> torch.Tensor:
        """
        ROI Pooling + fc6 cho một tập boxes.

        Args:
            feature_map: [1, 512, H', W']
            boxes: [N, 4] tọa độ pixel trên ảnh gốc [x1,y1,x2,y2]
            spatial_scale: tỷ lệ feature_map / ảnh gốc (thường = 1/16)

        Returns:
            features: [N, 4096]
        """
        # Chuẩn bị rois cho roi_pool: [N, 5] với col đầu là batch_idx=0
        batch_idx = torch.zeros(
            (boxes.shape[0], 1),
            dtype=boxes.dtype,
            device=boxes.device
        )
        rois = torch.cat([batch_idx, boxes], dim=1)  # [N, 5]

        # ROI Pooling
        pooled = ops.roi_pool(
            feature_map,
            rois,
            output_size=(self.roi_pool_size, self.roi_pool_size),
            spatial_scale=spatial_scale,
        )  # [N, 512, 7, 7]

        # Flatten và fc6
        flat = pooled.view(pooled.size(0), -1)  # [N, 512*7*7]
        feat = self.fc6(flat)                    # [N, 4096]
        return feat

    def forward(
        self,
        image: torch.Tensor,
        window_box: Optional[torch.Tensor] = None,
        feature_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trích xuất (global_feat, roi_feat) cho một ảnh.

        Args:
            image: [1, 3, H, W]
            window_box: [1, 4] tọa độ cửa sổ hiện tại
            feature_map: nếu đã pre-computed thì truyền vào để tái sử dụng

        Returns:
            global_feat: [4096] — feature của toàn bộ ảnh
            roi_feat: [4096] — feature của cửa sổ hiện tại
        """
        H, W = image.shape[2], image.shape[3]
        spatial_scale = 1.0 / 16.0  # VGG-16 stride = 16

        # Feature map
        if feature_map is None:
            feature_map = self.get_conv_feature_map(image)

        # Global feature (toàn ảnh = box [0,0,W,H])
        global_box = torch.tensor(
            [[0, 0, W, H]], dtype=torch.float32, device=image.device
        )
        global_feat = self.extract_roi_feature(
            feature_map, global_box, spatial_scale
        ).squeeze(0)  # [4096]

        # ROI feature (cửa sổ hiện tại)
        if window_box is None:
            roi_feat = global_feat
        else:
            if window_box.dim() == 1:
                window_box = window_box.unsqueeze(0)  # [1,4]
            roi_feat = self.extract_roi_feature(
                feature_map, window_box, spatial_scale
            ).squeeze(0)  # [4096]

        return global_feat, roi_feat
