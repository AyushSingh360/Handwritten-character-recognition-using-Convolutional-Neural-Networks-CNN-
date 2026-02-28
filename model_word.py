"""
Word-Level CNN Model for Handwriting Recognition

Designed for the Kaggle Handwriting Recognition dataset.
Input: grayscale word images (64 x 256)
Output: N-class softmax over word labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):
    """
    Deep CNN for word-image classification.

    Architecture:
        5 convolutional blocks with BatchNorm + MaxPool
        Adaptive pooling → fixed 512-dim feature vector
        2 fully-connected layers → num_classes output
    
    Input:  (B, 1, H, W)  — grayscale, default H=64 W=256
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.Dropout2d(p=0.1))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1,   64),          # -> 32 x 128
            conv_block(64,  128),         # -> 16 x 64
            conv_block(128, 256),         # -> 8  x 32
            conv_block(256, 512),         # -> 4  x 16
            conv_block(512, 512, pool=False),  # -> 4 x 16 (no pool)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))  # -> 512 x 2 x 4

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_word_model_summary(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"\n{'='*55}\n"
        f"  Word-Level CNN — Model Summary\n"
        f"{'='*55}\n"
        f"  Classes:              {model.num_classes:>10,}\n"
        f"  Total Parameters:     {total:>10,}\n"
        f"  Trainable Parameters: {trainable:>10,}\n"
        f"  Model Size:           {total*4/1024/1024:>10.2f} MB\n"
        f"{'='*55}\n"
    )
