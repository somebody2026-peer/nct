#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FER2013 情绪识别预训练 V2 - 深度学习增强版

V2 改进:
1. 使用预训练 ResNet18 替代轻量 CNN (预期准确率 50% 至 68%+)
2. 添加数据增强 (随机裁剪、翻转、颜色抖动)
3. 支持学习率调度和早停
4. 保存最佳模型到 checkpoints/education_v2/

与 V1 的区别:
- V1 文件: experiments/fer_pretrain.py (轻量 3 层 CNN)
- V2 文件: experiments/fer_pretrain_v2.py (预训练 ResNet18)
- V2 结果: results/education_v2/phase2_fer_v2.json
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 常量定义
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "fer2013"
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v2"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "education_v2"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# FER2013 情绪类别
EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 情绪与神经调质映射表（偏差量，基准 0.5）
EMOTION_NM_MAP: Dict[str, Dict[str, float]] = {
    "Happy":    {"DA":  0.3,  "5-HT":  0.2, "NE":  0.0, "ACh":  0.1},
    "Surprise": {"DA":  0.2,  "5-HT":  0.0, "NE":  0.3, "ACh":  0.0},
    "Fear":     {"DA": -0.1,  "5-HT": -0.3, "NE":  0.4, "ACh":  0.0},
    "Angry":    {"DA": -0.2,  "5-HT": -0.1, "NE":  0.3, "ACh":  0.0},
    "Sad":      {"DA": -0.25, "5-HT": -0.2, "NE": -0.1, "ACh": -0.1},
    "Disgust":  {"DA": -0.15, "5-HT": -0.2, "NE":  0.1, "ACh":  0.2},
    "Neutral":  {"DA":  0.0,  "5-HT":  0.0, "NE":  0.0, "ACh":  0.0},
}


def emotion_to_neuromodulator(emotion_idx: int) -> Dict[str, float]:
    """将情绪索引映射到神经调质字典"""
    name = EMOTION_NAMES[emotion_idx]
    deltas = EMOTION_NM_MAP.get(name, {})
    baseline = 0.5
    return {
        k: float(np.clip(baseline + deltas.get(k, 0.0), 0.0, 1.0))
        for k in ["DA", "5-HT", "NE", "ACh"]
    }


# ============================================================================
# 数据集
# ============================================================================

class FERDatasetV2(Dataset):
    """FER2013 数据集 V2 - 支持数据增强"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]  # [48, 48]
        label = self.labels[idx]
        
        # 转为 3 通道（ResNet 需要）
        img_3ch = np.stack([img, img, img], axis=0)  # [3, 48, 48]
        img_tensor = torch.from_numpy(img_3ch).float()
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)


def load_fer2013_data(max_samples: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 FER2013 数据集"""
    csv_path = DATA_DIR / "fer2013.csv"
    
    if csv_path.exists():
        logger.info(f"从本地 CSV 加载: {csv_path}")
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        train_df = df[df['Usage'] == 'Training']
        val_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]
        
        def parse_pixels(pixel_str):
            return np.array([int(p) for p in pixel_str.split()]).reshape(48, 48).astype(np.float32) / 255.0
        
        X_train = np.array([parse_pixels(p) for p in train_df['pixels'].values])
        y_train = train_df['emotion'].values
        X_val = np.array([parse_pixels(p) for p in val_df['pixels'].values])
        y_val = val_df['emotion'].values
        
    else:
        logger.info("尝试从 HuggingFace 加载 FER2013...")
        try:
            from datasets import load_dataset
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            ds = load_dataset("abhilash88/fer2013-enhanced", split="train")
            images = np.array([np.array(img.convert('L')).astype(np.float32) / 255.0 
                              for img in ds['image']])
            labels = np.array(ds['label'])
            
            # 8:2 划分
            n = len(labels)
            idx = np.random.permutation(n)
            split = int(0.8 * n)
            X_train, y_train = images[idx[:split]], labels[idx[:split]]
            X_val, y_val = images[idx[split:]], labels[idx[split:]]
            
        except Exception as e:
            logger.warning(f"HuggingFace 加载失败: {e}，生成模拟数据")
            X_train = np.random.rand(5000, 48, 48).astype(np.float32)
            y_train = np.random.randint(0, 7, 5000)
            X_val = np.random.rand(1000, 48, 48).astype(np.float32)
            y_val = np.random.randint(0, 7, 1000)
    
    if max_samples:
        X_train, y_train = X_train[:max_samples], y_train[:max_samples]
        X_val, y_val = X_val[:max_samples // 4], y_val[:max_samples // 4]
    
    logger.info(f"训练集: {len(y_train)}, 验证集: {len(y_val)}")
    return X_train, y_train, X_val, y_val


# ============================================================================
# V2 模型: 预训练 ResNet18
# ============================================================================

class FERResNet18(nn.Module):
    """
    FER ResNet18 V2 模型
    
    改进:
    1. 使用 ImageNet 预训练权重
    2. 调整第一层卷积支持 48x48 输入
    3. 添加 Dropout 防止过拟合
    """
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # 加载预训练 ResNet18
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = models.resnet18(weights=weights)
                logger.info("使用 ImageNet 预训练权重")
            except:
                self.backbone = models.resnet18(pretrained=True)
                logger.info("使用旧版预训练权重加载方式")
        else:
            self.backbone = models.resnet18(pretrained=False)
            logger.info("从头训练（无预训练权重）")
        
        # 修改第一层：适配 48x48 输入（原始为 224x224）
        # 保持 3 通道输入（灰度图复制 3 次）
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # 移除 maxpool，保持分辨率
        
        # 修改最后一层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """提取特征向量（不含分类头部）"""
        # 通过所有层直到 avgpool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # [B, 512]


# ============================================================================
# 训练函数
# ============================================================================

def train_fer_v2(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    max_samples: int = None,
    patience: int = 5,
    device: str = None,
) -> Dict:
    """
    训练 FER ResNet18 V2 模型
    
    Args:
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        max_samples: 最大样本数（用于快速测试）
        patience: 早停耐心值
        device: 设备
    
    Returns:
        results: 训练结果字典
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    X_train, y_train, X_val, y_val = load_fer2013_data(max_samples)
    
    # 数据增强（V2 改进点）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = FERDatasetV2(X_train, y_train, transform=train_transform)
    val_dataset = FERDatasetV2(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 模型
    model = FERResNet18(num_classes=7, pretrained=True, dropout=0.3).to(device)
    
    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练记录
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    
    logger.info(f"开始训练 FER V2 (ResNet18): epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        # 记录
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] "
                   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR / "fer_resnet18_best.pt")
            logger.info(f"  -> 保存最佳模型 (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
        
        # 早停
        if no_improve >= patience:
            logger.info(f"早停: {patience} 轮无改进")
            break
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(CKPT_DIR / "fer_resnet18_best.pt"))
    model.eval()
    
    # 计算各类别准确率
    class_correct = [0] * 7
    class_total = [0] * 7
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    per_class_acc = {
        EMOTION_NAMES[i]: round(class_correct[i] / max(class_total[i], 1), 4)
        for i in range(7)
    }
    
    # 汇总结果
    results = {
        "version": "V2",
        "model": "ResNet18 (pretrained)",
        "best_val_acc": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "per_class_acc": per_class_acc,
        "history": history,
        "checkpoint": str(CKPT_DIR / "fer_resnet18_best.pt"),
        "emotion_nm_map": EMOTION_NM_MAP,
        "n_train_samples": len(y_train),
        "n_val_samples": len(y_val),
        "improvements_over_v1": {
            "backbone": "ResNet18 vs 3-layer CNN",
            "data_augmentation": True,
            "lr_scheduler": "CosineAnnealing",
            "early_stopping": True,
        }
    }
    
    # 保存结果
    results_path = RESULTS_DIR / "phase2_fer_v2.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存至: {results_path}")
    
    return results


# ============================================================================
# 推理函数
# ============================================================================

def predict_emotion_v2(
    model: FERResNet18,
    face_crop: np.ndarray,
    device: torch.device = None
) -> Tuple[int, np.ndarray]:
    """
    使用 V2 模型预测情绪
    
    Args:
        model: FERResNet18 模型
        face_crop: 人脸裁剪图像 [H, W] 或 [H, W, 3]
        device: 设备
    
    Returns:
        (emotion_idx, probabilities)
    """
    import cv2
    
    if device is None:
        device = next(model.parameters()).device
    
    if face_crop.ndim == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop
    
    # 缩放到 48x48
    resized = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0
    
    # 复制 3 通道 + 归一化
    img_3ch = np.stack([resized, resized, resized], axis=0)
    tensor = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)
    tensor = (tensor - 0.5) / 0.5  # 归一化
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    return int(probs.argmax()), probs


def load_fer_model_v2(device: str = None) -> FERResNet18:
    """加载预训练的 FER V2 模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    model = FERResNet18(num_classes=7, pretrained=False)
    ckpt_path = CKPT_DIR / "fer_resnet18_best.pt"
    
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"已加载 FER V2 模型: {ckpt_path}")
    else:
        logger.warning(f"未找到 V2 模型检查点: {ckpt_path}，使用未训练模型")
    
    model.to(device)
    model.eval()
    return model


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FER2013 V2 训练 (ResNet18)")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FER2013 V2 训练 - ResNet18 深度学习增强版")
    logger.info("=" * 60)
    
    results = train_fer_v2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        patience=args.patience,
    )
    
    logger.info(f"\n最终结果: val_acc = {results['best_val_acc']:.4f}")
    logger.info(f"各类别准确率: {results['per_class_acc']}")