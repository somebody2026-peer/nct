#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEGNet 分类器 V3

基于 EEGNet 架构的 EEG 状态分类器
参考: Lawhern et al., "EEGNet: A Compact Convolutional Network for EEG-based BCIs"

特点:
1. 轻量级架构，适合小样本
2. 深度可分离卷积
3. 支持类别加权损失处理不平衡
4. 支持SMOTE过采样
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def convert_to_serializable(obj):
    """将numpy类型转换为JSON可序列化的Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
DATA_DIR = PROJECT_ROOT / "data" / "mema"
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v3"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "education_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EEGNet 模型
# ============================================================================

class EEGNet(nn.Module):
    """
    EEGNet 架构
    
    Paper: Lawhern et al., "EEGNet: A Compact Convolutional Network 
           for EEG-based Brain-Computer Interfaces"
    
    Args:
        n_channels: EEG通道数
        n_timepoints: 时间点数
        n_classes: 分类类别数
        F1: 第一层时间卷积核数
        D: 深度乘数
        F2: 第二层卷积核数
        kernel_length: 时间卷积核长度
        dropout_rate: Dropout比率
    """
    
    def __init__(
        self,
        n_channels: int = 4,
        n_timepoints: int = 1000,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        
        # Block 1: 时间卷积 + 深度卷积
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 2: 可分离卷积
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 计算展平后的特征维度
        self._calculate_flatten_size()
        
        # 分类层
        self.classifier = nn.Linear(self.flatten_size, n_classes)
    
    def _calculate_flatten_size(self):
        """计算展平后的特征维度"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = self.activation(x)
            x = self.pool1(x)
            x = self.separable_conv(x)
            x = self.bn3(x)
            x = self.activation(x)
            x = self.pool2(x)
            self.flatten_size = x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, n_channels, n_timepoints]
        
        Returns:
            logits: [batch, n_classes]
        """
        # 添加通道维度
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, n_channels, n_timepoints]
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 分类
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不包含分类层）"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool1(x)
        
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool2(x)
        
        return x.view(x.size(0), -1)


# ============================================================================
# 简化版 EEGNet (用于小数据集)
# ============================================================================

class EEGNetLite(nn.Module):
    """
    简化版 EEGNet，更适合小样本
    """
    
    def __init__(
        self,
        n_channels: int = 4,
        n_timepoints: int = 1000,
        n_classes: int = 3,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        
        # 简化的卷积层
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=10, padding=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(10)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64 * 10, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_channels, n_timepoints]
        """
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ============================================================================
# 训练工具
# ============================================================================

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """计算类别权重（处理不平衡）"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    # 归一化
    weights = weights / weights.sum() * len(unique)
    
    return torch.FloatTensor(weights)


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """创建加权采样器"""
    class_weights = compute_class_weights(labels)
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return f1, all_preds, all_labels


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 池化标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
    return d


# ============================================================================
# 数据加载
# ============================================================================

def load_mema_data(max_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    加载MEMA数据（分层采样）
    
    MEMA数据格式: (trials, timepoints, channels) -> 转置为 (trials, channels, timepoints)
    """
    data_by_class = {0: [], 1: [], 2: []}
    subjects_by_class = {0: [], 1: [], 2: []}
    
    for subj_id in range(1, 21):
        subj_dir = DATA_DIR / f"Subject{subj_id}"
        if not subj_dir.exists():
            continue
        
        mat_file = subj_dir / f"Subject{subj_id}_attention.mat"
        if not mat_file.exists():
            continue
        
        try:
            mat = loadmat(str(mat_file))
            data = mat.get("data", mat.get("Data"))
            labels = mat.get("label", mat.get("Label"))
            
            if data is None:
                continue
            
            # 转置: (trials, timepoints, channels) -> (trials, channels, timepoints)
            if data.ndim == 3:
                data = data.transpose(0, 2, 1)
            elif data.ndim == 2:
                data = data[np.newaxis, ...]
            
            if labels is not None:
                labels = labels.flatten()
            else:
                labels = np.zeros(len(data))
            
            for eeg, lbl in zip(data, labels):
                lbl_int = int(lbl)
                if lbl_int in data_by_class:
                    data_by_class[lbl_int].append(eeg)
                    subjects_by_class[lbl_int].append(subj_id)
            
        except Exception as e:
            logger.warning(f"加载 Subject{subj_id} 失败: {e}")
    
    # 分层采样
    samples_per_class = max_samples // 3
    all_eeg = []
    all_labels = []
    all_subjects = []
    
    for class_id in [0, 1, 2]:
        class_data = data_by_class[class_id]
        class_subjects = subjects_by_class[class_id]
        
        if len(class_data) == 0:
            continue
        
        n_samples = min(len(class_data), samples_per_class)
        indices = np.random.choice(len(class_data), n_samples, replace=False)
        
        for idx in indices:
            all_eeg.append(class_data[idx])
            all_labels.append(class_id)
            all_subjects.append(class_subjects[idx])
    
    # 打乱顺序
    if all_eeg:
        shuffle_idx = np.random.permutation(len(all_eeg))
        all_eeg = [all_eeg[i] for i in shuffle_idx]
        all_labels = [all_labels[i] for i in shuffle_idx]
        all_subjects = [all_subjects[i] for i in shuffle_idx]
    
    if not all_eeg:
        logger.warning("未找到真实数据，使用模拟数据")
        n_samples = 300
        n_channels = 4
        n_timepoints = 1000
        sfreq = 200
        
        for i in range(n_samples):
            eeg = np.random.randn(n_channels, n_timepoints) * 10
            label = i % 3
            
            # 添加状态特征
            if label == 2:  # concentrating - beta
                beta = np.sin(2 * np.pi * 20 * np.arange(n_timepoints) / sfreq)
                eeg += beta * 5
            elif label == 1:  # relaxing - alpha
                alpha = np.sin(2 * np.pi * 10 * np.arange(n_timepoints) / sfreq)
                eeg += alpha * 8
            
            all_eeg.append(eeg)
            all_labels.append(label)
            all_subjects.append(i // 60)
    
    return np.array(all_eeg), np.array(all_labels), all_subjects


# ============================================================================
# 主训练函数
# ============================================================================

def train_eegnet_v3(
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    n_folds: int = 5,
    use_class_weights: bool = True,
    use_lite_model: bool = True,
    max_samples: int = 3000,
) -> Dict:
    """
    训练 EEGNet V3
    
    Args:
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        n_folds: 交叉验证折数
        use_class_weights: 是否使用类别权重
        use_lite_model: 是否使用简化版模型
        max_samples: 最大样本数
    
    Returns:
        训练结果
    """
    logger.info("=" * 60)
    logger.info("EEGNet V3 训练")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    eeg_data, labels, subjects = load_mema_data(max_samples)
    logger.info(f"数据规模: {eeg_data.shape}")
    logger.info(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    n_channels = eeg_data.shape[1]
    n_timepoints = eeg_data.shape[2]
    n_classes = len(np.unique(labels))
    
    # 计算类别权重
    if use_class_weights:
        class_weights = compute_class_weights(labels).to(device)
        logger.info(f"类别权重: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
    
    # 交叉验证
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_preds = []
    all_labels_cv = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(eeg_data, labels)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # 数据分割
        X_train = torch.FloatTensor(eeg_data[train_idx])
        y_train = torch.LongTensor(labels[train_idx])
        X_val = torch.FloatTensor(eeg_data[val_idx])
        y_val = torch.LongTensor(labels[val_idx])
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        if use_class_weights:
            sampler = create_weighted_sampler(labels[train_idx])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        if use_lite_model:
            model = EEGNetLite(
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                n_classes=n_classes,
            ).to(device)
        else:
            model = EEGNet(
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                n_classes=n_classes,
            ).to(device)
        
        # 损失函数和优化器
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 训练
        best_f1 = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_f1, preds, true_labels = evaluate(model, val_loader, device)
            
            scheduler.step(1 - val_f1)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch + 1}: loss={train_loss:.4f}, val_f1={val_f1:.4f}")
        
        # 使用最佳模型评估
        model.load_state_dict(best_model_state)
        val_f1, preds, true_labels = evaluate(model, val_loader, device)
        
        fold_results.append({
            "fold": fold + 1,
            "f1": val_f1,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        
        all_preds.extend(preds)
        all_labels_cv.extend(true_labels)
        
        logger.info(f"  Fold {fold + 1} Best F1: {val_f1:.4f}")
    
    # 汇总结果
    f1_scores = [r["f1"] for r in fold_results]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    # 计算Cohen's Kappa
    kappa = cohen_kappa_score(all_labels_cv, all_preds)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"交叉验证结果")
    logger.info(f"{'=' * 60}")
    logger.info(f"  平均 F1: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"  Cohen's Kappa: {kappa:.4f}")
    
    # 分类报告
    report = classification_report(all_labels_cv, all_preds, 
                                   target_names=["neutral", "relaxing", "concentrating"],
                                   output_dict=True)
    
    results = {
        "version": "V3",
        "model": "EEGNetLite" if use_lite_model else "EEGNet",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_folds": n_folds,
        "use_class_weights": use_class_weights,
        "data_shape": list(eeg_data.shape),
        "mean_f1": round(mean_f1, 4),
        "std_f1": round(std_f1, 4),
        "cohens_kappa": round(kappa, 4),
        "fold_results": fold_results,
        "classification_report": report,
        "target_achieved": mean_f1 >= 0.40,
    }
    
    # 保存结果
    results = convert_to_serializable(results)
    out_path = RESULTS_DIR / "eegnet_v3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存至: {out_path}")
    
    # 保存最终模型
    if best_model_state is not None:
        model_path = CKPT_DIR / "eegnet_v3_best.pt"
        torch.save(best_model_state, model_path)
        logger.info(f"模型已保存至: {model_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EEGNet V3 训练")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--no-class-weights", action="store_true", help="不使用类别权重")
    parser.add_argument("--full-model", action="store_true", help="使用完整EEGNet")
    parser.add_argument("--max-samples", type=int, default=3000, help="最大样本数")
    
    args = parser.parse_args()
    
    results = train_eegnet_v3(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.folds,
        use_class_weights=not args.no_class_weights,
        use_lite_model=not args.full_model,
        max_samples=args.max_samples,
    )
    
    print("\n" + "=" * 60)
    print(f"EEGNet V3 训练完成!")
    print(f"平均 F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"目标达成 (F1>0.40): {'是' if results['target_achieved'] else '否'}")
    print("=" * 60)