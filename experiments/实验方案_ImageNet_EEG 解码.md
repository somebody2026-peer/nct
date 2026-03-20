# NCT ImageNet 与 EEG 解码实验方案

## 一、实验概述

### 1.1 实验目标

本实验方案旨在系统性地开展 NCT（NeuroConscious Transformer）在两个重要数据集上的验证：

1. **ImageNet 子集识别实验**：验证 NCT 在大规模自然图像数据集上的表征能力和扩展性
2. **EEG 神经信号解码实验**：验证 NCT 在脑电信号解码任务上的应用潜力，建立神经科学与 AI 的桥梁

### 1.2 预期成果

| 数据集 | 基线准确率 | 目标准确率 | 预期提升 | 完成时间 |
|--------|-----------|-----------|---------|---------|
| **ImageNet 子集** | 65.2% (d=768) | 75.0% (d=2048) | +9.8% | 2027 Q1 |
| **EEG 解码** | 71.3% (d=768) | 82.0% (d=2048) | +10.7% | 2027 Q2 |

---

## 二、ImageNet 子集识别实验

### 2.1 实验设计

#### 2.1.1 数据集选择策略

由于完整 ImageNet-1k（128 万张图像，1000 类）训练成本过高，采用以下渐进式策略：

**Phase 1: ImageNet Subset（小规模验证）**
- **数据集**: ImageNet-100（100 类，约 12.6 万张图像）
- **选择标准**: 
  - 选择常见的 100 个类别（如猫、狗、汽车、飞机等）
  - 保持类别平衡
- **训练/验证划分**: 官方验证集（5 万张）
- **预计训练时间**: 2-3 天（单卡 A100）

**Phase 2: ImageNet-200（中等规模扩展）**
- **数据集**: ImageNet-200（200 类，约 25 万张图像）
- **目标**: 验证 d_model 从 768 扩展到 2048 的效果
- **预计训练时间**: 4-5 天（单卡 A100）

**Phase 3: 完整 ImageNet-1k（可选）**
- 如果 Phase 1-2 结果理想，考虑完整数据集
- 需要多卡并行（4-8 卡 A100）

#### 2.1.2 数据增强策略

参考 ViT 和 ResNet 的成功实践：

```python
# 训练时增强
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
])

# 验证时增强
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
```

#### 2.1.3 NCT 架构配置

**配置 A（基线，d=768）**:
```python
nct_config = {
    'input_size': 224,
    'patch_size': 16,  # 14x14 patches
    'd_model': 768,
    'n_heads': 12,
    'n_layers': 12,
    'd_ff': 3072,
    'num_classes': 100,  # Phase 1
    'dropout': 0.1,
    'gamma_freq': 40,  # 40Hz γ同步
    'predictive_coding_layers': 6,
}
```

**配置 B（优化，d=2048）**:
```python
nct_config = {
    'input_size': 224,
    'patch_size': 16,
    'd_model': 2048,  # 扩大模型容量
    'n_heads': 16,
    'n_layers': 24,   # 加深网络
    'd_ff': 8192,
    'num_classes': 200,  # Phase 2
    'dropout': 0.15,
    'gamma_freq': 40,
    'predictive_coding_layers': 12,
}
```

### 2.2 训练策略

#### 2.2.1 学习率调度

```python
# Warmup + Cosine Annealing
optimizer = torch.optim.AdamW(model.parameters(), 
                               lr=1e-4, 
                               weight_decay=0.05)

scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                        T_0=10,  # 10 epochs 一个周期
                                        T_mult=2,  # 每个周期翻倍
                                        eta_min=1e-6)
```

#### 2.2.2 正则化技术

1. **Label Smoothing**: 0.1（防止过拟合）
2. **Mixup/CutMix**: α=0.8, β=1.0（数据级增强）
3. **Stochastic Depth**: drop_rate=0.1（针对深层网络）
4. **Gradient Clipping**: max_norm=1.0

#### 2.2.3 训练参数

```python
training_config = {
    'batch_size': 256,  # 根据 GPU 内存调整
    'epochs': 300,  # Phase 1
    'warmup_epochs': 10,
    'optimizer': 'AdamW',
    'lr': 1e-4,
    'weight_decay': 0.05,
    'label_smoothing': 0.1,
    'mixup_alpha': 0.8,
    'cutmix_alpha': 1.0,
    'grad_clip': 1.0,
    'ema_decay': 0.999,  # 指数移动平均
}
```

### 2.3 评估指标

#### 2.3.1 主要指标

1. **Top-1 Accuracy**: 预测概率最高的类别是否正确
2. **Top-5 Accuracy**: 正确类别是否在前 5 个预测中
3. **收敛速度**: 达到目标准确率所需的 epoch

#### 2.3.2 次要指标

1. **Φ值演化曲线**: 监控意识整合度随训练的变化
2. **预测误差（自由能）**: 验证自由能最小化原理
3. **注意力可视化**: Grad-CAM 展示关注区域
4. **计算效率**: FPS、GPU 利用率、参数量

### 2.4 实验步骤

#### Step 1: 环境准备（1 周）

```bash
# 安装依赖
pip install torch torchvision timm mmdet

# 下载 ImageNet-100
# 从官方渠道申请：https://image-net.org/download.php

# 验证数据集
python experiments/validate_imagenet_subset.py
```

#### Step 2: 基线模型实现（2 周）

```bash
# 创建实验脚本
python experiments/run_imagenet_nct_baseline.py \
    --dataset imagenet-100 \
    --config configs/nct_imagenet_base.yaml \
    --gpus 0,1 \
    --batch_size 256
```

#### Step 3: 模型优化迭代（4 周）

```bash
# 实验 1: d_model 扩展
python experiments/run_imagenet_nct_scaled.py \
    --d_model 2048 \
    --n_layers 24

# 实验 2: 超参数搜索
python experiments/run_imagenet_hyperparameter_search.py \
    --search_space imagenet_large

# 实验 3: 集成方法
python experiments/run_imagenet_ensemble.py \
    --models nct_vit_resnet \
    --fusion attention_weighted
```

#### Step 4: 结果分析与可视化（1 周）

```bash
# 生成对比图表
python visualization/plot_imagenet_results.py \
    --results_dir results/imagenet_nct/ \
    --output figures/imagenet_comparison.png

# 注意力可视化
python visualization/visualize_attention_maps.py \
    --checkpoint checkpoints/nct_imagenet_best.pth \
    --images samples/imagenet_test_images/
```

### 2.5 风险与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|-------|------|---------|
| 训练时间过长 | 高 | 高 | 使用混合精度训练（AMP）、梯度累积 |
| 过拟合严重 | 中 | 中 | 增强正则化、早停、数据增强 |
| GPU 资源不足 | 中 | 高 | 申请云平台资源、优化 batch_size |
| 收敛困难 | 低 | 高 | 调整学习率、使用预训练权重初始化 |

---

## 三、EEG 神经信号解码实验

### 3.1 实验设计

#### 3.1.1 数据集选择

**选项 A: 公开数据集（推荐起点）**

1. **SEED 数据集**（情感识别）
   - 15 名被试，3 种情感状态（正性、中性、负性）
   - 62 通道 EEG，采样率 1000Hz
   - 任务：观看视频片段诱发情感
   - 基线准确率：~70%（CNN/LSTM）

2. **BCI Competition IV 2a**（运动想象）
   - 9 名被试，4 类运动想象（左手、右手、脚、舌头）
   - 22 通道 EEG，采样率 250Hz
   - 基线准确率：~75%（CSP + LDA）

3. **TUH EEG Corpus**（临床 EEG）
   - 大规模临床脑电数据
   - 包含多种病理模式
   - 适合异常检测任务

**选项 B: 自主采集（长期目标）**

```python
# 实验范式
# 使用 NCT 实时处理 EEG 信号
# 任务：视觉刺激 Oddball 范式
# 设备：OpenBCI Cyton（32 通道，预算 5000 元）
```

#### 3.1.2 数据预处理流程

```python
import mne
from mne.preprocessing import ICA

def preprocess_eeg(raw_data):
    """EEG 标准化预处理"""
    
    # 1. 滤波
    raw = raw_data.filter(l_freq=0.5, h_freq=100.0)  # 带通滤波
    raw = raw.notch_filter(freqs=[50, 100])  # 工频陷波
    
    # 2. 重参考
    raw = raw.set_eeg_reference('average')
    
    # 3. 坏道检测与插值
    bad_channels = detect_bad_channels(raw)
    raw = raw.interpolate_bads(reset_bads=True)
    
    # 4. ICA 去除伪迹
    ica = ICA(n_components=0.95, method='fastica')
    ica.fit(raw)
    ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    ica.apply(raw)
    
    # 5. 分段
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, 
                        baseline=(-0.2, 0), reject=dict(eeg=150e-6))
    
    return epochs
```

#### 3.1.3 NCT 架构适配

**EEG 编码器设计**:

```python
class EEGEncoder(nn.Module):
    """将 EEG 时序信号转换为 NCT 可处理的 token 序列"""
    
    def __init__(self, n_channels=62, seq_len=200, d_model=768):
        super().__init__()
        
        # 空间编码（通道维度）
        self.spatial_conv = nn.Conv1d(n_channels, d_model, kernel_size=5, padding=2)
        
        # 时间编码（时序维度）
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=11, padding=5)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.spatial_conv(x)  # (batch, d_model, time)
        x = self.temporal_conv(x)  # (batch, d_model, time)
        x = x.transpose(1, 2)  # (batch, time, d_model)
        x = self.pos_encoder(x)
        return x
```

**NCT 配置**:

```python
nct_eeg_config = {
    'input_type': 'eeg',
    'n_channels': 62,
    'seq_len': 200,  # 1 秒 @ 200Hz
    'd_model': 768,
    'n_heads': 12,
    'n_layers': 8,   # EEG 不需要太深
    'd_ff': 2048,
    'num_classes': 3,  # 情感分类
    'gamma_freq': 40,
    'cross_modal_attention': True,  # 通道间注意力
    'temporal_prediction': True,    # 时序预测编码
}
```

### 3.2 训练策略

#### 3.2.1 特征工程

1. **频域特征**: 提取δ/θ/α/β/γ频段功率
2. **时域特征**: ERP 成分（P300, N400 等）
3. **功能连接**: 相位锁定值（PLV）、相干性
4. **微分熵特征**: DE（Differential Entropy）

```python
def extract_band_power(epochs, bands):
    """提取频段功率特征"""
    psd_features = []
    
    for band_name, (low, high) in bands.items():
        psd = epochs.compute_psd(method='welch', fmin=low, fmax=high)
        band_power = psd.get_data().mean(axis=-1)  # 平均功率
        psd_features.append(band_power)
    
    return np.concatenate(psd_features, axis=1)

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50),
}
```

#### 3.2.2 迁移学习策略

由于 EEG 数据量有限，采用迁移学习：

```python
# Phase 1: 在大规模 EEG 数据集上预训练
pretrain_dataset = 'TUH_EEG_Corpus'  # 数千小时临床 EEG
pretrain_task = 'reconstruction'     # 自监督重构任务

# Phase 2: 在目标任务上微调
finetune_dataset = 'SEED'  # 情感识别
finetune_task = 'classification'

# 冻结底层参数
for param in model.eeg_encoder.parameters():
    param.requires_grad = False
    
# 只训练顶层和分类器
optimizer = torch.optim.Adam([
    {'params': model.nct_layers.parameters()},
    {'params': model.classifier.parameters()},
], lr=1e-3)
```

#### 3.2.3 跨被试泛化

```python
# Leave-One-Subject-Out Cross-Validation
for test_subject in all_subjects:
    train_data = [s for s in all_subjects if s != test_subject]
    test_data = test_subject
    
    # 训练
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # 测试
    acc = trainer.test(model, test_dataloader)
    
# 报告平均跨被试准确率
print(f"Cross-subject accuracy: {acc.mean():.2f} ± {acc.std():.2f}")
```

### 3.3 评估指标

#### 3.3.1 分类性能

1. **Accuracy**: 总体分类准确率
2. **F1-Score**: 宏平均 F1
3. **Confusion Matrix**: 混淆矩阵分析
4. **ROC-AUC**: 多类别 AUC

#### 3.3.2 神经科学验证

1. **与人类表现对比**: vs 人类被试情感识别准确率
2. **与经典方法对比**: vs CSP+LDA, vs DeepConvNet, vs EEGNet
3. **可解释性分析**: 
   - 注意力权重 vs 已知神经标记物（如额叶θ波与情感）
   - 显著性图 vs ERP 成分时空分布

#### 3.3.3 Φ值分析

```python
# 计算不同情感状态下的Φ值
phi_positive = compute_phi(model, positive_samples)
phi_neutral = compute_phi(model, neutral_samples)
phi_negative = compute_phi(model, negative_samples)

# 假设：负性情感（高唤醒）→ 更高Φ值
# 验证：与神经科学文献一致性
```

### 3.4 实验步骤

#### Step 1: 数据准备与预处理（2 周）

```bash
# 下载 SEED 数据集
# 链接：http://bcmi.sjtu.edu.cn/~seed/

# 预处理
python experiments/preprocess_eeg.py \
    --dataset SEED \
    --raw_dir data/SEED/raw \
    --processed_dir data/SEED/processed \
    --preprocessing_pipeline standard
```

#### Step 2: 基线模型建立（3 周）

```bash
# 实验 1: 传统机器学习基线
python experiments/run_eeg_baseline.py \
    --method svm \
    --features de_feature \
    --cv loso  # leave-one-subject-out

# 实验 2: 深度学习基线
python experiments/run_eeg_baseline.py \
    --model eegnet \
    --dataset SEED

# 实验 3: NCT 基线
python experiments/run_eeg_nct.py \
    --config configs/nct_eeg_base.yaml \
    --dataset SEED
```

#### Step 3: 模型优化（4 周）

```bash
# 实验 1: 架构搜索
python experiments/run_eeg_architecture_search.py \
    --search_space eeg_nct_large

# 实验 2: 多模态融合（如果有眼动数据）
python experiments/run_eeg_multimodal.py \
    --modalities eeg_eye_tracking

# 实验 3: 个体化适配
python experiments/run_eeg_subject_adaptation.py \
    --adaptation_method finetune \
    --calibration_minutes 5
```

#### Step 4: 神经科学分析（2 周）

```bash
# 注意力可视化
python analysis/visualize_eeg_attention.py \
    --checkpoint checkpoints/nct_eeg_best.pth \
    --output figures/eeg_attention_maps.png

# 频段贡献分析
python analysis/analyze_frequency_contribution.py \
    --model nct \
    --ablation bands

# Φ值 - 情感相关性
python analysis/correlate_phi_emotion.py \
    --dataset SEED \
    --statistical_test anova
```

### 3.5 伦理考量

1. **数据隐私**: 使用公开数据集，确保符合伦理审批
2. **知情同意**: 自主采集时需获得被试书面同意
3. **数据安全**: 匿名化处理，加密存储
4. **结果解读**: 避免过度解读Φ值的"意识"含义

### 3.6 风险与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|-------|------|---------|
| 数据质量差 | 中 | 高 | 严格质控、增加 trial 数量 |
| 个体差异大 | 高 | 中 | 个体化校准、领域自适应 |
| 过拟合 | 高 | 中 | 数据增强、迁移学习、正则化 |
| 可解释性差 | 中 | 低 | 注意力可视化、消融实验 |

---

## 四、资源需求与预算

### 4.1 计算资源

| 项目 | 配置 | 数量 | 预计时长 | 成本估算 |
|------|------|------|---------|---------|
| **ImageNet 实验** | NVIDIA A100 80GB | 2 卡 | 2 个月 | 3 万元（云平台） |
| **EEG 实验** | NVIDIA RTX 4090 | 1 卡 | 1 个月 | 0.5 万元（本地） |
| **存储** | NVMe SSD | 2TB | - | 0.3 万元 |
| **合计** | - | - | - | **3.8 万元** |

### 4.2 数据资源

| 数据集 | 获取方式 | 成本 | 时间 |
|--------|---------|------|------|
| ImageNet-100 | 学术申请 | 免费 | 1 周审批 |
| SEED | 学术申请 | 免费 | 即时下载 |
| BCI Competition | 公开 | 免费 | 即时下载 |
| 自主采集 EEG | OpenBCI 设备 | 0.5 万元 | 2 周采集 |

### 4.3 人力资源

| 角色 | 职责 | 投入时间 |
|------|------|---------|
| 算法工程师 | 模型实现与优化 | 2 人月 |
| 数据分析师 | 数据预处理与可视化 | 1 人月 |
| 神经科学顾问 | EEG 实验设计与解读 | 0.5 人月 |

---

## 五、时间规划

### 5.1 ImageNet 实验时间表

```
2026 Q4:
├─ 第 1-2 周：环境准备、数据集下载
├─ 第 3-6 周：基线模型实现与调试
├─ 第 7-10 周：模型优化迭代
└─ 第 11-12 周：结果分析与论文撰写

2027 Q1:
└─ 第 1-4 周：补充实验、代码开源
```

### 5.2 EEG 实验时间表

```
2027 Q1:
├─ 第 1-2 周：数据申请与预处理
├─ 第 3-6 周：基线模型与 NCT 实现
├─ 第 7-10 周：模型优化与跨被试实验
└─ 第 11-12 周：神经科学分析

2027 Q2:
├─ 第 1-4 周：自主采集验证实验
└─ 第 5-8 周：论文撰写与投稿
```

### 5.3 里程碑节点

| 时间节点 | 交付物 | 验收标准 |
|---------|--------|---------|
| **2026-12-31** | ImageNet-100 基线结果 | Top-1 ≥ 65% |
| **2027-01-31** | ImageNet 优化版 | Top-1 ≥ 72% |
| **2027-02-28** | EEG 基线结果 | Accuracy ≥ 70% |
| **2027-04-30** | EEG 优化版 | Accuracy ≥ 80% |
| **2027-06-30** | 论文投稿 | SCI 一区 1 篇 |

---

## 六、预期产出

### 6.1 学术成果

1. **会议论文**: 
   - NeurIPS 2027（ImageNet 实验）
   - CVPR 2027（ImageNet 可视化）
   - OHBM 2027（EEG 神经科学）

2. **期刊论文**:
   - Nature Communications（综合成果）
   - IEEE TNNLS（EEG 方法学）

### 6.2 技术成果

1. **开源代码**: GitHub 仓库（目标 100+ stars）
2. **预训练模型**: HuggingFace Model Hub
3. **数据集**: 整理后的 EEG 基准数据集

### 6.3 应用前景

1. **医疗应用**: 意识障碍评估系统原型
2. **教育应用**: 注意力监测系统 Demo
3. **工业应用**: 视觉质检 API 服务

---

## 七、风险评估与备选方案

### 7.1 技术风险

**风险 1: ImageNet 训练不收敛**
- **原因**: 学习率不当、初始化问题
- **应对**: 
  - 使用预训练权重（ViT/RN50）初始化
  - 网格搜索学习率
  - 减小 batch_size 增加更新频率

**风险 2: EEG 解码准确率低于预期**
- **原因**: 个体差异、噪声干扰
- **应对**:
  - 增加个体校准环节
  - 使用领域自适应技术
  - 集成多个模型

### 7.2 资源风险

**风险 1: GPU 资源不足**
- **应对**:
  - 申请高校超算中心资源
  - 使用 Google Colab Pro/Kaggle Kernels
  - 与实验室合作共享资源

**风险 2: 数据采集延期**
- **应对**:
  - 优先使用公开数据集
  - 并行开展多个数据集实验

### 7.3 备选方案

如果 ImageNet 全规模实验困难：
- **Plan B**: 专注于 CIFAR-100 → Tiny-ImageNet → ImageNet 渐进路线
- **Plan C**: 与已有 ImageNet 结果的模型做公平对比（相同计算预算）

如果 EEG 自主采集困难：
- **Plan B**: 深度挖掘 3-5 个公开 EEG 数据集
- **Plan C**: 与医院/研究所合作获取临床数据

---

## 八、总结

本实验方案系统规划了 NCT 在 ImageNet 和 EEG 两个重要方向的验证路径：

1. **ImageNet 实验**: 通过渐进式策略（100 类→200 类→1000 类），在可控成本下验证 NCT 的大规模视觉识别能力，目标是在 d_model=2048 配置下达到 75% 准确率。

2. **EEG 实验**: 从公开数据集（SEED、BCI Competition）入手，结合迁移学习和跨被试泛化技术，验证 NCT 在神经信号解码上的应用潜力，同时探索Φ值与意识状态的关联。

两个实验相辅相成：ImageNet 验证通用表征能力，EEG 验证专业领域应用价值。预期成果将显著提升 NCT 的学术影响力和实用价值。

**关键成功要素**:
- ✅ 充足的计算资源保障
- ✅ 严谨的实验设计与对照
- ✅ 及时的阶段性复盘与调整
- ✅ 开放的科学态度（代码/数据开源）

**下一步行动**:
1. [ ] 申请 ImageNet 数据集访问权限
2. [ ] 采购/租赁 GPU 服务器
3. [ ] 组建实验团队（算法 + 神经科学）
4. [ ] 启动预实验验证流程

---

## 附录

### A. 关键参考文献

1. **ImageNet 相关**:
   - Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*.
   - He et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
   - Huang et al. (2017). "Densely Connected Convolutional Networks." *CVPR*.

2. **EEG 解码相关**:
   - Lawhern et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." *Journal of Neural Engineering*.
   - Schirrmeister et al. (2017). "Deep learning with convolutional neural networks for EEG decoding and visualization." *Human Brain Mapping*.
   - Zheng & Lu (2015). "Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition." *IEEE TAC*.

3. **NCT 理论基础**:
   - 内部技术报告：[NCT_Report_vs_Experiments_Analysis.md](file://docs/NCT_Report_vs_Experiments_Analysis.md)
   - 路线图规划：[17_NCT 发展路线图_短期中长期规划.md](file://docs/NCT技术博客专栏 16 篇/17_NCT 发展路线图_短期中长期规划.md)

### B. 实验记录模板

```markdown
## 实验记录

**实验 ID**: IMG-001
**日期**: 2026-XX-XX
**数据集**: ImageNet-100
**模型**: NCT-Base (d=768)

### 超参数
- batch_size: 256
- lr: 1e-4
- epochs: 100

### 结果
- Train Acc@1: 68.5%
- Val Acc@1: 65.2%
- Φ值：2.34 bits

### 观察
- 收敛速度正常
- 第 50 epoch 开始过拟合
- 注意力图显示合理关注区域

### 改进方向
- 增加正则化
- 尝试 cosine lr schedule
```

### C. 联系人信息

- **项目负责人**: [待填写]
- **技术支持**: [待填写]
- **神经科学顾问**: [待填写]

---

*文档版本：v1.0*  
*创建日期：2026 年 3 月 9 日*  
*最后更新：2026 年 3 月 9 日*
