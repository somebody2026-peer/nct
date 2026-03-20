
我来帮你梳理分析这个教育数据集实验脚本文件夹内的所有脚本。让我先查看每个脚本的内容，然后生成详细的使用清单。


现在我已经阅读了所有脚本的内容，让我生成一个详细的使用清单：

## 📚 教育数据集实验脚本使用清单

基于对 `educational_scripts` 文件夹内 18 个脚本的完整分析，以下是详细的使用清单：

---

### 🎯 **核心实验流程脚本**

#### 1. **run_education_experiments_v2.py** - V2 实验统一入口
- **作用**: NCT 教育领域 V2 实验的总控制脚本，协调所有 Phase 实验
- **输入**: 
  - 命令行参数 (`--phase all/fer/daisee/mema`)
  - FER2013 数据集 → `data/fer2013/`
  - DAiSEE 数据集 → `data/daisee/`
  - MEMA 数据集 → `data/mema/`
- **输出**: 
  - `results/education_v2/phase1_daisee_v2.json`
  - `results/education_v2/phase2_fer_v2.json`
  - `results/education_v2/phase3_mema_v2.json`
  - `results/education_v2/combined_report_v2.json`
- **使用方法**: `python run_education_experiments_v2.py --phase all`

#### 2. **run_education_experiments_v3.py** - V3 实验总控脚本
- **作用**: NCT 教育领域 V3 实验总控，包含 Day 0 验证和分阶段实验
- **输入**: 
  - 命令行参数 (`--day0/--phase1/--all/--skip-day0`)
  - MEMA 数据集 → `data/mema/`
- **输出**: 
  - `results/education_v3/day0_validation.json`
  - `results/education_v3/phase1_results.json`
  - `logs/v3_experiment.log`
- **使用方法**: `python run_education_experiments_v3.py --all`

---

### 📺 **Phase 1: 视觉验证实验（DAiSEE 数据集）**

#### 3. **daisee_nct_experiment.py** - V1 视觉验证实验
- **作用**: 使用 DAiSEE 视频数据验证 NCT 视觉编码器对学生情感状态的识别
- **输入**: 
  - DAiSEE 视频文件 → `data/daisee/DataSet/`
  - DAiSEE 标签 → `data/daisee/Labels/`
- **输出**: 
  - `results/education/phase1_daisee.json`
  - Φ值与注意力状态的相关性分析结果
- **处理流程**: 视频帧 → 视觉特征 → StudentState → NeuromodulatorState → NCT → Φ值

#### 4. **daisee_nct_experiment_v2.py** - V2 视觉参与度检测实验
- **作用**: V2 改进版，使用 ResNet18 + MediaPipe Face Mesh + LSTM 时序编码
- **输入**: 
  - DAiSEE 视频数据集
  - 预训练 FER 模型 → `checkpoints/education_v2/fer_resnet18.pth`
- **输出**: 
  - `results/education_v2/phase1_daisee_v2.json`
  - 参与度检测结果和时序情绪变化
- **V2 改进**: ResNet18 FER 模型、MediaPipe Face Mesh、LSTM 时序编码、可学习状态映射器

---

### 😊 **Phase 2: 面部表情预训练（FER2013 数据集）**

#### 5. **fer_pretrain.py** - V1 轻量 CNN 表情分类器
- **作用**: 训练轻量级 CNN 进行 7 类面部表情识别，映射到 NCT 神经调质参数
- **输入**: 
  - FER2013 CSV 数据 → `data/fer2013/fer2013.csv`
- **输出**: 
  - `checkpoints/fer_pretrain/fer_cnn.pth` (模型权重)
  - `results/education/phase2_fer.json` (分类准确率和混淆矩阵)
- **映射关系**: 7 种情绪 → 神经调质 (DA/5-HT/NE/ACh)

#### 6. **fer_pretrain_v2.py** - V2 ResNet18 预训练模型
- **作用**: V2 改进版，使用预训练 ResNet18 提升准确率至 68%+
- **输入**: 
  - FER2013 数据集 → `data/fer2013/`
- **输出**: 
  - `checkpoints/education_v2/fer_resnet18_best.pth`
  - `results/education_v2/phase2_fer_v2.json`
- **V2 改进**: 预训练 ResNet18、数据增强、学习率调度、早停机制

---

### 🧠 **Phase 3: EEG神经调质映射（MEMA 数据集）**

#### 7. **mema_nct_experiment.py** - V1 核心 EEG 映射实验
- **作用**: 验证 EEG 频带功率与 NCT 神经调质参数的生理合理性
- **输入**: 
  - MEMA 数据集 → `data/mema/Subject{1-20}_attention.mat`
- **输出**: 
  - `results/education/phase3_mema.json`
  - 实验 A: 3 类分类基准 (NCT vs SVM vs LSTM) F1 分数
  - 实验 B: Concentrating vs Relaxing 的Φ值 t 检验
  - 实验 C: 4 种神经调质时序曲线
- **映射关系**: Theta↔ACh, Alpha↔5-HT, Beta↔DA, Theta/Alpha↔NE

#### 8. **mema_nct_experiment_v2.py** - V2 CWT 频谱 + 神经网络映射
- **作用**: V2 改进版，使用 CWT 替代 Welch 频谱，神经网络替代固定 sigmoid
- **输入**: 
  - MEMA 数据集
  - 可选预训练 EEG 映射网络
- **输出**: 
  - `results/education_v2/phase3_mema_v2.json`
  - CWT 时频特征和可学习映射权重
- **V2 改进**: CWT 频谱估计、可学习神经网络、个体基线校准、时序特征提取

#### 9. **mema_nct_experiment_v3.py** - V3 EEGNet 端到端分类
- **作用**: V3 版本使用 EEGNet 进行端到端分类，添加替代指标和统计分析
- **输入**: 
  - MEMA 数据集
  - 预训练 EEGNet 模型 (可选)
- **输出**: 
  - `results/education_v3/phase3_mema_v3.json`
  - Cohen's d 效应量、谱熵、排列熵分析
- **V3 改进**: EEGNet 端到端分类、替代指标 (谱熵/排列熵/因果复杂度)、Cohen's d 效应量

---

### 🔬 **消融实验与分析脚本**

#### 10. **ablation_feature_dimension.py** - 特征维度消融实验
- **作用**: 验证不同特征维度对Φ区分能力的影响
- **输入**: 
  - MEMA 数据集
  - 测试维度：5/20/50/100/320/640 维
- **输出**: 
  - `results/education_v3/ablation_dimensionality.json`
  - 各维度下的Φ区分能力和统计显著性
- **实验设计**: 
  - 5 维：简单频带功率
  - 20 维：频带功率 + 统计量
  - 50/100/320 维：PCA 降维
  - 640 维：完整 EEGNet 特征

#### 11. **eegnet_phi_experiment.py** - EEGNet 特征→Φ计算实验
- **作用**: 验证深度学习特征能否改善Φ的状态区分能力
- **输入**: 
  - MEMA 数据集
  - 预训练 EEGNet 模型 → `checkpoints/education_v3/eegnet_v3_best.pt`
- **输出**: 
  - `results/education_v3/eegnet_phi_analysis.json`
  - 简单特征 vs EEGNet特征的Φ区分能力对比
- **对比内容**: 传统频带功率特征 vs EEGNet 高层特征

#### 12. **day0_validation_v3.py** - V3 Day 0 前置验证
- **作用**: 在正式实验前验证数据质量和因果假设
- **输入**: 
  - MEMA 数据集
- **输出**: 
  - `results/education_v3/day0_validation.json`
  - 数据质量报告 (SNR 信噪比)
  - 因果假设验证结果
  - 类别不平衡评估
- **决策规则**: 验证失败则建议跳过 Phase 1-2，直接 Phase 3

---

### 🛠️ **工具组件脚本**

#### 13. **education_state_detection.py** - 学生状态智能检测系统
- **作用**: 多模态感知的学生状态评估与 NCT 参数映射
- **输入**: 
  - 视觉数据 (摄像头)
  - 行为数据 (姿态/动作)
  - 交互数据 (鼠标/键盘)
- **输出**: 
  - StudentState 对象 (专注度/困惑度/疲劳度/参与度等)
  - NeuromodulatorState 对象 (DA/5-HT/NE/ACh水平)
- **核心功能**: 多模态数据采集、状态识别、神经调质映射、自适应学习策略

#### 14. **education_state_mapper_v2.py** - 学生状态→神经调质可学习映射器 V2
- **作用**: 使用神经网络实现从学生状态到神经调质的非线性映射
- **输入**: 
  - StudentState 对象 (6 维状态向量)
- **输出**: 
  - NeuromodulatorState 对象 (4 维神经调质向量)
  - 可解释性权重分析
- **V2 改进**: 神经网络替代固定线性加权、理论先验正则项、端到端训练

#### 15. **eeg_neuromodulator_net_v2.py** - EEG→神经调质非线性映射器 V2
- **作用**: 从 EEG 信号到神经调质参数的可学习非线性映射
- **输入**: 
  - EEG 原始信号或频带功率特征
- **输出**: 
  - 神经调质参数 (DA/5-HT/NE/ACh)
  - 个体基线校准后的参数
- **V2 改进**: 神经网络替代 sigmoid、CWT 时频特征、个体基线校准、滑动窗口时序特征

#### 16. **eegnet_classifier_v3.py** - EEGNet 分类器 V3
- **作用**: 基于 EEGNet 架构的 EEG 状态分类器
- **输入**: 
  - EEG 数据 (n_channels, n_timepoints)
  - 标签 (neutral/relaxing/concentrating)
- **输出**: 
  - `checkpoints/education_v3/eegnet_v3_best.pt`
  - `results/education_v3/eegnet_v3_results.json`
  - 交叉验证 F1 分数、Cohen's Kappa、分类报告
- **特点**: 轻量级架构、深度可分离卷积、类别加权损失、5 折交叉验证

#### 17. **face_landmark_extractor_v2.py** - MediaPipe 面部关键点提取器 V2
- **作用**: 使用 MediaPipe Face Mesh 提取 478 个 3D 面部关键点
- **输入**: 
  - BGR 图像帧 (摄像头或视频)
- **输出**: 
  - FacialFeatures 对象 (EAR、眉毛高度、嘴部特征、头部姿态、AU 特征)
  - 15 维特征向量
- **V2 改进**: MediaPipe Face Mesh、478 个精确关键点、AU 特征计算、实时性能优化

#### 18. **temporal_emotion_encoder_v2.py** - 时序情绪编码器 V2 (LSTM)
- **作用**: 使用双向 LSTM 捕捉时序情绪变化
- **输入**: 
  - 多帧 FER 特征序列
  - 面部关键点特征序列
- **输出**: 
  - 时序感知的情绪状态
  - 神经调质动态变化
- **V2 改进**: 双向 LSTM、注意力机制、多模态特征融合

---

### 📊 **实验依赖关系图**

```
run_education_experiments_v2.py/v3.py (总控)
├── Day 0 Validation (day0_validation_v3.py)
│   └── 数据质量检查、因果假设验证
│
├── Phase 1: 视觉验证 (DAiSEE)
│   ├── daisee_nct_experiment.py (V1)
│   ├── daisee_nct_experiment_v2.py (V2)
│   │   ├── fer_pretrain_v2.py (FER 预训练)
│   │   ├── face_landmark_extractor_v2.py (面部关键点)
│   │   ├── temporal_emotion_encoder_v2.py (LSTM 时序)
│   │   └── education_state_mapper_v2.py (状态映射)
│   └── education_state_detection.py (状态检测)
│
├── Phase 2: 面部表情 (FER2013)
│   ├── fer_pretrain.py (V1 CNN)
│   └── fer_pretrain_v2.py (V2 ResNet18)
│
└── Phase 3: EEG 映射 (MEMA)
    ├── mema_nct_experiment.py (V1)
    ├── mema_nct_experiment_v2.py (V2 CWT)
    ├── mema_nct_experiment_v3.py (V3 EEGNet)
    │   └── eegnet_classifier_v3.py (EEGNet 模型)
    ├── eeg_neuromodulator_net_v2.py (EEG→NM 映射)
    ├── eegnet_phi_experiment.py (Φ分析)
    └── ablation_feature_dimension.py (消融实验)
```

---

### 🔑 **关键输入/输出汇总**

| 脚本名称 | 主要输入 | 主要输出 | 执行时间估算 |
|---------|---------|---------|-------------|
| **run_education_experiments_v2.py** | 3 个数据集 | 4 个 JSON 报告 | 2-4 小时 |
| **run_education_experiments_v3.py** | MEMA 数据集 | 3 个 JSON 报告 + 日志 | 1-2 小时 |
| **daisee_nct_experiment_v2.py** | DAiSEE 视频 | phase1_daisee_v2.json | 30-60 分钟 |
| **fer_pretrain_v2.py** | FER2013 CSV | fer_resnet18_best.pth | 20-40 分钟 |
| **mema_nct_experiment_v3.py** | MEMA .mat 文件 | phase3_mema_v3.json | 30-50 分钟 |
| **eegnet_classifier_v3.py** | MEMA EEG 数据 | eegnet_v3_results.json | 15-30 分钟 |
| **ablation_feature_dimension.py** | MEMA 数据集 | ablation_dimensionality.json | 40-80 分钟 |

---

### 💡 **推荐使用顺序**

1. **初次运行**: 从 `run_education_experiments_v3.py --day0` 开始，验证数据质量
2. **Phase 1**: 运行 DAiSEE 实验前，先执行 `fer_pretrain_v2.py` 训练 FER 模型
3. **Phase 2**: 可直接运行 `fer_pretrain_v2.py`
4. **Phase 3**: 推荐先运行 `eegnet_classifier_v3.py` 训练 EEGNet，再运行 `mema_nct_experiment_v3.py`
5. **深入分析**: 运行 `ablation_feature_dimension.py` 和 `eegnet_phi_experiment.py` 进行消融实验

---

这个清单涵盖了所有 18 个脚本的完整使用说明，包括输入、输出、作用和相互依赖关系。 