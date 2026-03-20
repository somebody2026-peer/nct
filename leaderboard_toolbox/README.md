# 🏆 NCT 榜单申报工具箱

本目录包含将 NCT 模型提交到各大主流评测榜单所需的所有工具和指南。

---

## 📋 快速开始

### 3 步完成榜单提交

```bash
# 1️⃣ 训练模型 (如果还没有)
python experiments/run_full_mnist_training.py

# 2️⃣ 生成提交文件
python examples/quickstart_leaderboard_submission.py --dataset mnist

# 3️⃣ 上传到 Kaggle
# 访问 https://www.kaggle.com/competitions/digit-recognizer
# 点击 "Submit Prediction" 并上传生成的 CSV 文件
```

---

## 🛠️ 可用工具

### 1. **快速示例脚本** (推荐新手)

**位置**: `examples/quickstart_leaderboard_submission.py`

**用途**: 快速从已有模型生成 Kaggle 提交文件

**使用方法**:
```bash
# MNIST
python examples/quickstart_leaderboard_submission.py --dataset mnist

# CIFAR-10
python examples/quickstart_leaderboard_submission.py --dataset cifar10

# 指定模型路径
python examples/quickstart_leaderboard_submission.py \
    --dataset mnist \
    --model results/full_mnist_training/best_model.pt
```

**输出**:
- ✅ Kaggle 格式 CSV 文件
- ✅ 预测分布统计
- ✅ 下一步操作指引

---

### 2. **高级提交生成器** (完整功能)

**位置**: `tools/generate_leaderboard_submission.py`

**用途**: 生成多种格式的提交文件（Kaggle、Papers With Code 等）

**使用方法**:
```bash
# 完整模式：生成所有格式
python tools/generate_leaderboard_submission.py \
    --dataset mnist \
    --model results/full_mnist_training/best_model.pt

# 仅生成预测（不创建报告）
python tools/generate_leaderboard_submission.py \
    --dataset mnist \
    --model results/full_mnist_training/best_model.pt \
    --mode predict_only

# 仅生成 Papers With Code 报告
python tools/generate_leaderboard_submission.py \
    --dataset mnist \
    --mode report_only
```

**输出**:
- 📄 Kaggle submission CSV
- 📊 预测分布可视化 (PNG)
- 📝 Papers With Code 报告 (Markdown)
- 📈 统计信息 (JSON)

---

### 3. **完整指南文档**

**位置**: `docs/leaderboard_submission_guide.md`

**内容**:
- ✅ 主流榜单平台介绍
- ✅ 详细提交流程图解
- ✅ 提交文件格式要求
- ✅ 常见错误及避免方法
- ✅ 提高排名的技巧
- ✅ FAQ 常见问题解答

**在线阅读**: [查看文档](../docs/leaderboard_submission_guide.md)

---

## 📊 支持的榜单平台

### Kaggle (推荐 ⭐⭐⭐⭐⭐)
- **MNIST - Digit Recognizer**: https://www.kaggle.com/competitions/digit-recognizer
- **CIFAR-10 - Object Recognition**: https://www.kaggle.com/c/cifar-10

**特点**: 
- 实时排行榜
- 自动评分
- 全球最大数据科学社区

### Papers With Code (学术 ⭐⭐⭐⭐⭐)
- **MNIST Leaderboard**: https://paperswithcode.com/sota/image-classification-on-mnist
- **CIFAR-10 Leaderboard**: https://paperswithcode.com/sota/image-classification-on-cifar-10

**特点**:
- 学术界认可度高
- 需要关联论文/代码
- 便于同行对比

### Hugging Face Spaces (新兴 ⭐⭐⭐⭐)
- **Vision Tasks**: https://huggingface.co/spaces

**特点**:
- 可创建交互式演示
- 社区活跃
- 支持在线体验

---

## 📁 文件结构说明

```
submissions/                    # 提交文件输出目录
├── mnist_20260227_120000/     # 按时间戳组织的提交包
│   ├── mnist_submission.csv          # Kaggle 提交文件
│   ├── submission_stats.json         # 统计信息
│   ├── prediction_distribution.png   # 预测分布图
│   └── paperswithcode_report.md      # Papers With Code 报告
└── cifar10_20260227_150000/
    ├── cifar10_submission.csv
    ├── submission_stats.json
    ├── prediction_distribution.png
    └── paperswithcode_report.md
```

---

## 🎯 典型工作流程

### 场景 1: 首次提交

```bash
# 1. 训练模型
python experiments/run_full_mnist_training.py --epochs 50

# 2. 验证模型性能
python tests/test_complete.py

# 3. 生成提交文件
python examples/quickstart_leaderboard_submission.py --dataset mnist

# 4. 上传到 Kaggle
# 访问 https://www.kaggle.com/competitions/digit-recognizer
# 上传生成的 CSV 文件

# 5. 记录结果
# 在实验日志中记录分数和排名
```

### 场景 2: 优化后重新提交

```bash
# 1. 改进模型（例如调整超参数）
python experiments/run_optimized_training_v3.py --lr 0.0005

# 2. 生成新的提交文件
python tools/generate_leaderboard_submission.py \
    --dataset mnist \
    --model results/optimized_mnist/best_model.pt

# 3. 对比两次提交的结果
# Kaggle 会显示历史提交记录
# 分析哪次提交效果更好
```

### 场景 3: 多数据集提交

```bash
# 1. MNIST 提交
python examples/quickstart_leaderboard_submission.py --dataset mnist

# 2. CIFAR-10 提交（使用迁移学习）
python experiments/run_cifar10_full.py \
    --pretrained results/full_mnist_training/best_model.pt

python examples/quickstart_leaderboard_submission.py --dataset cifar10

# 3. Papers With Code 提交（两个数据集）
python tools/generate_leaderboard_submission.py \
    --dataset mnist \
    --mode full

python tools/generate_leaderboard_submission.py \
    --dataset cifar10 \
    --mode full
```

---

## 💡 提高榜单排名的技巧

### 1. 数据增强
```python
# 在训练脚本中添加
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### 2. 模型集成
```bash
# 训练多个不同初始化的模型
python experiments/run_full_mnist_training.py --seed 42
python experiments/run_full_mnist_training.py --seed 123
python experiments/run_full_mnist_training.py --seed 456

# 使用集成工具生成提交（开发中）
python tools/ensemble_predictions.py \
    --models model_seed42.pt model_seed123.pt model_seed456.pt
```

### 3. 测试时增强 (TTA)
```python
# 在推理阶段对同一图像进行多次增强
# 平均所有增强的预测结果
# 可以提高 1-2% 的准确率
```

### 4. 超参数优化
```bash
# 使用 Optuna 等工具自动搜索最优参数
python experiments/optimize_hyperparameters.py \
    --dataset mnist \
    --n_trials 100
```

---

## ⚠️ 常见错误及解决方案

### 错误 1: CSV 格式不正确
```
❌ ImageId 不连续
❌ Label 超出范围 (0-9)
❌ 缺少表头

✅ 使用提供的工具自动生成，避免手动编辑
```

### 错误 2: 模型加载失败
```
❌ 路径错误
❌ PyTorch 版本不兼容
❌ 模型配置文件缺失

✅ 确保使用绝对路径，检查 requirements.txt
```

### 错误 3: 违反比赛规则
```
❌ 使用外部数据（如果禁止）
❌ 使用测试集标签
❌ 重复提交相同结果

✅ 仔细阅读每个比赛的 Rules 页面
```

---

## 📈 提交后的操作

### 1. 监控排名
- 定期查看 Kaggle 排行榜
- 记录每次提交的分数
- 分析与其他参赛者的差距

### 2. 结果分析
```bash
# 生成混淆矩阵和误差分析
python tools/analyze_submission.py \
    --submission submissions/mnist_20260227_120000/mnist_submission.csv
```

### 3. 撰写技术报告
- 描述 NCT 架构创新
- 详细实验设置
- 对比传统方法
- 开源代码和数据

---

## 🔗 相关资源

### 官方文档
- [Kaggle 竞赛指南](https://www.kaggle.com/docs/competitions)
- [Papers With Code 提交指南](https://paperswithcode.com/about)

### 优秀案例
- [MNIST 99.6% 方案](https://www.kaggle.com/code/paulbacher/mnist-99-6-accuracy-top-10-leaderboard)
- [CIFAR-10 入门教程](https://www.kaggle.com/code/vakninmaor/cifar-10-for-beginners-score-90)

### 工具推荐
- [Weights & Biases](https://wandb.ai/) - 实验追踪
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 可视化
- [Optuna](https://optuna.org/) - 超参数优化

---

## ❓ 常见问题 (FAQ)

### Q: 我的模型应该提交到哪个榜单？
**A**: 
- **学术研究** → Papers With Code
- **工程实践** → Kaggle
- **两者兼顾** → 同时提交

### Q: 提交后多久能看到分数？
**A**:
- Kaggle: 几分钟内
- Papers With Code: 1-3 个工作日（人工审核）

### Q: 可以多次提交吗？
**A**:
- Kaggle: 每天最多 5 次
- Papers With Code: 无限制（需审核）

### Q: 榜单排名有什么实际用处？
**A**:
- ✅ 提升学术声誉
- ✅ 论文评审支撑材料
- ✅ 求职能力证明
- ✅ 吸引潜在合作
- ✅ 基金申请成果展示

---

## 🎓 总结

NCT 作为一个创新的神经形态架构，已经在 MNIST 和 CIFAR-10 等标准数据集上取得了优异成绩。通过积极参与榜单竞争，您可以：

1. **验证方法有效性** - 与 SOTA 方法直接对比
2. **获得社区反馈** - 发现改进方向
3. **提升影响力** - 吸引更多关注和合作
4. **促进可复现性** - 推动开放科学研究

**立即开始**:
```bash
python examples/quickstart_leaderboard_submission.py --dataset mnist
```

祝您好运！🚀

---

**最后更新**: 2026 年 2 月 27 日  
**维护者**: NeuroConscious Research Team  
**联系方式**: neuroconscious@example.com
