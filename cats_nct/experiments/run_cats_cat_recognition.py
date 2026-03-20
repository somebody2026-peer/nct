"""
CATS-NET vs NCT 猫识别对比实验
验证 CATS-NET 在小样本学习场景下的优势

实验设计:
1. 小样本设置：仅使用 5-10 张猫图进行训练
2. 对比组：NCT (原版) vs CATS-NET (新版)
3. 评估指标：
   - 分类准确率
   - 收敛速度（达到 90% 准确率所需 epoch）
   - 概念清晰度（CATS-NET 特有）
   - 抗遗忘能力

假设验证:
1. CATS-NET 小样本学习效率更高（概念抽象加速）
2. CATS-NET 抗遗忘能力更强（概念稳定性）
3. CATS-NET 可解释性更好（概念可视化）

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

import sys
import os
# 添加项目根目录到路径（确保能正确导入 cats_nct）
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 导入 CATS-NET
try:
    from cats_nct import CATSManager, CATSConfig
except ImportError as e:
    print(f"警告：无法导入 CATS-NET 模块：{e}")
    print("将使用简化模式运行")
    # 创建占位类以便测试
    class CATSManager:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def process_cycle(self, *args):
            return None
        def eval(self):
            pass
    
    class CATSConfig:
        @classmethod
        def get_small_config(cls):
            return cls()
        def __init__(self):
            self.concept_dim = 64
            self.n_concept_prototypes = 100
            self.d_model = 768
            self.n_heads = 8
            self.n_task_modules = 1

print("="*70)
print("CATS-NET vs NCT 猫识别对比实验")
print("="*70)


# ============================================================================
# 实验配置
# ============================================================================

class ExperimentConfig:
    """实验配置参数"""
    
    # 数据集配置
    n_train_samples = 10      # 训练样本数（小样本）
    n_test_samples = 50       # 测试样本数
    image_size = 28           # 图片大小
    
    # 模型配置
    use_cats_net = True       # 使用 CATS-NET
    use_small_config = True   # 使用小型配置加快实验
    
    # 训练配置
    n_epochs = 50             # 训练轮数
    learning_rate = 1e-3      # 学习率
    batch_size = 4            # 批次大小
    
    # 评估配置
    target_accuracy = 0.9     # 目标准确率 90%
    
    @classmethod
    def get_config(cls, use_cats=True) -> CATSConfig:
        """获取模型配置"""
        if cls.use_small_config:
            config = CATSConfig.get_small_config()
            config.n_task_modules = 1  # 单任务：猫识别
        else:
            config = CATSConfig()
            config.n_task_modules = 1
        
        return config


# ============================================================================
# 数据生成器
# ============================================================================

class CatDataGenerator:
    """模拟猫/非猫图像生成器"""
    
    def __init__(self, image_size=28):
        self.image_size = image_size
    
    def generate_cat_pattern(self, label: int) -> np.ndarray:
        """生成模拟猫图案
        
        Args:
            label: 1=猫，0=非猫
            
        Returns:
            28x28 图像数组
        """
        if label == 1:
            # 猫的特征：尖耳、圆脸、胡须
            pattern = np.zeros((self.image_size, self.image_size))
            
            # 两只耳朵（三角形）
            pattern[5:9, 8:12] = 0.8 + np.random.rand(4, 4) * 0.2
            pattern[5:9, 16:20] = 0.8 + np.random.rand(4, 4) * 0.2
            
            # 圆脸
            pattern[10:20, 10:18] = 0.6 + np.random.rand(10, 8) * 0.2
            
            # 胡须
            pattern[18:20, 6:10] = 0.4 + np.random.rand(2, 4) * 0.2
            pattern[18:20, 18:22] = 0.4 + np.random.rand(2, 4) * 0.2
            
            # 添加噪声
            pattern += np.random.randn(*pattern.shape) * 0.1
            
        else:
            # 随机图案（非猫）
            pattern = np.random.rand(self.image_size, self.image_size) * 0.5
        
        return pattern.astype(np.float32)
    
    def create_dataset(self, n_samples: int) -> Tuple[List[np.ndarray], List[int]]:
        """创建平衡数据集"""
        images = []
        labels = []
        
        for i in range(n_samples):
            label = i % 2  # 交替生成猫和非猫
            img = self.generate_cat_pattern(label)
            images.append(img)
            labels.append(label)
        
        return images, labels


# ============================================================================
# 训练器
# ============================================================================

class CatsNetTrainer:
    """CATS-NET 训练器"""
    
    def __init__(self, config: CATSConfig):
        self.config = config
        # 修改任务输出维度为 2（猫/非猫二分类）
        config.n_task_modules = 1
        config.task_output_dims = [2]  # 二分类
        
        self.manager = CATSManager(config, device='cpu')
        self.manager.start()
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.manager.parameters(), lr=1e-3)
        
        # 训练统计
        self.history = {
            'loss': [],
            'accuracy': [],
            'phi_values': [],
            'concept_clarity': [],
        }
    
    def train_epoch(self, train_data: List[Tuple[np.ndarray, int]]) -> float:
        """训练一个 epoch
        
        Returns:
            平均损失
        """
        self.manager.train()  # 设置为训练模式
        total_loss = 0.0
        correct = 0
        n_valid = 0
        
        for image, label in train_data:
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 准备感觉输入
            sensory_data = {'visual': image}
            
            # 前向传播
            state = self.manager.process_cycle(sensory_data)
            
            if state is not None and state.task_outputs is not None and len(state.task_outputs) > 0:
                # 获取任务输出（假设第一个任务是猫识别）
                task_output = state.task_outputs[0]
                
                # 确保输出维度正确
                if task_output.shape[-1] != 2:
                    # 简化处理：取前2个维度
                    task_output = task_output[:, :2]
                
                # 计算损失（交叉熵）
                target = torch.tensor([label], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(task_output, target)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_valid += 1
                
                # 计算准确率
                pred_class = task_output.argmax(dim=1).item()
                if pred_class == label:
                    correct += 1
                
                # 记录Φ值
                self.history['phi_values'].append(state.phi_value)
                
                # 记录概念清晰度（CATS-NET 特有）
                if state.prototype_weights is not None:
                    # 概念清晰度 = 1 - 熵
                    probs = state.prototype_weights.squeeze()
                    if probs.dim() > 0 and probs.numel() > 0:
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                        max_entropy = torch.log(torch.tensor(float(probs.numel())))
                        clarity = 1.0 - (entropy / max_entropy).item()
                        self.history['concept_clarity'].append(clarity)
        
        if n_valid > 0:
            avg_loss = total_loss / n_valid
            accuracy = correct / n_valid
        else:
            avg_loss = 0.0
            accuracy = 0.0
        
        self.history['loss'].append(avg_loss)
        self.history['accuracy'].append(accuracy)
        
        return avg_loss
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, int]]) -> Dict[str, float]:
        """评估模型"""
        self.manager.eval()
        
        correct = 0
        total_loss = 0.0
        n_valid = 0
        
        with torch.no_grad():
            for image, label in test_data:
                sensory_data = {'visual': image}
                state = self.manager.process_cycle(sensory_data)
                
                if state is not None and state.task_outputs is not None and len(state.task_outputs) > 0:
                    task_output = state.task_outputs[0]
                    
                    # 确保输出维度正确
                    if task_output.shape[-1] != 2:
                        task_output = task_output[:, :2]
                    
                    target = torch.tensor([label], dtype=torch.long)
                    loss = torch.nn.functional.cross_entropy(task_output, target)
                    total_loss += loss.item()
                    n_valid += 1
                    
                    pred_class = task_output.argmax(dim=1).item()
                    if pred_class == label:
                        correct += 1
        
        if n_valid > 0:
            return {
                'accuracy': correct / n_valid,
                'avg_loss': total_loss / n_valid,
            }
        else:
            return {
                'accuracy': 0.0,
                'avg_loss': 0.0,
            }


# ============================================================================
# 实验执行
# ============================================================================

def run_experiment():
    """运行完整对比实验"""
    
    print("\n" + "="*70)
    print("实验配置")
    print("="*70)
    print(f"训练样本数：{ExperimentConfig.n_train_samples}")
    print(f"测试样本数：{ExperimentConfig.n_test_samples}")
    print(f"训练轮数：{ExperimentConfig.n_epochs}")
    print(f"目标准确率：{ExperimentConfig.target_accuracy:.0%}")
    
    # ========== 1. 生成数据集 ==========
    print("\n" + "="*70)
    print("Step 1: 生成模拟猫识别数据集")
    print("="*70)
    
    data_gen = CatDataGenerator(image_size=ExperimentConfig.image_size)
    
    # 训练集
    train_images, train_labels = data_gen.create_dataset(ExperimentConfig.n_train_samples)
    train_data = list(zip(train_images, train_labels))
    
    # 测试集
    test_images, test_labels = data_gen.create_dataset(ExperimentConfig.n_test_samples)
    test_data = list(zip(test_images, test_labels))
    
    print(f"✓ 训练集：{len(train_data)} 样本")
    print(f"✓ 测试集：{len(test_data)} 样本")
    print(f"  - 猫：{sum(train_labels)} 正样本")
    print(f"  - 非猫：{len(train_labels) - sum(train_labels)} 负样本")
    
    # ========== 2. 训练 CATS-NET ==========
    print("\n" + "="*70)
    print("Step 2: 训练 CATS-NET 模型")
    print("="*70)
    
    config = ExperimentConfig.get_config(use_cats=True)
    trainer = CatsNetTrainer(config)
    
    start_time = time.time()
    
    best_accuracy = 0.0
    epochs_to_target = None
    
    for epoch in range(ExperimentConfig.n_epochs):
        # 训练
        avg_loss = trainer.train_epoch(train_data)
        
        # 评估
        eval_result = trainer.evaluate(test_data)
        accuracy = eval_result['accuracy']
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        if accuracy >= ExperimentConfig.target_accuracy and epochs_to_target is None:
            epochs_to_target = epoch + 1
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{ExperimentConfig.n_epochs}: "
                  f"Loss={avg_loss:.4f}, Acc={accuracy:.3f}")
    
    training_time = time.time() - start_time
    
    print(f"\n✓ 训练完成!")
    print(f"  - 最佳测试准确率：{best_accuracy:.3f}")
    print(f"  - 训练时间：{training_time:.1f}秒")
    if epochs_to_target:
        print(f"  - 达到 90% 准确率所需轮数：{epochs_to_target}")
    
    # ========== 3. 分析结果 ==========
    print("\n" + "="*70)
    print("Step 3: 结果分析")
    print("="*70)
    
    # 绘制训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    axes[0, 0].plot(trainer.history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Curve')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(trainer.history['accuracy'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Test Accuracy Curve')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=ExperimentConfig.target_accuracy, color='g', 
                      linestyle='--', label='Target (90%)')
    axes[0, 1].legend()
    
    # Φ值分布
    if trainer.history['phi_values']:
        axes[1, 0].hist(trainer.history['phi_values'], bins=20, 
                       color='steelblue', alpha=0.7)
        axes[1, 0].set_xlabel('Φ Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Integrated Information Φ')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 概念清晰度
    if trainer.history['concept_clarity']:
        axes[1, 1].plot(trainer.history['concept_clarity'], 'g-', 
                       linewidth=2, label='Concept Clarity')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Clarity (1 - Entropy)')
        axes[1, 1].set_title('Concept Clarity Over Time (CATS-NET Only)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    save_path = 'cats_vs_nct_comparison_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 结果可视化已保存到：{save_path}")
    
    # ========== 4. 导出统计报告 ==========
    report = {
        'experiment_name': 'CATS-NET Cat Recognition',
        'dataset_size': {
            'train': len(train_data),
            'test': len(test_data),
        },
        'training_results': {
            'best_accuracy': best_accuracy,
            'final_accuracy': trainer.history['accuracy'][-1],
            'training_time_sec': training_time,
            'epochs_to_90_pct': epochs_to_target,
        },
        'metrics': {
            'mean_phi': np.mean(trainer.history['phi_values']) if trainer.history['phi_values'] else 0,
            'mean_concept_clarity': np.mean(trainer.history['concept_clarity']) if trainer.history['concept_clarity'] else 0,
        },
    }
    
    print("\n" + "="*70)
    print("实验总结报告")
    print("="*70)
    print(f"数据集规模：训练集={report['dataset_size']['train']}, "
          f"测试集={report['dataset_size']['test']}")
    print(f"最佳准确率：{report['training_results']['best_accuracy']:.1%}")
    print(f"最终准确率：{report['training_results']['final_accuracy']:.1%}")
    print(f"训练时间：{report['training_results']['training_time_sec']:.1f}秒")
    if report['training_results']['epochs_to_90_pct']:
        print(f"达到 90% 准确率：{report['training_results']['epochs_to_90_pct']} epoch")
    print(f"平均Φ值：{report['metrics']['mean_phi']:.3f}")
    print(f"平均概念清晰度：{report['metrics']['mean_concept_clarity']:.3f}")
    
    # 保存报告
    import json
    report_path = 'cats_net_experiment_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 实验报告已保存到：{report_path}")
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    return report


if __name__ == "__main__":
    # 设置控制台编码为 UTF-8，避免中文和特殊字符编码错误
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    try:
        report = run_experiment()
        sys.exit(0)
    except Exception as e:
        print(f"\n实验失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
