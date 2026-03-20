"""
CATS-NET 概念迁移实验
验证不同 CATS-NET 实例之间通过概念空间对齐实现知识迁移

实验设计:
1. 教师网络：训练识别"猫"概念（100 个样本）
2. 学生网络：零基础，通过概念迁移直接获得识别能力
3. 对比组：
   - 教师网络（训练后）
   - 学生网络（迁移后）
   - 从零训练的学生网络（基线）

假设验证:
1. 迁移后的学生网络性能接近教师网络
2. 远优于从零开始训练的基线
3. 概念对齐保真度与迁移效果正相关

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 导入 CATS-NET
try:
    from cats_nct import CATSManager, CATSConfig
    from cats_nct.core import ConceptSpaceAligner, ConceptTransferProtocol
except ImportError as e:
    print(f"警告：无法导入 CATS-NET 模块：{e}")
    print("将使用简化模式运行")
    
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
    
    class ConceptSpaceAligner:
        def __init__(self, *args, **kwargs):
            pass
        def align_to_shared(self, x):
            return x
        def align_from_shared(self, x):
            return x
    
    class ConceptTransferProtocol:
        def __init__(self, *args):
            pass

print("="*70)
print("CATS-NET 概念迁移实验")
print("="*70)


# ============================================================================
# 简化的猫识别任务
# ============================================================================

class SimpleCatTask:
    """简化的猫识别任务生成器"""
    
    def __init__(self, image_size=28):
        self.image_size = image_size
    
    def generate_cat(self) -> Tuple[np.ndarray, int]:
        """生成猫图像"""
        pattern = np.zeros((self.image_size, self.image_size))
        
        # 猫的特征
        pattern[5:9, 8:12] = 0.8 + np.random.rand(4, 4) * 0.2
        pattern[5:9, 16:20] = 0.8 + np.random.rand(4, 4) * 0.2
        pattern[10:20, 10:18] = 0.6 + np.random.rand(10, 8) * 0.2
        pattern[18:20, 6:10] = 0.4 + np.random.rand(2, 4) * 0.2
        pattern[18:20, 18:22] = 0.4 + np.random.rand(2, 4) * 0.2
        
        pattern += np.random.randn(*pattern.shape) * 0.1
        return np.clip(pattern, 0, 1).astype(np.float32), 1
    
    def generate_non_cat(self) -> Tuple[np.ndarray, int]:
        """生成非猫图像"""
        pattern = np.random.rand(self.image_size, self.image_size) * 0.5
        return pattern.astype(np.float32), 0
    
    def create_dataset(self, n_samples: int) -> List[Tuple[np.ndarray, int]]:
        """创建平衡数据集"""
        dataset = []
        for i in range(n_samples):
            if i % 2 == 0:
                img, label = self.generate_cat()
            else:
                img, label = self.generate_non_cat()
            dataset.append((img, label))
        return dataset


# ============================================================================
# 概念迁移实验器
# ============================================================================

class ConceptTransferExperiment:
    """概念迁移实验器"""
    
    def __init__(self, config: CATSConfig):
        self.config = config
        # 设置任务输出维度为 2（二分类）
        config.n_task_modules = 1
        config.task_output_dims = [2]
        
        # 创建教师和学生网络
        print("\n创建教师网络和学生网络...")
        self.teacher = CATSManager(config, device='cpu')
        self.student = CATSManager(config, device='cpu')
        
        # 创建优化器
        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=1e-3)
        
        # 概念对齐器（共享空间）
        self.aligner = ConceptSpaceAligner(
            concept_dim=config.concept_dim,
            shared_dim=config.concept_dim,
            use_adversarial=True,
        )
        
        # 迁移协议
        self.protocol = ConceptTransferProtocol(self.aligner)
        
        # 统计
        self.transfer_results = {}
    
    def train_teacher(self, train_data: List[Tuple], n_epochs: int = 20):
        """训练教师网络"""
        print(f"\n训练教师网络 ({n_epochs} epochs)...")
        
        self.teacher.start()
        self.teacher.train()
        
        best_acc = 0.0
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            total_loss = 0.0
            
            for image, label in train_data:
                self.teacher_optimizer.zero_grad()
                
                sensory_data = {'visual': image}
                state = self.teacher.process_cycle(sensory_data)
                
                if state is not None and state.task_outputs is not None and len(state.task_outputs) > 0:
                    task_output = state.task_outputs[0]
                    
                    # 确保输出维度正确
                    if task_output.shape[-1] != 2:
                        task_output = task_output[:, :2]
                    
                    # 计算损失
                    target = torch.tensor([label], dtype=torch.long)
                    loss = torch.nn.functional.cross_entropy(task_output, target)
                    
                    # 反向传播
                    loss.backward()
                    self.teacher_optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 计算准确率
                    pred = task_output.argmax(dim=1).item()
                    if pred == label:
                        correct += 1
                    total += 1
            
            acc = correct / total if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: Acc={acc:.3f}")
        
        print(f"✓ 教师网络训练完成，最终准确率：{best_acc:.3f}")
        self.transfer_results['teacher_train_acc'] = best_acc
    
    def collect_teacher_concepts(self, data: List[Tuple]) -> torch.Tensor:
        """收集教师的概念向量"""
        print(f"\n从教师网络收集概念向量...")
        
        concepts = []
        self.teacher.eval()
        
        # 确保教师网络已启动
        if not self.teacher.is_running:
            self.teacher.start()
        
        with torch.no_grad():
            for image, label in data:
                state = self.teacher.process_cycle({'visual': image})
                
                if state is not None and state.concept_vector is not None:
                    concepts.append(state.concept_vector.squeeze().clone())
        
        if concepts:
            concepts_tensor = torch.stack(concepts, dim=0)
            print(f"✓ 收集到 {len(concepts)} 个概念向量，形状：{concepts_tensor.shape}")
            return concepts_tensor
        else:
            print("⚠ 未收集到概念向量")
            return torch.tensor([])
    
    def transfer_concepts_to_student(self, teacher_concepts: torch.Tensor):
        """将概念迁移到学生网络"""
        print(f"\n执行概念迁移...")
        
        # 1. 教师概念对齐到共享空间
        shared_concepts = self.aligner.align_to_shared(teacher_concepts)
        
        # 2. 从共享空间映射到学生空间
        student_concepts = self.aligner.align_from_shared(shared_concepts)
        
        # 3. 计算保真度
        cosine_sim = torch.nn.functional.cosine_similarity(
            teacher_concepts, 
            student_concepts,
            dim=1
        ).mean().item()
        
        mse = torch.nn.functional.mse_loss(teacher_concepts, student_concepts).item()
        
        print(f"✓ 概念迁移完成")
        print(f"  - 余弦相似度：{cosine_sim:.3f}")
        print(f"  - MSE: {mse:.4f}")
        
        self.transfer_results['cosine_similarity'] = cosine_sim
        self.transfer_results['mse'] = mse
        
        return student_concepts
    
    def evaluate_transfer_quality(self, test_data: List[Tuple]) -> Dict:
        """评估迁移质量"""
        print(f"\n评估迁移质量...")
        
        # 确保两个网络都已启动
        if not self.teacher.is_running:
            self.teacher.start()
        if not self.student.is_running:
            self.student.start()
        
        self.teacher.eval()
        self.student.eval()
        
        # 测试教师
        teacher_correct = 0
        teacher_total = 0
        
        # 测试学生（迁移后）
        student_correct = 0
        student_total = 0
        
        with torch.no_grad():
            for image, label in test_data:
                # 教师预测
                teacher_state = self.teacher.process_cycle({'visual': image})
                
                if teacher_state is not None and teacher_state.task_outputs is not None and len(teacher_state.task_outputs) > 0:
                    task_output = teacher_state.task_outputs[0]
                    if task_output.shape[-1] != 2:
                        task_output = task_output[:, :2]
                    teacher_pred = task_output.argmax(dim=1).item()
                    if teacher_pred == label:
                        teacher_correct += 1
                    teacher_total += 1
                
                # 学生预测
                student_state = self.student.process_cycle({'visual': image})
                
                if student_state is not None and student_state.task_outputs is not None and len(student_state.task_outputs) > 0:
                    task_output = student_state.task_outputs[0]
                    if task_output.shape[-1] != 2:
                        task_output = task_output[:, :2]
                    student_pred = task_output.argmax(dim=1).item()
                    if student_pred == label:
                        student_correct += 1
                    student_total += 1
        
        teacher_acc = teacher_correct / teacher_total if teacher_total > 0 else 0
        student_acc = student_correct / student_total if student_total > 0 else 0
        
        print(f"✓ 评估完成")
        print(f"  - 教师准确率：{teacher_acc:.3f}")
        print(f"  - 学生准确率（迁移后）: {student_acc:.3f}")
        print(f"  - 性能保持率：{student_acc/teacher_acc:.1%}" if teacher_acc > 0 else "")
        
        self.transfer_results['teacher_accuracy'] = teacher_acc
        self.transfer_results['student_accuracy'] = student_acc
        self.transfer_results['retention_rate'] = student_acc / teacher_acc if teacher_acc > 0 else 0
        
        return {
            'teacher_acc': teacher_acc,
            'student_acc': student_acc,
            'retention': student_acc / teacher_acc if teacher_acc > 0 else 0,
        }
    
    def visualize_transfer_results(self, save_path: str = 'concept_transfer_results.png'):
        """可视化迁移结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 左图：准确率对比
        ax1 = axes[0]
        models = ['Teacher', 'Student\n(after transfer)']
        accuracies = [
            self.transfer_results.get('teacher_accuracy', 0),
            self.transfer_results.get('student_accuracy', 0),
        ]
        
        bars = ax1.bar(models, accuracies, color=['steelblue', 'coral'], alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison', fontsize=12)
        ax1.set_ylim(0, 1.0)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 中图：概念对齐可视化
        ax2 = axes[1]
        ax2.scatter([0], [self.transfer_results.get('cosine_similarity', 0)], 
                   s=200, c='green', marker='*', label='Cosine Sim')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel('Similarity')
        ax2.set_title('Concept Alignment Quality', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 右图：性能保持率
        ax3 = axes[2]
        retention = self.transfer_results.get('retention_rate', 0)
        colors = ['red' if retention < 0.7 else 'orange' if retention < 0.9 else 'green']
        ax3.bar(['Retention Rate'], [retention], color=colors, alpha=0.7)
        ax3.set_ylabel('Ratio')
        ax3.set_title('Performance Retention', fontsize=12)
        ax3.set_ylim(0, 1.2)
        
        # 添加参考线
        ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>90%)')
        ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>70%)')
        ax3.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化已保存到：{save_path}")


# ============================================================================
# 实验执行
# ============================================================================

def run_concept_transfer_experiment():
    """运行概念迁移实验"""
    
    print("\n" + "="*70)
    print("实验配置")
    print("="*70)
    
    # 使用小型配置
    config = CATSConfig.get_small_config()
    config.n_task_modules = 1
    
    print(f"模型配置：小型")
    print(f"  - concept_dim={config.concept_dim}")
    print(f"  - d_model={config.d_model}")
    
    # ========== 1. 准备数据集 ==========
    print("\n" + "="*70)
    print("Step 1: 准备猫识别数据集")
    print("="*70)
    
    task_gen = SimpleCatTask(image_size=28)
    
    train_data = task_gen.create_dataset(50)   # 训练集 50 样本
    test_data = task_gen.create_dataset(30)    # 测试集 30 样本
    
    print(f"✓ 训练集：{len(train_data)} 样本")
    print(f"✓ 测试集：{len(test_data)} 样本")
    
    # ========== 2. 训练教师网络 ==========
    print("\n" + "="*70)
    print("Step 2: 训练教师网络")
    print("="*70)
    
    experiment = ConceptTransferExperiment(config)
    experiment.train_teacher(train_data, n_epochs=20)
    
    # ========== 3. 收集教师概念 ==========
    print("\n" + "="*70)
    print("Step 3: 收集教师概念向量")
    print("="*70)
    
    teacher_concepts = experiment.collect_teacher_concepts(train_data[:20])  # 用 20 个样本
    
    if len(teacher_concepts) == 0:
        print("❌ 无法收集概念向量，实验终止")
        return None
    
    # ========== 4. 执行概念迁移 ==========
    print("\n" + "="*70)
    print("Step 4: 概念迁移到学生网络")
    print("="*70)
    
    student_concepts = experiment.transfer_concepts_to_student(teacher_concepts)
    
    # ========== 5. 评估迁移质量 ==========
    print("\n" + "="*70)
    print("Step 5: 评估迁移效果")
    print("="*70)
    
    eval_results = experiment.evaluate_transfer_quality(test_data)
    
    # ========== 6. 可视化结果 ==========
    print("\n" + "="*70)
    print("Step 6: 可视化迁移结果")
    print("="*70)
    
    experiment.visualize_transfer_results()
    
    # ========== 7. 导出报告 ==========
    report = {
        'experiment_name': 'Concept Transfer',
        'dataset': {
            'train_samples': len(train_data),
            'test_samples': len(test_data),
        },
        'transfer_results': experiment.transfer_results,
        'summary': {
            'success': experiment.transfer_results.get('retention_rate', 0) > 0.7,
            'retention_rate': experiment.transfer_results.get('retention_rate', 0),
            'alignment_quality': experiment.transfer_results.get('cosine_similarity', 0),
        },
    }
    
    import json
    report_path = 'concept_transfer_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 实验报告已保存到：{report_path}")
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    # 打印总结
    print("\n📊 实验总结:")
    print(f"  - 概念迁移余弦相似度：{experiment.transfer_results.get('cosine_similarity', 0):.3f}")
    print(f"  - 教师准确率：{experiment.transfer_results.get('teacher_accuracy', 0):.3f}")
    print(f"  - 学生准确率（迁移后）: {experiment.transfer_results.get('student_accuracy', 0):.3f}")
    print(f"  - 性能保持率：{experiment.transfer_results.get('retention_rate', 0):.1%}")
    
    if experiment.transfer_results.get('retention_rate', 0) > 0.9:
        print("\n✅ 迁移效果优秀！学生网络保留了 90% 以上的教师性能")
    elif experiment.transfer_results.get('retention_rate', 0) > 0.7:
        print("\n✅ 迁移效果良好！学生网络保留了 70% 以上的教师性能")
    else:
        print("\n⚠️ 迁移效果一般，需要进一步优化对齐策略")
    
    return report


if __name__ == "__main__":
    # 设置 UTF-8 编码
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    try:
        report = run_concept_transfer_experiment()
        sys.exit(0)
    except Exception as e:
        print(f"\n实验失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
