"""
NCT Leaderboard Submission Generator
=====================================
用于生成 MNIST 和 CIFAR-10 榜单提交文件的工具

支持的榜单:
1. Kaggle - Digit Recognizer (MNIST)
2. Kaggle - CIFAR-10 Object Recognition
3. Papers With Code - Image Classification

使用方法:
    # 生成 MNIST 提交文件
    python tools/generate_leaderboard_submission.py --dataset mnist --model results/full_mnist_training/best_model.pt
    
    # 生成 CIFAR-10 提交文件
    python tools/generate_leaderboard_submission.py --dataset cifar10 --model results/cifar10/best_model.pt
    
    # 生成测试集预测（不提交，仅验证）
    python tools/generate_leaderboard_submission.py --dataset mnist --mode predict_only
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from nct_modules.nct_batched import BatchedNCTManager
from nct_modules.nct_cifar10 import NCTForCIFAR10


class SubmissionGenerator:
    """榜单提交文件生成器"""
    
    def __init__(self, dataset_name: str, model_path: Optional[str] = None):
        self.dataset_name = dataset_name.lower()
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建提交目录
        self.submission_dir = project_root / 'submissions' / f'{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.submission_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 数据集：{dataset_name}")
        print(f"📁 提交目录：{self.submission_dir}")
        print(f"💻 设备：{self.device}")
    
    def load_mnist_test_data(self) -> Tuple[Dataset, DataLoader]:
        """加载 MNIST 测试集"""
        print("\n📥 加载 MNIST 测试数据...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载测试集
        test_dataset = datasets.MNIST(
            root=project_root / 'data',
            train=False,
            download=True,
            transform=transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4
        )
        
        print(f"✅ 测试集大小：{len(test_dataset)}")
        return test_dataset, test_loader
    
    def load_cifar10_test_data(self) -> Tuple[Dataset, DataLoader]:
        """加载 CIFAR-10 测试集"""
        print("\n📥 加载 CIFAR-10 测试数据...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2470, 0.2435, 0.2616))
        ])
        
        # 加载测试集
        test_dataset = datasets.CIFAR10(
            root=project_root / 'data',
            train=False,
            download=True,
            transform=transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4
        )
        
        print(f"✅ 测试集大小：{len(test_dataset)}")
        return test_dataset, test_loader
    
    def load_nct_model(self, model_path: str, n_classes: int = 10) -> nn.Module:
        """加载 NCT 模型"""
        print(f"\n🔄 加载模型：{model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 根据模型结构判断类型
            if any('cifar' in key.lower() for key in state_dict.keys()):
                # CIFAR-10 模型
                model = NCTForCIFAR10(
                    d_model=checkpoint.get('config', {}).get('d_model', 512),
                    n_heads=checkpoint.get('config', {}).get('n_heads', 8),
                    n_layers=checkpoint.get('config', {}).get('n_layers', 4),
                    n_classes=n_classes
                ).to(self.device)
            else:
                # MNIST 模型 - 使用 BatchedNCTManager
                config = checkpoint.get('config', {})
                model = BatchedNCTManager(
                    n_classes=n_classes,
                    n_heads=config.get('n_heads', 7),
                    n_layers=config.get('n_layers', 4),
                    d_model=config.get('d_model', 504)
                ).model.to(self.device)
            
            model.load_state_dict(state_dict)
            model.eval()
            
            print(f"✅ 模型加载成功")
            if 'accuracy' in checkpoint:
                print(f"📈 验证集准确率：{checkpoint['accuracy']:.4f}")
            
            return model
            
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            raise
    
    @torch.no_grad()
    def predict_mnist(self, model: nn.Module, test_loader: DataLoader) -> List[Tuple[int, int]]:
        """生成 MNIST 预测结果"""
        print("\n🔮 生成 MNIST 预测...")
        
        predictions = []
        image_id = 1
        
        for batch_idx, (data, _) in enumerate(tqdm(test_loader, desc="Predicting")):
            data = data.to(self.device)
            
            # NCT 模型推理
            if hasattr(model, 'predict'):
                outputs = model.predict(data)
            else:
                outputs = model(data)
            
            # 获取预测类别
            if isinstance(outputs, tuple):
                pred = outputs[0].argmax(dim=1)
            else:
                pred = outputs.argmax(dim=1)
            
            # 收集预测结果
            for p in pred.cpu().numpy():
                predictions.append((image_id, int(p)))
                image_id += 1
        
        print(f"✅ 生成预测：{len(predictions)} 张图像")
        return predictions
    
    @torch.no_grad()
    def predict_cifar10(self, model: nn.Module, test_loader: DataLoader) -> List[Tuple[int, int]]:
        """生成 CIFAR-10 预测结果"""
        print("\n🔮 生成 CIFAR-10 预测...")
        
        predictions = []
        image_id = 1
        
        for batch_idx, (data, _) in enumerate(tqdm(test_loader, desc="Predicting")):
            data = data.to(self.device)
            
            # CIFAR-10 模型推理
            outputs = model(data)
            
            # 获取预测类别
            if isinstance(outputs, tuple):
                pred = outputs[0].argmax(dim=1)
            else:
                pred = outputs.argmax(dim=1)
            
            # 收集预测结果
            for p in pred.cpu().numpy():
                predictions.append((image_id, int(p)))
                image_id += 1
        
        print(f"✅ 生成预测：{len(predictions)} 张图像")
        return predictions
    
    def save_kaggle_submission(self, predictions: List[Tuple[int, int]], 
                               filename: str = 'submission.csv') -> str:
        """保存为 Kaggle 格式提交文件"""
        submission_path = self.submission_dir / filename
        
        print(f"\n💾 保存 Kaggle 提交文件：{submission_path}")
        
        # 创建 DataFrame
        df = pd.DataFrame(predictions, columns=['ImageId', 'Label'])
        df.to_csv(submission_path, index=False)
        
        # 统计信息
        stats = {
            'total_predictions': len(df),
            'class_distribution': df['Label'].value_counts().to_dict(),
            'submission_time': datetime.now().isoformat()
        }
        
        # 保存统计信息
        stats_path = self.submission_dir / 'submission_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ 提交文件已保存")
        print(f"   - 总预测数：{stats['total_predictions']}")
        print(f"   - 类别分布：{stats['class_distribution']}")
        
        return str(submission_path)
    
    def generate_papers_with_code_report(self, model_info: Dict, 
                                        results: Dict) -> str:
        """生成 Papers With Code 格式的报告"""
        report_path = self.submission_dir / 'paperswithcode_report.md'
        
        print(f"\n📝 生成 Papers With Code 报告：{report_path}")
        
        report = f"""# NCT on {self.dataset_name.upper()} - Submission Report

## Method Information

**Method Name**: NeuroConscious Transformer (NCT)
**Authors**: NeuroConscious Research Team
**Affiliation**: Universiti Teknologi Malaysia

## Model Configuration

{json.dumps(model_info, indent=2)}

## Results

| Metric | Value |
|--------|-------|
| Accuracy | {results.get('accuracy', 'N/A')} |
| Test Error | {results.get('test_error', 'N/A')} |
| Parameters | {results.get('parameters', 'N/A')} |

## Reproduction Details

**Code Repository**: https://github.com/your-repo/NCT
**Framework**: PyTorch
**Hardware**: {results.get('hardware', 'GPU')}

## Submission Files

- Submission CSV: `{self.submission_dir.name}/submission.csv`
- Model Checkpoint: `{self.submission_dir.name}/model.pt`
- Training Logs: `{self.submission_dir.name}/training_log.json`

## Notes

This submission uses the NeuroConscious Transformer architecture with multi-head attention and STDP-inspired plasticity rules.
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 报告已生成")
        return str(report_path)
    
    def create_visualization(self, predictions: List[Tuple[int, int]]) -> str:
        """创建预测可视化"""
        viz_path = self.submission_dir / 'prediction_distribution.png'
        
        print(f"\n📊 生成预测分布图：{viz_path}")
        
        labels = [p[1] for p in predictions]
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts, color='steelblue', alpha=0.7)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'{self.dataset_name.upper()} - Prediction Distribution', fontsize=14)
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(counts):
            plt.text(i, v + 50, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化已保存")
        return str(viz_path)


def main():
    parser = argparse.ArgumentParser(description='Generate leaderboard submission files')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['mnist', 'cifar10'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'predict_only', 'report_only'],
                       help='Generation mode')
    parser.add_argument('--n_classes', type=int, default=10,
                       help='Number of classes')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = SubmissionGenerator(args.dataset, args.model)
    
    # 加载数据
    if args.dataset == 'mnist':
        test_dataset, test_loader = generator.load_mnist_test_data()
    else:  # cifar10
        test_dataset, test_loader = generator.load_cifar10_test_data()
    
    # 生成预测
    if args.mode in ['full', 'predict_only']:
        if args.model is None:
            print("⚠️  需要模型路径才能生成预测")
            return
        
        # 加载模型
        model = generator.load_nct_model(args.model, args.n_classes)
        
        # 生成预测
        if args.dataset == 'mnist':
            predictions = generator.predict_mnist(model, test_loader)
        else:
            predictions = generator.predict_cifar10(model, test_loader)
        
        # 保存 Kaggle 提交文件
        submission_path = generator.save_kaggle_submission(
            predictions, 
            f'{args.dataset}_submission.csv'
        )
        
        # 创建可视化
        generator.create_visualization(predictions)
        
        print(f"\n{'='*60}")
        print(f"✅ 提交文件生成完成！")
        print(f"{'='*60}")
        print(f"📁 提交目录：{generator.submission_dir}")
        print(f"📄 Kaggle 提交：{submission_path}")
        print(f"\n下一步:")
        print(f"1. 访问 https://www.kaggle.com/competitions/digit-recognizer (MNIST)")
        print(f"   或 https://www.kaggle.com/c/cifar-10 (CIFAR-10)")
        print(f"2. 点击 'Submit Prediction'")
        print(f"3. 上传生成的 CSV 文件")
        print(f"4. 查看排行榜分数")
    
    if args.mode == 'full':
        # 生成 Papers With Code 报告
        model_info = {
            'architecture': 'NeuroConscious Transformer',
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 4,
            'training_epochs': 50
        }
        
        results = {
            'accuracy': '待测试',
            'test_error': '待计算',
            'parameters': '~10M',
            'hardware': str(generator.device)
        }
        
        generator.generate_papers_with_code_report(model_info, results)


if __name__ == '__main__':
    main()
