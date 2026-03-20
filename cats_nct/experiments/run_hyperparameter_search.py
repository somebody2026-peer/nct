"""
CATS-NET v2.0 超参数搜索实验

目标: 从 94.75% 提升至 96-97%

搜索空间:
- concept_dim: [64, 96, 128]
- n_concept_levels: [3, 4, 5]
- prototypes_per_level: [50, 75, 100]
- learning_rate: [5e-4, 1e-3, 2e-3]
- dropout: [0.1, 0.2, 0.3]

策略: 网格搜索 + 早停

作者：NeuroConscious Research Team
创建：2026-03-02
版本：v1.0
"""

import sys
import os
import time
import json
import logging
import itertools
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# 添加 cats_nct 目录到路径
cats_nct_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if cats_nct_dir not in sys.path:
    sys.path.insert(0, cats_nct_dir)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 导入端到端训练器
sys.path.insert(0, os.path.join(cats_nct_dir, 'experiments'))
from run_end_to_end_training import EndToEndConfig, EndToEndCATSTrainer

# ========== 实验元数据 ==========
EXPERIMENT_NAME = "hyperparameter_search"
VERSION = "v1"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_ID = f"{EXPERIMENT_NAME}_{VERSION}_{TIMESTAMP}"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 搜索空间定义 ==========
SEARCH_SPACE = {
    'concept_dim': [64, 96, 128],
    'n_concept_levels': [3, 4, 5],
    'prototypes_per_level': [50, 75, 100],
    'learning_rate': [5e-4, 1e-3, 2e-3],
    'dropout': [0.1, 0.2, 0.3],
}

# 精简搜索空间（快速模式）
QUICK_SEARCH_SPACE = {
    'concept_dim': [64, 96],
    'n_concept_levels': [3, 4],
    'prototypes_per_level': [50, 75],
    'learning_rate': [1e-3, 2e-3],
    'dropout': [0.1, 0.2],
}

# 固定参数
FIXED_PARAMS = {
    'input_dim': 784,
    'n_epochs': 30,  # 搜索阶段使用较少轮次
    'batch_size': 128,
    'weight_decay': 1e-4,
    'classification_weight': 1.0,
    'concept_consistency_weight': 0.5,
    'attention_entropy_weight': 0.01,
    'gradient_clip': 1.0,
    'seed': 42,
}


def load_mnist_data(root='data', n_samples_per_class=300, batch_size=128):
    """加载 MNIST 数据集（平衡采样）"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    
    # 平衡采样
    n_classes = 10
    train_indices = []
    for class_id in range(n_classes):
        class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_id]
        selected = np.random.choice(class_indices, min(n_samples_per_class, len(class_indices)), replace=False)
        train_indices.extend(selected)
    
    train_subset = Subset(train_dataset, train_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, test_loader


def run_single_experiment(params: dict, train_loader, test_loader, device, trial_id: int) -> dict:
    """运行单个超参数组合的实验"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial_id}: {params}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # 构建配置
    config = EndToEndConfig(
        input_dim=FIXED_PARAMS['input_dim'],
        concept_dim=params['concept_dim'],
        n_concept_levels=params['n_concept_levels'],
        prototypes_per_level=params['prototypes_per_level'],
        learning_rate=params['learning_rate'],
        weight_decay=FIXED_PARAMS['weight_decay'],
        n_epochs=FIXED_PARAMS['n_epochs'],
        batch_size=FIXED_PARAMS['batch_size'],
        classification_weight=FIXED_PARAMS['classification_weight'],
        concept_consistency_weight=FIXED_PARAMS['concept_consistency_weight'],
        attention_entropy_weight=FIXED_PARAMS['attention_entropy_weight'],
        dropout=params['dropout'],
        gradient_clip=FIXED_PARAMS['gradient_clip'],
        seed=FIXED_PARAMS['seed'],
        device=device,
    )
    
    try:
        # 创建训练器
        trainer = EndToEndCATSTrainer(config)
        
        # 计算参数量
        param_count = sum(p.numel() for p in trainer.concept_space.parameters())
        param_count += sum(p.numel() for p in trainer.classifier.parameters())
        
        # 训练
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=test_loader,
        )
        
        elapsed_time = time.time() - start_time
        
        # 记录结果
        result = {
            'trial_id': trial_id,
            'params': params,
            'best_val_acc': max(history['val_acc']) if history.get('val_acc') else 0,
            'final_val_acc': history['val_acc'][-1] if history.get('val_acc') else 0,
            'final_train_acc': history['train_acc'][-1] if history.get('train_acc') else 0,
            'param_count': param_count,
            'elapsed_time': elapsed_time,
            'status': 'completed',
        }
        
        logger.info(f"Trial {trial_id} 完成: Best Val Acc = {result['best_val_acc']:.2f}%, Time = {elapsed_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Trial {trial_id} 失败: {e}")
        result = {
            'trial_id': trial_id,
            'params': params,
            'best_val_acc': 0,
            'final_val_acc': 0,
            'final_train_acc': 0,
            'param_count': 0,
            'elapsed_time': time.time() - start_time,
            'status': 'failed',
            'error': str(e),
        }
    
    return result


def generate_param_combinations(search_space: dict) -> list:
    """生成所有参数组合"""
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)
    
    return combinations


def main(quick_mode: bool = False):
    """主函数"""
    
    print("=" * 80)
    print("CATS-NET v2.0 超参数搜索实验")
    print("=" * 80)
    
    # 选择搜索空间
    search_space = QUICK_SEARCH_SPACE if quick_mode else SEARCH_SPACE
    param_combinations = generate_param_combinations(search_space)
    
    logger.info(f"\n搜索空间: {search_space}")
    logger.info(f"总实验数: {len(param_combinations)}")
    
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"使用 GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        logger.info("使用 CPU")
    
    # 加载数据
    logger.info("\n加载 MNIST 数据集...")
    train_loader, test_loader = load_mnist_data(
        root=str(Path(cats_nct_dir) / 'data'),
        n_samples_per_class=300,  # 搜索阶段使用较少数据
        batch_size=FIXED_PARAMS['batch_size'],
    )
    logger.info(f"训练样本: {len(train_loader.dataset)}, 测试样本: {len(test_loader.dataset)}")
    
    # 创建结果目录
    results_dir = Path(cats_nct_dir) / 'experiments' / 'results' / EXPERIMENT_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行所有实验
    all_results = []
    best_result = None
    best_val_acc = 0
    
    total_start_time = time.time()
    
    for trial_id, params in enumerate(param_combinations, 1):
        result = run_single_experiment(
            params=params,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            trial_id=trial_id,
        )
        all_results.append(result)
        
        # 更新最佳结果
        if result['best_val_acc'] > best_val_acc:
            best_val_acc = result['best_val_acc']
            best_result = result
            logger.info(f"*** 新的最佳结果: {best_val_acc:.2f}% ***")
        
        # 定期保存中间结果
        if trial_id % 5 == 0:
            intermediate_path = results_dir / f'intermediate_results_trial_{trial_id}.json'
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    total_elapsed_time = time.time() - total_start_time
    
    # ========== 结果分析 ==========
    print("\n" + "=" * 80)
    print("超参数搜索完成！")
    print("=" * 80)
    
    # 排序结果
    sorted_results = sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True)
    
    print("\nTop 5 配置:")
    print("-" * 80)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. Val Acc: {result['best_val_acc']:.2f}% | Params: {result['param_count']:,}")
        print(f"   Config: {result['params']}")
        print()
    
    # 保存最终结果
    final_report = {
        'experiment_id': EXPERIMENT_ID,
        'search_space': search_space,
        'fixed_params': FIXED_PARAMS,
        'total_trials': len(param_combinations),
        'total_time_seconds': total_elapsed_time,
        'best_result': best_result,
        'top_5_results': sorted_results[:5],
        'all_results': sorted_results,
    }
    
    # 保存报告
    report_path = results_dir / f'{EXPERIMENT_ID}_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logger.info(f"\n报告已保存: {report_path}")
    
    # 保存最佳配置
    best_config_path = results_dir / f'{EXPERIMENT_ID}_best_config.json'
    with open(best_config_path, 'w', encoding='utf-8') as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)
    logger.info(f"最佳配置已保存: {best_config_path}")
    
    print("\n" + "=" * 80)
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最佳配置: {best_result['params']}")
    print(f"总耗时: {total_elapsed_time/60:.1f} 分钟")
    print(f"结果目录: {results_dir}")
    print("=" * 80)
    
    return best_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CATS-NET 超参数搜索')
    parser.add_argument('--quick', action='store_true', help='使用精简搜索空间 (快速模式)')
    args = parser.parse_args()
    
    main(quick_mode=args.quick)
