r"""
CATS-NET 实验启动脚本（在 NCT 根目录运行）

运行方式：
    cd d:\python_projects\NCT
    python run_cats_experiments.py

版本管理：
    - 自动检测已有版本并递增（v1, v2, v3...）
    - 结果保存到 cats_nct/experiments/results/{实验名}_v{版本}_{时间戳}/
    - 每次运行都会生成新的版本号，方便对比分析

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# 添加 cats_nct 到路径
sys.path.insert(0, str(Path(__file__).parent / "cats_nct"))

# 导入实验管理器
from cats_nct.run_all_experiments import ExperimentRunner

def main():
    """在 NCT 根目录运行所有 CATS-NET 实验"""
    
    print("="*70)
    print("CATS-NET 实验启动器")
    print("="*70)
    print(f"工作目录: {Path.cwd()}")
    print(f"Python: {sys.executable}")
    print(f"Python 版本: {sys.version}")
    print("="*70)
    
    # 导入测试
    try:
        from cats_nct import CATSManager, CATSConfig
        print("✓ CATS-NET 模块导入成功")
        
        # 测试配置创建
        config = CATSConfig.get_small_config()
        print(f"✓ 配置创建成功: concept_dim={config.concept_dim}")
    except ImportError as e:
        print(f"✗ CATS-NET 模块导入失败: {e}")
        print("  将使用简化模式运行实验")
    
    print("="*70)
    print()
    
    # 实验 1: 猫识别对比
    print("\n" + "="*70)
    print("实验 1: 猫识别对比实验")
    print("="*70)
    
    runner1 = ExperimentRunner("cats_cat_recognition", base_dir="cats_nct/experiments/results")
    runner1.setup_result_directory()
    
    config1 = {
        "n_train_samples": 10,
        "n_test_samples": 50,
        "n_epochs": 50,
        "target_accuracy": 0.9,
        "use_small_config": True,
    }
    runner1.save_experiment_config(config1)
    
    start_time = time.time()
    return_code = runner1.run_experiment("cats_nct/experiments/run_cats_cat_recognition.py")
    duration = time.time() - start_time
    
    runner1.move_output_files(['*.png', '*.json'])
    runner1.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # 实验 2: 概念形成
    print("\n" + "="*70)
    print("实验 2: 概念形成实验")
    print("="*70)
    
    runner2 = ExperimentRunner("concept_formation", base_dir="cats_nct/experiments/results")
    runner2.setup_result_directory()
    
    config2 = {
        "n_categories": 5,
        "samples_per_category": 20,
        "n_epochs": 5,
        "use_small_config": True,
    }
    runner2.save_experiment_config(config2)
    
    start_time = time.time()
    return_code = runner2.run_experiment("cats_nct/experiments/run_concept_formation.py")
    duration = time.time() - start_time
    
    runner2.move_output_files(['*.png', '*.json'])
    runner2.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # 实验 3: 概念迁移
    print("\n" + "="*70)
    print("实验 3: 概念迁移实验")
    print("="*70)
    
    runner3 = ExperimentRunner("concept_transfer", base_dir="cats_nct/experiments/results")
    runner3.setup_result_directory()
    
    config3 = {
        "train_samples": 50,
        "test_samples": 30,
        "teacher_epochs": 20,
        "use_adversarial": True,
    }
    runner3.save_experiment_config(config3)
    
    start_time = time.time()
    return_code = runner3.run_experiment("cats_nct/experiments/run_concept_transfer.py")
    duration = time.time() - start_time
    
    runner3.move_output_files(['*.png', '*.json'])
    runner3.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # 实验 4: 小样本学习对比
    print("\n" + "="*70)
    print("实验 4: 小样本学习对比实验")
    print("="*70)
    
    runner4 = ExperimentRunner("few_shot_learning", base_dir="cats_nct/experiments/results")
    runner4.setup_result_directory()
    
    config4 = {
        "sample_sizes": [5, 10, 20, 50, 100],
        "test_size": 100,
        "epochs_per_sample": 3,
    }
    runner4.save_experiment_config(config4)
    
    start_time = time.time()
    return_code = runner4.run_experiment("cats_nct/experiments/run_few_shot_learning.py")
    duration = time.time() - start_time
    
    runner4.move_output_files(['*.png', '*.json'])
    runner4.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # 总结
    print("\n" + "="*70)
    print("所有实验运行完成！")
    print("="*70)
    
    # 列出生成的结果目录
    results_base = Path("cats_nct/experiments/results")
    print(f"\n结果目录位置：{results_base.absolute()}")
    print("\n生成的运行记录:")
    
    experiments = [
        "cats_cat_recognition",
        "concept_formation",
        "concept_transfer",
        "few_shot_learning",
    ]
    
    for exp_name in experiments:
        runs = ExperimentRunner.list_all_runs(exp_name, base_dir=str(results_base))
        if runs:
            latest = runs[-1]
            print(f"\n  {exp_name}:")
            print(f"    最新版本：v{latest['version']} ({latest['timestamp']})")
            print(f"    路径：{latest['path']}")
    
    print(f"\n{'='*70}")
    print("✓ 实验管理器任务完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    # 设置 UTF-8 编码
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n实验管理器失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
