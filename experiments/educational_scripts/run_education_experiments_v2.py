#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCT 教育领域实验 V2 统一入口

V2 版本改进总结:
1. FER: ResNet18 预训练模型替代轻量CNN
2. 视觉: MediaPipe Face Mesh 替代 OpenCV Haar
3. 时序: LSTM 时序情绪编码
4. EEG: CWT 频谱 + 可学习神经网络映射
5. 状态: 可学习的学生状态→神经调质映射

使用方法:
    python run_education_experiments_v2.py --phase all
    python run_education_experiments_v2.py --phase fer  # 运行 FER 预训练
    python run_education_experiments_v2.py --phase daisee  # 运行 DAiSEE
    python run_education_experiments_v2.py --phase mema  # 运行 MEMA

结果输出:
    results/education_v2/phase1_daisee_v2.json
    results/education_v2/phase2_fer_v2.json
    results/education_v2/phase3_mema_v2.json
    results/education_v2/combined_report_v2.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "education_v2.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 结果目录
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase2_fer_v2(args) -> Dict:
    """Phase 2: FER ResNet18 预训练"""
    logger.info("=" * 60)
    logger.info("Phase 2 V2: FER ResNet18 预训练")
    logger.info("=" * 60)
    
    try:
        from experiments.fer_pretrain_v2 import train_fer_v2
        
        results = train_fer_v2(
            epochs=args.fer_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=args.max_samples,
            patience=args.patience,
        )
        
        logger.info(f"FER V2 完成: val_acc = {results['best_val_acc']:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"FER V2 失败: {e}")
        return {"error": str(e)}


def run_phase1_daisee_v2(args) -> Dict:
    """Phase 1: DAiSEE 视频参与度检测"""
    logger.info("=" * 60)
    logger.info("Phase 1 V2: DAiSEE 视频参与度检测")
    logger.info("=" * 60)
    
    try:
        from experiments.daisee_nct_experiment_v2 import run_daisee_experiment_v2
        
        results = run_daisee_experiment_v2(
            use_mock=args.mock,
            max_videos=args.max_videos,
            split=args.split,
        )
        
        r = results['correlations']['engagement_vs_phi']['r']
        p = results['correlations']['engagement_vs_phi']['p']
        logger.info(f"DAiSEE V2 完成: r={r:.4f}, p={p:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"DAiSEE V2 失败: {e}")
        return {"error": str(e)}


def run_phase3_mema_v2(args) -> Dict:
    """Phase 3: MEMA EEG 神经调质映射"""
    logger.info("=" * 60)
    logger.info("Phase 3 V2: MEMA EEG 神经调质映射")
    logger.info("=" * 60)
    
    try:
        from experiments.mema_nct_experiment_v2 import run_mema_experiment_v2
        
        results = run_mema_experiment_v2(
            use_mock=args.mock,
            max_samples=args.max_eeg_samples,
            max_per_class=args.max_per_class,
        )
        
        f1 = results['experiment_A_classification']['SVM_V2']['mean_f1']
        logger.info(f"MEMA V2 完成: SVM F1={f1:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"MEMA V2 失败: {e}")
        return {"error": str(e)}


def generate_combined_report(
    phase1_results: Optional[Dict],
    phase2_results: Optional[Dict],
    phase3_results: Optional[Dict],
) -> Dict:
    """生成综合报告"""
    report = {
        "version": "V2",
        "generated_at": datetime.now().isoformat(),
        "phases": {},
        "v2_improvements_summary": {
            "visual_feature_extraction": {
                "v1": "OpenCV Haar Cascade",
                "v2": "MediaPipe Face Mesh (478 landmarks) + ResNet18 FER",
            },
            "temporal_modeling": {
                "v1": "Single frame prediction",
                "v2": "Bidirectional LSTM with attention",
            },
            "eeg_mapping": {
                "v1": "Fixed sigmoid mapping",
                "v2": "Learnable neural network + CWT features + subject calibration",
            },
            "state_mapping": {
                "v1": "Fixed linear weights",
                "v2": "Learnable MLP with theory-guided prior",
            },
        },
    }
    
    # Phase 2: FER
    if phase2_results and "error" not in phase2_results:
        report["phases"]["phase2_fer"] = {
            "status": "success",
            "best_val_acc": phase2_results.get("best_val_acc"),
            "model": phase2_results.get("model"),
            "per_class_acc": phase2_results.get("per_class_acc"),
        }
    else:
        report["phases"]["phase2_fer"] = {"status": "failed", "error": phase2_results.get("error")}
    
    # Phase 1: DAiSEE
    if phase1_results and "error" not in phase1_results:
        report["phases"]["phase1_daisee"] = {
            "status": "success",
            "n_samples": phase1_results.get("n_samples"),
            "correlations": phase1_results.get("correlations"),
            "nm_means": phase1_results.get("nm_means"),
        }
    else:
        report["phases"]["phase1_daisee"] = {"status": "failed", "error": phase1_results.get("error") if phase1_results else "skipped"}
    
    # Phase 3: MEMA
    if phase3_results and "error" not in phase3_results:
        clf = phase3_results.get("experiment_A_classification", {})
        phi = phase3_results.get("experiment_B_phi_analysis", {})
        report["phases"]["phase3_mema"] = {
            "status": "success",
            "n_samples": phase3_results.get("n_samples"),
            "svm_f1_v2": clf.get("SVM_V2", {}).get("mean_f1"),
            "svm_f1_v1_baseline": clf.get("SVM_V1_baseline", {}).get("mean_f1"),
            "phi_t_test": phi.get("t_test_relax_vs_conc"),
        }
    else:
        report["phases"]["phase3_mema"] = {"status": "failed", "error": phase3_results.get("error") if phase3_results else "skipped"}
    
    # 保存
    report_path = RESULTS_DIR / "combined_report_v2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"综合报告已保存至: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="NCT 教育领域实验 V2 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_education_experiments_v2.py --phase all        # 运行所有实验
    python run_education_experiments_v2.py --phase fer        # 运行 FER 预训练
    python run_education_experiments_v2.py --phase daisee     # 运行 DAiSEE
    python run_education_experiments_v2.py --phase mema       # 运行 MEMA
    python run_education_experiments_v2.py --phase all --mock # 使用模拟数据
        """
    )
    
    # 实验选择
    parser.add_argument("--phase", type=str, default="all",
                       choices=["all", "fer", "daisee", "mema"],
                       help="运行哪个阶段")
    
    # 通用参数
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    
    # FER 参数
    parser.add_argument("--fer-epochs", type=int, default=30, help="FER 训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    
    # DAiSEE 参数
    parser.add_argument("--max-videos", type=int, default=100, help="DAiSEE 最大视频数")
    parser.add_argument("--split", type=str, default="Train", help="DAiSEE 数据划分")
    
    # MEMA 参数
    parser.add_argument("--max-eeg-samples", type=int, default=6000, help="MEMA 最大样本数")
    parser.add_argument("--max-per-class", type=int, default=50, help="Φ分析每类最大样本")
    
    args = parser.parse_args()
    
    # 日志目录
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("NCT 教育领域实验 V2 - 深度学习特征提取 + 非线性神经调质映射")
    logger.info("=" * 70)
    logger.info(f"运行阶段: {args.phase}")
    logger.info(f"模拟模式: {args.mock}")
    
    # 运行实验
    phase1_results = None
    phase2_results = None
    phase3_results = None
    
    if args.phase in ["all", "fer"]:
        phase2_results = run_phase2_fer_v2(args)
    
    if args.phase in ["all", "daisee"]:
        phase1_results = run_phase1_daisee_v2(args)
    
    if args.phase in ["all", "mema"]:
        phase3_results = run_phase3_mema_v2(args)
    
    # 生成综合报告
    if args.phase == "all":
        report = generate_combined_report(phase1_results, phase2_results, phase3_results)
        
        logger.info("\n" + "=" * 70)
        logger.info("V2 实验完成！综合结果:")
        logger.info("=" * 70)
        
        for phase_name, phase_data in report["phases"].items():
            status = phase_data.get("status", "unknown")
            logger.info(f"  {phase_name}: {status}")
            if status == "success":
                if "best_val_acc" in phase_data:
                    logger.info(f"    -> FER val_acc: {phase_data['best_val_acc']}")
                if "correlations" in phase_data:
                    corr = phase_data['correlations'].get('engagement_vs_phi', {})
                    logger.info(f"    -> Φ vs Engagement: r={corr.get('r')}, p={corr.get('p')}")
                if "svm_f1_v2" in phase_data:
                    logger.info(f"    -> SVM F1 (V2): {phase_data['svm_f1_v2']}")
                    logger.info(f"    -> SVM F1 (V1 baseline): {phase_data['svm_f1_v1_baseline']}")
    
    logger.info("\n完成！结果保存在: results/education_v2/")


if __name__ == "__main__":
    main()