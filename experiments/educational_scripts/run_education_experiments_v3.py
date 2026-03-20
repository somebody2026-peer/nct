#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCT 教育领域实验 V3 总控脚本

执行顺序:
1. Day 0: 前置验证 (数据质量、因果假设、类别不平衡)
2. Phase 1: EEGNet 分类 + MediaPipe 修复
3. Phase 2: 端到端映射 + 对比学习 (可选)
4. Phase 3: Φ方法改进 + 替代指标 (条件触发)

使用方法:
    python run_education_experiments_v3.py --day0        # 仅运行Day 0验证
    python run_education_experiments_v3.py --phase1     # 运行Phase 1
    python run_education_experiments_v3.py --all        # 运行所有阶段
    python run_education_experiments_v3.py --skip-day0  # 跳过验证直接运行
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "v3_experiment.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 结果目录
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_day0_validation() -> dict:
    """运行 Day 0 前置验证"""
    logger.info("=" * 70)
    logger.info("Day 0: 前置验证")
    logger.info("=" * 70)
    
    try:
        from experiments.day0_validation_v3 import run_day0_validation as day0_validate
        results = day0_validate(max_samples=1000)
        return results
    except Exception as e:
        logger.error(f"Day 0 验证失败: {e}")
        return {"error": str(e), "proceed_to_phase1": False}


def run_phase1_eegnet() -> dict:
    """运行 Phase 1: EEGNet 分类"""
    logger.info("=" * 70)
    logger.info("Phase 1: EEGNet 分类")
    logger.info("=" * 70)
    
    try:
        from experiments.eegnet_classifier_v3 import train_eegnet_v3
        results = train_eegnet_v3(
            n_epochs=50,
            batch_size=32,
            lr=0.001,
            n_folds=5,
            use_class_weights=True,
            use_lite_model=True,
            max_samples=3000,
        )
        return results
    except Exception as e:
        logger.error(f"Phase 1 EEGNet 训练失败: {e}")
        return {"error": str(e)}


def run_phase3_mema() -> dict:
    """运行 Phase 3: MEMA V3 实验 (含替代指标)"""
    logger.info("=" * 70)
    logger.info("Phase 3: MEMA V3 实验 (替代指标 + 效应量分析)")
    logger.info("=" * 70)
    
    try:
        from experiments.mema_nct_experiment_v3 import run_mema_experiment_v3
        results = run_mema_experiment_v3(
            use_mock=False,
            max_samples=3000,
            max_per_class=100,
            run_eegnet=False,  # 已在Phase 1运行
        )
        return results
    except Exception as e:
        logger.error(f"Phase 3 MEMA 实验失败: {e}")
        return {"error": str(e)}


def generate_combined_report(
    day0_results: dict,
    phase1_results: dict,
    phase3_results: dict,
) -> dict:
    """生成综合报告"""
    logger.info("=" * 70)
    logger.info("生成综合报告")
    logger.info("=" * 70)
    
    report = {
        "version": "V3",
        "timestamp": datetime.now().isoformat(),
        "day0_validation": {
            "passed": day0_results.get("proceed_to_phase1", False),
            "recommendation": day0_results.get("recommendation", "unknown"),
            "data_quality": day0_results.get("checks", {}).get("data_quality", {}),
            "causal_hypothesis": day0_results.get("checks", {}).get("causal_hypothesis", {}),
            "class_imbalance": day0_results.get("checks", {}).get("class_imbalance", {}),
        },
        "phase1_eegnet": {
            "mean_f1": phase1_results.get("mean_f1"),
            "std_f1": phase1_results.get("std_f1"),
            "target_achieved": phase1_results.get("target_achieved", False),
            "cohens_kappa": phase1_results.get("cohens_kappa"),
        },
        "phase3_mema": {
            "phi_analysis": phase3_results.get("experiment_C_phi_analysis", {}),
            "alternative_metrics": phase3_results.get("experiment_B_alternative_metrics", {}),
            "best_metric": phase3_results.get("summary", {}).get("best_discriminating_metric"),
            "best_p_value": phase3_results.get("summary", {}).get("best_p_value"),
        },
    }
    
    # 综合评估
    phase1_success = phase1_results.get("target_achieved", False)
    phi_significant = phase3_results.get("summary", {}).get("significant_at_005", False)
    
    if phase1_success and phi_significant:
        overall_status = "SUCCESS"
        overall_message = "V3 实验全部成功！EEGNet达标，指标显著区分状态。"
    elif phase1_success:
        overall_status = "PARTIAL"
        overall_message = "EEGNet分类达标，但Φ/替代指标未达显著性，建议深入Phase 3理论探索。"
    else:
        overall_status = "NEEDS_IMPROVEMENT"
        overall_message = "EEGNet未达标，建议检查数据质量或尝试更复杂模型。"
    
    report["overall_assessment"] = {
        "status": overall_status,
        "message": overall_message,
        "phase1_target_f1_040": phase1_success,
        "phi_significant_p005": phi_significant,
    }
    
    # 保存报告
    out_path = RESULTS_DIR / "combined_report_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"综合报告已保存: {out_path}")
    
    return report


def print_summary(report: dict):
    """打印摘要"""
    print("\n" + "=" * 70)
    print("V3 实验总结")
    print("=" * 70)
    
    # Day 0
    day0 = report.get("day0_validation", {})
    print(f"\n📋 Day 0 验证:")
    print(f"   数据质量: {'✅ 通过' if day0.get('data_quality', {}).get('passed') else '❌ 未通过'}")
    print(f"   因果假设: {'✅ 成立' if day0.get('causal_hypothesis', {}).get('passed') else '❌ 不成立'}")
    print(f"   类别平衡: {'✅ 平衡' if day0.get('class_imbalance', {}).get('passed') else '⚠️ 不平衡'}")
    
    # Phase 1
    p1 = report.get("phase1_eegnet", {})
    print(f"\n🧠 Phase 1 EEGNet:")
    if p1.get("mean_f1"):
        print(f"   F1 分数: {p1['mean_f1']:.4f} ± {p1.get('std_f1', 0):.4f}")
        print(f"   目标达成 (F1>0.40): {'✅' if p1.get('target_achieved') else '❌'}")
    else:
        print(f"   状态: 未运行或失败")
    
    # Phase 3
    p3 = report.get("phase3_mema", {})
    print(f"\n📊 Phase 3 MEMA:")
    print(f"   最佳区分指标: {p3.get('best_metric', 'N/A')}")
    print(f"   最佳 p 值: {p3.get('best_p_value', 'N/A')}")
    print(f"   显著性(p<0.05): {'✅' if p3.get('best_p_value', 1) < 0.05 else '❌'}")
    
    # 总体评估
    overall = report.get("overall_assessment", {})
    print(f"\n🎯 总体评估:")
    print(f"   状态: {overall.get('status', 'unknown')}")
    print(f"   {overall.get('message', '')}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="NCT 教育领域实验 V3 总控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_education_experiments_v3.py --day0        # 仅运行Day 0验证
    python run_education_experiments_v3.py --phase1     # 仅运行Phase 1
    python run_education_experiments_v3.py --phase3     # 仅运行Phase 3
    python run_education_experiments_v3.py --all        # 运行所有阶段
    python run_education_experiments_v3.py --skip-day0  # 跳过Day 0直接运行
        """
    )
    
    parser.add_argument("--day0", action="store_true", help="仅运行 Day 0 验证")
    parser.add_argument("--phase1", action="store_true", help="仅运行 Phase 1 (EEGNet)")
    parser.add_argument("--phase3", action="store_true", help="仅运行 Phase 3 (MEMA)")
    parser.add_argument("--all", action="store_true", help="运行所有阶段")
    parser.add_argument("--skip-day0", action="store_true", help="跳过 Day 0 验证")
    parser.add_argument("--force", action="store_true", help="忽略 Day 0 建议，强制执行")
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，显示帮助
    if not any([args.day0, args.phase1, args.phase3, args.all]):
        parser.print_help()
        print("\n💡 推荐: python run_education_experiments_v3.py --all")
        return
    
    logger.info("=" * 70)
    logger.info("NCT 教育领域实验 V3")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    
    day0_results = {}
    phase1_results = {}
    phase3_results = {}
    
    # Day 0 验证
    if args.day0 or (args.all and not args.skip_day0):
        day0_results = run_day0_validation()
        
        if args.day0:
            # 仅运行 Day 0，打印结果后退出
            print("\n" + "=" * 70)
            print(f"Day 0 验证完成!")
            print(f"建议: {day0_results.get('recommendation_text', 'N/A')}")
            print("=" * 70)
            return
        
        # 检查是否继续
        if not day0_results.get("proceed_to_phase1", False) and not args.force:
            logger.warning("Day 0 验证建议不继续 Phase 1-2")
            logger.warning("使用 --force 参数可强制继续")
            print("\n⚠️ Day 0 验证建议跳过 Phase 1-2，直接进入 Phase 3")
            print(f"   原因: {day0_results.get('recommendation_text', 'N/A')}")
            print("   使用 --force 参数可强制继续")
            
            # 直接运行 Phase 3
            phase3_results = run_phase3_mema()
            
            report = generate_combined_report(day0_results, phase1_results, phase3_results)
            print_summary(report)
            return
    
    # Phase 1: EEGNet
    if args.phase1 or args.all:
        phase1_results = run_phase1_eegnet()
        
        if args.phase1:
            print("\n" + "=" * 70)
            print(f"Phase 1 完成!")
            print(f"EEGNet F1: {phase1_results.get('mean_f1', 'N/A')}")
            print(f"目标达成: {'✅' if phase1_results.get('target_achieved') else '❌'}")
            print("=" * 70)
            return
    
    # Phase 3: MEMA (含替代指标)
    if args.phase3 or args.all:
        phase3_results = run_phase3_mema()
        
        if args.phase3:
            print("\n" + "=" * 70)
            print(f"Phase 3 完成!")
            print(f"最佳指标: {phase3_results.get('summary', {}).get('best_discriminating_metric', 'N/A')}")
            print(f"显著性: {'✅' if phase3_results.get('summary', {}).get('significant_at_005') else '❌'}")
            print("=" * 70)
            return
    
    # 生成综合报告
    if args.all:
        report = generate_combined_report(day0_results, phase1_results, phase3_results)
        print_summary(report)
    
    logger.info("=" * 70)
    logger.info(f"V3 实验完成! 结束时间: {datetime.now().isoformat()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
