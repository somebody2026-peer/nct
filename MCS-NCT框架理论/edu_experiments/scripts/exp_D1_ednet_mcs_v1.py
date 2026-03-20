"""
Phase D: EdNet MCS 自适应学习实验 v1
核心假设 H_D: MCS 6维约束驱动的自适应 > NCT Phi驱动 > Fixed
NCT 基线: Fixed AUC=0.4695, NCT-Phi AUC=0.4690, ΔAUC=-0.0005

实验设计: 三臂对比实验
- Fixed: 固定难度策略（随机出题）
- NCT-Phi: Phi 值驱动（Phi低→降低难度，Phi高→提高难度）
- MCS-6D: 6维约束驱动（综合 C2/C3/C6 调整难度）

评估方法: 离线模拟 + 题目重排 + 学习曲线AUC
"""

import sys
sys.path.insert(0, 'D:/python_projects/NCT/MCS-NCT框架理论')

import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# 导入项目模块
from edu_experiments.config import (
    setup_environment, D_MODEL, DEVICE, RANDOM_SEED, EDU_WEIGHTS,
    EDNET_DIR, RESULTS_ROOT, CONSTRAINT_KEYS
)
from mcs_solver import MCSConsciousnessSolver, MCSState
from edu_experiments.utils.stats import one_way_anova, paired_ttest, cohens_d
from edu_experiments.utils.plotting import (
    plot_learning_curves, plot_boxplot_comparison, plot_bar_comparison
)

# ============================================================================
# 配置
# ============================================================================
MAX_STUDENTS = 300        # 最大学生数
MIN_QUESTIONS = 50        # 每个学生最少题目数
SEQ_LEN = 20              # MCS 输入序列长度
WINDOW_SIZE = 10          # 滑动窗口大小（计算准确率）
RANDOM_SEED_EXP = 42

# 结果保存路径
EXP_NAME = "exp_D"
VERSION = "v1"
RESULTS_DIR = RESULTS_ROOT / EXP_NAME / VERSION
FIGURES_DIR = RESULTS_DIR / "figures"

# ============================================================================
# 简化的特征编码器（不使用 nn.Module，直接 numpy 处理）
# ============================================================================

class SimpleFeatureEncoder:
    """简化的特征编码器，将答题序列转换为 MCS Solver 输入"""
    
    def __init__(self, d_model: int = 128, device: str = 'cuda'):
        self.d_model = d_model
        self.device = device
        # 使用固定的随机投影矩阵
        np.random.seed(42)
        self.proj_matrix = np.random.randn(5, d_model).astype(np.float32) * 0.1
        
    def encode(self, questions: List[Dict], seq_len: int = 20) -> Dict[str, torch.Tensor]:
        """
        将答题序列编码为 MCS Solver 输入
        
        Args:
            questions: [{'correct': 0/1, 'elapsed_norm': float, 'part_norm': float}, ...]
            seq_len: 序列长度
            
        Returns:
            {"visual": [1,T,D], "auditory": [1,T,D], "current_state": [1,D]}
        """
        # 取最近 seq_len 个题目
        if len(questions) > seq_len:
            questions = questions[-seq_len:]
        elif len(questions) < seq_len:
            # 填充
            padding = [{'correct': 0.5, 'elapsed_norm': 0.5, 'part_norm': 0.5}] * (seq_len - len(questions))
            questions = padding + questions
        
        features = []
        for q in questions:
            # 特征: [correct, elapsed_norm, part_norm, knowledge_est, 1.0]
            feat = np.array([
                q.get('correct', 0.5),
                q.get('elapsed_norm', 0.5),
                q.get('part_norm', 0.5),
                q.get('knowledge_est', 0.5),
                1.0  # bias
            ], dtype=np.float32)
            # 投影到 d_model
            projected = feat @ self.proj_matrix
            features.append(projected)
        
        features = np.stack(features, axis=0)  # [T, D]
        
        # 添加位置信息（正弦位置编码简化版）
        positions = np.arange(seq_len, dtype=np.float32) / seq_len
        pos_features = np.sin(positions[:, None] * np.arange(self.d_model)[None, :] * 0.1)
        features = features + pos_features * 0.1
        
        # 转为 torch tensor
        visual = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        auditory = visual.clone()  # 听觉通道使用相同输入
        current_state = visual.mean(dim=1)  # [1, D]
        
        return {
            "visual": visual,
            "auditory": auditory,
            "current_state": current_state
        }


# ============================================================================
# 数据加载
# ============================================================================

def load_contents() -> pd.DataFrame:
    """加载题目信息"""
    contents_path = EDNET_DIR / "contents.csv"
    df = pd.read_csv(str(contents_path))
    print(f"[Data] Loaded {len(df)} questions from contents.csv")
    return df


def load_student_data(max_students: int = 300, min_questions: int = 50) -> List[Dict]:
    """
    加载学生答题数据
    
    Returns:
        List[Dict]: [{'user_id': str, 'interactions': pd.DataFrame}, ...]
    """
    kt1_dir = EDNET_DIR / "KT1"
    if not kt1_dir.exists():
        raise FileNotFoundError(f"KT1 directory not found: {kt1_dir}")
    
    students = []
    skipped = 0
    for user_file in sorted(kt1_dir.glob("u*.csv")):
        if len(students) >= max_students:
            break
        
        try:
            df = pd.read_csv(str(user_file))
            # 检查必要的列是否存在
            required_cols = ['question_id', 'elapsed_time', 'answered_correctly']
            if not all(col in df.columns for col in required_cols):
                skipped += 1
                continue
            if len(df) >= min_questions:
                students.append({
                    'user_id': user_file.stem,
                    'interactions': df
                })
        except Exception as e:
            continue
    
    print(f"[Data] Loaded {len(students)} students with >= {min_questions} questions (skipped {skipped} incompatible files)")
    return students


def prepare_student_questions(
    student_df: pd.DataFrame,
    contents_df: pd.DataFrame
) -> List[Dict]:
    """
    准备单个学生的答题记录
    
    Returns:
        List[Dict]: [{'question_id': str, 'correct': int, 'elapsed_norm': float, 
                      'part': int, 'part_norm': float, 'timestamp': int}, ...]
    """
    # 创建题目难度映射
    part_map = dict(zip(contents_df['question_id'], contents_df['part']))
    
    questions = []
    knowledge_est = 0.5  # 初始知识估计
    
    for _, row in student_df.iterrows():
        qid = row['question_id']
        correct = int(row['answered_correctly'])
        elapsed = row['elapsed_time'] / 1000.0  # 转秒
        elapsed_norm = min(elapsed / 300.0, 1.0)  # 5分钟为上限
        part = part_map.get(qid, 4)  # 默认中间难度
        
        # 更新知识估计（指数移动平均）
        if correct == 1:
            delta = 0.1 * (1 - elapsed_norm)
        else:
            delta = -0.1 * (1 + elapsed_norm)
        knowledge_est = 0.3 * (knowledge_est + delta) + 0.7 * knowledge_est
        knowledge_est = max(0, min(1, knowledge_est))
        
        questions.append({
            'question_id': qid,
            'correct': correct,
            'elapsed_norm': elapsed_norm,
            'part': part,
            'part_norm': part / 7.0,
            'timestamp': row['timestamp'],
            'knowledge_est': knowledge_est
        })
    
    return questions


# ============================================================================
# 策略实现
# ============================================================================

def categorize_by_difficulty(questions: List[Dict]) -> Dict[str, List[Dict]]:
    """按难度分类题目"""
    easy = [q for q in questions if q['part'] <= 2]
    medium = [q for q in questions if 3 <= q['part'] <= 5]
    hard = [q for q in questions if q['part'] >= 6]
    
    # 确保每个类别都有题目
    all_questions = questions.copy()
    random.shuffle(all_questions)
    
    if not easy:
        easy = all_questions[:len(all_questions)//3]
    if not medium:
        medium = all_questions[len(all_questions)//3:2*len(all_questions)//3]
    if not hard:
        hard = all_questions[2*len(all_questions)//3:]
    
    return {'easy': easy, 'medium': medium, 'hard': hard}


def fixed_strategy_reorder(questions: List[Dict], seed: int = 42) -> List[Dict]:
    """
    Fixed 策略：随机顺序
    """
    reordered = questions.copy()
    random.seed(seed)
    random.shuffle(reordered)
    return reordered


def phi_strategy_reorder(
    questions: List[Dict],
    solver: MCSConsciousnessSolver,
    encoder: SimpleFeatureEncoder
) -> List[Dict]:
    """
    NCT-Phi 策略：根据 Phi 值调整难度
    - Phi 低 → 选择简单题
    - Phi 高 → 选择难题
    """
    categorized = categorize_by_difficulty(questions)
    available = {k: v.copy() for k, v in categorized.items()}
    
    reordered = []
    history = []
    
    # 重置时间约束历史
    solver.c2_temporal.reset_history(1)
    
    while len(reordered) < len(questions):
        # 计算当前 Phi 值
        if len(history) >= 5:
            mcs_input = encoder.encode(history[-SEQ_LEN:], SEQ_LEN)
            with torch.no_grad():
                mcs_state = solver(
                    visual=mcs_input['visual'],
                    auditory=mcs_input['auditory'],
                    current_state=mcs_input['current_state']
                )
            phi = mcs_state.phi_value
        else:
            phi = 0.5  # 初始默认
        
        # 根据 Phi 选择难度
        if phi < 0.3:
            target_diff = 'easy'
        elif phi > 0.5:
            target_diff = 'hard'
        else:
            target_diff = 'medium'
        
        # 从目标难度选择题目
        for diff in [target_diff, 'medium', 'easy', 'hard']:
            if available[diff]:
                q = available[diff].pop(0)
                reordered.append(q)
                history.append(q)
                break
        else:
            # 所有题目用完
            break
    
    return reordered


def mcs_strategy_reorder(
    questions: List[Dict],
    solver: MCSConsciousnessSolver,
    encoder: SimpleFeatureEncoder
) -> List[Dict]:
    """
    MCS-6D 策略：根据 6 维约束向量调整难度
    - C2 temporal 高违反 → 降低难度（思路不连贯）
    - C6 phi 高违反 → 增加练习题（整合能力不足）
    - C3 self 低违反 → 尝试难题（掌握稳定）
    - C1 sensory 高违反 → 简化呈现
    """
    categorized = categorize_by_difficulty(questions)
    available = {k: v.copy() for k, v in categorized.items()}
    
    reordered = []
    history = []
    
    # 重置时间约束历史
    solver.c2_temporal.reset_history(1)
    
    while len(reordered) < len(questions):
        # 计算当前约束违反情况
        if len(history) >= 5:
            mcs_input = encoder.encode(history[-SEQ_LEN:], SEQ_LEN)
            with torch.no_grad():
                mcs_state = solver(
                    visual=mcs_input['visual'],
                    auditory=mcs_input['auditory'],
                    current_state=mcs_input['current_state']
                )
            cv = mcs_state.constraint_violations
            c2 = cv['temporal_continuity']
            c3 = cv['self_consistency']
            c6 = cv['integrated_information']
        else:
            c2, c3, c6 = 0.5, 0.5, 0.5
        
        # MCS 6D 决策逻辑
        if c2 > 0.5:
            # 时间连续性差 → 降难度
            target_diff = 'easy'
        elif c3 < 0.3 and c6 < 0.3:
            # 自我一致 + 整合好 → 提难度
            target_diff = 'hard'
        elif c6 > 0.5:
            # 整合差 → 练习题（简单到中等）
            target_diff = 'easy' if random.random() < 0.5 else 'medium'
        else:
            target_diff = 'medium'
        
        # 从目标难度选择题目
        for diff in [target_diff, 'medium', 'easy', 'hard']:
            if available[diff]:
                q = available[diff].pop(0)
                reordered.append(q)
                history.append(q)
                break
        else:
            break
    
    return reordered


# ============================================================================
# 评估指标
# ============================================================================

def compute_learning_curve(questions: List[Dict], window_size: int = 10) -> List[float]:
    """
    计算学习曲线（滑动窗口准确率）
    """
    if len(questions) < window_size:
        return [sum(q['correct'] for q in questions) / len(questions)]
    
    curve = []
    for i in range(window_size, len(questions) + 1):
        window = questions[i-window_size:i]
        acc = sum(q['correct'] for q in window) / window_size
        curve.append(acc)
    
    return curve


def compute_auc(curve: List[float]) -> float:
    """计算学习曲线下面积（归一化到 0-1）"""
    if not curve:
        return 0.0
    return np.trapezoid(curve) / len(curve)


def evaluate_strategy(questions: List[Dict], window_size: int = 10) -> Dict:
    """
    评估单个策略的效果
    
    Returns:
        {'auc': float, 'final_acc': float, 'curve': List[float]}
    """
    curve = compute_learning_curve(questions, window_size)
    auc = compute_auc(curve)
    final_acc = curve[-1] if curve else 0.0
    overall_acc = sum(q['correct'] for q in questions) / len(questions) if questions else 0.0
    
    return {
        'auc': auc,
        'final_acc': final_acc,
        'overall_acc': overall_acc,
        'curve': curve
    }


# ============================================================================
# 约束动态追踪
# ============================================================================

def track_constraint_dynamics(
    questions: List[Dict],
    solver: MCSConsciousnessSolver,
    encoder: SimpleFeatureEncoder,
    sample_interval: int = 5
) -> Dict[str, List[float]]:
    """
    追踪学习过程中 6 维约束的变化
    
    Returns:
        {'C1': [v1, v2, ...], 'C2': [...], ...}
    """
    dynamics = {f'C{i+1}': [] for i in range(6)}
    dynamics['phi'] = []
    dynamics['consciousness'] = []
    
    history = []
    solver.c2_temporal.reset_history(1)
    
    for i, q in enumerate(questions):
        history.append(q)
        
        if i > 0 and i % sample_interval == 0 and len(history) >= 5:
            mcs_input = encoder.encode(history[-SEQ_LEN:], SEQ_LEN)
            with torch.no_grad():
                mcs_state = solver(
                    visual=mcs_input['visual'],
                    auditory=mcs_input['auditory'],
                    current_state=mcs_input['current_state']
                )
            
            cv = mcs_state.constraint_violations
            dynamics['C1'].append(1 - cv['sensory_coherence'])  # 满足度 = 1 - 违反度
            dynamics['C2'].append(1 - cv['temporal_continuity'])
            dynamics['C3'].append(1 - cv['self_consistency'])
            dynamics['C4'].append(1 - cv['action_feasibility'])
            dynamics['C5'].append(1 - cv['social_interpretability'])
            dynamics['C6'].append(1 - cv['integrated_information'])
            dynamics['phi'].append(mcs_state.phi_value)
            dynamics['consciousness'].append(mcs_state.consciousness_level)
    
    return dynamics


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment():
    """运行 Phase D 实验"""
    print("=" * 80)
    print("Phase D: EdNet MCS Adaptive Learning Experiment v1")
    print("=" * 80)
    
    # 1. 环境初始化
    setup_environment()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    random.seed(RANDOM_SEED_EXP)
    np.random.seed(RANDOM_SEED_EXP)
    
    # 2. 加载数据
    print("\n[Step 1] Loading data...")
    contents_df = load_contents()
    students = load_student_data(MAX_STUDENTS, MIN_QUESTIONS)
    
    if not students:
        raise RuntimeError("No valid students found!")
    
    # 3. 初始化模型
    print("\n[Step 2] Initializing models...")
    solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS)
    solver = solver.to(DEVICE)
    solver.eval()
    encoder = SimpleFeatureEncoder(d_model=D_MODEL, device=DEVICE)
    
    # 4. 对每个学生应用三种策略
    print(f"\n[Step 3] Running strategies on {len(students)} students...")
    
    results = {
        'Fixed': {'aucs': [], 'final_accs': [], 'curves': []},
        'NCT-Phi': {'aucs': [], 'final_accs': [], 'curves': []},
        'MCS-6D': {'aucs': [], 'final_accs': [], 'curves': []}
    }
    
    # 用于约束动态追踪（只追踪前10个学生以节省时间）
    mcs_dynamics_all = []
    
    for idx, student in enumerate(students):
        if (idx + 1) % 50 == 0:
            print(f"  Processing student {idx + 1}/{len(students)}...")
        
        # 准备该学生的答题数据
        questions = prepare_student_questions(student['interactions'], contents_df)
        
        # Fixed 策略
        fixed_questions = fixed_strategy_reorder(questions, seed=RANDOM_SEED_EXP + idx)
        fixed_eval = evaluate_strategy(fixed_questions, WINDOW_SIZE)
        results['Fixed']['aucs'].append(fixed_eval['auc'])
        results['Fixed']['final_accs'].append(fixed_eval['final_acc'])
        results['Fixed']['curves'].append(fixed_eval['curve'])
        
        # NCT-Phi 策略
        phi_questions = phi_strategy_reorder(questions, solver, encoder)
        phi_eval = evaluate_strategy(phi_questions, WINDOW_SIZE)
        results['NCT-Phi']['aucs'].append(phi_eval['auc'])
        results['NCT-Phi']['final_accs'].append(phi_eval['final_acc'])
        results['NCT-Phi']['curves'].append(phi_eval['curve'])
        
        # MCS-6D 策略
        mcs_questions = mcs_strategy_reorder(questions, solver, encoder)
        mcs_eval = evaluate_strategy(mcs_questions, WINDOW_SIZE)
        results['MCS-6D']['aucs'].append(mcs_eval['auc'])
        results['MCS-6D']['final_accs'].append(mcs_eval['final_acc'])
        results['MCS-6D']['curves'].append(mcs_eval['curve'])
        
        # 追踪约束动态（前10个学生）
        if idx < 10:
            dynamics = track_constraint_dynamics(mcs_questions, solver, encoder)
            mcs_dynamics_all.append(dynamics)
    
    # 5. 统计分析
    print("\n[Step 4] Statistical analysis...")
    
    # 转换为 numpy 数组
    fixed_aucs = np.array(results['Fixed']['aucs'])
    phi_aucs = np.array(results['NCT-Phi']['aucs'])
    mcs_aucs = np.array(results['MCS-6D']['aucs'])
    
    # ANOVA
    anova_result = one_way_anova({
        'Fixed': fixed_aucs,
        'NCT-Phi': phi_aucs,
        'MCS-6D': mcs_aucs
    })
    
    # 配对 t 检验
    fixed_vs_mcs = paired_ttest(fixed_aucs, mcs_aucs)
    phi_vs_mcs = paired_ttest(phi_aucs, mcs_aucs)
    fixed_vs_phi = paired_ttest(fixed_aucs, phi_aucs)
    
    # 效应量
    d_fixed_mcs = cohens_d(fixed_aucs, mcs_aucs)
    d_phi_mcs = cohens_d(phi_aucs, mcs_aucs)
    
    # 6. 输出统计报告
    print("\n" + "=" * 60)
    print("Phase D: EdNet MCS Adaptive Learning Results")
    print("=" * 60)
    print(f"Students: N={len(students)}")
    print(f"Questions per student: {MIN_QUESTIONS}+")
    
    print("\n--- AUC Comparison ---")
    print(f"| {'Strategy':<10} | {'AUC Mean':>10} | {'AUC Std':>10} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*12}|")
    for name in ['Fixed', 'NCT-Phi', 'MCS-6D']:
        aucs = np.array(results[name]['aucs'])
        print(f"| {name:<10} | {np.mean(aucs):>10.4f} | {np.std(aucs):>10.4f} |")
    
    print("\n--- ANOVA ---")
    print(f"F={anova_result['F']:.4f}, p={anova_result['p']:.6f}, eta²={anova_result['eta_squared']:.4f}")
    
    print("\n--- Pairwise Comparisons ---")
    print(f"Fixed vs MCS-6D: t={fixed_vs_mcs['t']:.4f}, p={fixed_vs_mcs['p']:.6f}, d={d_fixed_mcs:.4f}")
    print(f"NCT-Phi vs MCS-6D: t={phi_vs_mcs['t']:.4f}, p={phi_vs_mcs['p']:.6f}, d={d_phi_mcs:.4f}")
    print(f"Fixed vs NCT-Phi: t={fixed_vs_phi['t']:.4f}, p={fixed_vs_phi['p']:.6f}")
    
    print("\n--- Final Accuracy ---")
    for name in ['Fixed', 'NCT-Phi', 'MCS-6D']:
        final_accs = np.array(results[name]['final_accs'])
        print(f"{name} final accuracy: {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}")
    
    # 成功标准评估
    mcs_better_fixed = np.mean(mcs_aucs) > np.mean(fixed_aucs)
    mcs_sig_phi = phi_vs_mcs['p'] < 0.05
    
    print("\n--- Success Criteria ---")
    print(f"[{'PASS' if mcs_better_fixed else 'FAIL'}] MCS-6D AUC ({np.mean(mcs_aucs):.4f}) > Fixed AUC ({np.mean(fixed_aucs):.4f})")
    print(f"[{'PASS' if mcs_sig_phi else 'FAIL'}] MCS-6D vs NCT-Phi statistically significant (p={phi_vs_mcs['p']:.6f})")
    print("=" * 60)
    
    # 7. 生成图表
    print("\n[Step 5] Generating figures...")
    
    # 图1: 学习曲线对比
    # 计算平均学习曲线（对齐到最短长度）
    min_len = min(
        min(len(c) for c in results['Fixed']['curves']),
        min(len(c) for c in results['NCT-Phi']['curves']),
        min(len(c) for c in results['MCS-6D']['curves'])
    )
    
    avg_curves = {}
    for name in ['Fixed', 'NCT-Phi', 'MCS-6D']:
        curves_aligned = [c[:min_len] for c in results[name]['curves']]
        avg_curves[name] = np.mean(curves_aligned, axis=0)
    
    plot_learning_curves(
        curves_dict=avg_curves,
        save_path=FIGURES_DIR / "fig_D_learning_curves_v1.png",
        xlabel="Question Index (windowed)",
        ylabel="Accuracy (sliding window)",
        title="Learning Curves: Fixed vs NCT-Phi vs MCS-6D",
        smooth=True
    )
    
    # 图2: AUC 箱线图
    plot_boxplot_comparison(
        groups_dict={
            'Fixed': fixed_aucs,
            'NCT-Phi': phi_aucs,
            'MCS-6D': mcs_aucs
        },
        xlabel="Strategy",
        ylabel="AUC (Learning Curve)",
        save_path=FIGURES_DIR / "fig_D_auc_comparison_v1.png",
        title="AUC Comparison: Three Adaptive Strategies"
    )
    
    # 图3: 约束动态变化（MCS-6D 策略下）
    if mcs_dynamics_all:
        # 平均动态
        avg_dynamics = {}
        for key in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
            all_values = [d[key] for d in mcs_dynamics_all if d[key]]
            if all_values:
                min_len_dyn = min(len(v) for v in all_values)
                aligned = [v[:min_len_dyn] for v in all_values]
                avg_dynamics[key] = np.mean(aligned, axis=0)
        
        if avg_dynamics:
            plot_learning_curves(
                curves_dict=avg_dynamics,
                save_path=FIGURES_DIR / "fig_D_constraint_dynamics_v1.png",
                xlabel="Learning Progress (sampled)",
                ylabel="Constraint Satisfaction (1 - violation)",
                title="MCS 6-Constraint Dynamics During Learning",
                smooth=True
            )
    
    # 8. 保存结果
    print("\n[Step 6] Saving results...")
    
    metrics = {
        'experiment': 'Phase D: EdNet MCS Adaptive Learning',
        'version': VERSION,
        'n_students': len(students),
        'min_questions': MIN_QUESTIONS,
        'window_size': WINDOW_SIZE,
        'results': {
            'Fixed': {
                'auc_mean': float(np.mean(fixed_aucs)),
                'auc_std': float(np.std(fixed_aucs)),
                'final_acc_mean': float(np.mean(results['Fixed']['final_accs'])),
                'final_acc_std': float(np.std(results['Fixed']['final_accs']))
            },
            'NCT-Phi': {
                'auc_mean': float(np.mean(phi_aucs)),
                'auc_std': float(np.std(phi_aucs)),
                'final_acc_mean': float(np.mean(results['NCT-Phi']['final_accs'])),
                'final_acc_std': float(np.std(results['NCT-Phi']['final_accs']))
            },
            'MCS-6D': {
                'auc_mean': float(np.mean(mcs_aucs)),
                'auc_std': float(np.std(mcs_aucs)),
                'final_acc_mean': float(np.mean(results['MCS-6D']['final_accs'])),
                'final_acc_std': float(np.std(results['MCS-6D']['final_accs']))
            }
        },
        'statistics': {
            'anova': {
                'F': float(anova_result['F']),
                'p': float(anova_result['p']),
                'eta_squared': float(anova_result['eta_squared']),
                'significant': bool(anova_result['p'] < 0.05)
            },
            'fixed_vs_mcs': {
                't': float(fixed_vs_mcs['t']),
                'p': float(fixed_vs_mcs['p']),
                'cohens_d': float(d_fixed_mcs),
                'significant': bool(fixed_vs_mcs['p'] < 0.05)
            },
            'phi_vs_mcs': {
                't': float(phi_vs_mcs['t']),
                'p': float(phi_vs_mcs['p']),
                'cohens_d': float(d_phi_mcs),
                'significant': bool(phi_vs_mcs['p'] < 0.05)
            }
        },
        'success_criteria': {
            'mcs_better_than_fixed': bool(mcs_better_fixed),
            'mcs_sig_vs_phi': bool(mcs_sig_phi),
            'overall_pass': bool(mcs_better_fixed)  # 主要标准
        },
        'nct_baseline': {
            'fixed_auc': 0.4695,
            'nct_phi_auc': 0.4690,
            'delta_auc': -0.0005
        }
    }
    
    with open(RESULTS_DIR / "metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Done] Results saved to {RESULTS_DIR}")
    print(f"[Done] Figures saved to {FIGURES_DIR}")
    
    return metrics


if __name__ == "__main__":
    run_experiment()
