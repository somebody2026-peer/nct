"""
Phase D V2: EdNet MCS 自适应学习实验
=====================================
核心改进:
1. LearnableFeatureEncoder 替代 SimpleFeatureEncoder 的随机矩阵
2. ProfileBasedAdaptiveStrategy 替代简单的 consciousness_level 阈值
3. SimpleIRTBaseline 作为新基线

实验设计: 5种策略对比
- Fixed: 固定难度策略（随机出题）
- NCT-Phi: Phi 值驱动
- MCS-6D-V1: V1 的约束驱动策略
- MCS-Profile-V2: 基于约束 profile 的自适应策略
- IRT: Item Response Theory 基线

评估方法: 5折交叉验证 + AUC
成功标准:
- MCS-Profile-V2 AUC > 0.30 (V1: 0.244)
- MCS vs Fixed: Wilcoxon p < 0.05
"""

import sys
sys.path.insert(0, 'D:/python_projects/NCT/MCS-NCT框架理论')

import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import matplotlib.pyplot as plt

# 导入项目模块
from edu_experiments.config import (
    setup_environment, D_MODEL, DEVICE, RANDOM_SEED,
    EDNET_DIR, RESULTS_ROOT, EDU_WEIGHTS_V2_EXP_D, EDU_WEIGHTS
)
from mcs_solver import MCSConsciousnessSolver
from edu_experiments.mcs_edu_wrapper_v2 import MCSEduWrapperV2
from edu_experiments.utils.stats import one_way_anova, paired_ttest, cohens_d

# ============================================================================
# 配置
# ============================================================================
MAX_STUDENTS = 300
MIN_QUESTIONS = 50
SEQ_LEN = 20
WINDOW_SIZE = 10
N_FOLDS = 5
RANDOM_SEED_EXP = 42

# 结果保存路径
EXP_NAME = "exp_D"
VERSION = "v2"
RESULTS_DIR = RESULTS_ROOT / EXP_NAME / VERSION
FIGURES_DIR = RESULTS_DIR

# ============================================================================
# 可学习特征编码器
# ============================================================================

class LearnableFeatureEncoder(nn.Module):
    """可学习的特征编码器，替代 SimpleFeatureEncoder 的随机矩阵"""
    
    def __init__(self, input_dim: int = 5, d_model: int = 128, device: str = 'cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.device = device
        
        # 特征投影层（替代随机投影）
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # GRU 时序混合层
        self.temporal_mixer = nn.GRU(d_model, d_model, batch_first=True)
        
        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model) * 0.02)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.to(device)
    
    def encode(self, questions: List[Dict], seq_len: int = 20) -> Dict[str, torch.Tensor]:
        """
        从答题记录提取特征 → 投影 → 时序混合
        
        Args:
            questions: [{'correct': 0/1, 'elapsed_norm': float, 'part_norm': float, ...}, ...]
            seq_len: 序列长度
            
        Returns:
            dict with visual [1,T,D], auditory [1,T,D], current_state [1,D]
        """
        # 取最近 seq_len 个题目
        if len(questions) > seq_len:
            questions = questions[-seq_len:]
        elif len(questions) < seq_len:
            # 填充
            padding = [{'correct': 0.5, 'elapsed_norm': 0.5, 'part_norm': 0.5, 
                       'knowledge_est': 0.5, 'difficulty_norm': 0.5}] * (seq_len - len(questions))
            questions = padding + questions
        
        # 提取特征
        features = []
        for q in questions:
            feat = np.array([
                q.get('correct', 0.5),
                q.get('elapsed_norm', 0.5),
                q.get('part_norm', 0.5),
                q.get('knowledge_est', 0.5),
                q.get('difficulty_norm', 0.5) if 'difficulty_norm' in q else q.get('part_norm', 0.5),
            ], dtype=np.float32)
            features.append(feat)
        
        features = np.stack(features, axis=0)  # [T, input_dim]
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, input_dim]
        
        # 投影到 d_model
        projected = self.feature_proj(features_t)  # [1, T, d_model]
        
        # 添加位置编码
        projected = projected + self.pos_embed[:, :seq_len, :]
        
        # GRU 时序混合
        temporal_out, hidden = self.temporal_mixer(projected)  # [1, T, d_model]
        
        # 构建输出
        visual = temporal_out
        auditory = temporal_out.clone()  # 听觉通道使用相同特征
        current_state = hidden.squeeze(0)  # [1, d_model]
        
        return {
            "visual": visual,
            "auditory": auditory,
            "current_state": current_state
        }


# ============================================================================
# 基于约束 Profile 的自适应策略
# ============================================================================

class ProfileBasedAdaptiveStrategy:
    """基于约束 profile 的自适应策略"""
    
    def __init__(self):
        # 难度映射
        self.difficulty_levels = ['easy', 'medium', 'hard']
        # 历史决策记录（用于图表）
        self.decision_history = []
        # 约束 profile 历史
        self.profile_history = []
    
    def reset_history(self):
        """重置历史记录"""
        self.decision_history = []
        self.profile_history = []
    
    def select_difficulty(self, wrapper_result) -> str:
        """
        根据约束 profile 选择难度
        
        Args:
            wrapper_result: MCSEduWrapperV2Result 对象
            
        Returns:
            'easy', 'medium', 或 'hard'
        """
        # 获取关键指标
        consciousness_level = wrapper_result.consciousness_level
        dominant_constraint = wrapper_result.dominant_violation
        weighted_violations = wrapper_result.constraint_violations_weighted
        
        # 保存 profile 历史
        self.profile_history.append({
            'consciousness_level': consciousness_level,
            'dominant_constraint': dominant_constraint,
            'C1': weighted_violations.get('sensory_coherence', 0),
            'C2': weighted_violations.get('temporal_continuity', 0),
            'C6': weighted_violations.get('integrated_information', 0),
        })
        
        # 决策逻辑
        decision = 'medium'  # 默认
        reason = 'default'
        
        # 1. 根据 dominant constraint 类型选择策略
        if dominant_constraint == 'temporal_continuity':
            # C2 temporal 高违反 → 降低难度（学习连贯性差）
            decision = 'easy'
            reason = 'C2_temporal_high'
        elif dominant_constraint == 'integrated_information':
            # C6 phi 高违反 → 切换到复习/练习模式
            decision = 'easy' if random.random() < 0.7 else 'medium'
            reason = 'C6_phi_high'
        elif dominant_constraint == 'sensory_coherence':
            # C1 sensory 高违反 → 维持或简化
            decision = 'easy' if consciousness_level < 0.3 else 'medium'
            reason = 'C1_sensory_high'
        
        # 2. 考虑 consciousness_level 的绝对值进行调整
        if consciousness_level > 0.6:
            # 高意识水平 → 可以尝试提升难度
            if decision == 'easy':
                decision = 'medium'
            elif decision == 'medium':
                decision = 'hard'
            reason = f'{reason}_high_cl'
        elif consciousness_level < 0.3:
            # 低意识水平 → 确保降低难度
            if decision != 'easy':
                decision = 'easy'
            reason = f'{reason}_low_cl'
        
        # 记录决策
        self.decision_history.append({
            'decision': decision,
            'reason': reason,
            'consciousness_level': consciousness_level,
            'dominant_constraint': dominant_constraint
        })
        
        return decision


# ============================================================================
# 简化的 IRT 基线
# ============================================================================

class SimpleIRTBaseline:
    """简化的 1PL IRT 模型"""
    
    def __init__(self):
        self.theta = 0.0  # 学生能力（初始为0）
        self.learning_rate = 0.1
        self.history = []
    
    def reset(self):
        """重置状态"""
        self.theta = 0.0
        self.history = []
    
    def estimate_ability(self, question_history: List[Dict]) -> float:
        """
        使用答题正确率的 logit 变换估计能力值
        
        Args:
            question_history: 答题历史
            
        Returns:
            theta: 能力估计值
        """
        if not question_history:
            return 0.0
        
        # 计算近期正确率（使用权重衰减）
        weights = [0.8 ** (len(question_history) - 1 - i) for i in range(len(question_history))]
        weighted_correct = sum(w * q['correct'] for w, q in zip(weights, question_history))
        total_weight = sum(weights)
        
        correct_rate = weighted_correct / total_weight if total_weight > 0 else 0.5
        
        # Logit 变换，防止除零
        correct_rate = max(0.01, min(0.99, correct_rate))
        theta = np.log(correct_rate / (1 - correct_rate))
        
        # 平滑更新
        self.theta = 0.7 * self.theta + 0.3 * theta
        
        return self.theta
    
    def predict_correctness(self, beta: float) -> float:
        """
        预测在给定难度下的正确概率
        
        P(correct) = sigmoid(theta - beta)
        """
        return 1.0 / (1.0 + np.exp(-(self.theta - beta)))
    
    def select_difficulty(self, question_history: List[Dict]) -> str:
        """
        选择难度：选择使 P(correct) ≈ 0.5 的难度（最大化信息量）
        
        Args:
            question_history: 答题历史
            
        Returns:
            'easy', 'medium', 或 'hard'
        """
        theta = self.estimate_ability(question_history)
        
        # 难度参数映射: easy=-1, medium=0, hard=1
        difficulty_params = {'easy': -1.0, 'medium': 0.0, 'hard': 1.0}
        
        # 选择使 |P(correct) - 0.5| 最小的难度
        best_difficulty = 'medium'
        min_dist = float('inf')
        
        for diff, beta in difficulty_params.items():
            p = self.predict_correctness(beta)
            dist = abs(p - 0.5)
            if dist < min_dist:
                min_dist = dist
                best_difficulty = diff
        
        # 记录历史
        self.history.append({
            'theta': theta,
            'selected': best_difficulty
        })
        
        return best_difficulty


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
    """加载学生答题数据"""
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
            required_cols = ['question_id', 'elapsed_time', 'answered_correctly']
            if not all(col in df.columns for col in required_cols):
                skipped += 1
                continue
            if len(df) >= min_questions:
                students.append({
                    'user_id': user_file.stem,
                    'interactions': df
                })
        except Exception:
            continue
    
    print(f"[Data] Loaded {len(students)} students with >= {min_questions} questions")
    return students


def prepare_student_questions(student_df: pd.DataFrame, contents_df: pd.DataFrame) -> List[Dict]:
    """准备单个学生的答题记录"""
    part_map = dict(zip(contents_df['question_id'], contents_df['part']))
    
    questions = []
    knowledge_est = 0.5
    
    for _, row in student_df.iterrows():
        qid = row['question_id']
        correct = int(row['answered_correctly'])
        elapsed = row['elapsed_time'] / 1000.0
        elapsed_norm = min(elapsed / 300.0, 1.0)
        part = part_map.get(qid, 4)
        
        # 更新知识估计
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
            'knowledge_est': knowledge_est,
            'difficulty_norm': part / 7.0
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
    """Fixed 策略：随机顺序"""
    reordered = questions.copy()
    random.seed(seed)
    random.shuffle(reordered)
    return reordered


def phi_strategy_reorder(questions: List[Dict], solver, encoder: LearnableFeatureEncoder) -> List[Dict]:
    """NCT-Phi 策略"""
    categorized = categorize_by_difficulty(questions)
    available = {k: v.copy() for k, v in categorized.items()}
    
    reordered = []
    history = []
    solver.c2_temporal.reset_history(1)
    
    while len(reordered) < len(questions):
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
            phi = 0.5
        
        if phi < 0.3:
            target_diff = 'easy'
        elif phi > 0.5:
            target_diff = 'hard'
        else:
            target_diff = 'medium'
        
        for diff in [target_diff, 'medium', 'easy', 'hard']:
            if available[diff]:
                q = available[diff].pop(0)
                reordered.append(q)
                history.append(q)
                break
        else:
            break
    
    return reordered


def mcs_v1_strategy_reorder(questions: List[Dict], solver, encoder: LearnableFeatureEncoder) -> List[Dict]:
    """MCS-6D V1 策略（使用原有逻辑但新编码器）"""
    categorized = categorize_by_difficulty(questions)
    available = {k: v.copy() for k, v in categorized.items()}
    
    reordered = []
    history = []
    solver.c2_temporal.reset_history(1)
    
    while len(reordered) < len(questions):
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
        
        if c2 > 0.5:
            target_diff = 'easy'
        elif c3 < 0.3 and c6 < 0.3:
            target_diff = 'hard'
        elif c6 > 0.5:
            target_diff = 'easy' if random.random() < 0.5 else 'medium'
        else:
            target_diff = 'medium'
        
        for diff in [target_diff, 'medium', 'easy', 'hard']:
            if available[diff]:
                q = available[diff].pop(0)
                reordered.append(q)
                history.append(q)
                break
        else:
            break
    
    return reordered


def mcs_profile_v2_strategy_reorder(
    questions: List[Dict], 
    wrapper: MCSEduWrapperV2,
    encoder: LearnableFeatureEncoder,
    strategy: ProfileBasedAdaptiveStrategy
) -> Tuple[List[Dict], List[Dict]]:
    """MCS-Profile V2 策略"""
    categorized = categorize_by_difficulty(questions)
    available = {k: v.copy() for k, v in categorized.items()}
    
    reordered = []
    history = []
    profile_records = []
    
    strategy.reset_history()
    wrapper.reset_normalizer()
    wrapper.solver.c2_temporal.reset_history(1)
    
    while len(reordered) < len(questions):
        if len(history) >= 5:
            mcs_input = encoder.encode(history[-SEQ_LEN:], SEQ_LEN)
            with torch.no_grad():
                result = wrapper.process(
                    visual=mcs_input['visual'],
                    auditory=mcs_input['auditory'],
                    current_state=mcs_input['current_state']
                )
            target_diff = strategy.select_difficulty(result)
            profile_records.append({
                'step': len(reordered),
                'consciousness_level': result.consciousness_level,
                'dominant_violation': result.dominant_violation,
                'weighted_violations': result.constraint_violations_weighted.copy()
            })
        else:
            target_diff = 'medium'
        
        for diff in [target_diff, 'medium', 'easy', 'hard']:
            if available[diff]:
                q = available[diff].pop(0)
                reordered.append(q)
                history.append(q)
                break
        else:
            break
    
    return reordered, profile_records


def irt_strategy_reorder(questions: List[Dict], irt: SimpleIRTBaseline) -> List[Dict]:
    """IRT 策略"""
    categorized = categorize_by_difficulty(questions)
    available = {k: v.copy() for k, v in categorized.items()}
    
    reordered = []
    history = []
    irt.reset()
    
    while len(reordered) < len(questions):
        target_diff = irt.select_difficulty(history)
        
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
    """计算学习曲线"""
    if len(questions) < window_size:
        return [sum(q['correct'] for q in questions) / len(questions)]
    
    curve = []
    for i in range(window_size, len(questions) + 1):
        window = questions[i-window_size:i]
        acc = sum(q['correct'] for q in window) / window_size
        curve.append(acc)
    
    return curve


def compute_auc(curve: List[float]) -> float:
    """计算学习曲线下面积"""
    if not curve:
        return 0.0
    return np.trapezoid(curve) / len(curve)


def evaluate_strategy(questions: List[Dict], window_size: int = 10) -> Dict:
    """评估单个策略"""
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
# 图表生成
# ============================================================================

def plot_strategy_comparison_v2(results: Dict, save_path: Path):
    """5 种策略 AUC 对比条形图"""
    plt.figure(figsize=(10, 6))
    
    strategies = list(results.keys())
    means = [np.mean(results[s]['aucs']) for s in strategies]
    stds = [np.std(results[s]['aucs']) for s in strategies]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    bars = plt.bar(strategies, means, yerr=stds, capsize=5, color=colors[:len(strategies)])
    
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('Strategy Comparison V2: 5-Fold Cross-Validation AUC', fontsize=14)
    plt.ylim(0, max(means) * 1.3)
    
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_auc_v1_vs_v2(v1_results: Dict, v2_results: Dict, save_path: Path):
    """V1 vs V2 AUC 对比"""
    plt.figure(figsize=(10, 6))
    
    # 共同策略
    common_strategies = ['Fixed', 'NCT-Phi', 'MCS-6D']
    x = np.arange(len(common_strategies))
    width = 0.35
    
    v1_means = [v1_results[s]['auc_mean'] for s in common_strategies]
    v2_means = [np.mean(v2_results[s]['aucs']) for s in common_strategies]
    
    bars1 = plt.bar(x - width/2, v1_means, width, label='V1', color='#3498db', alpha=0.8)
    bars2 = plt.bar(x + width/2, v2_means, width, label='V2', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('AUC Comparison: V1 vs V2', fontsize=14)
    plt.xticks(x, common_strategies)
    plt.legend()
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_adaptation_trajectory(decision_history: List[Dict], save_path: Path):
    """典型学生的自适应难度轨迹"""
    if not decision_history:
        print(f"  Skipped adaptation_trajectory (no data)")
        return
    
    plt.figure(figsize=(12, 5))
    
    # 难度映射
    diff_map = {'easy': 1, 'medium': 2, 'hard': 3}
    difficulties = [diff_map.get(d['decision'], 2) for d in decision_history]
    cl_values = [d['consciousness_level'] for d in decision_history]
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # 绘制难度轨迹
    ax1.plot(difficulties, 'b-o', label='Difficulty', alpha=0.7, markersize=3)
    ax1.set_xlabel('Question Index', fontsize=12)
    ax1.set_ylabel('Difficulty Level', color='b', fontsize=12)
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Easy', 'Medium', 'Hard'])
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 绘制意识水平
    ax2 = ax1.twinx()
    ax2.plot(cl_values, 'r-', label='Consciousness Level', alpha=0.7)
    ax2.set_ylabel('Consciousness Level', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Adaptive Difficulty Trajectory (MCS-Profile-V2)', fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_constraint_profile_evolution(profile_records: List[Dict], save_path: Path):
    """约束 profile 随学习进度的变化"""
    if not profile_records:
        print(f"  Skipped constraint_profile_evolution (no data)")
        return
    
    plt.figure(figsize=(12, 6))
    
    steps = [r['step'] for r in profile_records]
    cl_values = [r['consciousness_level'] for r in profile_records]
    c1_values = [r['weighted_violations'].get('sensory_coherence', 0) for r in profile_records]
    c2_values = [r['weighted_violations'].get('temporal_continuity', 0) for r in profile_records]
    c6_values = [r['weighted_violations'].get('integrated_information', 0) for r in profile_records]
    
    plt.plot(steps, c1_values, 'g-', label='C1 (Sensory)', alpha=0.7)
    plt.plot(steps, c2_values, 'b-', label='C2 (Temporal)', alpha=0.7)
    plt.plot(steps, c6_values, 'r-', label='C6 (Phi)', alpha=0.7)
    plt.plot(steps, cl_values, 'k--', label='Consciousness Level', alpha=0.8, linewidth=2)
    
    plt.xlabel('Learning Progress (Question Index)', fontsize=12)
    plt.ylabel('Weighted Violation / Consciousness Level', fontsize=12)
    plt.title('Constraint Profile Evolution During Learning', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment():
    """运行 Phase D V2 实验"""
    print("=" * 80)
    print("Phase D V2: EdNet MCS Adaptive Learning Experiment")
    print("=" * 80)
    
    # 1. 环境初始化
    setup_environment()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    random.seed(RANDOM_SEED_EXP)
    np.random.seed(RANDOM_SEED_EXP)
    torch.manual_seed(RANDOM_SEED_EXP)
    
    # 2. 加载数据
    print("\n[Step 1] Loading data...")
    contents_df = load_contents()
    students = load_student_data(MAX_STUDENTS, MIN_QUESTIONS)
    
    if not students:
        raise RuntimeError("No valid students found!")
    
    # 3. 初始化模型
    print("\n[Step 2] Initializing models...")
    device = torch.device(DEVICE)
    
    # MCS Solver
    solver = MCSConsciousnessSolver(d_model=D_MODEL, constraint_weights=EDU_WEIGHTS)
    solver = solver.to(device)
    solver.eval()
    
    # V2 Wrapper
    wrapper = MCSEduWrapperV2(
        solver=solver,
        weight_profile=EDU_WEIGHTS_V2_EXP_D,
        consciousness_formula='exp',
        normalization_warmup=100
    )
    
    # 可学习编码器
    encoder = LearnableFeatureEncoder(input_dim=5, d_model=D_MODEL, device=DEVICE)
    encoder.eval()
    
    # Profile 策略
    profile_strategy = ProfileBasedAdaptiveStrategy()
    
    # IRT 基线
    irt_baseline = SimpleIRTBaseline()
    
    # 4. 5折交叉验证
    print(f"\n[Step 3] Running 5-fold cross-validation on {len(students)} students...")
    
    results = {
        'Fixed': {'aucs': [], 'final_accs': [], 'curves': []},
        'NCT-Phi': {'aucs': [], 'final_accs': [], 'curves': []},
        'MCS-6D': {'aucs': [], 'final_accs': [], 'curves': []},
        'MCS-Profile-V2': {'aucs': [], 'final_accs': [], 'curves': []},
        'IRT': {'aucs': [], 'final_accs': [], 'curves': []}
    }
    
    # 用于图表的样本数据
    sample_decision_history = []
    sample_profile_records = []
    
    # 随机打乱学生
    random.shuffle(students)
    fold_size = len(students) // N_FOLDS
    
    for fold in range(N_FOLDS):
        print(f"\n  Fold {fold+1}/{N_FOLDS}...")
        
        # 获取当前折的学生
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < N_FOLDS - 1 else len(students)
        fold_students = students[start_idx:end_idx]
        
        for idx, student in enumerate(fold_students):
            questions = prepare_student_questions(student['interactions'], contents_df)
            
            # Fixed 策略
            fixed_questions = fixed_strategy_reorder(questions, seed=RANDOM_SEED_EXP + idx + fold*1000)
            fixed_eval = evaluate_strategy(fixed_questions, WINDOW_SIZE)
            results['Fixed']['aucs'].append(fixed_eval['auc'])
            results['Fixed']['final_accs'].append(fixed_eval['final_acc'])
            
            # NCT-Phi 策略
            phi_questions = phi_strategy_reorder(questions, solver, encoder)
            phi_eval = evaluate_strategy(phi_questions, WINDOW_SIZE)
            results['NCT-Phi']['aucs'].append(phi_eval['auc'])
            results['NCT-Phi']['final_accs'].append(phi_eval['final_acc'])
            
            # MCS-6D V1 策略
            mcs_v1_questions = mcs_v1_strategy_reorder(questions, solver, encoder)
            mcs_v1_eval = evaluate_strategy(mcs_v1_questions, WINDOW_SIZE)
            results['MCS-6D']['aucs'].append(mcs_v1_eval['auc'])
            results['MCS-6D']['final_accs'].append(mcs_v1_eval['final_acc'])
            
            # MCS-Profile V2 策略
            mcs_v2_questions, profile_records = mcs_profile_v2_strategy_reorder(
                questions, wrapper, encoder, profile_strategy
            )
            mcs_v2_eval = evaluate_strategy(mcs_v2_questions, WINDOW_SIZE)
            results['MCS-Profile-V2']['aucs'].append(mcs_v2_eval['auc'])
            results['MCS-Profile-V2']['final_accs'].append(mcs_v2_eval['final_acc'])
            
            # 保存第一个学生的详细记录用于图表
            if fold == 0 and idx == 0:
                sample_decision_history = profile_strategy.decision_history.copy()
                sample_profile_records = profile_records.copy()
            
            # IRT 策略
            irt_questions = irt_strategy_reorder(questions, irt_baseline)
            irt_eval = evaluate_strategy(irt_questions, WINDOW_SIZE)
            results['IRT']['aucs'].append(irt_eval['auc'])
            results['IRT']['final_accs'].append(irt_eval['final_acc'])
    
    # 5. 统计分析
    print("\n[Step 4] Statistical analysis...")
    
    fixed_aucs = np.array(results['Fixed']['aucs'])
    phi_aucs = np.array(results['NCT-Phi']['aucs'])
    mcs_v1_aucs = np.array(results['MCS-6D']['aucs'])
    mcs_v2_aucs = np.array(results['MCS-Profile-V2']['aucs'])
    irt_aucs = np.array(results['IRT']['aucs'])
    
    # ANOVA
    anova_result = one_way_anova({
        'Fixed': fixed_aucs,
        'NCT-Phi': phi_aucs,
        'MCS-6D': mcs_v1_aucs,
        'MCS-Profile-V2': mcs_v2_aucs,
        'IRT': irt_aucs
    })
    
    # Wilcoxon signed-rank tests
    wilcoxon_mcs_fixed = stats.wilcoxon(mcs_v2_aucs, fixed_aucs)
    wilcoxon_mcs_irt = stats.wilcoxon(mcs_v2_aucs, irt_aucs)
    wilcoxon_mcs_phi = stats.wilcoxon(mcs_v2_aucs, phi_aucs)
    
    # 效应量
    d_mcs_fixed = cohens_d(mcs_v2_aucs, fixed_aucs)
    d_mcs_irt = cohens_d(mcs_v2_aucs, irt_aucs)
    
    # 6. 输出统计报告
    print("\n" + "=" * 70)
    print("Phase D V2: EdNet MCS Adaptive Learning Results")
    print("=" * 70)
    print(f"Students: N={len(students)}, Folds: {N_FOLDS}")
    print(f"Total samples: {len(fixed_aucs)}")
    
    print("\n--- AUC Comparison ---")
    print(f"| {'Strategy':<16} | {'AUC Mean':>10} | {'AUC Std':>10} |")
    print(f"|{'-'*18}|{'-'*12}|{'-'*12}|")
    for name in ['Fixed', 'NCT-Phi', 'MCS-6D', 'MCS-Profile-V2', 'IRT']:
        aucs = np.array(results[name]['aucs'])
        print(f"| {name:<16} | {np.mean(aucs):>10.4f} | {np.std(aucs):>10.4f} |")
    
    print("\n--- ANOVA ---")
    print(f"F={anova_result['F']:.4f}, p={anova_result['p']:.6f}")
    
    print("\n--- Wilcoxon Tests (MCS-Profile-V2 vs Others) ---")
    print(f"MCS-V2 vs Fixed: W={wilcoxon_mcs_fixed.statistic:.4f}, p={wilcoxon_mcs_fixed.pvalue:.6f}, d={d_mcs_fixed:.4f}")
    print(f"MCS-V2 vs IRT: W={wilcoxon_mcs_irt.statistic:.4f}, p={wilcoxon_mcs_irt.pvalue:.6f}, d={d_mcs_irt:.4f}")
    print(f"MCS-V2 vs NCT-Phi: W={wilcoxon_mcs_phi.statistic:.4f}, p={wilcoxon_mcs_phi.pvalue:.6f}")
    
    # 成功标准评估
    mcs_v2_mean = np.mean(mcs_v2_aucs)
    success_auc = mcs_v2_mean > 0.30
    success_wilcoxon = wilcoxon_mcs_fixed.pvalue < 0.05
    success_vs_irt = mcs_v2_mean >= np.mean(irt_aucs) - 0.01  # 允许1%误差
    
    print("\n--- Success Criteria ---")
    print(f"[{'PASS' if success_auc else 'FAIL'}] MCS-Profile-V2 AUC ({mcs_v2_mean:.4f}) > 0.30")
    print(f"[{'PASS' if success_wilcoxon else 'FAIL'}] MCS vs Fixed: Wilcoxon p={wilcoxon_mcs_fixed.pvalue:.6f} < 0.05")
    print(f"[{'PASS' if success_vs_irt else 'FAIL'}] MCS vs IRT: comparable or better ({mcs_v2_mean:.4f} vs {np.mean(irt_aucs):.4f})")
    print("=" * 70)
    
    # 7. 生成图表
    print("\n[Step 5] Generating figures...")
    
    # 图1: 5种策略对比
    plot_strategy_comparison_v2(results, FIGURES_DIR / "strategy_comparison_v2.png")
    
    # 图2: V1 vs V2 对比（需要 V1 结果）
    v1_results = {
        'Fixed': {'auc_mean': 0.244},
        'NCT-Phi': {'auc_mean': 0.243},
        'MCS-6D': {'auc_mean': 0.242}
    }
    plot_auc_v1_vs_v2(v1_results, results, FIGURES_DIR / "auc_v1_vs_v2.png")
    
    # 图3: 自适应难度轨迹
    plot_adaptation_trajectory(sample_decision_history, FIGURES_DIR / "adaptation_trajectory.png")
    
    # 图4: 约束 profile 演化
    plot_constraint_profile_evolution(sample_profile_records, FIGURES_DIR / "constraint_profile_evolution.png")
    
    # 8. 保存结果
    print("\n[Step 6] Saving results...")
    
    metrics = {
        'experiment': 'Phase D V2: EdNet MCS Adaptive Learning',
        'version': VERSION,
        'n_students': len(students),
        'n_folds': N_FOLDS,
        'total_samples': len(fixed_aucs),
        'results': {
            name: {
                'auc_mean': float(np.mean(results[name]['aucs'])),
                'auc_std': float(np.std(results[name]['aucs'])),
                'final_acc_mean': float(np.mean(results[name]['final_accs'])),
                'final_acc_std': float(np.std(results[name]['final_accs']))
            }
            for name in results.keys()
        },
        'statistics': {
            'anova': {
                'F': float(anova_result['F']),
                'p': float(anova_result['p']),
                'significant': bool(anova_result['p'] < 0.05)
            },
            'wilcoxon_mcs_v2_vs_fixed': {
                'W': float(wilcoxon_mcs_fixed.statistic),
                'p': float(wilcoxon_mcs_fixed.pvalue),
                'cohens_d': float(d_mcs_fixed),
                'significant': bool(wilcoxon_mcs_fixed.pvalue < 0.05)
            },
            'wilcoxon_mcs_v2_vs_irt': {
                'W': float(wilcoxon_mcs_irt.statistic),
                'p': float(wilcoxon_mcs_irt.pvalue),
                'cohens_d': float(d_mcs_irt),
                'significant': bool(wilcoxon_mcs_irt.pvalue < 0.05)
            }
        },
        'success_criteria': {
            'auc_above_0.30': bool(success_auc),
            'wilcoxon_significant': bool(success_wilcoxon),
            'comparable_to_irt': bool(success_vs_irt),
            'overall_pass': bool(success_auc and success_wilcoxon)
        },
        'v1_baseline': {
            'Fixed_auc': 0.244,
            'NCT-Phi_auc': 0.243,
            'MCS-6D_auc': 0.242,
            'anova_p': 0.957
        },
        'improvements': {
            'encoder': 'LearnableFeatureEncoder (vs SimpleFeatureEncoder random projection)',
            'strategy': 'ProfileBasedAdaptiveStrategy (vs simple threshold)',
            'baseline': 'Added IRT baseline'
        }
    }
    
    with open(RESULTS_DIR / "metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Done] Results saved to {RESULTS_DIR}")
    print(f"[Done] Figures saved to {FIGURES_DIR}")
    
    return metrics


if __name__ == "__main__":
    run_experiment()
