"""
MCS-NCT V2 教育场景包装器
===========================
通过后处理层解决 V1 实验发现的问题:
- R1: C5 violation 主导问题 → 运行时归一化
- R2: consciousness_level 区分力不足 → 新公式
- R3: C3/C4 在教育场景无意义 → 权重控制
- R4: 约束共线性 → 加权策略

作者: NCT Team
日期: 2026年3月
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


# ============================================================================
# 约束名称常量
# ============================================================================

CONSTRAINT_NAMES = [
    'sensory_coherence',
    'temporal_continuity', 
    'self_consistency',
    'action_feasibility',
    'social_interpretability',
    'integrated_information'
]


# ============================================================================
# RunningConstraintNormalizer - 运行时约束归一化器
# ============================================================================

class RunningConstraintNormalizer:
    """
    运行时约束违反值归一化器
    
    解决问题: C5 (social_interpretability) 的 violation 值约 0.92，
    远高于其他约束 (0.25-0.52)，导致 dominant_violation 100% 为 C5
    
    方法:
    1. 收集每个约束的 violation 统计量 (running mean/std)
    2. warmup 阶段：前 N 个样本只收集不归一化
    3. 归一化后：z = (v - mean) / (std + eps)，然后用 sigmoid 映射到 [0,1]
    """
    
    def __init__(
        self,
        warmup_samples: int = 100,
        eps: float = 1e-6,
        momentum: float = 0.1
    ):
        """
        Args:
            warmup_samples: 预热样本数，在此期间只收集统计量不归一化
            eps: 防止除零的小值
            momentum: 移动平均动量 (0-1)，越大越重视新样本
        """
        self.warmup_samples = warmup_samples
        self.eps = eps
        self.momentum = momentum
        
        # 每个约束的统计量
        self.running_mean: Dict[str, float] = {name: 0.0 for name in CONSTRAINT_NAMES}
        self.running_var: Dict[str, float] = {name: 1.0 for name in CONSTRAINT_NAMES}
        self.sample_count: Dict[str, int] = {name: 0 for name in CONSTRAINT_NAMES}
        
        # 缓存的标准差（避免重复计算）
        self._running_std: Dict[str, float] = {name: 1.0 for name in CONSTRAINT_NAMES}
    
    def reset(self):
        """重置所有统计量"""
        for name in CONSTRAINT_NAMES:
            self.running_mean[name] = 0.0
            self.running_var[name] = 1.0
            self.sample_count[name] = 0
            self._running_std[name] = 1.0
    
    def _update_stats(self, name: str, value: float):
        """
        更新指定约束的运行统计量
        使用 Welford 在线算法计算方差
        """
        count = self.sample_count[name]
        
        if count == 0:
            # 第一个样本
            self.running_mean[name] = value
            self.running_var[name] = 0.0
        else:
            # 使用动量更新
            old_mean = self.running_mean[name]
            # 指数移动平均
            new_mean = (1 - self.momentum) * old_mean + self.momentum * value
            # 增量方差更新
            delta = value - old_mean
            delta2 = value - new_mean
            new_var = (1 - self.momentum) * self.running_var[name] + self.momentum * delta * delta2
            
            self.running_mean[name] = new_mean
            self.running_var[name] = max(new_var, self.eps)  # 防止方差为负
        
        self.sample_count[name] += 1
        self._running_std[name] = np.sqrt(self.running_var[name] + self.eps)
    
    def is_warmed_up(self) -> bool:
        """检查是否完成预热"""
        min_count = min(self.sample_count.values())
        return min_count >= self.warmup_samples
    
    def get_warmup_progress(self) -> float:
        """获取预热进度 (0-1)"""
        min_count = min(self.sample_count.values())
        return min(1.0, min_count / self.warmup_samples)
    
    def normalize(
        self,
        violations: Dict[str, float],
        update_stats: bool = True
    ) -> Dict[str, float]:
        """
        归一化约束违反值
        
        Args:
            violations: 原始违反值字典 {约束名: 值}
            update_stats: 是否更新统计量（训练时 True，推理时可以 False）
        
        Returns:
            normalized: 归一化后的违反值字典
        """
        normalized = {}
        
        for name in CONSTRAINT_NAMES:
            value = violations.get(name, 0.0)
            
            # 更新统计量
            if update_stats:
                self._update_stats(name, value)
            
            # 预热阶段返回原始值
            if not self.is_warmed_up():
                normalized[name] = value
            else:
                # z-score 归一化
                z = (value - self.running_mean[name]) / self._running_std[name]
                # sigmoid 映射到 [0, 1]
                normalized[name] = 1.0 / (1.0 + np.exp(-z))
        
        return normalized
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取当前统计量（用于调试）"""
        return {
            name: {
                'mean': self.running_mean[name],
                'std': self._running_std[name],
                'count': self.sample_count[name]
            }
            for name in CONSTRAINT_NAMES
        }


# ============================================================================
# MCSEduWrapperV2Result - V2 结果数据类
# ============================================================================

@dataclass
class MCSEduWrapperV2Result:
    """V2 包装器处理结果"""
    consciousness_level: float           # 新公式计算的意识水平
    consciousness_level_v1: float        # 保留 V1 原始值用于对比
    constraint_violations_raw: Dict[str, float]       # 原始 violation 值
    constraint_violations_normalized: Dict[str, float] # 归一化后的 violation 值
    constraint_violations_weighted: Dict[str, float]   # 加权后的 violation 值
    dominant_violation: str              # 加权后的主导约束名
    dominant_violation_v1: str           # V1 原始的主导约束名
    total_weighted_violation: float      # 加权总违反
    phi_value: float                     # 原始 phi 值
    active_constraints: List[str]        # 权重>0的约束列表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'consciousness_level': self.consciousness_level,
            'consciousness_level_v1': self.consciousness_level_v1,
            'constraint_violations_raw': self.constraint_violations_raw,
            'constraint_violations_normalized': self.constraint_violations_normalized,
            'constraint_violations_weighted': self.constraint_violations_weighted,
            'dominant_violation': self.dominant_violation,
            'dominant_violation_v1': self.dominant_violation_v1,
            'total_weighted_violation': self.total_weighted_violation,
            'phi_value': self.phi_value,
            'active_constraints': self.active_constraints,
        }


# ============================================================================
# MCSEduWrapperV2 - 主包装器类
# ============================================================================

class MCSEduWrapperV2:
    """
    MCS-NCT V2 教育场景包装器
    
    功能:
    1. 包装 MCSConsciousnessSolver，不修改原始实现
    2. 对约束违反值进行运行时归一化
    3. 应用实验专用权重配置
    4. 使用新公式计算 consciousness_level
    5. 返回增强的结果用于分析
    """
    
    def __init__(
        self,
        solver,
        weight_profile: Dict[str, float],
        consciousness_formula: str = "exp",
        normalization_warmup: int = 100,
        normalization_eps: float = 1e-6,
        normalization_momentum: float = 0.1
    ):
        """
        Args:
            solver: MCSConsciousnessSolver 实例
            weight_profile: 实验专用权重，如 {'sensory_coherence': 1.0, ...}
            consciousness_formula: "exp" 或 "geometric"
            normalization_warmup: 归一化预热样本数
            normalization_eps: 归一化 epsilon
            normalization_momentum: 归一化动量
        """
        self.solver = solver
        self.weight_profile = weight_profile
        self.consciousness_formula = consciousness_formula
        
        # 验证公式类型
        if consciousness_formula not in ["exp", "geometric"]:
            raise ValueError(f"consciousness_formula must be 'exp' or 'geometric', got '{consciousness_formula}'")
        
        # 验证权重配置
        for name in CONSTRAINT_NAMES:
            if name not in weight_profile:
                raise ValueError(f"Missing weight for constraint: {name}")
        
        # 初始化归一化器
        self.normalizer = RunningConstraintNormalizer(
            warmup_samples=normalization_warmup,
            eps=normalization_eps,
            momentum=normalization_momentum
        )
        
        # 缓存活跃约束列表
        self._active_constraints = [
            name for name in CONSTRAINT_NAMES 
            if weight_profile.get(name, 0) > 0
        ]
    
    @property
    def active_constraints(self) -> List[str]:
        """获取权重>0的活跃约束列表"""
        return self._active_constraints
    
    def reset_normalizer(self):
        """重置归一化器统计量"""
        self.normalizer.reset()
    
    def is_warmed_up(self) -> bool:
        """检查归一化器是否预热完成"""
        return self.normalizer.is_warmed_up()
    
    def process(
        self,
        visual: torch.Tensor,
        auditory: torch.Tensor,
        current_state: torch.Tensor,
        update_normalizer: bool = True,
        **kwargs
    ) -> MCSEduWrapperV2Result:
        """
        处理输入并返回增强的 MCS 结果
        
        Args:
            visual: 视觉输入 [B, T, D]
            auditory: 听觉输入 [B, T, D]
            current_state: 当前状态 [B, D]
            update_normalizer: 是否更新归一化器统计量
            **kwargs: 传递给 solver.forward() 的其他参数
        
        Returns:
            MCSEduWrapperV2Result: 增强的结果
        """
        # Step 1: 调用原始 solver
        mcs_state = self.solver(
            visual=visual,
            auditory=auditory,
            current_state=current_state,
            **kwargs
        )
        
        # Step 2: 提取原始违反值
        raw_violations = dict(mcs_state.constraint_violations)
        
        # Step 3: 归一化
        normalized_violations = self.normalizer.normalize(
            raw_violations,
            update_stats=update_normalizer
        )
        
        # Step 4: 应用权重
        weighted_violations = {}
        for name in CONSTRAINT_NAMES:
            weight = self.weight_profile.get(name, 0)
            if weight > 0:
                weighted_violations[name] = weight * normalized_violations[name]
            else:
                weighted_violations[name] = 0.0
        
        # Step 5: 计算新的 consciousness_level
        consciousness_level_v2 = self._compute_consciousness_level(
            normalized_violations,
            self.weight_profile
        )
        
        # Step 6: 计算加权主导违反
        dominant_violation_v2 = self._weighted_dominant_violation(
            normalized_violations,
            self.weight_profile
        )
        
        # Step 7: 计算总加权违反
        total_weighted = sum(
            self.weight_profile[name] * normalized_violations[name]
            for name in self._active_constraints
        )
        
        return MCSEduWrapperV2Result(
            consciousness_level=consciousness_level_v2,
            consciousness_level_v1=mcs_state.consciousness_level,
            constraint_violations_raw=raw_violations,
            constraint_violations_normalized=normalized_violations,
            constraint_violations_weighted=weighted_violations,
            dominant_violation=dominant_violation_v2,
            dominant_violation_v1=mcs_state.dominant_violation,
            total_weighted_violation=total_weighted,
            phi_value=mcs_state.phi_value if mcs_state.phi_value is not None else 0.0,
            active_constraints=self._active_constraints.copy()
        )
    
    def _compute_consciousness_level(
        self,
        normalized_violations: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        使用新公式计算意识水平
        
        公式选项:
        - exp: C = exp(-Σ(w_i * v_i))
        - geometric: C = ∏((1 - v_i)^w_i)^(1/Σw_i)
        """
        active_constraints = self._active_constraints
        
        if not active_constraints:
            return 1.0  # 没有活跃约束，返回最大值
        
        if self.consciousness_formula == "exp":
            # 指数公式
            total = sum(
                weights[name] * normalized_violations[name]
                for name in active_constraints
            )
            return float(np.exp(-total))
        
        elif self.consciousness_formula == "geometric":
            # 几何平均公式
            total_weight = sum(weights[name] for name in active_constraints)
            if total_weight <= 0:
                return 1.0
            
            log_sum = 0.0
            for name in active_constraints:
                v = normalized_violations[name]
                w = weights[name]
                # 防止 log(0)
                factor = max(1.0 - v, 1e-10)
                log_sum += w * np.log(factor)
            
            return float(np.exp(log_sum / total_weight))
        
        else:
            raise ValueError(f"Unknown formula: {self.consciousness_formula}")
    
    def _weighted_dominant_violation(
        self,
        normalized_violations: Dict[str, float],
        weights: Dict[str, float]
    ) -> str:
        """
        计算加权后的主导违反约束
        
        只在权重>0的约束中选择 argmax(w_i * v_i)
        """
        if not self._active_constraints:
            return "none"
        
        max_weighted_value = -1.0
        dominant = self._active_constraints[0]
        
        for name in self._active_constraints:
            weighted_value = weights[name] * normalized_violations[name]
            if weighted_value > max_weighted_value:
                max_weighted_value = weighted_value
                dominant = name
        
        return dominant
    
    def get_normalizer_stats(self) -> Dict[str, Dict[str, float]]:
        """获取归一化器统计量（用于调试和分析）"""
        return self.normalizer.get_stats()
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            'weight_profile': self.weight_profile.copy(),
            'consciousness_formula': self.consciousness_formula,
            'active_constraints': self._active_constraints.copy(),
            'normalization_warmup': self.normalizer.warmup_samples,
            'normalization_eps': self.normalizer.eps,
            'normalization_momentum': self.normalizer.momentum,
        }


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_edu_wrapper_v2(
    solver,
    experiment_phase: str,
    consciousness_formula: str = "exp",
    normalization_warmup: int = 100,
    custom_weights: Optional[Dict[str, float]] = None
) -> MCSEduWrapperV2:
    """
    便捷工厂函数：根据实验阶段创建包装器
    
    Args:
        solver: MCSConsciousnessSolver 实例
        experiment_phase: 实验阶段 ("A", "B", "C", "D")
        consciousness_formula: 意识水平公式
        normalization_warmup: 预热样本数
        custom_weights: 自定义权重（覆盖默认值）
    
    Returns:
        MCSEduWrapperV2 实例
    """
    # 延迟导入避免循环依赖
    from config import (
        EDU_WEIGHTS_V2_EXP_A,
        EDU_WEIGHTS_V2_EXP_B,
        EDU_WEIGHTS_V2_EXP_C,
        EDU_WEIGHTS_V2_EXP_D
    )
    
    weight_map = {
        'A': EDU_WEIGHTS_V2_EXP_A,
        'B': EDU_WEIGHTS_V2_EXP_B,
        'C': EDU_WEIGHTS_V2_EXP_C,
        'D': EDU_WEIGHTS_V2_EXP_D,
    }
    
    if experiment_phase not in weight_map:
        raise ValueError(f"Unknown experiment phase: {experiment_phase}. Must be one of {list(weight_map.keys())}")
    
    weights = weight_map[experiment_phase].copy()
    
    # 应用自定义权重
    if custom_weights:
        weights.update(custom_weights)
    
    return MCSEduWrapperV2(
        solver=solver,
        weight_profile=weights,
        consciousness_formula=consciousness_formula,
        normalization_warmup=normalization_warmup
    )


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'CONSTRAINT_NAMES',
    'RunningConstraintNormalizer',
    'MCSEduWrapperV2Result',
    'MCSEduWrapperV2',
    'create_edu_wrapper_v2',
]
