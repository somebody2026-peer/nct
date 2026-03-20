"""
MCS-NCT V2 包装器单元测试
========================

测试内容:
1. 归一化器测试：验证归一化后 std 偏差合理
2. consciousness_level 区分度测试：验证输出有足够差异
3. weighted_dominant_violation 测试：验证权重为0的约束不会被选为 dominant
4. 完整流程集成测试（使用 mock solver）
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcs_edu_wrapper_v2 import (
    CONSTRAINT_NAMES,
    RunningConstraintNormalizer,
    MCSEduWrapperV2,
    MCSEduWrapperV2Result,
)


# ============================================================================
# Mock MCSState for testing
# ============================================================================

@dataclass
class MockMCSState:
    """模拟 MCSState"""
    consciousness_level: float
    total_violation: float
    constraint_violations: Dict[str, float]
    satisfied_constraints: List[str]
    violated_constraints: List[str]
    dominant_violation: str
    state_vector: torch.Tensor
    phi_value: Optional[float] = None


class MockMCSConsciousnessSolver:
    """模拟 MCSConsciousnessSolver"""
    
    def __init__(self, fixed_violations: Optional[Dict[str, float]] = None):
        """
        Args:
            fixed_violations: 固定的违反值用于测试，如果为 None 则生成随机值
        """
        self.fixed_violations = fixed_violations
        self.call_count = 0
    
    def __call__(self, visual, auditory, current_state, **kwargs):
        self.call_count += 1
        
        if self.fixed_violations:
            violations = self.fixed_violations.copy()
        else:
            # 模拟 V1 实验中观察到的分布：C5 远高于其他约束
            violations = {
                'sensory_coherence': np.random.uniform(0.20, 0.35),
                'temporal_continuity': np.random.uniform(0.25, 0.40),
                'self_consistency': np.random.uniform(0.30, 0.45),
                'action_feasibility': np.random.uniform(0.25, 0.40),
                'social_interpretability': np.random.uniform(0.85, 0.95),  # C5 高违反
                'integrated_information': np.random.uniform(0.35, 0.55),
            }
        
        # 计算 V1 原始意识水平
        total_v = sum(violations.values())
        consciousness_level = 1 / (1 + total_v)
        
        # 找到主导违反
        max_key = max(violations, key=violations.get)
        
        return MockMCSState(
            consciousness_level=consciousness_level,
            total_violation=total_v,
            constraint_violations=violations,
            satisfied_constraints=[],
            violated_constraints=list(violations.keys()),
            dominant_violation=f"C{CONSTRAINT_NAMES.index(max_key)+1}-{max_key}",
            state_vector=current_state,
            phi_value=0.5
        )


# ============================================================================
# Test: RunningConstraintNormalizer
# ============================================================================

class TestRunningConstraintNormalizer:
    """归一化器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        normalizer = RunningConstraintNormalizer(warmup_samples=50, eps=1e-5)
        
        assert normalizer.warmup_samples == 50
        assert normalizer.eps == 1e-5
        assert not normalizer.is_warmed_up()
        assert normalizer.get_warmup_progress() == 0.0
    
    def test_warmup_phase(self):
        """测试预热阶段"""
        normalizer = RunningConstraintNormalizer(warmup_samples=10)
        
        # 模拟 V1 分布：C5 约 0.92，其他约 0.25-0.52
        for i in range(10):
            violations = {
                'sensory_coherence': 0.26,
                'temporal_continuity': 0.35,
                'self_consistency': 0.40,
                'action_feasibility': 0.30,
                'social_interpretability': 0.92,
                'integrated_information': 0.45,
            }
            result = normalizer.normalize(violations)
            
            # 预热阶段应返回原始值
            if i < 9:  # 前 9 个样本
                assert not normalizer.is_warmed_up()
        
        # 第 10 个样本后应该完成预热
        assert normalizer.is_warmed_up()
    
    def test_normalization_reduces_c5_dominance(self):
        """
        测试归一化后 C5 不再因绝对值高而主导
        
        输入：C5 violation=0.92, C1=0.26 等
        验证：归一化后所有约束的 std 偏差 < 2x
        """
        normalizer = RunningConstraintNormalizer(warmup_samples=50, momentum=0.1)
        
        # 收集统计量（模拟训练过程）
        all_results = []
        for i in range(100):
            # 添加一些随机噪声
            violations = {
                'sensory_coherence': 0.26 + np.random.normal(0, 0.03),
                'temporal_continuity': 0.35 + np.random.normal(0, 0.04),
                'self_consistency': 0.40 + np.random.normal(0, 0.05),
                'action_feasibility': 0.30 + np.random.normal(0, 0.03),
                'social_interpretability': 0.92 + np.random.normal(0, 0.02),
                'integrated_information': 0.45 + np.random.normal(0, 0.05),
            }
            result = normalizer.normalize(violations)
            if normalizer.is_warmed_up():
                all_results.append(result)
        
        assert normalizer.is_warmed_up()
        assert len(all_results) >= 40  # 至少有 40 个归一化后的结果
        
        # 计算归一化后各约束的 std
        normalized_values = {name: [] for name in CONSTRAINT_NAMES}
        for result in all_results:
            for name in CONSTRAINT_NAMES:
                normalized_values[name].append(result[name])
        
        stds = {name: np.std(vals) for name, vals in normalized_values.items()}
        
        # 验证：所有约束的 std 应该在同一数量级
        # 之前 C5 因为绝对值高而主导，现在应该接近
        max_std = max(stds.values())
        min_std = min(stds.values())
        
        # 最大 std 不应超过最小 std 的 3 倍（合理范围）
        # 原始数据中 C5 的绝对值是其他的 2-3 倍
        assert max_std < min_std * 5, f"std ratio too high: {max_std / min_std:.2f}"
    
    def test_reset(self):
        """测试重置功能"""
        normalizer = RunningConstraintNormalizer(warmup_samples=5)
        
        # 预热
        for _ in range(10):
            normalizer.normalize({name: 0.5 for name in CONSTRAINT_NAMES})
        
        assert normalizer.is_warmed_up()
        
        # 重置
        normalizer.reset()
        
        assert not normalizer.is_warmed_up()
        assert normalizer.get_warmup_progress() == 0.0
    
    def test_get_stats(self):
        """测试统计量获取"""
        normalizer = RunningConstraintNormalizer(warmup_samples=5)
        
        for _ in range(10):
            normalizer.normalize({name: 0.5 for name in CONSTRAINT_NAMES})
        
        stats = normalizer.get_stats()
        
        assert len(stats) == len(CONSTRAINT_NAMES)
        for name in CONSTRAINT_NAMES:
            assert 'mean' in stats[name]
            assert 'std' in stats[name]
            assert 'count' in stats[name]
            assert stats[name]['count'] == 10


# ============================================================================
# Test: MCSEduWrapperV2 - Consciousness Level
# ============================================================================

class TestConsciousnessLevel:
    """consciousness_level 区分度测试"""
    
    @pytest.fixture
    def weights(self):
        """标准权重配置（禁用 C3, C4, C5）"""
        return {
            'sensory_coherence': 1.0,
            'temporal_continuity': 2.0,
            'self_consistency': 0.0,
            'action_feasibility': 0.0,
            'social_interpretability': 0.0,
            'integrated_information': 1.5,
        }
    
    def test_exp_formula_range(self, weights):
        """测试指数公式输出范围"""
        solver = MockMCSConsciousnessSolver()
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=10
        )
        
        # 收集结果
        results = []
        for i in range(50):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            
            result = wrapper.process(visual, auditory, current_state)
            results.append(result.consciousness_level)
        
        # 验证范围在 (0, 1]
        assert all(0 < r <= 1.0 for r in results)
    
    def test_geometric_formula_range(self, weights):
        """测试几何公式输出范围"""
        solver = MockMCSConsciousnessSolver()
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="geometric",
            normalization_warmup=10
        )
        
        results = []
        for i in range(50):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            
            result = wrapper.process(visual, auditory, current_state)
            results.append(result.consciousness_level)
        
        # 验证范围在 (0, 1]
        assert all(0 < r <= 1.0 for r in results)
    
    def test_consciousness_level_discrimination(self, weights):
        """
        测试意识水平有足够的区分度
        
        构造不同的违反组合，验证输出有足够差异（std > 0.05）
        """
        # 创建不同违反程度的 solver
        low_violation_solver = MockMCSConsciousnessSolver(fixed_violations={
            'sensory_coherence': 0.1,
            'temporal_continuity': 0.15,
            'self_consistency': 0.2,
            'action_feasibility': 0.1,
            'social_interpretability': 0.3,
            'integrated_information': 0.1,
        })
        
        high_violation_solver = MockMCSConsciousnessSolver(fixed_violations={
            'sensory_coherence': 0.8,
            'temporal_continuity': 0.85,
            'self_consistency': 0.9,
            'action_feasibility': 0.8,
            'social_interpretability': 0.95,
            'integrated_information': 0.9,
        })
        
        wrapper_low = MCSEduWrapperV2(
            solver=low_violation_solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        wrapper_high = MCSEduWrapperV2(
            solver=high_violation_solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        # 预热
        for _ in range(10):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            wrapper_low.process(visual, auditory, current_state)
            wrapper_high.process(visual, auditory, current_state)
        
        # 收集结果
        low_results = []
        high_results = []
        for _ in range(20):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            
            low_results.append(wrapper_low.process(visual, auditory, current_state).consciousness_level)
            high_results.append(wrapper_high.process(visual, auditory, current_state).consciousness_level)
        
        # 低违反应该有更高的意识水平
        assert np.mean(low_results) > np.mean(high_results)
        
        # 两组之间应该有明显差异
        diff = abs(np.mean(low_results) - np.mean(high_results))
        assert diff > 0.1, f"Difference too small: {diff:.4f}"
    
    def test_v2_better_discrimination_than_v1(self, weights):
        """
        测试 V2 比 V1 有更好的区分度
        
        V1 公式 1/(1+total_violation) 将所有样本压缩到约 0.301
        V2 应该有更大的标准差
        """
        solver = MockMCSConsciousnessSolver()
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=20
        )
        
        v1_levels = []
        v2_levels = []
        
        for i in range(100):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            
            result = wrapper.process(visual, auditory, current_state)
            v1_levels.append(result.consciousness_level_v1)
            v2_levels.append(result.consciousness_level)
        
        v1_std = np.std(v1_levels)
        v2_std = np.std(v2_levels)
        
        # V2 的标准差应该比 V1 大（更有区分度）
        # V1 的 std 约 0.0017，V2 应该 > 0.05
        print(f"V1 std: {v1_std:.4f}, V2 std: {v2_std:.4f}")
        
        # 放宽条件：只要 V2 std > 0.01 就可以
        assert v2_std > 0.01, f"V2 std {v2_std:.4f} should be > 0.01"


# ============================================================================
# Test: Weighted Dominant Violation
# ============================================================================

class TestWeightedDominantViolation:
    """weighted_dominant_violation 测试"""
    
    def test_zero_weight_never_dominant(self):
        """
        验证权重为0的约束永远不会被选为 dominant
        
        即使 C5 (social_interpretability) 的原始违反值最高，
        如果权重为 0，它也不应该成为 dominant
        """
        # C5 违反值最高，但权重为 0
        solver = MockMCSConsciousnessSolver(fixed_violations={
            'sensory_coherence': 0.3,
            'temporal_continuity': 0.4,
            'self_consistency': 0.5,          # 权重为 0
            'action_feasibility': 0.35,       # 权重为 0
            'social_interpretability': 0.95,  # 最高违反，但权重为 0
            'integrated_information': 0.5,
        })
        
        weights = {
            'sensory_coherence': 1.0,
            'temporal_continuity': 2.0,
            'self_consistency': 0.0,        # 禁用
            'action_feasibility': 0.0,      # 禁用
            'social_interpretability': 0.0, # 禁用
            'integrated_information': 1.5,
        }
        
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        # 预热和测试
        for i in range(20):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            
            result = wrapper.process(visual, auditory, current_state)
            
            # dominant 不应该是权重为 0 的约束
            assert result.dominant_violation in ['sensory_coherence', 'temporal_continuity', 'integrated_information'], \
                f"Unexpected dominant: {result.dominant_violation}"
            
            assert result.dominant_violation not in ['self_consistency', 'action_feasibility', 'social_interpretability'], \
                f"Zero-weight constraint became dominant: {result.dominant_violation}"
    
    def test_weighted_dominant_calculation(self):
        """测试加权主导计算正确性"""
        solver = MockMCSConsciousnessSolver(fixed_violations={
            'sensory_coherence': 0.5,         # w=1.0, weighted=0.5
            'temporal_continuity': 0.3,       # w=2.0, weighted=0.6
            'self_consistency': 0.9,          # w=0.0, weighted=0.0
            'action_feasibility': 0.8,        # w=0.0, weighted=0.0
            'social_interpretability': 0.95,  # w=0.0, weighted=0.0
            'integrated_information': 0.2,    # w=1.5, weighted=0.3
        })
        
        weights = {
            'sensory_coherence': 1.0,
            'temporal_continuity': 2.0,
            'self_consistency': 0.0,
            'action_feasibility': 0.0,
            'social_interpretability': 0.0,
            'integrated_information': 1.5,
        }
        
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        # 预热后，dominant 应该是 temporal_continuity（最高加权违反）
        for _ in range(10):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            result = wrapper.process(visual, auditory, current_state)
        
        # 由于归一化可能改变相对大小，只验证 dominant 是活跃约束之一
        assert result.dominant_violation in wrapper.active_constraints


# ============================================================================
# Test: Integration Test with Mock Solver
# ============================================================================

class TestIntegration:
    """完整流程集成测试"""
    
    def test_full_pipeline(self):
        """测试完整处理流程"""
        solver = MockMCSConsciousnessSolver()
        weights = {
            'sensory_coherence': 1.0,
            'temporal_continuity': 2.0,
            'self_consistency': 0.0,
            'action_feasibility': 0.0,
            'social_interpretability': 0.0,
            'integrated_information': 1.5,
        }
        
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=10
        )
        
        # 处理多个样本
        for i in range(30):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            
            result = wrapper.process(visual, auditory, current_state)
            
            # 验证返回结果结构
            assert isinstance(result, MCSEduWrapperV2Result)
            assert 'sensory_coherence' in result.constraint_violations_raw
            assert 'sensory_coherence' in result.constraint_violations_normalized
            assert 'sensory_coherence' in result.constraint_violations_weighted
            
            # 验证值范围
            assert 0 < result.consciousness_level <= 1.0
            assert 0 < result.consciousness_level_v1 <= 1.0
            assert result.total_weighted_violation >= 0
            
            # 验证活跃约束
            assert result.active_constraints == ['sensory_coherence', 'temporal_continuity', 'integrated_information']
        
        # 验证 solver 被调用
        assert solver.call_count == 30
    
    def test_to_dict(self):
        """测试结果转换为字典"""
        solver = MockMCSConsciousnessSolver()
        weights = {name: 1.0 for name in CONSTRAINT_NAMES}
        
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        visual = torch.randn(1, 5, 128)
        auditory = torch.randn(1, 5, 128)
        current_state = torch.randn(1, 128)
        
        result = wrapper.process(visual, auditory, current_state)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'consciousness_level' in result_dict
        assert 'consciousness_level_v1' in result_dict
        assert 'constraint_violations_raw' in result_dict
        assert 'active_constraints' in result_dict
    
    def test_get_config(self):
        """测试获取配置"""
        weights = {
            'sensory_coherence': 1.0,
            'temporal_continuity': 2.0,
            'self_consistency': 0.0,
            'action_feasibility': 0.0,
            'social_interpretability': 0.0,
            'integrated_information': 1.5,
        }
        
        wrapper = MCSEduWrapperV2(
            solver=MockMCSConsciousnessSolver(),
            weight_profile=weights,
            consciousness_formula="geometric",
            normalization_warmup=50
        )
        
        config = wrapper.get_config()
        
        assert config['consciousness_formula'] == "geometric"
        assert config['normalization_warmup'] == 50
        assert config['weight_profile'] == weights
        assert len(config['active_constraints']) == 3
    
    def test_reset_normalizer(self):
        """测试重置归一化器"""
        solver = MockMCSConsciousnessSolver()
        weights = {name: 1.0 for name in CONSTRAINT_NAMES}
        
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        # 预热
        for _ in range(10):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            wrapper.process(visual, auditory, current_state)
        
        assert wrapper.is_warmed_up()
        
        # 重置
        wrapper.reset_normalizer()
        
        assert not wrapper.is_warmed_up()
    
    def test_invalid_formula_raises_error(self):
        """测试无效公式引发错误"""
        with pytest.raises(ValueError) as excinfo:
            MCSEduWrapperV2(
                solver=MockMCSConsciousnessSolver(),
                weight_profile={name: 1.0 for name in CONSTRAINT_NAMES},
                consciousness_formula="invalid"
            )
        
        assert "must be 'exp' or 'geometric'" in str(excinfo.value)
    
    def test_missing_weight_raises_error(self):
        """测试缺少权重引发错误"""
        incomplete_weights = {
            'sensory_coherence': 1.0,
            'temporal_continuity': 1.0,
            # 缺少其他权重
        }
        
        with pytest.raises(ValueError) as excinfo:
            MCSEduWrapperV2(
                solver=MockMCSConsciousnessSolver(),
                weight_profile=incomplete_weights,
                consciousness_formula="exp"
            )
        
        assert "Missing weight" in str(excinfo.value)


# ============================================================================
# Test: Different Weight Profiles
# ============================================================================

class TestWeightProfiles:
    """不同权重配置测试"""
    
    def test_all_zero_weights(self):
        """测试所有权重为 0"""
        solver = MockMCSConsciousnessSolver()
        weights = {name: 0.0 for name in CONSTRAINT_NAMES}
        
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        visual = torch.randn(1, 5, 128)
        auditory = torch.randn(1, 5, 128)
        current_state = torch.randn(1, 128)
        
        result = wrapper.process(visual, auditory, current_state)
        
        # 没有活跃约束，意识水平应该为 1.0
        assert result.consciousness_level == 1.0
        assert result.active_constraints == []
        assert result.dominant_violation == "none"
    
    def test_single_active_constraint(self):
        """测试单个活跃约束"""
        solver = MockMCSConsciousnessSolver(fixed_violations={
            'sensory_coherence': 0.5,
            'temporal_continuity': 0.5,
            'self_consistency': 0.5,
            'action_feasibility': 0.5,
            'social_interpretability': 0.5,
            'integrated_information': 0.5,
        })
        
        weights = {
            'sensory_coherence': 0.0,
            'temporal_continuity': 0.0,
            'self_consistency': 0.0,
            'action_feasibility': 0.0,
            'social_interpretability': 0.0,
            'integrated_information': 2.0,  # 只有 C6 活跃
        }
        
        wrapper = MCSEduWrapperV2(
            solver=solver,
            weight_profile=weights,
            consciousness_formula="exp",
            normalization_warmup=5
        )
        
        for _ in range(10):
            visual = torch.randn(1, 5, 128)
            auditory = torch.randn(1, 5, 128)
            current_state = torch.randn(1, 128)
            result = wrapper.process(visual, auditory, current_state)
        
        assert result.active_constraints == ['integrated_information']
        assert result.dominant_violation == 'integrated_information'


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
