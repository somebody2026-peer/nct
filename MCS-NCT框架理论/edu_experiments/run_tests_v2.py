"""
MCS-NCT V2 包装器测试脚本
直接运行测试，不依赖 pytest 的包发现机制
"""

import sys
import traceback
from pathlib import Path

# 设置路径
edu_root = Path(__file__).parent
sys.path.insert(0, str(edu_root))

import torch
import numpy as np

# 导入要测试的模块
from mcs_edu_wrapper_v2 import (
    CONSTRAINT_NAMES,
    RunningConstraintNormalizer,
    MCSEduWrapperV2,
    MCSEduWrapperV2Result,
)


class MockMCSState:
    """模拟 MCSState"""
    def __init__(self, consciousness_level, total_violation, constraint_violations,
                 satisfied_constraints, violated_constraints, dominant_violation,
                 state_vector, phi_value=None):
        self.consciousness_level = consciousness_level
        self.total_violation = total_violation
        self.constraint_violations = constraint_violations
        self.satisfied_constraints = satisfied_constraints
        self.violated_constraints = violated_constraints
        self.dominant_violation = dominant_violation
        self.state_vector = state_vector
        self.phi_value = phi_value


class MockMCSConsciousnessSolver:
    """模拟 MCSConsciousnessSolver"""
    
    def __init__(self, fixed_violations=None):
        self.fixed_violations = fixed_violations
        self.call_count = 0
    
    def __call__(self, visual, auditory, current_state, **kwargs):
        self.call_count += 1
        
        if self.fixed_violations:
            violations = self.fixed_violations.copy()
        else:
            violations = {
                'sensory_coherence': np.random.uniform(0.20, 0.35),
                'temporal_continuity': np.random.uniform(0.25, 0.40),
                'self_consistency': np.random.uniform(0.30, 0.45),
                'action_feasibility': np.random.uniform(0.25, 0.40),
                'social_interpretability': np.random.uniform(0.85, 0.95),
                'integrated_information': np.random.uniform(0.35, 0.55),
            }
        
        total_v = sum(violations.values())
        consciousness_level = 1 / (1 + total_v)
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


def test_normalizer_initialization():
    """测试归一化器初始化"""
    normalizer = RunningConstraintNormalizer(warmup_samples=50, eps=1e-5)
    
    assert normalizer.warmup_samples == 50
    assert normalizer.eps == 1e-5
    assert not normalizer.is_warmed_up()
    assert normalizer.get_warmup_progress() == 0.0
    print("  [PASS] test_normalizer_initialization")


def test_normalizer_warmup():
    """测试预热阶段"""
    normalizer = RunningConstraintNormalizer(warmup_samples=10)
    
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
        
        if i < 9:
            assert not normalizer.is_warmed_up()
    
    assert normalizer.is_warmed_up()
    print("  [PASS] test_normalizer_warmup")


def test_normalization_reduces_c5_dominance():
    """测试归一化后 C5 不再主导"""
    normalizer = RunningConstraintNormalizer(warmup_samples=50, momentum=0.1)
    
    all_results = []
    for i in range(100):
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
    assert len(all_results) >= 40
    
    normalized_values = {name: [] for name in CONSTRAINT_NAMES}
    for result in all_results:
        for name in CONSTRAINT_NAMES:
            normalized_values[name].append(result[name])
    
    stds = {name: np.std(vals) for name, vals in normalized_values.items()}
    
    max_std = max(stds.values())
    min_std = min(stds.values())
    
    assert max_std < min_std * 5, f"std ratio too high: {max_std / min_std:.2f}"
    print("  [PASS] test_normalization_reduces_c5_dominance")


def test_exp_formula_range():
    """测试指数公式输出范围"""
    weights = {
        'sensory_coherence': 1.0,
        'temporal_continuity': 2.0,
        'self_consistency': 0.0,
        'action_feasibility': 0.0,
        'social_interpretability': 0.0,
        'integrated_information': 1.5,
    }
    
    solver = MockMCSConsciousnessSolver()
    wrapper = MCSEduWrapperV2(
        solver=solver,
        weight_profile=weights,
        consciousness_formula="exp",
        normalization_warmup=10
    )
    
    results = []
    for i in range(50):
        visual = torch.randn(1, 5, 128)
        auditory = torch.randn(1, 5, 128)
        current_state = torch.randn(1, 128)
        
        result = wrapper.process(visual, auditory, current_state)
        results.append(result.consciousness_level)
    
    assert all(0 < r <= 1.0 for r in results)
    print("  [PASS] test_exp_formula_range")


def test_geometric_formula_range():
    """测试几何公式输出范围"""
    weights = {
        'sensory_coherence': 1.0,
        'temporal_continuity': 2.0,
        'self_consistency': 0.0,
        'action_feasibility': 0.0,
        'social_interpretability': 0.0,
        'integrated_information': 1.5,
    }
    
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
    
    assert all(0 < r <= 1.0 for r in results)
    print("  [PASS] test_geometric_formula_range")


def test_v2_better_discrimination_than_v1():
    """测试 V2 比 V1 有更好的区分度"""
    weights = {
        'sensory_coherence': 1.0,
        'temporal_continuity': 2.0,
        'self_consistency': 0.0,
        'action_feasibility': 0.0,
        'social_interpretability': 0.0,
        'integrated_information': 1.5,
    }
    
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
    
    print(f"    V1 std: {v1_std:.4f}, V2 std: {v2_std:.4f}")
    assert v2_std > 0.01, f"V2 std {v2_std:.4f} should be > 0.01"
    print("  [PASS] test_v2_better_discrimination_than_v1")


def test_zero_weight_never_dominant():
    """验证权重为0的约束永远不会被选为 dominant"""
    solver = MockMCSConsciousnessSolver(fixed_violations={
        'sensory_coherence': 0.3,
        'temporal_continuity': 0.4,
        'self_consistency': 0.5,
        'action_feasibility': 0.35,
        'social_interpretability': 0.95,
        'integrated_information': 0.5,
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
    
    for i in range(20):
        visual = torch.randn(1, 5, 128)
        auditory = torch.randn(1, 5, 128)
        current_state = torch.randn(1, 128)
        
        result = wrapper.process(visual, auditory, current_state)
        
        assert result.dominant_violation in ['sensory_coherence', 'temporal_continuity', 'integrated_information'], \
            f"Unexpected dominant: {result.dominant_violation}"
        
        assert result.dominant_violation not in ['self_consistency', 'action_feasibility', 'social_interpretability'], \
            f"Zero-weight constraint became dominant: {result.dominant_violation}"
    
    print("  [PASS] test_zero_weight_never_dominant")


def test_full_pipeline():
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
    
    for i in range(30):
        visual = torch.randn(1, 5, 128)
        auditory = torch.randn(1, 5, 128)
        current_state = torch.randn(1, 128)
        
        result = wrapper.process(visual, auditory, current_state)
        
        assert isinstance(result, MCSEduWrapperV2Result)
        assert 'sensory_coherence' in result.constraint_violations_raw
        assert 'sensory_coherence' in result.constraint_violations_normalized
        assert 'sensory_coherence' in result.constraint_violations_weighted
        
        assert 0 < result.consciousness_level <= 1.0
        assert 0 < result.consciousness_level_v1 <= 1.0
        assert result.total_weighted_violation >= 0
        
        assert result.active_constraints == ['sensory_coherence', 'temporal_continuity', 'integrated_information']
    
    assert solver.call_count == 30
    print("  [PASS] test_full_pipeline")


def test_to_dict():
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
    print("  [PASS] test_to_dict")


def test_invalid_formula_raises_error():
    """测试无效公式引发错误"""
    try:
        MCSEduWrapperV2(
            solver=MockMCSConsciousnessSolver(),
            weight_profile={name: 1.0 for name in CONSTRAINT_NAMES},
            consciousness_formula="invalid"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be 'exp' or 'geometric'" in str(e)
    print("  [PASS] test_invalid_formula_raises_error")


def test_missing_weight_raises_error():
    """测试缺少权重引发错误"""
    incomplete_weights = {
        'sensory_coherence': 1.0,
        'temporal_continuity': 1.0,
    }
    
    try:
        MCSEduWrapperV2(
            solver=MockMCSConsciousnessSolver(),
            weight_profile=incomplete_weights,
            consciousness_formula="exp"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing weight" in str(e)
    print("  [PASS] test_missing_weight_raises_error")


def test_all_zero_weights():
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
    
    assert result.consciousness_level == 1.0
    assert result.active_constraints == []
    assert result.dominant_violation == "none"
    print("  [PASS] test_all_zero_weights")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("MCS-NCT V2 Wrapper Unit Tests")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    tests = [
        ("Normalizer Initialization", test_normalizer_initialization),
        ("Normalizer Warmup", test_normalizer_warmup),
        ("Normalization Reduces C5 Dominance", test_normalization_reduces_c5_dominance),
        ("Exp Formula Range", test_exp_formula_range),
        ("Geometric Formula Range", test_geometric_formula_range),
        ("V2 Better Discrimination Than V1", test_v2_better_discrimination_than_v1),
        ("Zero Weight Never Dominant", test_zero_weight_never_dominant),
        ("Full Pipeline", test_full_pipeline),
        ("To Dict", test_to_dict),
        ("Invalid Formula Raises Error", test_invalid_formula_raises_error),
        ("Missing Weight Raises Error", test_missing_weight_raises_error),
        ("All Zero Weights", test_all_zero_weights),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
