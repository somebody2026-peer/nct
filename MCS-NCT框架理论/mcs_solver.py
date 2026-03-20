"""
MCS 意识理论：多重约束满足框架
Consciousness as Multi-Constraint Satisfaction (MCS) Theory

完整 PyTorch 实现

作者：NCT Team
日期：2026 年 3 月
版本：v1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MCSState:
    """MCS 意识状态数据类"""
    consciousness_level: float
    total_violation: float
    constraint_violations: Dict[str, float]
    satisfied_constraints: List[str]
    violated_constraints: List[str]
    dominant_violation: str
    state_vector: torch.Tensor
    phi_value: Optional[float] = None
    
    def __str__(self):
        return (f"MCS State - Level: {self.consciousness_level:.3f}, "
                f"Violations: {self.constraint_violations}")


# ============================================================================
# C1: 感觉一致性约束
# ============================================================================

class SensoryCoherenceConstraint(nn.Module):
    """
    C1: 感觉一致性约束 (Sensory Coherence Constraint)
    
    多模态感觉输入必须在时空上对齐
    """
    
    def __init__(
        self, 
        d_model: int = 768, 
        temporal_window_ms: float = 50.0,
        spatial_threshold_deg: float = 5.0,
        feature_similarity_threshold: float = 0.7
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_window = temporal_window_ms
        self.spatial_threshold = spatial_threshold_deg
        self.feature_threshold = feature_similarity_threshold
        
        # 跨模态对齐网络（可学习）
        self.cross_modal_aligner = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        # 一致性评估器（可学习）
        self.coherence_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        visual: torch.Tensor, 
        auditory: torch.Tensor,
        tactile: Optional[torch.Tensor] = None,
        timestamps: Optional[Dict[str, float]] = None,
        locations: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        计算感觉一致性约束违反程度
        
        Args:
            visual: 视觉表征 [B, T, D]
            auditory: 听觉表征 [B, T, D]
            tactile: 触觉表征（可选）[B, T, D]
            timestamps: 时间戳字典 {'visual': t1, 'auditory': t2, ...}
            locations: 空间位置字典
        
        Returns:
            violation: 违反程度 [B, 1]
        """
        B = visual.size(0)
        
        # Step 1: 跨模态注意力对齐
        aligned_auditory, attn_weights = self.cross_modal_aligner(
            query=visual,
            key=auditory,
            value=auditory
        )
        
        # Step 2: 计算特征一致性
        visual_mean = visual.mean(dim=1)  # [B, D]
        aligned_mean = aligned_auditory.mean(dim=1)  # [B, D]
        
        combined = torch.cat([visual_mean, aligned_mean], dim=-1)  # [B, 2D]
        feature_coherence = self.coherence_scorer(combined)  # [B, 1]
        
        # Step 3: 时间对齐检查（如果有时间戳）
        temporal_violation = torch.zeros(B, 1, device=visual.device)
        if timestamps:
            t_vis = timestamps.get('visual', 0)
            t_aud = timestamps.get('auditory', 0)
            temporal_diff = abs(t_vis - t_aud)
            
            if temporal_diff > self.temporal_window:
                normalized_diff = (temporal_diff - self.temporal_window) / self.temporal_window
                temporal_violation = torch.ones(B, 1, device=visual.device) * min(normalized_diff, 1.0)
        
        # Step 4: 空间对齐检查（如果有位置信息）
        spatial_violation = torch.zeros(B, 1, device=visual.device)
        if locations:
            loc_vis = locations.get('visual', torch.zeros(3))
            loc_aud = locations.get('auditory', torch.zeros(3))
            spatial_diff = torch.norm(loc_vis - loc_aud)
            
            if spatial_diff > self.spatial_threshold:
                normalized_diff = (spatial_diff - self.spatial_threshold) / self.spatial_threshold
                spatial_violation = torch.ones(B, 1, device=visual.device) * min(normalized_diff, 1.0)
        
        # Step 5: 综合违反程度
        # violation = 1 - coherence + temporal_penalty + spatial_penalty
        feature_violation = 1 - feature_coherence
        
        # 加权求和（特征 60% + 时间 25% + 空间 15%）
        total_violation = (
            0.60 * feature_violation + 
            0.25 * temporal_violation + 
            0.15 * spatial_violation
        )
        
        return total_violation.squeeze(-1)  # [B]


# ============================================================================
# C2: 时间连续性约束
# ============================================================================

class TemporalContinuityConstraint(nn.Module):
    """
    C2: 时间连续性约束 (Temporal Continuity Constraint)
    
    当前意识状态必须从前一状态可达
    """
    
    def __init__(
        self, 
        d_model: int = 768, 
        history_length: int = 10,
        sigma: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.history_length = history_length
        self.sigma = sigma
        
        # GRU 状态转移模型
        self.transition_predictor = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 连续性评分器
        self.continuity_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 状态历史缓冲
        self.register_buffer('state_history', torch.zeros(1, history_length, d_model))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def reset_history(self, batch_size: int = 1):
        """重置状态历史"""
        self.state_history = torch.zeros(batch_size, self.history_length, self.d_model, 
                                         device=self.state_history.device)
        self.history_ptr = torch.zeros(batch_size, dtype=torch.long, device=self.history_ptr.device)
    
    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        计算时间连续性约束违反程度
        
        Args:
            current_state: 当前状态 [B, D]
        
        Returns:
            violation: 违反程度 [B]
        """
        B, D = current_state.shape
        
        # 动态调整历史缓冲区大小
        if self.state_history.shape[0] != B:
            self.state_history = torch.zeros(B, self.history_length, self.d_model, 
                                             device=current_state.device)
            self.history_ptr = torch.tensor([0], device=current_state.device)
        
        # 更新历史
        ptr = self.history_ptr[0].item() if self.history_ptr.dim() > 0 else self.history_ptr.item()
        self.state_history[:, ptr, :] = current_state.detach()  # [B, D] 直接赋值
        self.history_ptr = torch.tensor([(ptr + 1) % self.history_length], device=self.history_ptr.device)
        
        # 构建历史序列（按时间顺序）
        indices = [(ptr - i - 1) % self.history_length for i in range(self.history_length)]
        history_seq = torch.stack([self.state_history[:, idx, :] for idx in indices], dim=1)  # [B, T, D]
        
        # 从历史预测当前状态
        predicted, _ = self.transition_predictor(history_seq)
        predicted_current = predicted[:, -1, :]  # [B, D]
        
        # 计算预测与实际的一致性
        combined = torch.cat([predicted_current, current_state], dim=-1)
        continuity = self.continuity_scorer(combined)  # [B, 1]
        
        # 违反程度 = 1 - 连续性
        violation = 1 - continuity
        return violation.squeeze(-1)  # [B]


# ============================================================================
# C3: 自我一致性约束
# ============================================================================

class SelfConsistencyConstraint(nn.Module):
    """
    C3: 自我一致性约束 (Self-Consistency Constraint)
    
    信念系统必须内部逻辑自洽
    """
    
    def __init__(
        self, 
        d_model: int = 768, 
        n_beliefs: int = 64,
        contradiction_threshold: float = 0.8
    ):
        super().__init__()
        self.d_model = d_model
        self.n_beliefs = n_beliefs
        self.threshold = contradiction_threshold
        
        # 信念嵌入（可学习）
        self.belief_embeddings = nn.Parameter(torch.randn(n_beliefs, d_model))
        
        # 一致性检测 Transformer
        self.consistency_checker = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=2
        )
        
        # 矛盾检测器
        self.contradiction_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.belief_embeddings)
        for p in self.consistency_checker.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, current_experience: torch.Tensor) -> torch.Tensor:
        """
        计算自我一致性约束违反程度
        
        Args:
            current_experience: 当前体验 [B, D]
        
        Returns:
            violation: 违反程度 [B]
        """
        B = current_experience.shape[0]
        
        # 将当前体验与信念系统对比
        beliefs = self.belief_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        experience = current_experience.unsqueeze(1)  # [B, 1, D]
        
        # 合并并检测矛盾
        combined = torch.cat([beliefs, experience], dim=1)  # [B, N+1, D]
        checked = self.consistency_checker(combined)
        
        # 提取矛盾分数（取最后一个 token）
        contradiction_scores = self.contradiction_detector(checked[:, -1:, :])  # [B, 1]
        
        # 取最大矛盾作为违反程度
        violation = contradiction_scores.squeeze(-1)
        return violation


# ============================================================================
# C4: 行动可行性约束
# ============================================================================

class ActionFeasibilityConstraint(nn.Module):
    """
    C4: 行动可行性约束 (Action Feasibility Constraint)
    
    意图必须映射到可执行的运动计划
    """
    
    def __init__(
        self, 
        d_model: int = 768, 
        action_dim: int = 32,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        
        # 意图→动作映射
        self.intention_to_action = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 可行性评估器
        self.feasibility_evaluator = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(action_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.intention_to_action, self.feasibility_evaluator]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        intention: torch.Tensor, 
        motor_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算行动可行性约束违反程度
        
        Args:
            intention: 意图向量 [B, D]
            motor_state: 当前运动状态（可选）
        
        Returns:
            violation: 违反程度 [B]
        """
        # 映射到动作空间
        action_plan = self.intention_to_action(intention)
        
        # 评估可行性
        feasibility = self.feasibility_evaluator(action_plan)
        
        # 违反程度 = 1 - 可行性
        violation = 1 - feasibility
        return violation.squeeze(-1)


# ============================================================================
# C5: 社会可解释性约束
# ============================================================================

class SocialInterpretabilityConstraint(nn.Module):
    """
    C5: 社会可解释性约束 (Social Interpretability Constraint)
    
    体验必须可向他人传达
    """
    
    def __init__(
        self, 
        d_model: int = 768, 
        vocab_size: int = 10000,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 体验→语言编码器
        self.experience_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=2
        )
        
        # 语言→体验解码器（重建）
        self.experience_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=2
        )
        
        self.reconstruction_loss = nn.MSELoss(reduction='none')
        self._init_weights()
    
    def _init_weights(self):
        for p in self.experience_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.experience_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, experience: torch.Tensor) -> torch.Tensor:
        """
        计算社会可解释性约束违反程度
        
        Args:
            experience: 体验张量 [B, D]
        
        Returns:
            violation: 违反程度 [B]
        """
        B = experience.shape[0]
        
        # 编码体验
        experience_expanded = experience.unsqueeze(1)  # [B, 1, D]
        encoded = self.experience_encoder(experience_expanded)
        
        # 解码重建
        reconstructed = self.experience_decoder(encoded, encoded)
        reconstructed = reconstructed.squeeze(1)  # [B, D]
        
        # 重建误差
        reconstruction_error = self.reconstruction_loss(reconstructed, experience)
        normalized_error = reconstruction_error.mean(dim=-1) / (experience.pow(2).mean(dim=-1) + 1e-6)
        
        # 归一化到 [0, 1]
        violation = torch.sigmoid(normalized_error)
        return violation


# ============================================================================
# C6: 整合信息量约束
# ============================================================================

class IntegratedInformationConstraint(nn.Module):
    """
    C6: 整合信息量约束 (Integrated Information Constraint)
    
    系统的 Φ 值必须超过阈值
    """
    
    def __init__(
        self, 
        d_model: int = 768, 
        phi_threshold: float = 0.3,
        approximation_method: str = 'attention_flow'
    ):
        super().__init__()
        self.phi_threshold = phi_threshold
        self.approximation_method = approximation_method
        
        # Φ 值近似计算器
        self.phi_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.phi_estimator.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        neural_state: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算整合信息量约束违反程度
        
        Args:
            neural_state: 神经状态 [B, D]
            attention_weights: 注意力权重（可选）
        
        Returns:
            violation: 违反程度 [B]
            phi: Φ 值估计 [B]
        """
        # 估计 Φ 值
        phi = self.phi_estimator(neural_state).squeeze(-1)
        
        # 低于阈值则违反
        violation = torch.relu(self.phi_threshold - phi)
        
        # 归一化违反程度
        normalized_violation = violation / self.phi_threshold
        
        return normalized_violation, phi


# ============================================================================
# MCS 核心求解器
# ============================================================================

class MCSConsciousnessSolver(nn.Module):
    """
    MCS 意识求解器：多重约束满足优化
    
    核心方程：
        C(t) = argmin_S [ Σᵢ wᵢ·Vᵢ(S,t) ]
        Consciousness_Level = 1 / (1 + Total_Violation)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        constraint_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        
        # 六类约束模块
        self.c1_sensory = SensoryCoherenceConstraint(d_model, **kwargs)
        self.c2_temporal = TemporalContinuityConstraint(d_model, **kwargs)
        self.c3_self = SelfConsistencyConstraint(d_model, **kwargs)
        self.c4_action = ActionFeasibilityConstraint(d_model, **kwargs)
        self.c5_social = SocialInterpretabilityConstraint(d_model, **kwargs)
        self.c6_phi = IntegratedInformationConstraint(d_model, **kwargs)
        
        # 约束权重（默认权重基于理论重要性）
        default_weights = {
            'sensory_coherence': 1.0,      # w1
            'temporal_continuity': 1.0,     # w2
            'self_consistency': 1.0,        # w3
            'action_feasibility': 0.5,      # w4
            'social_interpretability': 0.5, # w5
            'integrated_information': 1.5   # w6 (最重要)
        }
        
        if constraint_weights:
            default_weights.update(constraint_weights)
        
        self.weights = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v), requires_grad=False) 
            for k, v in default_weights.items()
        })
        
        # 可学习的权重调制（可选）
        self.learnable_modulation = nn.Parameter(torch.ones(len(default_weights)))
    
    def set_weights(self, new_weights: Dict[str, float]):
        """动态调整约束权重"""
        for key, weight in new_weights.items():
            if key in self.weights:
                self.weights[key].data.fill_(weight)
    
    def forward(
        self,
        visual: torch.Tensor,
        auditory: torch.Tensor,
        current_state: torch.Tensor,
        intention: Optional[torch.Tensor] = None,
        timestamps: Optional[Dict[str, float]] = None,
        locations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> MCSState:
        """
        计算 MCS 意识状态
        
        Args:
            visual: 视觉输入 [B, T, D]
            auditory: 听觉输入 [B, T, D]
            current_state: 当前状态 [B, D]
            intention: 意图向量（可选）[B, D]
            timestamps: 时间戳
            locations: 空间位置
        
        Returns:
            mcs_state: MCS 意识状态
        """
        B = current_state.shape[0]
        
        # 如果没有提供 intention，使用 current_state 代替
        if intention is None:
            intention = current_state
        
        # ===== 计算各约束违反程度 =====
        
        # C1: 感觉一致性
        v1 = self.c1_sensory(
            visual=visual,
            auditory=auditory,
            timestamps=timestamps,
            locations=locations
        )
        
        # C2: 时间连续性
        v2 = self.c2_temporal(current_state=current_state)
        
        # C3: 自我一致性
        v3 = self.c3_self(current_experience=current_state)
        
        # C4: 行动可行性
        v4 = self.c4_action(intention=intention)
        
        # C5: 社会可解释性
        v5 = self.c5_social(experience=current_state)
        
        # C6: 整合信息量
        v6, phi = self.c6_phi(neural_state=current_state)
        
        # ===== 加权求和 =====
        
        violations_dict = {
            'sensory_coherence': v1.detach(),
            'temporal_continuity': v2.detach(),
            'self_consistency': v3.detach(),
            'action_feasibility': v4.detach(),
            'social_interpretability': v5.detach(),
            'integrated_information': v6.detach()
        }
        
        # 加权总违反
        total_violation = (
            self.weights['sensory_coherence'] * v1 +
            self.weights['temporal_continuity'] * v2 +
            self.weights['self_consistency'] * v3 +
            self.weights['action_feasibility'] * v4 +
            self.weights['social_interpretability'] * v5 +
            self.weights['integrated_information'] * v6
        )
        
        # ===== 计算意识水平 =====
        
        consciousness_level = 1 / (1 + total_violation + 1e-6)
        
        # ===== 分类约束满足情况 =====
        
        threshold = 0.3  # 满足/违反的阈值
        satisfied = []
        violated = []
        
        constraint_names = {
            'sensory_coherence': 'C1-感觉一致性',
            'temporal_continuity': 'C2-时间连续性',
            'self_consistency': 'C3-自我一致性',
            'action_feasibility': 'C4-行动可行性',
            'social_interpretability': 'C5-社会可解释性',
            'integrated_information': 'C6-整合信息量'
        }
        
        for key, violation in violations_dict.items():
            if violation.mean().item() < threshold:
                satisfied.append(constraint_names[key])
            else:
                violated.append(constraint_names[key])
        
        # 主导违反（最大的违反项）
        violation_values = list(violations_dict.values())
        violation_keys = list(violations_dict.keys())
        max_idx = np.argmax([v.mean().item() for v in violation_values])
        dominant_violation = constraint_names[violation_keys[max_idx]]
        
        # ===== 构建并返回状态 =====
        
        mcs_state = MCSState(
            consciousness_level=consciousness_level.mean().item(),
            total_violation=total_violation.mean().item(),
            constraint_violations={k: v.mean().item() for k, v in violations_dict.items()},
            satisfied_constraints=satisfied,
            violated_constraints=violated,
            dominant_violation=dominant_violation,
            state_vector=current_state,
            phi_value=phi.mean().item()
        )
        
        return mcs_state


# ============================================================================
# 辅助函数
# ============================================================================

def compute_phi_approximation(neural_state: torch.Tensor) -> torch.Tensor:
    """
    简化的 Φ 值近似计算（用于调试）
    
    基于神经状态的复杂度估计
    """
    B, D = neural_state.shape
    
    # 计算方差（复杂度的代理）
    variance = neural_state.var(dim=-1)
    
    # 计算熵（不确定性的度量）
    normalized = F.softmax(neural_state, dim=-1)
    entropy = -torch.sum(normalized * torch.log(normalized + 1e-6), dim=-1)
    
    # Φ ≈ 方差 × 熵
    phi = variance * entropy
    
    return phi


def consciousness_level_to_label(level: float) -> str:
    """
    将连续意识水平转换为离散标签
    
    Level 0 (Vegetative):   J > 0.8  → level < 0.55
    Level 1 (Anesthesia):   0.6 < J ≤ 0.8 → 0.55 ≤ level < 0.625
    Level 2 (Deep Sleep):   0.4 < J ≤ 0.6 → 0.625 ≤ level < 0.71
    Level 3 (Drowsy):       0.2 < J ≤ 0.4 → 0.71 ≤ level < 0.83
    Level 4 (Awake):        0.0 < J ≤ 0.2 → 0.83 ≤ level < 1.0
    Level 5 (Full):         J = 0 → level = 1.0
    """
    if level >= 0.99:
        return "Level 5 (Full)"
    elif level >= 0.83:
        return "Level 4 (Awake)"
    elif level >= 0.71:
        return "Level 3 (Drowsy)"
    elif level >= 0.625:
        return "Level 2 (Deep Sleep)"
    elif level >= 0.55:
        return "Level 1 (Anesthesia)"
    else:
        return "Level 0 (Vegetative)"


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MCS 意识理论：多重约束满足框架 - 单元测试")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模拟数据
    B = 4  # batch size
    T = 10  # time steps
    D = 768  # model dimension
    
    visual = torch.randn(B, T, D)
    auditory = torch.randn(B, T, D)
    current_state = torch.randn(B, D)
    
    # 创建求解器
    solver = MCSConsciousnessSolver(d_model=D)
    
    # 前向传播
    print("\n执行 MCS 求解...")
    mcs_state = solver(
        visual=visual,
        auditory=auditory,
        current_state=current_state
    )
    
    # 打印结果
    print(f"\n{'='*40}")
    print(f"MCS 意识状态:")
    print(f"{'='*40}")
    print(f"意识水平：{mcs_state.consciousness_level:.3f}")
    print(f"总违反：{mcs_state.total_violation:.3f}")
    print(f"Φ 值：{mcs_state.phi_value:.3f}")
    print(f"\n约束违反详情:")
    for key, value in mcs_state.constraint_violations.items():
        print(f"  {key}: {value:.3f}")
    print(f"\n满足的约束：{mcs_state.satisfied_constraints}")
    print(f"违反的约束：{mcs_state.violated_constraints}")
    print(f"主导违反：{mcs_state.dominant_violation}")
    print(f"{'='*40}\n")
    
    # 测试不同场景
    print("测试不同场景...")
    
    # 场景 1: 高一致性输入
    consistent_visual = torch.ones(B, T, D) * 0.5
    consistent_auditory = torch.ones(B, T, D) * 0.5
    state1 = solver(consistent_visual, consistent_auditory, current_state)
    print(f"\n场景 1 (高一致性): 意识水平 = {state1.consciousness_level:.3f}")
    
    # 场景 2: 低一致性输入
    inconsistent_visual = torch.randn(B, T, D)
    inconsistent_auditory = torch.randn(B, T, D) * 10  # 差异很大
    state2 = solver(inconsistent_visual, inconsistent_auditory, current_state)
    print(f"场景 2 (低一致性): 意识水平 = {state2.consciousness_level:.3f}")
    
    # 场景 3: 重复刺激（习惯化）
    repeated_state = torch.ones(B, D) * 0.1
    solver.c2_temporal.reset_history(B)
    for i in range(5):
        state3 = solver(consistent_visual, consistent_auditory, repeated_state)
    print(f"场景 3 (习惯化): 意识水平 = {state3.consciousness_level:.3f}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
