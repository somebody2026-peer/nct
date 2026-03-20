"""
MCS-NCT 集成模块
将 MCS 理论整合到 NCT框架中

注意：这是独立版本，不影响原始 NCT 代码
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time

from mcs_solver import MCSConsciousnessSolver, MCSState


class MCS_NCT_Integrated(nn.Module):
    """
    MCS 理论与 NCT框架的完整集成
    
    架构流程：
    1. NCT 多模态编码
    2. NCT 预测编码
    3. NCT 全局工作空间
    4. 【MCS 新增】多重约束求解
    5. γ同步
    6. 整合输出
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        gamma_freq: float = 40.0,
        consciousness_threshold: float = 0.7,
        mcs_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.gamma_freq = gamma_freq
        
        # ===== NCT 原有组件（简化版用于测试）=====
        
        # 1. 多模态编码器
        self.multimodal_encoder = SimpleMultimodalEncoder(d_model)
        
        # 2. 预测层级
        self.predictive_hierarchy = SimplePredictiveHierarchy(d_model)
        
        # 3. 注意力全局工作空间
        self.attention_workspace = AttentionGlobalWorkspace(
            d_model=d_model,
            n_heads=n_heads,
            dim_ff=dim_feedforward,
            dropout=dropout,
            gamma_freq=gamma_freq,
            consciousness_threshold=consciousness_threshold
        )
        
        # 4. γ振荡器
        self.gamma_synchronizer = GammaSynchronizer(frequency=gamma_freq)
        
        # ===== MCS 新增组件 =====
        
        # 5. MCS 约束求解器
        self.mcs_solver = MCSConsciousnessSolver(
            d_model=d_model,
            constraint_weights=mcs_weights
        )
        
        # 状态跟踪
        self.total_cycles = 0
        self.is_running = False
    
    def start(self):
        """启动系统"""
        self.is_running = True
        self.total_cycles = 0
    
    def stop(self):
        """停止系统"""
        self.is_running = False
    
    def process_cycle(
        self,
        sensory_data: Dict[str, torch.Tensor],
        neurotransmitter_state: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        处理一个意识周期（~100ms）
        
        Args:
            sensory_data: 感觉输入字典
                - 'visual': [B, T, D] 或 [B, H, W]
                - 'auditory': [B, T, F] 语谱图
                - 'interoceptive': [B, D_int] 内感受向量
            neurotransmitter_state: 神经递质状态
        
        Returns:
            state: 整合后的意识状态
        """
        if not self.is_running:
            print("[MCS_NCT] 系统未运行，调用 start()")
            self.start()
        
        current_time = time.time()
        self.total_cycles += 1
        
        B = list(sensory_data.values())[0].shape[0]
        
        # Step 1: NCT 多模态编码
        embeddings = self.multimodal_encoder(sensory_data)
        
        # Step 2: NCT 预测编码
        prediction_results = self.predictive_hierarchy(embeddings)
        prediction_error = prediction_results.get('total_free_energy', 0.5)
        
        # Step 3: NCT 注意力全局工作空间
        winner_state, workspace_info = self.attention_workspace(
            candidates=[embeddings['integrated'].squeeze(1)],  # [B, D]
            neuromodulator_state=neurotransmitter_state
        )
        
        # Step 4: 【MCS 核心】多重约束求解
        mcs_state = self.mcs_solver(
            visual=embeddings.get('visual_emb', embeddings['integrated']),
            auditory=embeddings.get('audio_emb', embeddings['integrated']),
            current_state=embeddings['integrated'].squeeze(1),
            intention=workspace_info.get('intention'),
            timestamps=None,  # 可选
            locations=None    # 可选
        )
        
        # Step 5: γ同步
        gamma_info = self.gamma_synchronizer.get_current_phase(current_time)
        
        # Step 6: 整合输出
        output_state = {
            # NCT 原有输出
            'phi_value': workspace_info.get('phi_value', 0),
            'free_energy': prediction_error,
            'consciousness_level': workspace_info.get('consciousness_level', 'unconscious'),
            'gamma_phase': gamma_info,
            'winner_salience': workspace_info.get('winner_salience', 0),
            'attention_distribution': workspace_info.get('attention_weights'),
            
            # MCS 新增输出
            'mcs_consciousness_level': mcs_state.consciousness_level,
            'mcs_total_violation': mcs_state.total_violation,
            'mcs_constraint_violations': mcs_state.constraint_violations,
            'mcs_satisfied_constraints': mcs_state.satisfied_constraints,
            'mcs_violated_constraints': mcs_state.violated_constraints,
            'mcs_dominant_violation': mcs_state.dominant_violation,
            'mcs_phi_value': mcs_state.phi_value,
            
            # 诊断信息
            'diagnostics': {
                'mcs_state': mcs_state,
                'workspace_info': workspace_info,
                'prediction_error': prediction_error,
            }
        }
        
        return output_state
    
    def get_state_summary(self, output_state: Dict[str, Any]) -> str:
        """生成状态摘要"""
        summary = []
        summary.append(f"Cycle {self.total_cycles}")
        summary.append(f"NCT Φ={output_state['phi_value']:.3f}")
        summary.append(f"MCS Level={output_state['mcs_consciousness_level']:.3f}")
        summary.append(f"MCS Violation={output_state['mcs_total_violation']:.3f}")
        summary.append(f"Dominant={output_state['mcs_dominant_violation']}")
        return " | ".join(summary)


# ============================================================================
# 简化的 NCT 组件（用于演示和测试）
# ============================================================================

class SimpleMultimodalEncoder(nn.Module):
    """简化的多模态编码器"""
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # 视觉编码 - 使用自适应池化，可接受任意输入维度
        self.visual_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(d_model),  # 自适应池化到 d_model
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # 听觉编码 - 使用自适应池化
        self.audio_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # 整合层
        self.integration_layer = nn.Linear(d_model * 2, d_model)
    
    def forward(self, sensory_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = list(sensory_data.values())[0].shape[0]
        device = next(self.parameters()).device
        
        # 视觉编码
        if 'visual' in sensory_data:
            visual = sensory_data['visual']
            # 将输入展平并转置以适应 AdaptiveAvgPool1d
            # 输入: [B, T, D] 或 [B, H, W] -> [B, D, -1]
            if visual.dim() == 3:
                visual_flat = visual.reshape(B, -1)  # [B, T*D]
            else:
                visual_flat = visual.reshape(B, -1)
            visual_flat = visual_flat.unsqueeze(1)  # [B, 1, T*D]
            visual_emb = self.visual_encoder(visual_flat).squeeze(1).unsqueeze(1)  # [B, 1, D]
        else:
            visual_emb = torch.zeros(B, 1, self.d_model, device=device)
        
        # 听觉编码
        if 'auditory' in sensory_data:
            audio = sensory_data['auditory']
            audio_flat = audio.reshape(B, -1).unsqueeze(1)  # [B, 1, T*D]
            audio_emb = self.audio_encoder(audio_flat).squeeze(1).unsqueeze(1)  # [B, 1, D]
        else:
            audio_emb = torch.zeros(B, 1, self.d_model, device=device)
        
        # 整合
        combined = torch.cat([visual_emb, audio_emb], dim=-1)
        integrated = self.integration_layer(combined)
        
        return {
            'visual_emb': visual_emb,
            'audio_emb': audio_emb,
            'integrated': integrated
        }


class SimplePredictiveHierarchy(nn.Module):
    """简化的预测层级"""
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.predictor = nn.Linear(d_model, d_model)
        self.error_computer = nn.MSELoss(reduction='none')
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        integrated = embeddings['integrated']
        
        # 简单预测
        prediction = self.predictor(integrated)
        
        # 预测误差（简化）
        error = self.error_computer(prediction, integrated).mean()
        
        return {
            'total_free_energy': error.item(),
            'prediction': prediction
        }


class AttentionGlobalWorkspace(nn.Module):
    """注意力全局工作空间"""
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        gamma_freq: float = 40.0,
        consciousness_threshold: float = 0.7
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.gamma_freq = gamma_freq
        self.threshold = consciousness_threshold
        
        # Workspace Query
        self.workspace_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Salience 评估
        self.salience_scorer = nn.Linear(d_model, 1)
    
    def forward(
        self,
        candidates: List[torch.Tensor],
        neuromodulator_state: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        处理候选意识内容
        
        Returns:
            winner_state: 获胜者状态
            workspace_info: 工作空间信息
        """
        # 堆叠候选
        candidates_tensor = torch.stack(candidates, dim=1)  # [B, N, D]
        if candidates_tensor.dim() == 4:
            candidates_tensor = candidates_tensor.squeeze(1)  # 移除多余维度
        B, N, D = candidates_tensor.shape
        
        # Self-Attention
        query = self.workspace_query.expand(B, -1, -1)
        attended, attn_weights = self.self_attention(
            query=query,
            key=candidates_tensor,
            value=candidates_tensor
        )
        
        # Residual + Norm
        attended = self.norm1(attended + query)
        
        # FFN
        attended = self.norm2(attended + self.ffn(attended))
        
        # Salience 评分
        salience = torch.sigmoid(self.salience_scorer(attended)).squeeze(-1)
        
        # 意识阈值判断
        consciousness_level = 'conscious' if salience.mean() > self.threshold else 'unconscious'
        
        # 构建输出
        workspace_info = {
            'attention_weights': attn_weights.mean(dim=1).detach().cpu().numpy(),
            'winner_salience': salience.mean().item(),
            'consciousness_level': consciousness_level,
            'phi_value': torch.rand(1).item() * 0.5 + 0.2,  # 模拟 Φ 值
            'salience': salience.detach().cpu().numpy()
        }
        
        winner_state = type('WinnerState', (), {
            'representation': attended.squeeze(1),
            'salience': salience.mean().item()
        })()
        
        return winner_state, workspace_info


class GammaSynchronizer:
    """γ同步器"""
    
    def __init__(self, frequency: float = 40.0):
        self.frequency = frequency
        self.period_ms = 1000.0 / frequency
        self.start_time = None
    
    def get_current_phase(self, current_time: float) -> float:
        """获取当前γ相位"""
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed_ms = (current_time - self.start_time) * 1000
        phase = 2 * np.pi * (elapsed_ms / self.period_ms) % (2 * np.pi)
        return phase


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MCS-NCT 集成系统 - 单元测试")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模拟数据
    B = 2
    T = 5
    D = 768
    
    sensory_data = {
        'visual': torch.randn(B, T, D),
        'auditory': torch.randn(B, T, D)
    }
    
    # 创建集成系统
    system = MCS_NCT_Integrated(d_model=D)
    
    # 启动
    system.start()
    
    # 执行多个周期
    print("\n执行 5 个意识周期...")
    for i in range(5):
        output = system.process_cycle(sensory_data)
        summary = system.get_state_summary(output)
        print(f"周期 {i+1}: {summary}")
    
    # 打印详细输出
    print(f"\n{'='*40}")
    print("最终状态详情:")
    print(f"{'='*40}")
    print(f"NCT Φ值：{output['phi_value']:.3f}")
    print(f"MCS 意识水平：{output['mcs_consciousness_level']:.3f}")
    print(f"MCS 总违反：{output['mcs_total_violation']:.3f}")
    print(f"\nMCS 约束违反:")
    for key, value in output['mcs_constraint_violations'].items():
        print(f"  {key}: {value:.3f}")
    print(f"\n满足的约束：{output['mcs_satisfied_constraints']}")
    print(f"违反的约束：{output['mcs_violated_constraints']}")
    print(f"主导违反：{output['mcs_dominant_violation']}")
    print(f"{'='*40}\n")
    
    print("测试完成！")
