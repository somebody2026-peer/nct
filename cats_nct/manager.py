"""
CATS-NET 总控制器 (CATS Manager)
集成所有子模块，提供统一的 process_cycle 接口

架构流程:
```
sensory_data → MultiModalEncoder → CrossModalIntegration 
              → AttentionGlobalWorkspace → Consciousness Winner
              → ConceptAbstractionModule → Concept Vector
              → HierarchicalGating → Task Modules
              → PredictiveCoding → Learning & Update
              → output (CATSConsciousnessState)
```

核心特性:
1. 概念抽象与任务求解的双模块协同
2. 分层门控精确控制下游任务
3. 复用 NCT 的神经科学机制（STDP、γ同步等）
4. 支持概念迁移与交流

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CATSConsciousnessState:
    """CATS-NET 意识状态
    
    扩展自 NCT 的意识状态，增加概念相关字段
    """
    content_id: str
    representation: torch.Tensor          # 意识表征 [B, D]
    salience: float                       # 显著性
    gamma_phase: float                    # γ波相位
    timestamp: float = field(default_factory=time.time)
    
    # CATS 特有：概念层面
    concept_vector: Optional[torch.Tensor] = None      # 概念向量 [B, C]
    prototype_weights: Optional[torch.Tensor] = None   # 原型权重 [B, n_proto]
    gate_signals: Optional[torch.Tensor] = None        # 门控信号 [B, n_tasks]
    
    # 任务输出
    task_outputs: Optional[List[torch.Tensor]] = None
    
    # 继承 NCT 的指标
    phi_value: float = 0.0                # 整合信息量Φ
    prediction_error: float = 0.0         # 预测误差（自由能）
    awareness_level: str = "minimal"      # 意识水平
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'content_id': self.content_id,
            'representation': self.representation.cpu().numpy() if self.representation is not None else None,
            'salience': self.salience,
            'gamma_phase': self.gamma_phase,
            'timestamp': self.timestamp,
            'concept_vector': self.concept_vector.cpu().numpy() if self.concept_vector is not None else None,
            'prototype_weights': self.prototype_weights.cpu().numpy() if self.prototype_weights is not None else None,
            'phi_value': self.phi_value,
            'prediction_error': self.prediction_error,
            'awareness_level': self.awareness_level,
        }


# ============================================================================
# 核心模块：CATS Manager
# ============================================================================

class CATSManager(nn.Module):
    """CATS-NET 总控制器
    
    完整集成 CATS Net 双模块架构与 NCT 神经科学特性
    
    使用示例:
    ```python
    from cats_nct import CATSManager, CATSConfig
    
    config = CATSConfig()
    manager = CATSManager(config)
    manager.start()
    
    sensory_data = {'visual': image, 'auditory': audio}
    state = manager.process_cycle(sensory_data)
    
    print(f"概念向量：{state.concept_vector.shape}")
    print(f"门控信号：{state.gate_signals.shape}")
    ```
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        """初始化 CATS 管理器
        
        Args:
            config: CATSConfig 配置
            device: 设备类型（None=自动选择）
        """
        super().__init__()
        
        # ========== 1. 配置与设备 ==========
        try:
            from cats_nct.core.config import CATSConfig
            self.config = config if config is not None else CATSConfig()
        except ImportError:
            logger.error("无法导入 CATSConfig，请检查安装")
            raise
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"[CATSManager] 初始化，设备：{self.device}")
        
        # ========== 2. 导入并初始化 NCT 模块 ==========
        try:
            from cats_nct.integration import IntegratedNCTModules
            
            self.nct_modules = IntegratedNCTModules(self.config)
            nct_components = self.nct_modules.create_all_modules()
            
            # 提取各组件
            self.multimodal_encoder = nct_components.get('multimodal_encoder')
            self.cross_modal = nct_components.get('cross_modal')
            self.predictive_hierarchy = nct_components.get('predictive_hierarchy')
            self.metrics = nct_components.get('consciousness_metrics')
            self.gamma_sync = nct_components.get('gamma_synchronizer')
            self.hybrid_learner = nct_components.get('hybrid_learner')
            
            logger.info("✓ NCT 模块加载成功")
            
        except Exception as e:
            logger.warning(f"NCT 模块加载失败：{e}")
            logger.warning("将使用简化模式（仅 CATS 核心模块）")
            self.multimodal_encoder = None
            self.cross_modal = None
            self.predictive_hierarchy = None
            self.metrics = None
            self.gamma_sync = None
            self.hybrid_learner = None
        
        # ========== 3. 从 NCT 导入全局工作空间 ==========
        try:
            from nct_modules.nct_workspace import AttentionGlobalWorkspace
            
            self.attention_workspace = AttentionGlobalWorkspace(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dim_ff=self.config.dim_ff,
                gamma_freq=self.config.gamma_freq,
                consciousness_threshold=self.config.consciousness_threshold,
            )
            logger.info("✓ 加载 AttentionGlobalWorkspace")
            
        except ImportError as e:
            logger.warning(f"无法加载全局工作空间：{e}")
            self.attention_workspace = None
        
        # ========== 4. CATS 核心模块 ==========
        try:
            from cats_nct.core import (
                ConceptAbstractionModule,
                MultiTaskSolver,
                HierarchicalGatingController,
            )
            
            # 概念抽象模块
            self.concept_module = ConceptAbstractionModule(
                d_model=self.config.d_model,
                concept_dim=self.config.concept_dim,
                n_prototypes=self.config.n_concept_prototypes,
                hidden_dim=self.config.concept_encoder_hidden,
                activation=self.config.concept_activation,
                reconstruction_loss_lambda=getattr(self.config, 'reconstruction_loss_lambda', 1.0),
                sparsity_lambda=getattr(self.config, 'concept_sparsity_lambda', 0.01),
                diversity_lambda=getattr(self.config, 'prototype_diversity_lambda', 0.1),
            )
            
            # 多任务求解器
            task_output_dims = getattr(self.config, 'task_output_dims', None)
            if task_output_dims is None:
                task_output_dims = [10] * self.config.n_task_modules
            # 确保输出维度列表与任务数一致
            while len(task_output_dims) < self.config.n_task_modules:
                task_output_dims.append(task_output_dims[-1] if task_output_dims else 10)
            
            self.task_solver = MultiTaskSolver(
                n_tasks=self.config.n_task_modules,
                input_dim=self.config.d_model,
                task_hidden_dim=self.config.task_hidden_dim,
                task_output_dims=task_output_dims[:self.config.n_task_modules],
            )
            
            # 分层门控控制器
            self.gating_controller = HierarchicalGatingController(
                concept_dim=self.config.concept_dim,
                n_task_modules=self.config.n_task_modules,
                n_levels=self.config.n_gating_levels,
                hidden_dim=self.config.task_hidden_dim // 4,
                gating_type=self.config.gating_type,
            )
            
            logger.info("✓ CATS 核心模块加载成功")
            
        except ImportError as e:
            logger.error(f"CATS 核心模块加载失败：{e}")
            raise
        
        # ========== 5. 运行状态 ==========
        self.is_running = False
        self.total_cycles = 0
        self.concept_history = []  # 概念历史（用于分析）
        
        logger.info("[CATSManager] 初始化完成")
    
    def start(self):
        """启动系统"""
        self.is_running = True
        self.total_cycles = 0
        self.concept_history = []
        logger.info("[CATSManager] 系统启动")
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        logger.info("[CATSManager] 系统停止")
    
    def process_cycle(
        self,
        sensory_data: Dict[str, Any],
        neurotransmitter_state: Optional[Dict[str, float]] = None,
    ) -> CATSConsciousnessState:
        """处理一个意识周期
        
        Args:
            sensory_data: 感觉输入字典
                - 'visual': [H, W] 或 [T, H, W]
                - 'auditory': [T, F] 语谱图
                - 'interoceptive': [10] 内感受向量
            neurotransmitter_state: 神经递质状态（可选）
        
        Returns:
            state: 意识状态（包含概念向量和任务输出）
        """
        if not self.is_running:
            logger.warning("[CATSManager] 系统未启动，调用 start()")
            return None
        
        current_time = time.time()
        
        # ========== 预处理：将 numpy array 转换为 tensor ==========
        processed_data = {}
        for key, value in sensory_data.items():
            if isinstance(value, np.ndarray):
                # 转换为 tensor
                tensor = torch.from_numpy(value).float()
                if tensor.dim() == 2:  # [H, W] -> [B, C, H, W]
                    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif tensor.dim() == 3:  # [C, H, W] 或 [T, H, W] -> [B, C, H, W]
                    tensor = tensor.unsqueeze(0)  # [1, C, H, W]
                processed_data[key] = tensor
            elif isinstance(value, torch.Tensor):
                tensor = value
                if tensor.dim() == 2:  # [H, W] -> [B, C, H, W]
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                elif tensor.dim() == 3:  # [C, H, W] -> [B, C, H, W]
                    tensor = tensor.unsqueeze(0)
                processed_data[key] = tensor
            else:
                processed_data[key] = value
        
        # ========== Step 1: 多模态编码 ==========
        if self.multimodal_encoder is not None:
            embeddings = self.multimodal_encoder(processed_data)
        else:
            # 简化模式：使用随机特征
            embeddings = {'visual_emb': torch.randn(1, 64, self.config.d_model)}
        
        # ========== Step 2: 跨模态整合 ==========
        if self.cross_modal is not None:
            integrated = self.cross_modal(embeddings)
            # 处理返回值可能是 tuple 的情况
            if isinstance(integrated, tuple):
                integrated = integrated[0]  # 取第一个元素作为整合表征
        else:
            integrated = embeddings.get('visual_emb', None)
        
        if integrated is None:
            logger.error("没有有效的整合表征")
            return None
        
        # ========== Step 3: 全局工作空间竞争 ==========
        candidates = self._extract_candidates(integrated)
        
        if self.attention_workspace is not None:
            winner_state, workspace_info = self.attention_workspace(
                candidates,
                neuromodulator_state=neurotransmitter_state,
            )
            
            if winner_state is None:
                # 无意识内容
                return self._create_minimal_state(current_time, workspace_info)
            
            winner_representation = winner_state.representation
            winner_salience = winner_state.salience
            gamma_phase = winner_state.gamma_phase
            
        else:
            # 简化模式：直接取平均
            winner_representation = integrated.mean(dim=1)
            winner_salience = 0.8
            gamma_phase = 0.0
            workspace_info = {}
        
        # ========== Step 4: 概念抽象（CATS 核心创新） ==========
        concept_output = self.concept_module(winner_representation)
        
        concept_vector = concept_output['concept_vector']
        prototype_weights = concept_output['prototype_weights']
        
        # 记录概念历史
        self.concept_history.append(concept_vector.detach().cpu())
        
        # ========== Step 5: 分层门控控制 ==========
        gate_output = self.gating_controller(concept_vector)
        
        gate_signals = gate_output['combined_gates']
        gate_diagnostics = gate_output['diagnostics']
        
        # ========== Step 6: 任务求解 ==========
        # 准备任务输入（这里使用相同的整合表征，实际可以不同）
        task_inputs = [winner_representation] * self.config.n_task_modules
        
        # 应用门控
        gated_inputs = self.gating_controller.apply_gates(task_inputs, gate_output)
        
        # 执行任务
        task_results = self.task_solver(gated_inputs, gate_signals=None)
        task_outputs = task_results.get('outputs', [])
        
        # ========== Step 7: 预测编码与学习 ==========
        if self.predictive_hierarchy is not None:
            prediction_results = self.predictive_hierarchy(winner_representation)
            prediction_error = prediction_results.get('total_free_energy', 0.0)
        else:
            prediction_error = 0.0
        
        # ========== Step 8: STDP 学习（如果启用） ==========
        if self.hybrid_learner is not None and self.config.use_stdp:
            # 这里需要根据具体事件触发 STDP
            pass
        
        # ========== Step 9: 计算意识度量 ==========
        if self.metrics is not None:
            # TODO: 实现Φ值计算
            phi_value = 0.0
            awareness_level = self._compute_awareness_level(
                winner_salience, 
                gate_diagnostics,
            )
        else:
            phi_value = 0.0
            awareness_level = "normal"
        
        # ========== Step 10: 构建并返回状态 ==========
        state = CATSConsciousnessState(
            content_id=f"cats_t_{current_time}",
            representation=winner_representation,
            salience=winner_salience,
            gamma_phase=gamma_phase,
            timestamp=current_time,
            concept_vector=concept_vector,
            prototype_weights=prototype_weights,
            gate_signals=gate_signals,
            task_outputs=task_outputs,
            phi_value=phi_value,
            prediction_error=prediction_error,
            awareness_level=awareness_level,
        )
        
        # 更新周期计数
        self.total_cycles += 1
        
        # 日志
        if self.total_cycles % 10 == 0:
            logger.info(
                f"[CATSManager] 周期 #{self.total_cycles}: "
                f"salience={winner_salience:.3f}, "
                f"concept_norm={concept_vector.norm().item():.3f}, "
                f"gate_sparsity={gate_diagnostics['sparsity']:.3f}"
            )
        
        return state
    
    def _extract_candidates(
        self,
        integrated: torch.Tensor,
    ) -> List[torch.Tensor]:
        """从整合表征中提取候选意识内容
        
        Args:
            integrated: 整合后的表征 [B, N, D]
        
        Returns:
            候选列表
        """
        # 简单策略：每个时间步作为一个候选
        # 确保是 2D 张量 [B, D]
        if integrated.dim() == 3:
            candidates = [integrated[:, i, :] for i in range(integrated.shape[1])]
        else:
            # 如果已经是 2D，直接返回
            candidates = [integrated]
        return candidates
    
    def _create_minimal_state(
        self,
        timestamp: float,
        info: Dict[str, Any],
    ) -> CATSConsciousnessState:
        """创建最小意识状态（当无显著内容时）"""
        return CATSConsciousnessState(
            content_id=f"minimal_t_{timestamp}",
            representation=torch.zeros(1, self.config.d_model),
            salience=0.0,
            gamma_phase=0.0,
            timestamp=timestamp,
            awareness_level="minimal",
        )
    
    def _compute_awareness_level(
        self,
        salience: float,
        gate_diagnostics: Dict[str, Any],
    ) -> str:
        """根据显著性和门控计算意识水平"""
        if salience < self.config.consciousness_threshold:
            return "minimal"
        
        # 根据门控稀疏性判断
        sparsity = gate_diagnostics.get('sparsity', 0.5)
        
        if sparsity < 0.2:
            return "highly_focused"  # 高度聚焦
        elif sparsity < 0.5:
            return "normal"  # 正常
        else:
            return "diffuse"  # 弥散
    
    def get_concept_stats(self) -> Dict[str, Any]:
        """获取概念统计信息"""
        if len(self.concept_history) == 0:
            return {'error': 'No concept history available'}
        
        # 堆叠历史
        all_concepts = torch.cat(self.concept_history, dim=0)
        
        stats = {
            'n_samples': len(all_concepts),
            'concept_dim': all_concepts.shape[1],
            'mean_norm': all_concepts.norm(p=2, dim=1).mean().item(),
            'std_norm': all_concepts.norm(p=2, dim=1).std().item(),
        }
        
        # 原型使用分析
        if hasattr(self.concept_module, 'prototypes'):
            with torch.no_grad():
                prototypes = self.concept_module.prototypes
                
                # 计算每个样本最接近的原型
                distances = torch.cdist(all_concepts, prototypes)
                nearest_proto = distances.argmin(dim=1)
                
                # 原型使用频率
                usage_counts = torch.bincount(
                    nearest_proto.flatten(), 
                    minlength=len(prototypes),
                )
                
                stats['prototype_usage'] = {
                    'active_prototypes': (usage_counts > 0).sum().item(),
                    'most_used_prototype': usage_counts.argmax().item(),
                    'most_used_count': usage_counts.max().item(),
                }
        
        return stats
    
    def export_concept_package(self) -> Dict[str, Any]:
        """导出概念包（用于迁移到其他实例）"""
        from cats_nct.core.concept_space import ConceptTransferProtocol
        
        if len(self.concept_history) == 0:
            return {'error': 'No concepts to export'}
        
        # 使用最近的 K 个概念
        K = min(100, len(self.concept_history))
        recent_concepts = torch.cat(self.concept_history[-K:], dim=0)
        
        # 创建协议
        protocol = ConceptTransferProtocol(self.concept_module)
        
        # 导出
        package = protocol.export_concepts(
            self.concept_module,
            recent_concepts,
        )
        
        logger.info(f"[CATSManager] 已导出概念包，包含 {len(recent_concepts)} 个概念")
        
        return package


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'CATSConsciousnessState',
    'CATSManager',
]
