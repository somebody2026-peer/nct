
"""
CATS-NET: Concept Abstraction & Task Solving in NeuroConscious Transformer
融合 CATS Net 双模块架构与 NCT 神经科学特性的创新框架

版本：v1.0.0
创建日期：2026-02-28
作者：NeuroConscious 研发团队

核心特性:
1. 概念抽象模块（CA）：从高维感知压缩为低维概念
2. 任务求解模块（TS）：在概念门控控制下执行具体任务
3. 分层门控机制：概念向量作为"钥匙"精确调控下游模块
4. 概念空间对齐：支持跨网络知识迁移与交流
5. 继承 NCT 神经科学特性：STDP、γ同步、预测编码等
"""

__version__ = "1.0.0"
__author__ = "NeuroConscious Research Team"
__date__ = "2026-02-28"

# 导入核心组件
from cats_nct.core.config import CATSConfig
from cats_nct.manager import CATSManager, CATSConsciousnessState

# 导出核心模块（方便用户直接使用）
from cats_nct.core import (
    ConceptAbstractionModule,
    HierarchicalGatingController,
    MultiTaskSolver,
    ConceptSpaceAligner,
)

__all__ = [
    # 配置与管理器
    "CATSConfig",
    "CATSManager",
    "CATSConsciousnessState",
    
    # 核心模块
    "ConceptAbstractionModule",
    "HierarchicalGatingController",
    "MultiTaskSolver",
    "ConceptSpaceAligner",
    
    # 版本信息
    "__version__",
]
