# CATS-NET: Concept Abstraction & Task Solving Module
# 核心架构模块

from .config import CATSConfig
from .concept_abstraction import ConceptAbstractionModule, PrototypeAnalyzer
from .differentiable_concept import (
    DifferentiableConceptFormation,
    DifferentiableConceptSpace,
    visualize_attention_weights,
)
from .task_solving import (
    TaskSolvingModule,
    MultiTaskSolver,
    CatRecognitionTask,
    MNISTClassificationTask,
    CIFAR10ClassificationTask,
    AnomalyDetectionTask,
)
from .hierarchical_gating import HierarchicalGatingController, GatingVisualizer
from .concept_space import (
    ConceptSpaceAligner,
    ConceptTransferProtocol,
    ConceptRetriever,
)

__all__ = [
    # 配置
    'CATSConfig',
    
    # 概念抽象（原始版本）
    'ConceptAbstractionModule',
    'PrototypeAnalyzer',
    
    # 可微分概念形成（新版本）
    'DifferentiableConceptFormation',
    'DifferentiableConceptSpace',
    'visualize_attention_weights',
    
    # 任务求解
    'TaskSolvingModule',
    'MultiTaskSolver',
    'CatRecognitionTask',
    'MNISTClassificationTask',
    'CIFAR10ClassificationTask',
    'AnomalyDetectionTask',
    
    # 分层门控
    'HierarchicalGatingController',
    'GatingVisualizer',
    
    # 概念空间
    'ConceptSpaceAligner',
    'ConceptTransferProtocol',
    'ConceptRetriever',
]
