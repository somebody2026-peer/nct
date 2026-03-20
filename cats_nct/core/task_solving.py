"""
任务求解模块 (Task Solving Module)
在概念门控控制下执行具体任务

核心功能:
1. 接收概念向量的门控信号
2. 并行处理多个任务（视觉、语言、运动等）
3. 动态资源分配（根据门控强度）
4. 任务特定参数学习

架构设计:
```
Concept vector [B, C]
        ↓
Gating Controller → Gate signals [B, n_tasks]
        ↓              ↓
Input [B, D] ─→ Task Module 1 → Output 1
              ├→ Task Module 2 → Output 2
              ├→ Task Module 3 → Output 3
              └→ Task Module 4 → Output 4
```

生物合理性:
- 门控机制 ↔ 前额叶对感觉运动皮层的 top-down 调控
- 多任务并行 ↔ 大脑多系统分工合作
- 动态资源分配 ↔ 注意力资源有限理论

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 核心模块：Task Solving
# ============================================================================

class TaskSolvingModule(nn.Module):
    """单个任务求解模块
    
    负责执行一个具体任务（如猫识别、数字分类等）
    """
    
    def __init__(
        self,
        task_name: str,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 10,
        task_type: str = "classification",
        dropout: float = 0.1,
    ):
        """初始化任务模块
        
        Args:
            task_name: 任务名称（如"cat_recognition"）
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（类别数或回归维度）
            task_type: 任务类型（classification/regression）
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.task_name = task_name
        self.task_type = task_type
        
        # 任务处理网络
        self.task_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        logger.info(
            f"[TaskSolvingModule.{task_name}] 初始化："
            f"{input_dim} → {hidden_dim} → {output_dim}"
        )
    
    def forward(self, gated_input: torch.Tensor) -> torch.Tensor:
        """处理任务
        
        Args:
            gated_input: 经过门控调制的输入 [B, D]
            
        Returns:
            任务输出 [B, output_dim]
        """
        return self.task_network(gated_input)


# ============================================================================
# 多任务管理器
# ============================================================================

class MultiTaskSolver(nn.Module):
    """多任务求解器
    
    管理多个任务模块，支持并行处理和门控控制
    """
    
    def __init__(
        self,
        n_tasks: int = 4,
        input_dim: int = 768,
        task_hidden_dim: int = 512,
        task_output_dims: Optional[List[int]] = None,
        task_names: Optional[List[str]] = None,
        share_lower_layers: bool = False,
    ):
        """初始化多任务求解器
        
        Args:
            n_tasks: 任务数量
            input_dim: 共享输入维度
            task_hidden_dim: 任务隐藏层维度
            task_output_dims: 各任务的输出维度列表（None 则默认相同）
            task_names: 各任务名称（None 则使用默认名）
            share_lower_layers: 是否共享低层特征提取器
        """
        super().__init__()
        
        self.n_tasks = n_tasks
        self.input_dim = input_dim
        
        # 任务配置
        if task_output_dims is None:
            task_output_dims = [10] * n_tasks  # 默认 10 类分类
        
        if task_names is None:
            task_names = [f"task_{i}" for i in range(n_tasks)]
        
        assert len(task_output_dims) == n_tasks
        assert len(task_names) == n_tasks
        
        # ========== 方案 1: 完全独立的任务网络 ===========
        if not share_lower_layers:
            self.task_modules = nn.ModuleList([
                TaskSolvingModule(
                    task_name=task_names[i],
                    input_dim=input_dim,
                    hidden_dim=task_hidden_dim,
                    output_dim=task_output_dims[i],
                )
                for i in range(n_tasks)
            ])
        
        # ========== 方案 2: 共享低层 + 任务特定头 ===========
        else:
            # 共享的特征提取器
            self.shared_encoder = nn.Sequential(
                nn.Linear(input_dim, task_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(task_hidden_dim),
            )
            
            # 任务特定的解码器
            self.task_decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(task_hidden_dim, task_hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(task_hidden_dim // 2, task_output_dims[i]),
                )
                for i in range(n_tasks)
            ])
            
            self.task_names = task_names
        
        logger.info(
            f"[MultiTaskSolver] 初始化："
            f"n_tasks={n_tasks}, input_dim={input_dim}, "
            f"share_lower_layers={share_lower_layers}"
        )
    
    def forward(
        self,
        inputs: List[torch.Tensor],
        gate_signals: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """处理多任务
        
        Args:
            inputs: 各任务的输入列表，每个形状 [B, D]
            gate_signals: 门控信号列表（可选），每个形状 [B, 1]
            
        Returns:
            包含以下字段的字典:
            - outputs: 各任务输出列表
            - gated_outputs: 门控调制后的输出（如果有门控）
        """
        outputs = []
        gated_outputs = []
        
        for i, task_input in enumerate(inputs):
            # 获取任务模块
            if hasattr(self, 'shared_encoder'):
                # 共享编码器模式
                shared_feat = self.shared_encoder(task_input)
                task_output = self.task_decoders[i](shared_feat)
            else:
                # 独立网络模式
                task_module = self.task_modules[i]
                task_output = task_module(task_input)
            
            outputs.append(task_output)
            
            # 应用门控（如果提供）
            if gate_signals is not None:
                gate = gate_signals[i]  # [B, 1]
                gated_output = task_output * gate
                gated_outputs.append(gated_output)
        
        result = {
            'outputs': outputs,
        }
        
        if gate_signals is not None:
            result['gated_outputs'] = gated_outputs
        
        return result
    
    def get_task_stats(self) -> List[Dict[str, Any]]:
        """获取各任务的统计信息"""
        stats = []
        
        for i in range(self.n_tasks):
            if hasattr(self, 'task_modules'):
                # 独立网络
                module = self.task_modules[i]
                n_params = sum(p.numel() for p in module.parameters())
            else:
                # 共享编码器 + 独立解码器
                n_params = (
                    sum(p.numel() for p in self.shared_encoder.parameters())
                    + sum(p.numel() for p in self.task_decoders[i].parameters())
                )
            
            stats.append({
                'task_id': i,
                'task_name': self.task_names[i] if hasattr(self, 'task_names') else f"task_{i}",
                'num_parameters': n_params,
            })
        
        return stats


# ============================================================================
# 任务特定处理器（预定义常用任务）
# ============================================================================

class CatRecognitionTask(TaskSolvingModule):
    """猫识别任务（专门化）"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__(
            task_name="cat_recognition",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2,  # 二分类：猫/非猫
            task_type="classification",
        )


class MNISTClassificationTask(TaskSolvingModule):
    """MNIST 手写数字分类"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__(
            task_name="mnist_classification",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=10,  # 0-9 十个数字
            task_type="classification",
        )


class CIFAR10ClassificationTask(TaskSolvingModule):
    """CIFAR-10 图像分类"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__(
            task_name="cifar10_classification",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=10,  # 10 个类别
            task_type="classification",
        )


class AnomalyDetectionTask(TaskSolvingModule):
    """异常检测任务"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__(
            task_name="anomaly_detection",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # 异常分数（标量）
            task_type="regression",
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'TaskSolvingModule',
    'MultiTaskSolver',
    'CatRecognitionTask',
    'MNISTClassificationTask',
    'CIFAR10ClassificationTask',
    'AnomalyDetectionTask',
]
