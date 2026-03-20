"""
Integration Module - 复用和增强 NCT 模块

本模块负责从 NCT 导入并适配已有组件:
1. MultiModalEncoder - 多模态编码
2. CrossModalIntegration - 跨模态整合
3. PredictiveCoding - 预测编码层次
4. ConsciousnessMetrics - 意识度量
5. GammaSynchronizer - γ同步器
6. TransformerSTDP - 混合学习规则

使用方式:
```python
from cats_nct.integration import (
    MultiModalEncoder,
    CrossModalIntegration,
    PredictiveHierarchy,
    ConsciousnessMetrics,
    GammaSynchronizer,
    TransformerSTDP,
)
```

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ============================================================================
# 从 NCT 导入模块
# ============================================================================

try:
    # 多模态编码器
    from nct_modules.nct_core import (
        MultiModalEncoder,
        VisionTransformer,
        AudioSpectrogramTransformer,
        NCTConfig,
    )
    
    logger.info("✓ 成功从 NCT 导入 MultiModalEncoder")
    
except ImportError as e:
    logger.warning(f"无法从 NCT 导入多模态编码器：{e}")
    logger.warning("将使用简化版本或需要手动配置 NCT 路径")
    
    # 定义占位类，避免导入错误
    class MultiModalEncoder(nn.Module):
        """占位版本 - 需要从 NCT 导入"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("请确保 nct_modules 在 Python 路径中")


try:
    # 跨模态整合
    from nct_modules.nct_cross_modal import CrossModalIntegration
    
    logger.info("✓ 成功从 NCT 导入 CrossModalIntegration")
    
except ImportError as e:
    logger.warning(f"无法从 NCT 导入跨模态整合：{e}")
    
    class CrossModalIntegration(nn.Module):
        """占位版本"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("请确保 nct_modules 在 Python 路径中")


try:
    # 预测编码层次
    from nct_modules.nct_predictive_coding import (
        PredictiveCodingDecoder,
        PredictiveHierarchy,
        SelfModelInference,
    )
    
    logger.info("✓ 成功从 NCT 导入 PredictiveHierarchy")
    
except ImportError as e:
    logger.warning(f"无法从 NCT 导入预测编码：{e}")
    
    class PredictiveHierarchy(nn.Module):
        """占位版本"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("请确保 nct_modules 在 Python 路径中")


try:
    # 意识度量
    from nct_modules.nct_metrics import ConsciousnessMetrics, ConsciousnessLevel
    
    logger.info("✓ 成功从 NCT 导入 ConsciousnessMetrics")
    
except ImportError as e:
    logger.warning(f"无法从 NCT 导入意识度量：{e}")
    
    class ConsciousnessMetrics:
        """占位版本"""
        def __init__(self, *args, **kwargs):
            raise ImportError("请确保 nct_modules 在 Python 路径中")


try:
    # γ同步器
    from nct_modules.nct_gamma_sync import GammaSynchronizer
    
    logger.info("✓ 成功从 NCT 导入 GammaSynchronizer")
    
except ImportError as e:
    logger.warning(f"无法从 NCT 导入γ同步器：{e}")
    
    class GammaSynchronizer:
        """占位版本"""
        def __init__(self, *args, **kwargs):
            raise ImportError("请确保 nct_modules 在 Python 路径中")


try:
    # Transformer-STDP 混合学习
    from nct_modules.nct_hybrid_learning import (
        TransformerSTDP,
        STDPEvent,
        NeuromodulatorGate,
    )
    
    logger.info("✓ 成功从 NCT 导入 TransformerSTDP")
    
except ImportError as e:
    logger.warning(f"无法从 NCT 导入混合学习规则：{e}")
    
    class TransformerSTDP(nn.Module):
        """占位版本"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("请确保 nct_modules 在 Python 路径中")


# ============================================================================
# CATS-NET 特有的集成封装
# ============================================================================

class IntegratedNCTModules:
    """NCT 模块集成封装
    
    提供统一的接口来访问所有从 NCT 导入的模块，
    并处理可能的版本兼容性问题
    """
    
    def __init__(self, config: Any):
        """初始化集成模块
        
        Args:
            config: CATSConfig 或 NCTConfig
        """
        self.config = config
        
        # 检查必要模块是否可用
        self._check_required_modules()
        
        logger.info("[IntegratedNCTModules] 初始化完成")
    
    def _check_required_modules(self):
        """检查必要模块的可用性"""
        required = {
            'MultiModalEncoder': MultiModalEncoder,
            'CrossModalIntegration': CrossModalIntegration,
            'PredictiveHierarchy': PredictiveHierarchy,
            'ConsciousnessMetrics': ConsciousnessMetrics,
            'GammaSynchronizer': GammaSynchronizer,
            'TransformerSTDP': TransformerSTDP,
        }
        
        missing = []
        for name, cls in required.items():
            # 检查是否是占位类
            if hasattr(cls, '__module__') and cls.__module__ is None:
                missing.append(name)
            elif 'ImportError' in str(cls.__init__):
                missing.append(name)
        
        if missing:
            logger.error(
                f"缺少必要的 NCT 模块：{missing}\n"
                f"请确保 nct_modules 目录在 Python 路径中"
            )
        else:
            logger.info("✓ 所有 NCT 模块可用")
    
    def create_multimodal_encoder(self) -> MultiModalEncoder:
        """创建多模态编码器"""
        # 尝试使用 CATSConfig，如果失败则使用 NCTConfig
        try:
            encoder = MultiModalEncoder(self.config)
        except Exception as e:
            logger.warning(f"使用 CATSConfig 创建编码器失败：{e}，尝试 NCTConfig")
            from nct_modules import NCTConfig as NCTConfigOrig
            encoder = MultiModalEncoder(NCTConfigOrig())
        
        return encoder
    
    def create_all_modules(self) -> Dict[str, nn.Module]:
        """创建所有集成的 NCT 模块
        
        Returns:
            模块字典
        """
        modules = {}
        
        # 1. 多模态编码器
        modules['multimodal_encoder'] = self.create_multimodal_encoder()
        
        # 2. 跨模态整合
        try:
            modules['cross_modal'] = CrossModalIntegration(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
            )
        except Exception as e:
            logger.warning(f"无法创建跨模态整合模块：{e}")
        
        # 3. 预测编码层次
        try:
            modules['predictive_hierarchy'] = PredictiveHierarchy(
                config={
                    f'layer{i}_dim': self.config.d_model
                    for i in range(4)
                }
            )
        except Exception as e:
            logger.warning(f"无法创建预测编码层次：{e}")
        
        # 4. 意识度量
        try:
            modules['consciousness_metrics'] = ConsciousnessMetrics()
        except Exception as e:
            logger.warning(f"无法创建意识度量模块：{e}")
        
        # 5. γ同步器
        try:
            modules['gamma_synchronizer'] = GammaSynchronizer(
                frequency=self.config.gamma_freq
            )
        except Exception as e:
            logger.warning(f"无法创建γ同步器：{e}")
        
        # 6. 混合学习规则
        try:
            modules['hybrid_learner'] = TransformerSTDP(
                n_neurons=self.config.d_model,
                d_model=self.config.d_model,
                stdp_learning_rate=self.config.stdp_learning_rate,
                attention_modulation_lambda=self.config.attention_modulation_lambda,
            )
        except Exception as e:
            logger.warning(f"无法创建混合学习规则：{e}")
        
        return modules


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 直接从 NCT 导入的模块
    'MultiModalEncoder',
    'CrossModalIntegration',
    'PredictiveHierarchy',
    'ConsciousnessMetrics',
    'GammaSynchronizer',
    'TransformerSTDP',
    
    # CATS-NET 封装
    'IntegratedNCTModules',
]
