"""
CATS-NET 配置类
融合 CATS Net 双模块架构与 NCT 神经科学特性
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class CATSConfig:
    """CATS-NET 完整配置
    
    设计原则:
    1. 概念抽象维度显著低于原始表征（压缩率~8%）
    2. 保持与 NCT 相同的 d_model、n_heads 等参数，便于对比
    3. 支持 STDP、γ同步等 NCT 特性
    
    示例用法:
    ```python
    config = CATSConfig(
        concept_dim=64,
        n_concept_prototypes=100,
        d_model=768,
        n_heads=8,
    )
    manager = CATSManager(config)
    ```
    """
    
    # ========== CA 模块（概念抽象）参数 ==========
    concept_dim: int = 64
    """概念向量维度（低维压缩），建议为 d_model 的 5-10%"""
    
    n_concept_prototypes: int = 100
    """可学习的概念原型数量，覆盖基础语义类别"""
    
    concept_compression_ratio: float = 0.08
    """压缩率目标 (concept_dim / d_model)，用于自动调参"""
    
    concept_encoder_hidden: int = 256
    """概念编码器隐藏层维度"""
    
    concept_activation: str = "gelu"
    """概念激活函数：gelu/relu/tanh"""
    
    # ========== TS 模块（任务求解）参数 ==========
    n_task_modules: int = 4
    """并行任务模块数量（如视觉、语言、运动等）"""
    
    task_output_dims: List[int] = field(default_factory=lambda: [10])
    """各任务的输出维度列表（默认 10 类分类）"""
    
    task_hidden_dim: int = 512
    """任务网络隐藏层维度"""
    
    gating_type: str = "sigmoid"
    """门控类型：sigmoid（独立开关）/ softmax（竞争分配）/ linear（线性调制）"""
    
    n_gating_levels: int = 3
    """分层门控的层级数（粗粒度→细粒度）"""
    
    # ========== 继承 NCT 的架构参数 ==========
    d_model: int = 768
    """模型表征维度（必须与 NCT 一致，便于对比实验）"""
    
    n_heads: int = 8
    """注意力头数（Miller's Law 7±2）"""
    
    n_layers: int = 4
    """Transformer 层数（皮层 4 层结构）"""
    
    dim_ff: int = 3072
    """前馈网络维度（4×d_model）"""
    
    dropout: float = 0.1
    """Dropout 比率"""
    
    max_seq_len: int = 512
    """最大序列长度"""
    
    # ========== 多模态编码参数 ==========
    visual_patch_size: int = 4
    """视觉 patch 大小"""
    
    visual_embed_dim: int = 256
    """视觉 embedding 维度"""
    
    audio_embed_dim: int = 256
    """音频 embedding 维度"""
    
    intero_embed_dim: int = 256
    """内感受 embedding 维度"""
    
    # ========== 学习参数 ==========
    concept_learning_rate: float = 1e-3
    """概念抽象模块学习率"""
    
    alignment_learning_rate: float = 1e-4
    """概念空间对齐学习率"""
    
    use_stdp: bool = True
    """是否使用 Transformer-STDP 混合学习（继承 NCT）"""
    
    stdp_learning_rate: float = 0.01
    """STDP 学习率"""
    
    attention_modulation_lambda: float = 0.1
    """注意力对 STDP 的调制强度λ"""
    
    # ========== 预测编码参数 ==========
    use_predictive_coding: bool = True
    """是否使用预测编码层次（Friston 自由能原理）"""
    
    predictive_hierarchy_layers: int = 4
    """预测编码层次结构的层数"""
    
    # ========== 神经生物学参数 ==========
    gamma_freq: float = 40.0
    """γ波频率（Hz），控制更新周期"""
    
    consciousness_threshold: float = 0.7
    """意识阈值（注意力权重超过此值才进入意识）"""
    
    # ========== 概念空间对齐参数 ==========
    use_adversarial_alignment: bool = True
    """是否使用对抗训练进行概念空间对齐"""
    
    alignment_hidden_dim: int = 128
    """对齐网络隐藏层维度"""
    
    alignment_discriminator_layers: int = 2
    """判别器层数"""
    
    # ========== 正则化参数 ==========
    concept_sparsity_lambda: float = 0.01
    """概念稀疏性约束权重"""
    
    prototype_diversity_lambda: float = 0.1
    """原型多样性约束权重（防止所有原型趋同）"""
    
    reconstruction_loss_lambda: float = 1.0
    """重构损失权重（确保概念保留关键信息）"""
    
    # ========== 训练参数 ==========
    batch_first: bool = True
    """batch 优先格式"""
    
    gradient_clip_norm: float = 1.0
    """梯度裁剪范数"""
    
    warmup_steps: int = 1000
    """学习率预热步数"""
    
    # ========== 设备与日志 ==========
    device: Optional[str] = None
    """训练设备（None=自动选择 cuda/cpu）"""
    
    log_level: str = "info"
    """日志级别：debug/info/warning/error"""
    
    use_wandb: bool = False
    """是否使用 WandB 跟踪实验"""
    
    project_name: str = "CATS-NET"
    """项目名称（用于 WandB）"""
    
    verbose: bool = False
    """是否打印配置摘要"""
    
    def __post_init__(self):
        """配置验证与自动调整
        
        确保各参数满足生物合理性和数值稳定性要求
        """
        # 架构验证
        assert self.concept_dim < self.d_model, (
            f"概念维度 ({self.concept_dim}) 必须小于模型维度 ({self.d_model})"
        )
        
        assert self.n_heads > 0 and self.n_layers > 0, (
            "注意力头数和层数必须为正整数"
        )
        
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
        )
        
        # 压缩率检查
        actual_ratio = self.concept_dim / self.d_model
        if abs(actual_ratio - self.concept_compression_ratio) > 0.05:
            print(
                f"[警告] 实际压缩率 ({actual_ratio:.2f}) 与目标 "
                f"({self.concept_compression_ratio}) 偏差较大"
            )
        
        # 门控类型验证
        valid_gating_types = ["sigmoid", "softmax", "linear"]
        assert self.gating_type in valid_gating_types, (
            f"gating_type 必须是 {valid_gating_types} 之一"
        )
        
        # 自动计算衍生参数
        self._compute_derived_params()
        
        # 打印配置摘要（仅在 verbose=True 时）
        if self.verbose:
            self._print_config_summary()
    
    def _compute_derived_params(self):
        """计算衍生参数"""
        # 实际压缩率
        self.actual_compression_ratio = self.concept_dim / self.d_model
        
        # 每个注意力头的维度
        self.head_dim = self.d_model // self.n_heads
        
        # 前馈网络维度（如果未显式设置）
        if not hasattr(self, 'dim_ff') or self.dim_ff is None:
            self.dim_ff = 4 * self.d_model
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("=" * 70)
        print("CATS-NET 配置摘要")
        print("=" * 70)
        print(f"✓ 概念抽象模块:")
        print(f"  - 概念维度：{self.concept_dim}")
        print(f"  - 原型数量：{self.n_concept_prototypes}")
        print(f"  - 压缩率：{self.actual_compression_ratio:.2%}")
        print(f"✓ 任务求解模块:")
        print(f"  - 任务模块数：{self.n_task_modules}")
        print(f"  - 门控类型：{self.gating_type}")
        print(f"✓ 全局架构:")
        print(f"  - d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers}")
        print(f"  - head_dim={self.head_dim}")
        print(f"✓ 学习机制:")
        print(f"  - STDP: {self.use_stdp}")
        print(f"  - 预测编码：{self.use_predictive_coding}")
        print(f"  - γ频率：{self.gamma_freq}Hz")
        print(f"✓ 概念对齐:")
        print(f"  - 对抗训练：{self.use_adversarial_alignment}")
        print("=" * 70)
    
    @classmethod
    def get_small_config(cls) -> "CATSConfig":
        """小型配置（用于快速测试）
        
        Returns:
            缩小版的配置，减少计算资源消耗
        """
        return cls(
            concept_dim=32,
            n_concept_prototypes=50,
            d_model=256,
            n_heads=4,
            n_layers=2,
            dim_ff=1024,
            n_task_modules=2,
        )
    
    @classmethod
    def get_medium_config(cls) -> "CATSConfig":
        """中型配置（平衡性能与速度）"""
        return cls(
            concept_dim=64,
            n_concept_prototypes=100,
            d_model=512,
            n_heads=8,
            n_layers=4,
            dim_ff=2048,
            n_task_modules=4,
        )
    
    @classmethod
    def get_large_config(cls) -> "CATSConfig":
        """大型配置（用于最终实验）"""
        return cls(
            concept_dim=128,
            n_concept_prototypes=200,
            d_model=768,
            n_heads=8,
            n_layers=6,
            dim_ff=3072,
            n_task_modules=6,
        )
    
    def to_dict(self) -> Dict:
        """转换为字典格式（用于序列化）"""
        import json
        # 只序列化基本类型，排除方法和特殊属性
        return {
            'concept_dim': self.concept_dim,
            'n_concept_prototypes': self.n_concept_prototypes,
            'concept_compression_ratio': self.concept_compression_ratio,
            'concept_encoder_hidden': self.concept_encoder_hidden,
            'concept_activation': self.concept_activation,
            'n_task_modules': self.n_task_modules,
            'task_hidden_dim': self.task_hidden_dim,
            'gating_type': self.gating_type,
            'n_gating_levels': self.n_gating_levels,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dim_ff': self.dim_ff,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'visual_patch_size': self.visual_patch_size,
            'visual_embed_dim': self.visual_embed_dim,
            'audio_embed_dim': self.audio_embed_dim,
            'intero_embed_dim': self.intero_embed_dim,
            'concept_learning_rate': self.concept_learning_rate,
            'alignment_learning_rate': self.alignment_learning_rate,
            'use_stdp': self.use_stdp,
            'stdp_learning_rate': self.stdp_learning_rate,
            'attention_modulation_lambda': self.attention_modulation_lambda,
            'use_predictive_coding': self.use_predictive_coding,
            'predictive_hierarchy_layers': self.predictive_hierarchy_layers,
            'gamma_freq': self.gamma_freq,
            'consciousness_threshold': self.consciousness_threshold,
            'use_adversarial_alignment': self.use_adversarial_alignment,
            'alignment_hidden_dim': self.alignment_hidden_dim,
            'alignment_discriminator_layers': self.alignment_discriminator_layers,
            'concept_sparsity_lambda': self.concept_sparsity_lambda,
            'prototype_diversity_lambda': self.prototype_diversity_lambda,
            'reconstruction_loss_lambda': self.reconstruction_loss_lambda,
            'batch_first': self.batch_first,
            'gradient_clip_norm': self.gradient_clip_norm,
            'warmup_steps': self.warmup_steps,
            'device': self.device,
            'log_level': self.log_level,
            'use_wandb': self.use_wandb,
            'project_name': self.project_name,
        }
    
    def save_to_json(self, filepath: str):
        """保存配置到 JSON 文件
        
        Args:
            filepath: 保存路径
        """
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[CATSConfig] 配置已保存到 {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> "CATSConfig":
        """从 JSON 文件加载配置
        
        Args:
            filepath: JSON 文件路径
            
        Returns:
            CATSConfig 实例
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
