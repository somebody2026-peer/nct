"""
MEMA EEG 数据适配器
将 MEMA 数据集 EEG 数据转换为 MCS Solver 输入格式

数据源: d:/data/mema/Subject{1..20}/*.mat
数据格式:
    - data: (samples, 500, 32) - EEG数据，32通道，500时间点
    - label: (1, samples) - 标签 0/1/2 (neutral/relaxing/concentrating)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edu_experiments.config import MEMA_DIR, D_MODEL, DEVICE


# 标签映射
ATTENTION_LABELS = {0: 'neutral', 1: 'relaxing', 2: 'concentrating'}


class MEMAAdapter(nn.Module):
    """
    MEMA EEG 数据 → MCS Solver 输入格式适配器
    
    适配策略:
        - 前16通道 → visual (视觉皮层区域代理)
        - 后16通道 → auditory (听觉皮层区域代理)  
        - 全32通道 → current_state
    """
    
    def __init__(
        self,
        d_model: int = D_MODEL,
        time_steps: int = 10,
        device: str = DEVICE
    ):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.device = device
        
        # 投影层: 16通道 → d_model
        self.visual_proj = nn.Linear(16, d_model)
        self.auditory_proj = nn.Linear(16, d_model)
        # 全通道 → state
        self.state_proj = nn.Linear(32, d_model)
        
        # Xavier初始化
        nn.init.xavier_uniform_(self.visual_proj.weight)
        nn.init.xavier_uniform_(self.auditory_proj.weight)
        nn.init.xavier_uniform_(self.state_proj.weight)
        
        self.to(device)
        
    def load_all_data(
        self,
        max_subjects: int = 20,
        max_samples_per_subject: int = 500
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
        """
        加载所有受试者数据
        
        Args:
            max_subjects: 最大受试者数量
            max_samples_per_subject: 每个受试者最大样本数
            
        Returns:
            (mcs_inputs_list, labels_list)
        """
        try:
            import scipy.io as sio
        except ImportError:
            print("[MEMA Warning] scipy not installed, returning synthetic data")
            return self._generate_synthetic_data(100)
        
        mcs_inputs = []
        labels = []
        
        for subj_idx in range(1, max_subjects + 1):
            subj_dir = MEMA_DIR / f"Subject{subj_idx}"
            if not subj_dir.exists():
                continue
                
            # 加载 attention 类型数据
            mat_file = subj_dir / f"Subject{subj_idx}_attention.mat"
            if not mat_file.exists():
                # 尝试其他命名格式
                mat_files = list(subj_dir.glob("*.mat"))
                if not mat_files:
                    continue
                mat_file = mat_files[0]
            
            try:
                mat_data = sio.loadmat(str(mat_file))
                data = mat_data['data']  # (samples, 500, 32)
                label = mat_data['label'].flatten()  # (samples,)
                
                n_samples = min(len(label), max_samples_per_subject)
                
                for i in range(n_samples):
                    eeg_segment = data[i]  # (500, 32)
                    lbl = int(label[i])
                    
                    mcs_input = self.adapt_single(eeg_segment, lbl)
                    mcs_inputs.append(mcs_input)
                    labels.append(lbl)
                    
            except Exception as e:
                print(f"[MEMA Warning] Failed to load {mat_file}: {e}")
                continue
                
        if len(mcs_inputs) == 0:
            print("[MEMA Warning] No data loaded, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        print(f"[MEMA] Loaded {len(mcs_inputs)} samples from {max_subjects} subjects")
        return mcs_inputs, labels
    
    def adapt_single(
        self,
        eeg_segment: np.ndarray,
        label: int
    ) -> Dict[str, torch.Tensor]:
        """
        单条EEG片段 → MCS输入格式
        
        Args:
            eeg_segment: EEG数据 (timepoints, channels) = (500, 32)
            label: 标签 0/1/2
            
        Returns:
            {"visual": [1,T,D], "auditory": [1,T,D], "current_state": [1,D]}
        """
        if isinstance(eeg_segment, torch.Tensor):
            eeg_segment = eeg_segment.cpu().numpy()
            
        # 确保数据形状正确
        if eeg_segment.shape[0] == 32 and eeg_segment.shape[1] != 32:
            # 转置: (32, T) → (T, 32)
            eeg_segment = eeg_segment.T
            
        T_original, n_channels = eeg_segment.shape
        
        # 重采样到目标时间步
        if T_original != self.time_steps:
            indices = np.linspace(0, T_original - 1, self.time_steps, dtype=int)
            eeg_segment = eeg_segment[indices]  # (time_steps, 32)
        
        # 转为tensor
        eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32, device=self.device)
        
        # 标准化
        eeg_tensor = (eeg_tensor - eeg_tensor.mean()) / (eeg_tensor.std() + 1e-6)
        
        # 分通道
        visual_raw = eeg_tensor[:, :16]   # 前16通道 → visual
        auditory_raw = eeg_tensor[:, 16:]  # 后16通道 → auditory
        
        # 投影到 d_model
        with torch.no_grad():
            visual = self.visual_proj(visual_raw)      # (T, D)
            auditory = self.auditory_proj(auditory_raw)  # (T, D)
            
            # 全通道均值 → state
            state_raw = eeg_tensor.mean(dim=0, keepdim=True)  # (1, 32)
            current_state = self.state_proj(state_raw)  # (1, D)
        
        # 添加 batch 维度
        return {
            "visual": visual.unsqueeze(0),          # [1, T, D]
            "auditory": auditory.unsqueeze(0),      # [1, T, D]
            "current_state": current_state,         # [1, D]
            "label": label
        }
    
    def _generate_synthetic_data(
        self,
        n_samples: int
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
        """生成合成数据作为降级方案"""
        print(f"[MEMA] Generating {n_samples} synthetic samples")
        
        mcs_inputs = []
        labels = []
        
        for i in range(n_samples):
            label = i % 3  # 均匀分布的标签
            
            # 根据标签生成不同特征的数据
            base_freq = [0.5, 0.3, 0.8][label]  # 不同状态的基础频率
            noise_level = [0.5, 0.3, 0.2][label]  # 不同状态的噪声水平
            
            t = np.linspace(0, 1, 500)
            eeg = np.zeros((500, 32))
            
            for ch in range(32):
                freq = base_freq + 0.1 * np.sin(ch * 0.5)
                eeg[:, ch] = (
                    np.sin(2 * np.pi * freq * t * 10) +
                    noise_level * np.random.randn(500)
                )
            
            mcs_input = self.adapt_single(eeg, label)
            mcs_inputs.append(mcs_input)
            labels.append(label)
            
        return mcs_inputs, labels
    
    def collate_batch(
        self,
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """将多个样本合并为批次"""
        return {
            "visual": torch.cat([b["visual"] for b in batch], dim=0),
            "auditory": torch.cat([b["auditory"] for b in batch], dim=0),
            "current_state": torch.cat([b["current_state"] for b in batch], dim=0),
            "labels": torch.tensor([b["label"] for b in batch], device=self.device)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("MEMA EEG 适配器测试")
    print("=" * 60)
    
    # 创建适配器
    adapter = MEMAAdapter(d_model=D_MODEL, time_steps=10, device=DEVICE)
    print(f"[Test] 适配器创建成功, device={DEVICE}, d_model={D_MODEL}")
    
    # 加载数据
    mcs_inputs, labels = adapter.load_all_data(max_subjects=2, max_samples_per_subject=10)
    
    print(f"\n[Test] 加载样本数: {len(mcs_inputs)}")
    print(f"[Test] 标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 检查输出形状
    if mcs_inputs:
        sample = mcs_inputs[0]
        print(f"\n[Test] 输出形状:")
        print(f"  visual: {sample['visual'].shape}")
        print(f"  auditory: {sample['auditory'].shape}")
        print(f"  current_state: {sample['current_state'].shape}")
        
        # 测试批处理
        batch = adapter.collate_batch(mcs_inputs[:4])
        print(f"\n[Test] 批处理后形状 (batch_size=4):")
        print(f"  visual: {batch['visual'].shape}")
        print(f"  auditory: {batch['auditory'].shape}")
        print(f"  current_state: {batch['current_state'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        
    print("\n" + "=" * 60)
    print("MEMA 适配器测试完成!")
    print("=" * 60)
