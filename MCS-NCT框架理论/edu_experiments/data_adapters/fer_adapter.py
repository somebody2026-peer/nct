"""
FER2013 情绪数据适配器
将 FER2013 面部表情数据转换为 MCS Solver 输入格式

数据源: d:/data/fer2013/fer2013.csv
数据格式:
    - emotion: 0-6 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    - pixels: 空格分隔的2304个值 (48x48灰度图像)
    - Usage: Training/PublicTest/PrivateTest
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edu_experiments.config import FER_CSV, D_MODEL, DEVICE


# 情绪标签映射
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust', 
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


class FERAdapter(nn.Module):
    """
    FER2013 面部表情数据 → MCS Solver 输入格式适配器
    
    适配策略:
        - 将48x48图像分为6个区域(3x2网格)作为时间步序列
        - 每个区域的原始像素 → visual
        - 每个区域的梯度/边缘特征 → auditory
        - 全图特征 → current_state
    """
    
    def __init__(
        self,
        d_model: int = D_MODEL,
        time_steps: int = 6,
        device: str = DEVICE
    ):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps  # 6个区域 (3行 x 2列)
        self.device = device
        
        # 每个区域大小: 16x24 = 384 像素
        self.region_size = 16 * 24
        
        # 投影层
        self.visual_proj = nn.Linear(self.region_size, d_model)
        self.auditory_proj = nn.Linear(self.region_size, d_model)
        self.state_proj = nn.Linear(48 * 48, d_model)
        
        # Xavier初始化
        nn.init.xavier_uniform_(self.visual_proj.weight)
        nn.init.xavier_uniform_(self.auditory_proj.weight)
        nn.init.xavier_uniform_(self.state_proj.weight)
        
        self.to(device)
        
    def load_dataset(
        self,
        max_samples: int = 5000,
        usage: str = 'Training'
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
        """
        加载FER2013数据集
        
        Args:
            max_samples: 最大样本数
            usage: 'Training', 'PublicTest', 或 'PrivateTest'
            
        Returns:
            (mcs_inputs_list, labels_list)
        """
        try:
            import pandas as pd
        except ImportError:
            print("[FER Warning] pandas not installed, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        if not FER_CSV.exists():
            print(f"[FER Warning] {FER_CSV} not found, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        mcs_inputs = []
        labels = []
        
        try:
            # 分块读取大文件
            chunk_size = 1000
            total_loaded = 0
            
            for chunk in pd.read_csv(str(FER_CSV), chunksize=chunk_size):
                # 过滤指定用途的数据
                filtered = chunk[chunk['Usage'] == usage]
                
                for _, row in filtered.iterrows():
                    if total_loaded >= max_samples:
                        break
                        
                    emotion = int(row['emotion'])
                    pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.float32)
                    pixels = pixels.reshape(48, 48)
                    
                    mcs_input = self.adapt_single(pixels, emotion)
                    mcs_inputs.append(mcs_input)
                    labels.append(emotion)
                    total_loaded += 1
                    
                if total_loaded >= max_samples:
                    break
                    
        except Exception as e:
            print(f"[FER Warning] Failed to load data: {e}")
            return self._generate_synthetic_data(100)
            
        if len(mcs_inputs) == 0:
            print("[FER Warning] No data loaded, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        print(f"[FER] Loaded {len(mcs_inputs)} samples ({usage})")
        return mcs_inputs, labels
    
    def adapt_single(
        self,
        pixels_array: np.ndarray,
        label: int
    ) -> Dict[str, torch.Tensor]:
        """
        单张图像 → MCS输入格式
        
        Args:
            pixels_array: 图像数据 (48, 48)
            label: 情绪标签 0-6
            
        Returns:
            {"visual": [1,T,D], "auditory": [1,T,D], "current_state": [1,D]}
        """
        if isinstance(pixels_array, torch.Tensor):
            pixels_array = pixels_array.cpu().numpy()
            
        # 确保形状正确
        pixels_array = pixels_array.reshape(48, 48).astype(np.float32)
        
        # 归一化到 [-1, 1]
        pixels_norm = (pixels_array - 127.5) / 127.5
        
        # 计算梯度特征 (Sobel边缘)
        gradient = self._compute_gradient(pixels_norm)
        
        # 分割为6个区域 (3行 x 2列, 每个区域 16x24)
        visual_regions = []
        auditory_regions = []
        
        for row in range(3):  # 3行
            for col in range(2):  # 2列
                r_start = row * 16
                r_end = r_start + 16
                c_start = col * 24
                c_end = c_start + 24
                
                region_pixels = pixels_norm[r_start:r_end, c_start:c_end].flatten()
                region_gradient = gradient[r_start:r_end, c_start:c_end].flatten()
                
                visual_regions.append(region_pixels)
                auditory_regions.append(region_gradient)
        
        # 转为tensor
        visual_raw = torch.tensor(np.array(visual_regions), dtype=torch.float32, device=self.device)  # (6, 384)
        auditory_raw = torch.tensor(np.array(auditory_regions), dtype=torch.float32, device=self.device)  # (6, 384)
        state_raw = torch.tensor(pixels_norm.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 2304)
        
        # 投影到 d_model
        with torch.no_grad():
            visual = self.visual_proj(visual_raw)      # (6, D)
            auditory = self.auditory_proj(auditory_raw)  # (6, D)
            current_state = self.state_proj(state_raw)  # (1, D)
        
        # 添加 batch 维度
        return {
            "visual": visual.unsqueeze(0),          # [1, 6, D]
            "auditory": auditory.unsqueeze(0),      # [1, 6, D]
            "current_state": current_state,         # [1, D]
            "label": label
        }
    
    def _compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """计算图像梯度 (Sobel边缘检测)"""
        # 简化的Sobel算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # 手动卷积 (避免依赖scipy)
        h, w = image.shape
        gradient = np.zeros_like(image)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                region = image[i-1:i+2, j-1:j+2]
                gx = np.sum(region * sobel_x)
                gy = np.sum(region * sobel_y)
                gradient[i, j] = np.sqrt(gx**2 + gy**2)
        
        # 归一化
        gradient = gradient / (gradient.max() + 1e-6)
        return gradient
    
    def _generate_synthetic_data(
        self,
        n_samples: int
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
        """生成合成数据作为降级方案"""
        print(f"[FER] Generating {n_samples} synthetic samples")
        
        mcs_inputs = []
        labels = []
        
        for i in range(n_samples):
            label = i % 7  # 7种情绪
            
            # 根据情绪生成不同特征的面部模式
            # 简化版：不同情绪有不同的亮度分布
            base_intensity = [100, 80, 90, 180, 100, 150, 128][label]
            noise_level = [30, 40, 35, 20, 35, 25, 15][label]
            
            # 生成简单的面部模式
            pixels = np.zeros((48, 48), dtype=np.float32)
            
            # 眼睛区域
            pixels[15:20, 10:18] = base_intensity + 30
            pixels[15:20, 30:38] = base_intensity + 30
            
            # 嘴巴区域 (根据情绪调整)
            if label == 3:  # Happy - 微笑
                for x in range(20, 28):
                    y_offset = int(2 * np.sin((x - 20) * np.pi / 8))
                    pixels[35 + y_offset, x + 10] = base_intensity + 50
            elif label == 4:  # Sad - 下垂
                for x in range(20, 28):
                    y_offset = -int(2 * np.sin((x - 20) * np.pi / 8))
                    pixels[35 + y_offset, x + 10] = base_intensity + 50
            else:
                pixels[34:37, 18:30] = base_intensity + 20
            
            # 添加噪声
            pixels += np.random.randn(48, 48) * noise_level
            pixels = np.clip(pixels, 0, 255)
            
            mcs_input = self.adapt_single(pixels, label)
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
    print("FER2013 情绪适配器测试")
    print("=" * 60)
    
    # 创建适配器
    adapter = FERAdapter(d_model=D_MODEL, time_steps=6, device=DEVICE)
    print(f"[Test] 适配器创建成功, device={DEVICE}, d_model={D_MODEL}")
    
    # 加载数据
    mcs_inputs, labels = adapter.load_dataset(max_samples=100, usage='Training')
    
    print(f"\n[Test] 加载样本数: {len(mcs_inputs)}")
    label_counts = {}
    for lbl in labels:
        label_counts[EMOTION_LABELS[lbl]] = label_counts.get(EMOTION_LABELS[lbl], 0) + 1
    print(f"[Test] 标签分布: {label_counts}")
    
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
    print("FER2013 适配器测试完成!")
    print("=" * 60)
