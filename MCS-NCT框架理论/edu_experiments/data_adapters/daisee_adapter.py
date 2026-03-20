"""
DAiSEE 视频参与度数据适配器
将 DAiSEE 数据集视频转换为 MCS Solver 输入格式

数据源: d:/data/daisee/DAiSEE/DAiSEE/
结构:
    - Labels/TrainLabels.csv: ClipID, Boredom, Engagement, Confusion, Frustration (0-3)
    - DataSet/Train/{StudentID}/{ClipID}/{ClipID}.avi: 视频文件
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edu_experiments.config import DAISEE_DIR, D_MODEL, DEVICE


# 标签名称
LABEL_NAMES = ['Boredom', 'Engagement', 'Confusion', 'Frustration']


class DAiSEEAdapter(nn.Module):
    """
    DAiSEE 视频数据 → MCS Solver 输入格式适配器
    
    适配策略:
        - 使用 ResNet18 提取帧特征 (或简化CNN)
        - 连续帧特征序列 → visual
        - 帧间差分特征 → auditory
        - 帧特征均值 → current_state
    """
    
    def __init__(
        self,
        d_model: int = D_MODEL,
        num_frames: int = 8,
        device: str = DEVICE,
        use_pretrained: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_frames = num_frames
        self.device = device
        
        # 特征维度 (ResNet18 输出512维, 简化CNN输出256维)
        self.feature_dim = 512 if use_pretrained else 256
        
        # 尝试加载预训练ResNet18
        self.feature_extractor = self._build_feature_extractor(use_pretrained)
        
        # 投影层
        self.visual_proj = nn.Linear(self.feature_dim, d_model)
        self.auditory_proj = nn.Linear(self.feature_dim, d_model)
        self.state_proj = nn.Linear(self.feature_dim, d_model)
        
        # Xavier初始化投影层
        nn.init.xavier_uniform_(self.visual_proj.weight)
        nn.init.xavier_uniform_(self.auditory_proj.weight)
        nn.init.xavier_uniform_(self.state_proj.weight)
        
        self.to(device)
        
    def _build_feature_extractor(self, use_pretrained: bool) -> nn.Module:
        """构建特征提取器"""
        try:
            if use_pretrained:
                from torchvision import models
                resnet = models.resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)
                # 去掉最后的FC层
                modules = list(resnet.children())[:-1]
                extractor = nn.Sequential(*modules)
                self.feature_dim = 512
                print("[DAiSEE] Using pretrained ResNet18 feature extractor")
                return extractor
        except Exception as e:
            print(f"[DAiSEE Warning] Failed to load ResNet18: {e}")
        
        # 降级方案: 简化CNN
        print("[DAiSEE] Using simplified CNN feature extractor")
        self.feature_dim = 256
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )
        
    def load_dataset(
        self,
        max_clips: int = 500,
        split: str = 'Train'
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, int]]]:
        """
        加载DAiSEE数据集
        
        Args:
            max_clips: 最大视频片段数
            split: 'Train', 'Validation', 或 'Test'
            
        Returns:
            (mcs_inputs_list, labels_dicts_list)
        """
        try:
            import pandas as pd
        except ImportError:
            print("[DAiSEE Warning] pandas not installed, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        # 加载标签
        labels_file = DAISEE_DIR / "Labels" / f"{split}Labels.csv"
        if not labels_file.exists():
            print(f"[DAiSEE Warning] {labels_file} not found, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        try:
            labels_df = pd.read_csv(str(labels_file))
            # 修复列名中的空格
            labels_df.columns = [c.strip() for c in labels_df.columns]
        except Exception as e:
            print(f"[DAiSEE Warning] Failed to load labels: {e}")
            return self._generate_synthetic_data(100)
            
        dataset_dir = DAISEE_DIR / "DataSet" / split
        
        mcs_inputs = []
        labels_list = []
        loaded = 0
        
        for _, row in labels_df.iterrows():
            if loaded >= max_clips:
                break
                
            clip_id = row['ClipID'].replace('.avi', '')
            
            # 构建视频路径: DataSet/Train/{StudentID}/{ClipID}/{ClipID}.avi
            # StudentID 是 ClipID 的前6位
            student_id = clip_id[:6]
            video_path = dataset_dir / student_id / clip_id / f"{clip_id}.avi"
            
            if not video_path.exists():
                continue
                
            # 提取标签
            label_dict = {
                'Boredom': int(row['Boredom']),
                'Engagement': int(row['Engagement']),
                'Confusion': int(row['Confusion']),
                'Frustration': int(row['Frustration'])
            }
            
            try:
                mcs_input = self.adapt_single(str(video_path), label_dict)
                if mcs_input is not None:
                    mcs_inputs.append(mcs_input)
                    labels_list.append(label_dict)
                    loaded += 1
            except Exception as e:
                print(f"[DAiSEE Warning] Failed to process {video_path}: {e}")
                continue
                
        if len(mcs_inputs) == 0:
            print("[DAiSEE Warning] No data loaded, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        print(f"[DAiSEE] Loaded {len(mcs_inputs)} clips ({split})")
        return mcs_inputs, labels_list
    
    def adapt_single(
        self,
        video_path: str,
        labels: Dict[str, int]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        单个视频片段 → MCS输入格式
        
        Args:
            video_path: 视频文件路径
            labels: 标签字典
            
        Returns:
            {"visual": [1,T,D], "auditory": [1,T,D], "current_state": [1,D]}
        """
        # 读取视频帧
        frames = self._read_video_frames(video_path)
        
        if frames is None or len(frames) < 2:
            return None
            
        # 提取特征
        features = self._extract_features(frames)  # (num_frames, feature_dim)
        
        if features is None:
            return None
            
        # 计算帧间差分
        diff_features = self._compute_temporal_diff(features)  # (num_frames, feature_dim)
        
        # 投影到 d_model
        with torch.no_grad():
            visual = self.visual_proj(features)      # (T, D)
            auditory = self.auditory_proj(diff_features)  # (T, D)
            
            # 均值 → state
            state_raw = features.mean(dim=0, keepdim=True)  # (1, feature_dim)
            current_state = self.state_proj(state_raw)  # (1, D)
        
        # 添加 batch 维度
        return {
            "visual": visual.unsqueeze(0),          # [1, T, D]
            "auditory": auditory.unsqueeze(0),      # [1, T, D]
            "current_state": current_state,         # [1, D]
            "labels": labels
        }
    
    def _read_video_frames(self, video_path: str) -> Optional[torch.Tensor]:
        """读取视频并均匀采样帧"""
        try:
            import cv2
        except ImportError:
            print("[DAiSEE Warning] cv2 not installed, generating synthetic frames")
            return self._generate_synthetic_frames()
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            cap.release()
            return None
            
        # 均匀采样帧索引
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR → RGB, resize to 224x224
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                # 如果读取失败，使用上一帧或零帧
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                    
        cap.release()
        
        # 转为tensor: (T, H, W, C) → (T, C, H, W)
        frames_array = np.array(frames, dtype=np.float32) / 255.0
        frames_tensor = torch.tensor(frames_array, device=self.device).permute(0, 3, 1, 2)
        
        return frames_tensor
    
    def _generate_synthetic_frames(self) -> torch.Tensor:
        """生成合成帧（降级方案）"""
        frames = torch.randn(self.num_frames, 3, 224, 224, device=self.device) * 0.5 + 0.5
        return frames.clamp(0, 1)
    
    def _extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """提取帧特征"""
        # frames: (T, C, H, W)
        with torch.no_grad():
            self.feature_extractor.eval()
            features = self.feature_extractor(frames)  # (T, feature_dim, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (T, feature_dim)
        return features
    
    def _compute_temporal_diff(self, features: torch.Tensor) -> torch.Tensor:
        """计算帧间差分特征"""
        T = features.shape[0]
        
        # 计算相邻帧差分
        diff = torch.zeros_like(features)
        diff[1:] = features[1:] - features[:-1]
        diff[0] = diff[1] if T > 1 else torch.zeros_like(diff[0])
        
        return diff
    
    def _generate_synthetic_data(
        self,
        n_samples: int
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, int]]]:
        """生成合成数据作为降级方案"""
        print(f"[DAiSEE] Generating {n_samples} synthetic samples")
        
        mcs_inputs = []
        labels_list = []
        
        for i in range(n_samples):
            # 随机生成标签 (0-3)
            labels = {
                'Boredom': np.random.randint(0, 4),
                'Engagement': np.random.randint(0, 4),
                'Confusion': np.random.randint(0, 4),
                'Frustration': np.random.randint(0, 4)
            }
            
            # 根据Engagement标签调整特征
            engagement = labels['Engagement']
            base_activity = 0.3 + 0.2 * engagement  # 高参与度 → 高活动
            
            # 生成合成帧特征
            features = torch.randn(self.num_frames, self.feature_dim, device=self.device) * base_activity
            
            # 添加时间模式
            for t in range(self.num_frames):
                features[t] += 0.1 * np.sin(t * np.pi / self.num_frames)
                
            # 计算差分
            diff_features = self._compute_temporal_diff(features)
            
            # 投影
            with torch.no_grad():
                visual = self.visual_proj(features)
                auditory = self.auditory_proj(diff_features)
                state_raw = features.mean(dim=0, keepdim=True)
                current_state = self.state_proj(state_raw)
            
            mcs_input = {
                "visual": visual.unsqueeze(0),
                "auditory": auditory.unsqueeze(0),
                "current_state": current_state,
                "labels": labels
            }
            
            mcs_inputs.append(mcs_input)
            labels_list.append(labels)
            
        return mcs_inputs, labels_list
    
    def collate_batch(
        self,
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """将多个样本合并为批次"""
        # 提取多维标签
        label_tensors = {}
        for name in LABEL_NAMES:
            label_tensors[name] = torch.tensor(
                [b["labels"][name] for b in batch], 
                device=self.device
            )
            
        return {
            "visual": torch.cat([b["visual"] for b in batch], dim=0),
            "auditory": torch.cat([b["auditory"] for b in batch], dim=0),
            "current_state": torch.cat([b["current_state"] for b in batch], dim=0),
            "labels": label_tensors
        }


if __name__ == "__main__":
    print("=" * 60)
    print("DAiSEE 视频参与度适配器测试")
    print("=" * 60)
    
    # 创建适配器
    adapter = DAiSEEAdapter(d_model=D_MODEL, num_frames=8, device=DEVICE)
    print(f"[Test] 适配器创建成功, device={DEVICE}, d_model={D_MODEL}")
    print(f"[Test] 特征维度: {adapter.feature_dim}")
    
    # 加载数据
    mcs_inputs, labels = adapter.load_dataset(max_clips=20, split='Train')
    
    print(f"\n[Test] 加载样本数: {len(mcs_inputs)}")
    
    # 统计标签分布
    for name in LABEL_NAMES:
        values = [l[name] for l in labels]
        print(f"[Test] {name} 分布: {dict(zip(*np.unique(values, return_counts=True)))}")
    
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
        for name in LABEL_NAMES:
            print(f"  labels[{name}]: {batch['labels'][name].shape}")
        
    print("\n" + "=" * 60)
    print("DAiSEE 适配器测试完成!")
    print("=" * 60)
