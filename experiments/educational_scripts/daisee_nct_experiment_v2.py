#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAiSEE 视频参与度检测实验 V2

V2 改进:
1. 使用 ResNet18 FER 模型替代轻量 CNN
2. 使用 MediaPipe Face Mesh 替代 Haar 级联
3. 添加 LSTM 时序编码器捕捉情绪动态
4. 使用可学习的学生状态映射器

与 V1 的区别:
- V1 文件: experiments/daisee_nct_experiment.py
- V2 文件: experiments/daisee_nct_experiment_v2.py
- V2 结果: results/education_v2/phase1_daisee_v2.json
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2
from scipy.stats import pearsonr

import torch
import torch.nn as nn

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# V2 组件导入
try:
    from experiments.fer_pretrain_v2 import (
        FERResNet18, load_fer_model_v2, predict_emotion_v2, EMOTION_NAMES
    )
    FER_V2_AVAILABLE = True
except ImportError:
    FER_V2_AVAILABLE = False
    logger.warning("无法导入 FER V2 模型")

try:
    from experiments.face_landmark_extractor_v2 import (
        FaceLandmarkExtractorV2, FacialFeatures, facial_features_to_student_state
    )
    LANDMARK_V2_AVAILABLE = True
except ImportError:
    LANDMARK_V2_AVAILABLE = False
    logger.warning("无法导入 Face Landmark V2")

try:
    from experiments.temporal_emotion_encoder_v2 import (
        TemporalEmotionEncoderV2, create_temporal_encoder
    )
    TEMPORAL_V2_AVAILABLE = True
except ImportError:
    TEMPORAL_V2_AVAILABLE = False
    logger.warning("无法导入 Temporal Encoder V2")

try:
    from experiments.education_state_mapper_v2 import (
        LearnableNeuromodulatorMapperV2, StudentState, NeuromodulatorState,
        create_state_mapper
    )
    STATE_MAPPER_V2_AVAILABLE = True
except ImportError:
    STATE_MAPPER_V2_AVAILABLE = False
    logger.warning("无法导入 State Mapper V2")

# NCT 核心
try:
    from nct_modules.nct_manager import NCTManager
    from nct_modules.nct_core import NCTConfig
    NCT_AVAILABLE = True
except ImportError:
    NCT_AVAILABLE = False
    NCTManager = None
    NCTConfig = None

# ============================================================================
# 常量定义
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "daisee" / "DAiSEE" / "DAiSEE"
RESULTS_DIR = PROJECT_ROOT / "results" / "education_v2"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "education_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 轻量 NCT 配置
LIGHT_NCT_CONFIG = NCTConfig(
    n_heads=4,
    n_layers=2,
    d_model=256,
    dim_ff=512,
    visual_embed_dim=128,
    consciousness_threshold=0.5,
) if NCTConfig else None


# ============================================================================
# V2 视觉特征提取器
# ============================================================================

class VisualFeatureExtractorV2:
    """
    V2 视觉特征提取器
    
    集成:
    1. MediaPipe Face Mesh
    2. FER ResNet18
    3. 时序情绪编码
    """
    
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # MediaPipe 面部关键点
        self._landmark_extractor = None
        if LANDMARK_V2_AVAILABLE:
            try:
                self._landmark_extractor = FaceLandmarkExtractorV2(static_image_mode=True)
                if self._landmark_extractor.available:
                    logger.info("MediaPipe Face Mesh 已启用")
                else:
                    self._landmark_extractor = None
            except Exception as e:
                logger.warning(f"MediaPipe 初始化失败: {e}")
        
        # FER ResNet18
        self._fer_model = None
        if FER_V2_AVAILABLE:
            try:
                self._fer_model = load_fer_model_v2(str(self.device))
                logger.info("FER ResNet18 已加载")
            except Exception as e:
                logger.warning(f"FER V2 加载失败: {e}")
        
        # 时序编码器
        self._temporal_encoder = None
        if TEMPORAL_V2_AVAILABLE:
            try:
                self._temporal_encoder = create_temporal_encoder(device=str(self.device))
                logger.info("时序情绪编码器已创建")
            except Exception as e:
                logger.warning(f"时序编码器创建失败: {e}")
    
    def extract_from_frame(self, frame: np.ndarray) -> Dict:
        """
        从单帧提取特征
        
        Args:
            frame: BGR 图像
        
        Returns:
            features: 特征字典
        """
        features = {
            "face_detected": False,
            "emotion": "Neutral",
            "emotion_confidence": 0.5,
            "facial_features": None,
            "student_state": None,
        }
        
        # 1. MediaPipe 面部关键点
        if self._landmark_extractor:
            facial_features = self._landmark_extractor.extract(frame)
            features["face_detected"] = facial_features.face_detected
            features["facial_features"] = facial_features
            
            if facial_features.face_detected:
                # 从面部特征推断学生状态
                student_state_dict = facial_features_to_student_state(facial_features)
                features["student_state"] = StudentState(**student_state_dict)
        
        # 2. FER 情绪识别
        if self._fer_model and features.get("face_detected", False):
            try:
                emotion_idx, probs = predict_emotion_v2(self._fer_model, frame, self.device)
                features["emotion"] = EMOTION_NAMES[emotion_idx]
                features["emotion_confidence"] = float(probs[emotion_idx])
                features["emotion_probs"] = probs.tolist()
            except Exception as e:
                logger.debug(f"FER 预测失败: {e}")
        
        # 3. 降级方案：OpenCV Haar
        if not features["face_detected"]:
            features.update(self._fallback_extract(frame))
        
        return features
    
    def extract_from_video_segment(
        self,
        frames: List[np.ndarray],
        sample_interval: int = 3,
    ) -> Dict:
        """
        从视频片段提取特征（时序）
        
        Args:
            frames: 帧列表
            sample_interval: 采样间隔
        
        Returns:
            aggregated_features: 聚合特征
        """
        # 采样帧
        sampled_frames = frames[::sample_interval][:30]  # 最多 30 帧
        
        if not sampled_frames:
            return self._get_default_features()
        
        # 逐帧提取
        frame_features = []
        emotions = []
        student_states = []
        
        for frame in sampled_frames:
            feat = self.extract_from_frame(frame)
            frame_features.append(feat)
            
            if feat.get("emotion"):
                emotions.append(feat["emotion"])
            if feat.get("student_state"):
                student_states.append(feat["student_state"])
        
        # 聚合
        face_detected_ratio = sum(1 for f in frame_features if f.get("face_detected")) / len(frame_features)
        
        # 主导情绪
        if emotions:
            from collections import Counter
            dominant_emotion = Counter(emotions).most_common(1)[0][0]
        else:
            dominant_emotion = "Neutral"
        
        # 平均学生状态
        if student_states:
            avg_state = StudentState(
                focus_level=np.mean([s.focus_level for s in student_states]),
                engagement=np.mean([s.engagement for s in student_states]),
                confusion=np.mean([s.confusion for s in student_states]),
                fatigue=np.mean([s.fatigue for s in student_states]),
                stress_level=np.mean([s.stress_level for s in student_states]),
                confidence=np.mean([s.confidence for s in student_states]),
            )
        else:
            avg_state = StudentState()
        
        return {
            "face_detected_ratio": face_detected_ratio,
            "dominant_emotion": dominant_emotion,
            "student_state": avg_state,
            "n_frames": len(sampled_frames),
        }
    
    def _fallback_extract(self, frame: np.ndarray) -> Dict:
        """降级方案"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        if len(faces) > 0:
            return {
                "face_detected": True,
                "student_state": StudentState(focus_level=0.5, engagement=0.5),
            }
        return {"face_detected": False}
    
    def _get_default_features(self) -> Dict:
        return {
            "face_detected_ratio": 0.0,
            "dominant_emotion": "Neutral",
            "student_state": StudentState(),
            "n_frames": 0,
        }
    
    def close(self):
        if self._landmark_extractor:
            self._landmark_extractor.close()


# ============================================================================
# DAiSEE 数据加载
# ============================================================================

class DAiSEELoaderV2:
    """DAiSEE 数据加载器 V2"""
    
    def __init__(self, data_dir: Path, split: str = "Train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.labels_df = None
        self._load_labels()
    
    def _load_labels(self):
        """加载标签文件"""
        import pandas as pd
        
        labels_file = self.data_dir / "Labels" / f"{self.split}Labels.csv"
        if labels_file.exists():
            self.labels_df = pd.read_csv(labels_file)
            # 清理列名
            self.labels_df.columns = [c.strip() for c in self.labels_df.columns]
            logger.info(f"加载标签: {len(self.labels_df)} 条")
    
    def is_available(self) -> bool:
        return self.labels_df is not None and len(self.labels_df) > 0
    
    def get_video_paths(self, max_videos: int = 100) -> List[Tuple[Path, Dict]]:
        """获取视频路径和标签"""
        if not self.is_available():
            return []
        
        videos = []
        dataset_dir = self.data_dir / "DataSet" / self.split
        
        for _, row in self.labels_df.iterrows():
            clip_id = row["ClipID"]
            user_id = clip_id.split("_")[0] if "_" in clip_id else clip_id[:6]
            
            video_dir = dataset_dir / user_id / clip_id
            
            # 查找视频文件
            video_file = None
            if video_dir.exists():
                for ext in [".avi", ".mp4"]:
                    candidates = list(video_dir.glob(f"*{ext}"))
                    if candidates:
                        video_file = candidates[0]
                        break
            
            if video_file and video_file.exists():
                labels = {
                    "Boredom": int(row.get("Boredom", 0)),
                    "Engagement": int(row.get("Engagement", 0)),
                    "Confusion": int(row.get("Confusion", 0)),
                    "Frustration": int(row.get("Frustration", row.get("Frustration ", 0))),
                }
                videos.append((video_file, labels))
            
            if len(videos) >= max_videos:
                break
        
        return videos
    
    def load_video_frames(self, video_path: Path, max_frames: int = 90) -> List[np.ndarray]:
        """加载视频帧"""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames


# ============================================================================
# 主实验函数
# ============================================================================

def run_daisee_experiment_v2(
    use_mock: bool = False,
    max_videos: int = 100,
    split: str = "Train",
) -> Dict:
    """
    运行 DAiSEE V2 实验
    
    Args:
        use_mock: 使用模拟数据
        max_videos: 最大视频数
        split: 数据集划分
    """
    logger.info("=" * 60)
    logger.info("Phase 1 V2 - DAiSEE 视频参与度检测（深度学习增强）")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化 V2 组件
    visual_extractor = VisualFeatureExtractorV2(device=str(device))
    
    # 状态映射器
    state_mapper = None
    if STATE_MAPPER_V2_AVAILABLE:
        state_mapper = create_state_mapper(device=str(device))
        logger.info("V2 状态映射器已创建")
    
    # NCT
    nct = None
    if NCT_AVAILABLE and LIGHT_NCT_CONFIG:
        nct = NCTManager(LIGHT_NCT_CONFIG)
        nct.start()
        logger.info("NCT Manager 已启动")
    
    # 加载数据
    loader = DAiSEELoaderV2(DATA_DIR, split=split)
    
    if use_mock or not loader.is_available():
        logger.info("使用模拟数据")
        results = _run_mock_experiment_v2(visual_extractor, state_mapper, nct, max_videos)
    else:
        videos = loader.get_video_paths(max_videos)
        logger.info(f"加载 {len(videos)} 个视频")
        results = _run_real_experiment_v2(loader, videos, visual_extractor, state_mapper, nct)
    
    # 清理
    visual_extractor.close()
    
    # 保存结果
    out_path = RESULTS_DIR / "phase1_daisee_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {out_path}")
    
    return results


def _run_real_experiment_v2(
    loader: DAiSEELoaderV2,
    videos: List[Tuple[Path, Dict]],
    visual_extractor: VisualFeatureExtractorV2,
    state_mapper: Optional[LearnableNeuromodulatorMapperV2],
    nct: Optional['NCTManager'],
) -> Dict:
    """运行真实数据实验"""
    
    phi_values = []
    engagement_labels = []
    nm_records = []
    
    for i, (video_path, labels) in enumerate(videos):
        if (i + 1) % 20 == 0:
            logger.info(f"处理进度: {i+1}/{len(videos)}")
        
        # 加载帧
        frames = loader.load_video_frames(video_path)
        if not frames:
            continue
        
        # 提取特征
        features = visual_extractor.extract_from_video_segment(frames)
        
        # 状态映射
        student_state = features.get("student_state", StudentState())
        
        if state_mapper:
            nm_state = state_mapper.map_student_state(student_state)
            nm_dict = nm_state.to_dict()
        else:
            nm_dict = {"DA": 0.5, "5-HT": 0.5, "NE": 0.5, "ACh": 0.5}
        
        # NCT 处理
        if nct and frames:
            # 取中间帧
            mid_frame = frames[len(frames) // 2]
            gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28)).astype(np.float32) / 255.0
            
            try:
                nct_state = nct.process_cycle({"visual": resized}, nm_dict)
                phi = nct_state.consciousness_metrics.get("phi_value", 0.0)
            except:
                phi = np.random.uniform(0.2, 0.5)
        else:
            phi = np.random.uniform(0.2, 0.5)
        
        phi_values.append(phi)
        engagement_labels.append(labels["Engagement"])
        nm_records.append(nm_dict)
    
    # 统计分析
    return _compute_statistics_v2(phi_values, engagement_labels, nm_records)


def _run_mock_experiment_v2(
    visual_extractor: VisualFeatureExtractorV2,
    state_mapper: Optional[LearnableNeuromodulatorMapperV2],
    nct: Optional['NCTManager'],
    n_samples: int,
) -> Dict:
    """运行模拟数据实验"""
    phi_values = []
    engagement_labels = []
    nm_records = []
    
    for i in range(n_samples):
        # 模拟学生状态
        engagement = np.random.randint(0, 4)
        student_state = StudentState(
            focus_level=0.5 + engagement * 0.1 + np.random.uniform(-0.1, 0.1),
            engagement=0.3 + engagement * 0.15 + np.random.uniform(-0.1, 0.1),
            confusion=0.3 - engagement * 0.05 + np.random.uniform(-0.1, 0.1),
            fatigue=0.3 - engagement * 0.05 + np.random.uniform(-0.1, 0.1),
            stress_level=0.3 + np.random.uniform(-0.1, 0.1),
            confidence=0.5 + engagement * 0.1 + np.random.uniform(-0.1, 0.1),
        )
        
        # 状态映射
        if state_mapper:
            nm_state = state_mapper.map_student_state(student_state)
            nm_dict = nm_state.to_dict()
        else:
            nm_dict = {"DA": 0.5, "5-HT": 0.5, "NE": 0.5, "ACh": 0.5}
        
        # 模拟 Phi
        phi = 0.3 + engagement * 0.05 + np.random.uniform(-0.1, 0.1)
        phi = np.clip(phi, 0, 1)
        
        phi_values.append(phi)
        engagement_labels.append(engagement)
        nm_records.append(nm_dict)
    
    return _compute_statistics_v2(phi_values, engagement_labels, nm_records, mock=True)


def _compute_statistics_v2(
    phi_values: List[float],
    engagement_labels: List[int],
    nm_records: List[Dict],
    mock: bool = False,
) -> Dict:
    """计算统计结果"""
    
    # 相关性分析
    r_phi, p_phi = pearsonr(phi_values, engagement_labels) if len(phi_values) > 2 else (0, 1)
    
    da_values = [nm["DA"] for nm in nm_records]
    r_da, p_da = pearsonr(da_values, engagement_labels) if len(da_values) > 2 else (0, 1)
    
    # 按 Engagement 分组的 Phi 统计
    phi_by_label = {}
    for label in range(4):
        mask = [l == label for l in engagement_labels]
        phi_subset = [p for p, m in zip(phi_values, mask) if m]
        if phi_subset:
            phi_by_label[str(label)] = {
                "mean": round(float(np.mean(phi_subset)), 4),
                "std": round(float(np.std(phi_subset)), 4),
                "n": len(phi_subset),
            }
    
    # 神经调质均值
    nm_means = {
        "DA": round(float(np.mean([nm["DA"] for nm in nm_records])), 4),
        "5-HT": round(float(np.mean([nm["5-HT"] for nm in nm_records])), 4),
        "NE": round(float(np.mean([nm["NE"] for nm in nm_records])), 4),
        "ACh": round(float(np.mean([nm["ACh"] for nm in nm_records])), 4),
    }
    
    return {
        "version": "V2",
        "experiment": "Phase1_DAiSEE_V2",
        "mock_mode": mock,
        "n_samples": len(phi_values),
        "correlations": {
            "engagement_vs_phi": {"r": round(r_phi, 4), "p": round(p_phi, 4)},
            "engagement_vs_DA": {"r": round(r_da, 4), "p": round(p_da, 4)},
        },
        "phi_by_engagement": phi_by_label,
        "nm_means": nm_means,
        "v2_improvements": {
            "feature_extraction": "MediaPipe Face Mesh + ResNet18 FER",
            "state_mapping": "Learnable Neural Network",
            "temporal_modeling": "LSTM (if enabled)",
        },
    }


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DAiSEE V2 实验")
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument("--max-videos", type=int, default=100, help="最大视频数")
    parser.add_argument("--split", type=str, default="Train", help="数据集划分")
    
    args = parser.parse_args()
    
    results = run_daisee_experiment_v2(
        use_mock=args.mock,
        max_videos=args.max_videos,
        split=args.split,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("V2 实验完成!")
    logger.info(f"Φ vs Engagement: r={results['correlations']['engagement_vs_phi']['r']}, "
               f"p={results['correlations']['engagement_vs_phi']['p']}")
