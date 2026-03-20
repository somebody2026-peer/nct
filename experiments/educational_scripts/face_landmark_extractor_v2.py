#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面部关键点提取器 V2 - MediaPipe Face Mesh

V2 改进:
1. 使用 MediaPipe Face Mesh 提取 478 个 3D 面部关键点
2. 计算面部动作单元 (AU) 特征
3. 支持头部姿态估计
4. 实时性能优化

与 V1 的区别:
- V1: OpenCV Haar 级联（仅人脸/眼睛框）
- V2: MediaPipe Face Mesh 478 个精确关键点 + AU 特征
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 关键点索引定义（MediaPipe Face Mesh 标准）
# ============================================================================

# 眼睛关键点
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# 眉毛关键点
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# 嘴唇关键点
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_LIP_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# 鼻子关键点
NOSE_INDICES = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4]

# 面部轮廓关键点
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class FacialFeatures:
    """面部特征数据类"""
    # 基础检测
    face_detected: bool = False
    confidence: float = 0.0
    
    # 眼睛特征
    left_ear: float = 0.3       # 左眼宽高比 (Eye Aspect Ratio)
    right_ear: float = 0.3      # 右眼宽高比
    avg_ear: float = 0.3        # 平均 EAR
    
    # 眉毛特征
    left_brow_height: float = 0.0   # 左眉高度
    right_brow_height: float = 0.0  # 右眉高度
    brow_furrow: float = 0.0        # 眉间距（皱眉程度）
    
    # 嘴部特征
    mouth_aspect_ratio: float = 0.0  # 嘴巴宽高比
    mouth_open: float = 0.0          # 张嘴程度
    smile_intensity: float = 0.0     # 微笑强度
    
    # 头部姿态
    head_pitch: float = 0.0  # 俯仰角（点头）
    head_yaw: float = 0.0    # 偏航角（摇头）
    head_roll: float = 0.0   # 翻滚角（歪头）
    
    # AU 特征（动作单元）
    AU1: float = 0.0   # Inner Brow Raise（专注）
    AU2: float = 0.0   # Outer Brow Raise（惊讶）
    AU4: float = 0.0   # Brow Lowerer（困惑/愤怒）
    AU6: float = 0.0   # Cheek Raise（真笑）
    AU12: float = 0.0  # Lip Corner Pull（微笑）
    AU15: float = 0.0  # Lip Corner Depress（悲伤）
    AU25: float = 0.0  # Lips Part（嘴张开）
    AU43: float = 0.0  # Eye Closure（闭眼/疲劳）
    
    # 原始关键点（可选）
    landmarks: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "face_detected": self.face_detected,
            "confidence": self.confidence,
            "ear": {"left": self.left_ear, "right": self.right_ear, "avg": self.avg_ear},
            "brow": {"left_height": self.left_brow_height, "right_height": self.right_brow_height, "furrow": self.brow_furrow},
            "mouth": {"aspect_ratio": self.mouth_aspect_ratio, "open": self.mouth_open, "smile": self.smile_intensity},
            "head_pose": {"pitch": self.head_pitch, "yaw": self.head_yaw, "roll": self.head_roll},
            "AU": {"AU1": self.AU1, "AU2": self.AU2, "AU4": self.AU4, "AU6": self.AU6, 
                   "AU12": self.AU12, "AU15": self.AU15, "AU25": self.AU25, "AU43": self.AU43}
        }
    
    def to_vector(self) -> np.ndarray:
        """转为特征向量（用于神经网络输入）"""
        return np.array([
            self.avg_ear, self.brow_furrow, self.mouth_open, self.smile_intensity,
            self.head_pitch, self.head_yaw, self.head_roll,
            self.AU1, self.AU2, self.AU4, self.AU6, self.AU12, self.AU15, self.AU25, self.AU43
        ], dtype=np.float32)


class FaceLandmarkExtractorV2:
    """
    MediaPipe Face Mesh 面部关键点提取器 V2
    
    特点:
    - 478 个 3D 面部关键点
    - 实时性能（CPU 可达 30+ FPS）
    - 自动计算 AU 特征
    - 头部姿态估计
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        初始化 MediaPipe Face Mesh
        
        Args:
            static_image_mode: True 用于静态图像，False 用于视频流
            max_num_faces: 最大检测人脸数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self._available = False
        self._face_mesh = None
        
        try:
            import mediapipe as mp
            self._mp = mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=True,  # 启用虹膜关键点
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._available = True
            logger.info("MediaPipe Face Mesh 初始化成功")
        except ImportError:
            logger.warning("MediaPipe 未安装，将使用降级方案。请安装: pip install mediapipe")
        except Exception as e:
            logger.warning(f"MediaPipe 初始化失败: {e}")
    
    @property
    def available(self) -> bool:
        return self._available
    
    def extract(self, frame: np.ndarray) -> FacialFeatures:
        """
        从图像帧提取面部特征
        
        Args:
            frame: BGR 图像 [H, W, 3]
        
        Returns:
            FacialFeatures 对象
        """
        features = FacialFeatures()
        
        if not self._available:
            return self._fallback_extract(frame)
        
        try:
            import cv2
            
            # MediaPipe 需要 RGB 输入
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return features
            
            # 获取第一张人脸的关键点
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # 转换为 numpy 数组 [478, 3]
            landmarks = np.array([
                [lm.x * w, lm.y * h, lm.z * w]
                for lm in face_landmarks.landmark
            ])
            
            features.face_detected = True
            features.confidence = 1.0  # MediaPipe 不直接返回置信度
            features.landmarks = landmarks
            
            # 计算各项特征
            features.left_ear = self._compute_ear(landmarks, LEFT_EYE_INDICES)
            features.right_ear = self._compute_ear(landmarks, RIGHT_EYE_INDICES)
            features.avg_ear = (features.left_ear + features.right_ear) / 2
            
            features.left_brow_height, features.right_brow_height, features.brow_furrow = \
                self._compute_brow_features(landmarks)
            
            features.mouth_aspect_ratio, features.mouth_open, features.smile_intensity = \
                self._compute_mouth_features(landmarks)
            
            features.head_pitch, features.head_yaw, features.head_roll = \
                self._compute_head_pose(landmarks)
            
            # 计算 AU 特征
            self._compute_au_features(features, landmarks)
            
        except Exception as e:
            logger.debug(f"特征提取异常: {e}")
        
        return features
    
    def _compute_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """计算眼睛宽高比 (EAR)"""
        try:
            pts = landmarks[eye_indices]
            
            # 计算垂直距离（上下眼睑）
            vertical_1 = np.linalg.norm(pts[1] - pts[5])
            vertical_2 = np.linalg.norm(pts[2] - pts[4])
            
            # 计算水平距离（眼角）
            horizontal = np.linalg.norm(pts[0] - pts[3])
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
            return float(np.clip(ear, 0.1, 0.5))
        except:
            return 0.3
    
    def _compute_brow_features(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """计算眉毛特征"""
        try:
            # 眉毛中心高度（相对于眼睛）
            left_brow_center = landmarks[LEFT_EYEBROW_INDICES].mean(axis=0)
            right_brow_center = landmarks[RIGHT_EYEBROW_INDICES].mean(axis=0)
            left_eye_center = landmarks[LEFT_EYE_INDICES].mean(axis=0)
            right_eye_center = landmarks[RIGHT_EYE_INDICES].mean(axis=0)
            
            left_height = float(left_eye_center[1] - left_brow_center[1])
            right_height = float(right_eye_center[1] - right_brow_center[1])
            
            # 眉间距（皱眉程度）
            inner_left = landmarks[107]  # 左眉内侧
            inner_right = landmarks[336]  # 右眉内侧
            furrow = float(np.linalg.norm(inner_left[:2] - inner_right[:2]))
            
            # 归一化
            face_width = landmarks[FACE_OVAL_INDICES].ptp(axis=0)[0]
            left_height = np.clip(left_height / (face_width + 1e-6), 0, 1)
            right_height = np.clip(right_height / (face_width + 1e-6), 0, 1)
            furrow = np.clip(furrow / (face_width + 1e-6), 0, 1)
            
            return left_height, right_height, furrow
        except:
            return 0.0, 0.0, 0.0
    
    def _compute_mouth_features(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """计算嘴部特征"""
        try:
            # 嘴巴宽度和高度
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            top_lip = landmarks[13]
            bottom_lip = landmarks[14]
            
            width = np.linalg.norm(left_corner[:2] - right_corner[:2])
            height = np.linalg.norm(top_lip[:2] - bottom_lip[:2])
            
            mar = height / (width + 1e-6)
            mouth_open = float(np.clip(mar / 0.5, 0, 1))  # 归一化
            
            # 微笑强度（嘴角上扬程度）
            mouth_center_y = (top_lip[1] + bottom_lip[1]) / 2
            avg_corner_y = (left_corner[1] + right_corner[1]) / 2
            smile = float(np.clip((mouth_center_y - avg_corner_y) / 20, 0, 1))
            
            return float(mar), mouth_open, smile
        except:
            return 0.0, 0.0, 0.0
    
    def _compute_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """计算头部姿态（简化版）"""
        try:
            # 使用鼻尖和面部轮廓点估计
            nose_tip = landmarks[1]
            left_face = landmarks[234]
            right_face = landmarks[454]
            forehead = landmarks[10]
            chin = landmarks[152]
            
            # 偏航角（左右）
            face_center_x = (left_face[0] + right_face[0]) / 2
            yaw = float((nose_tip[0] - face_center_x) / (right_face[0] - left_face[0] + 1e-6))
            
            # 俯仰角（上下）
            face_height = chin[1] - forehead[1]
            pitch = float((nose_tip[1] - (forehead[1] + face_height * 0.4)) / (face_height + 1e-6))
            
            # 翻滚角（歪头）
            roll = float(np.arctan2(right_face[1] - left_face[1], 
                                    right_face[0] - left_face[0]) * 180 / np.pi)
            roll = np.clip(roll / 45, -1, 1)
            
            return float(np.clip(pitch, -1, 1)), float(np.clip(yaw, -1, 1)), roll
        except:
            return 0.0, 0.0, 0.0
    
    def _compute_au_features(self, features: FacialFeatures, landmarks: np.ndarray):
        """计算 AU (Action Unit) 特征"""
        try:
            # AU1: Inner Brow Raise（专注/担忧）
            # 眉毛内侧抬起
            features.AU1 = float(np.clip(features.left_brow_height + features.right_brow_height, 0, 1))
            
            # AU2: Outer Brow Raise（惊讶）
            # 眉毛外侧抬起
            outer_left = landmarks[46]
            outer_right = landmarks[276]
            eye_left = landmarks[LEFT_EYE_INDICES].mean(axis=0)
            eye_right = landmarks[RIGHT_EYE_INDICES].mean(axis=0)
            features.AU2 = float(np.clip((eye_left[1] - outer_left[1] + eye_right[1] - outer_right[1]) / 40, 0, 1))
            
            # AU4: Brow Lowerer（困惑/愤怒）
            # 眉间距减少
            features.AU4 = float(np.clip(1.0 - features.brow_furrow * 5, 0, 1))
            
            # AU6: Cheek Raise（真笑）
            # 这里简化为与微笑强度相关
            features.AU6 = features.smile_intensity * 0.8
            
            # AU12: Lip Corner Pull（微笑）
            features.AU12 = features.smile_intensity
            
            # AU15: Lip Corner Depress（悲伤）
            features.AU15 = float(np.clip(1.0 - features.smile_intensity, 0, 1)) * 0.5
            
            # AU25: Lips Part（嘴张开）
            features.AU25 = features.mouth_open
            
            # AU43: Eye Closure（闭眼/疲劳）
            features.AU43 = float(np.clip((0.3 - features.avg_ear) / 0.15, 0, 1))
            
        except Exception as e:
            logger.debug(f"AU 计算异常: {e}")
    
    def _fallback_extract(self, frame: np.ndarray) -> FacialFeatures:
        """降级方案：使用 OpenCV Haar 级联"""
        features = FacialFeatures()
        
        try:
            import cv2
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                features.face_detected = True
                features.confidence = 0.7
                
                # 眼睛检测
                face_roi = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi)
                if len(eyes) >= 2:
                    ear = np.mean([ey[3] / (ey[2] + 1e-6) for ey in eyes[:2]])
                    features.avg_ear = float(np.clip(ear, 0.1, 0.5))
                    features.left_ear = features.avg_ear
                    features.right_ear = features.avg_ear
                
                # 头部偏航
                face_cx = x + w / 2
                img_cx = frame.shape[1] / 2
                features.head_yaw = float((face_cx - img_cx) / (img_cx + 1e-6))
                
                # 简单的 AU 估计
                features.AU43 = float(np.clip((0.3 - features.avg_ear) / 0.15, 0, 1))
                
        except Exception as e:
            logger.debug(f"降级方案异常: {e}")
        
        return features
    
    def close(self):
        """释放资源"""
        if self._face_mesh:
            self._face_mesh.close()


# ============================================================================
# 便捷函数
# ============================================================================

def extract_facial_features(frame: np.ndarray) -> FacialFeatures:
    """
    便捷函数：提取面部特征
    
    Args:
        frame: BGR 图像
    
    Returns:
        FacialFeatures 对象
    """
    extractor = FaceLandmarkExtractorV2(static_image_mode=True)
    features = extractor.extract(frame)
    extractor.close()
    return features


def facial_features_to_student_state(features: FacialFeatures) -> Dict[str, float]:
    """
    将面部特征映射到学生状态
    
    Args:
        features: 面部特征
    
    Returns:
        学生状态字典
    """
    if not features.face_detected:
        return {
            "focus_level": 0.5,
            "engagement": 0.5,
            "confusion": 0.2,
            "fatigue": 0.3,
            "stress_level": 0.2,
            "confidence": 0.5,
        }
    
    # 基于 AU 和其他特征计算学生状态
    focus = np.clip(0.5 + features.AU1 * 0.3 - features.AU43 * 0.4 - abs(features.head_yaw) * 0.3, 0, 1)
    engagement = np.clip(0.5 + features.smile_intensity * 0.3 + features.AU6 * 0.2 - features.AU43 * 0.3, 0, 1)
    confusion = np.clip(features.AU4 * 0.6 + features.brow_furrow * 0.4, 0, 1)
    fatigue = np.clip(features.AU43 * 0.7 + (0.3 - features.avg_ear) * 2, 0, 1)
    stress = np.clip(features.AU4 * 0.4 + features.AU15 * 0.3 + abs(features.head_roll) * 0.3, 0, 1)
    confidence = np.clip(0.5 + features.smile_intensity * 0.3 - confusion * 0.2, 0, 1)
    
    return {
        "focus_level": float(focus),
        "engagement": float(engagement),
        "confusion": float(confusion),
        "fatigue": float(fatigue),
        "stress_level": float(stress),
        "confidence": float(confidence),
    }


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    import cv2
    
    logger.info("=" * 60)
    logger.info("Face Landmark Extractor V2 测试")
    logger.info("=" * 60)
    
    # 创建提取器
    extractor = FaceLandmarkExtractorV2(static_image_mode=True)
    
    if extractor.available:
        logger.info("MediaPipe Face Mesh 可用")
        
        # 测试图像（创建一个空白图像）
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        features = extractor.extract(test_frame)
        
        logger.info(f"测试结果: face_detected={features.face_detected}")
        logger.info(f"特征向量维度: {features.to_vector().shape}")
    else:
        logger.warning("MediaPipe 不可用，请安装: pip install mediapipe")
    
    extractor.close()