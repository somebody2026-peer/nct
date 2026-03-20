"""
教育场景学生状态智能检测系统
==============================
基于多模态感知的学生状态评估与 NCT 参数映射

核心功能：
1. 多模态数据采集（视觉、行为、交互）
2. 学生状态识别（专注度、困惑度、疲劳度、参与度）
3. 映射到 NCT 神经调质参数（DA/5-HT/NE/ACh）
4. 自适应学习策略生成

Author: NeuroConscious Lab
Date: 2026-03-12
"""

import torch
import numpy as np
import cv2
try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    dlib = None
    _DLIB_AVAILABLE = False
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import time
from datetime import datetime


# ==================== 数据结构定义 ====================

@dataclass
class StudentState:
    """学生状态数据类"""
    timestamp: float
    focus_level: float      # 专注度(0-1)
    engagement: float       # 参与度(0-1)
    confusion: float        # 困惑度(0-1)
    fatigue: float          # 疲劳度(0-1)
    stress_level: float     # 压力水平 (0-1)
    confidence: float       # 自信度(0-1)
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'focus_level': self.focus_level,
            'engagement': self.engagement,
            'confusion': self.confusion,
            'fatigue': self.fatigue,
            'stress_level': self.stress_level,
            'confidence': self.confidence,
        }


@dataclass
class NeuromodulatorState:
    """神经调质状态数据类"""
    DA: float    # 多巴胺
    _5_HT: float # 血清素
    NE: float    # 去甲肾上腺素
    ACh: float   # 乙酰胆碱
    
    def to_dict(self):
        return {
            'DA': self.DA,
            '5-HT': self._5_HT,
            'NE': self.NE,
            'ACh': self.ACh,
        }


# ==================== 多模态数据采集 ====================

class MultiModalSensor:
    """
    多模态传感器数据采集中枢
    
    采集通道：
    1. 视觉通道：摄像头（表情、gaze、头部姿态）
    2. 行为通道：鼠标/键盘行为、答题行为
    3. 生理通道（可选）：心率、皮电反应
    4. 交互通道：答题正确率、响应时间
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        
        # 面部特征点检测器（使用 dlib）
        try:
            if _DLIB_AVAILABLE and dlib is not None:
                self.face_detector = dlib.get_frontal_face_detector()
                self.landmark_predictor = dlib.shape_predictor(
                    'shape_predictor_68_face_landmarks.dat'
                )
                self.face_tracking_enabled = True
            else:
                raise ImportError("dlib 未安装")
        except Exception as e:
            print(f"⚠️  面部追踪未启用：{e}")
            self.face_tracking_enabled = False
        
        # 眼动追踪（简化版，基于瞳孔位置）
        self.eye_tracking_enabled = True
        
        # 数据缓冲区（用于时序分析）
        self.buffer_size = 30  # 30 帧 ≈ 1 秒
        self.data_buffer = {
            'gaze_direction': [],
            'blink_rate': [],
            'head_pose': [],
            'facial_expression': [],
            'interaction_log': [],
        }
    
    def capture_visual_data(self, frame: np.ndarray) -> Dict:
        """
        从摄像头帧提取视觉特征
        
        Args:
            frame: BGR 图像帧
            
        Returns:
            visual_features: 视觉特征字典
        """
        features = {}
        
        if self.face_tracking_enabled:
            # 1. 面部关键点检测
            landmarks = self.detect_facial_landmarks(frame)
            
            if landmarks is not None:
                # 2. 眼睛纵横比（EAR）- 检测眨眼
                left_ear = self.calculate_eye_aspect_ratio(landmarks, 'left')
                right_ear = self.calculate_eye_aspect_ratio(landmarks, 'right')
                features['eye_aspect_ratio'] = (left_ear + right_ear) / 2
                
                # 3. 嘴巴张开度 - 检测惊讶/困惑
                features['mouth_opening'] = self.calculate_mouth_opening(landmarks)
                
                # 4. 眉毛倾斜度 - 检测困惑
                features['eyebrow_slope'] = self.calculate_eyebrow_slope(landmarks)
                
                # 5. 头部姿态（简化版）
                features['head_pose'] = self.estimate_head_pose(landmarks)
        
        # 6. Gaze 方向（简化版）
        if self.eye_tracking_enabled:
            features['gaze_direction'] = self.estimate_gaze_direction(frame)
        
        # 添加到缓冲区
        self._update_buffer(features)
        
        return features
    
    def capture_behavioral_data(self, interaction_events: List[Dict]) -> Dict:
        """
        捕获行为数据
        
        Args:
            interaction_events: 交互事件列表
                [
                    {'type': 'click', 'timestamp': t, 'position': (x,y)},
                    {'type': 'answer', 'timestamp': t, 'correct': True, 'response_time': 2.5},
                    ...
                ]
        """
        features = {}
        
        if len(interaction_events) > 0:
            # 1. 响应时间统计
            response_times = [
                e['response_time'] for e in interaction_events 
                if 'response_time' in e
            ]
            features['avg_response_time'] = np.mean(response_times) if response_times else 0
            features['response_time_std'] = np.std(response_times) if response_times else 0
            
            # 2. 正确率
            correct_count = sum(
                1 for e in interaction_events 
                if e.get('correct', False)
            )
            features['accuracy'] = correct_count / len(interaction_events)
            
            # 3. 交互频率
            features['interaction_rate'] = len(interaction_events) / 60.0  # 假设 60 秒窗口
        
        return features
    
    def detect_facial_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测面部 68 个关键点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return None
        
        face = faces[0]  # 假设只有一个人脸
        landmarks = self.landmark_predictor(gray, face)
        
        points = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)
        
        return points
    
    def calculate_eye_aspect_ratio(self, landmarks: np.ndarray, eye: str) -> float:
        """
        计算眼睛纵横比（EAR）
        EAR 降低 → 闭眼 → 可能疲劳或眨眼
        """
        if eye == 'left':
            indices = [36, 37, 38, 39, 40, 41]
        else:  # right
            indices = [42, 43, 44, 45, 46, 47]
        
        eye_points = landmarks[indices]
        
        # 垂直距离
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # 水平距离
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_mouth_opening(self, landmarks: np.ndarray) -> float:
        """计算嘴巴张开度"""
        mouth_top = landmarks[58]
        mouth_bottom = landmarks[66]
        
        opening = np.linalg.norm(mouth_top - mouth_bottom)
        return opening
    
    def calculate_eyebrow_slope(self, landmarks: np.ndarray) -> float:
        #计算眉毛倾斜度（皱眉检测）
        left_eyebrow_inner = landmarks[21]
        left_eyebrow_outer = landmarks[17]
        
        right_eyebrow_inner = landmarks[22]
        right_eyebrow_outer = landmarks[26]
        
        # 眉毛倾斜角度
        left_slope = (left_eyebrow_inner[1] - left_eyebrow_outer[1]) / \
                     (left_eyebrow_inner[0] - left_eyebrow_outer[0] + 1e-6)
        right_slope = (right_eyebrow_inner[1] - right_eyebrow_outer[1]) / \
                      (right_eyebrow_inner[0] - right_eyebrow_outer[0] + 1e-6)
        
        # 皱眉时斜率会变化
        slope_diff = abs(left_slope - right_slope)
        return slope_diff
        left_eyebrow_inner = landmarks[21]
        left_eyebrow_outer = landmarks[17]
        
        right_eyebrow_inner = landmarks[22]
        right_eyebrow_outer = landmarks[26]
        
        # 眉毛倾斜角度
        left_slope = (left_eyebrow_inner[1] - left_eyebrow_outer[1]) / \
                     (left_eyebrow_inner[0] - left_eyebrow_outer[0] + 1e-6)
        right_slope = (right_eyebrow_inner[1] - right_eyebrow_outer[1]) / \
                      (right_eyebrow_inner[0] - right_eyebrow_outer[0] + 1e-6)
        
        # 皱眉时斜率会变化
        slope_diff = abs(left_slope - right_slope)
        return slope_diff
    
    def estimate_head_pose(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        估计头部姿态（简化版）
        返回：pitch, yaw, roll
        """
        nose_tip = landmarks[30]
        chin = landmarks[8]
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        
        # 俯仰角（pitch）- 点头
        pitch = nose_tip[1] - chin[1]
        
        # 偏航角（yaw）- 摇头
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        yaw = nose_tip[0] - eye_center_x
        
        # 翻滚角（roll）- 歪头
        roll = np.arctan2(
            right_eye_center[1] - left_eye_center[1],
            right_eye_center[0] - left_eye_center[0]
        )
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': np.degrees(roll),
        }
    
    def estimate_gaze_direction(self, frame: np.ndarray) -> Dict[str, float]:
        """
        估计视线方向（简化版）
        返回：gaze_x, gaze_y（相对于屏幕中心）
        """
        # 实际应用中应该使用专门的眼动仪或深度学习模型
        # 这里简化为随机值（仅用于演示）
        return {
            'gaze_x': np.random.uniform(-0.5, 0.5),
            'gaze_y': np.random.uniform(-0.5, 0.5),
        }
    
    def _update_buffer(self, features: Dict):
        """更新数据缓冲区"""
        for key, value in features.items():
            if key in self.data_buffer:
                self.data_buffer[key].append(value)
                if len(self.data_buffer[key]) > self.buffer_size:
                    self.data_buffer[key].pop(0)
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 480,
            'fps': 30,
            'buffer_size': 30,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return default_config


# ==================== 状态识别引擎 ====================

class StudentStateRecognizer:
    """
    学生状态识别引擎
    
    输入：多模态特征
    输出：学生状态（专注度、困惑度、疲劳度等）
    
    使用方法：
    1. 规则基方法（快速原型）
    2. 机器学习方法（需要标注数据训练）
    3. 深度学习方法（端到端，需要大量数据）
    """
    
    def __init__(self, method='rule_based'):
        self.method = method
        
        # 阈值配置（可通过校准确定）
        self.thresholds = {
            'low_ear': 0.25,      # EAR < 0.25 → 闭眼/疲劳
            'high_mouth': 0.15,   # 嘴巴张开 > 0.15 → 惊讶/困惑
            'high_eyebrow': 0.3,  # 眉毛倾斜 > 0.3 → 皱眉
            'low_accuracy': 0.5,  # 正确率 < 0.5 → 困难
            'slow_response': 5.0, # 响应时间 > 5s → 犹豫
        }
        
        # 如果选择机器学习方法，需要训练分类器
        if method == 'ml_based':
            self.state_classifier = self.train_state_classifier()
    
    def recognize_state(self, visual_features: Dict, behavioral_features: Dict) -> StudentState:
        """
        识别学生状态
        
        Args:
            visual_features: 视觉特征
            behavioral_features: 行为特征
            
        Returns:
            state: 学生状态对象
        """
        
        if self.method == 'rule_based':
            return self.rule_based_recognition(visual_features, behavioral_features)
        elif self.method == 'ml_based':
            return self.ml_based_recognition(visual_features, behavioral_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def rule_based_recognition(self, visual: Dict, behavioral: Dict) -> StudentState:
        """
        基于规则的状态识别（快速原型）
        
        规则来源：教育心理学研究 + 专家经验
        """
        
        # 1. 疲劳度检测
        fatigue = self.detect_fatigue(visual)
        
        # 2. 专注度检测
        focus = self.detect_focus(visual, behavioral)
        
        # 3. 困惑度检测
        confusion = self.detect_confusion(visual, behavioral)
        
        # 4. 参与度检测
        engagement = self.detect_engagement(behavioral)
        
        # 5. 压力水平检测
        stress = self.detect_stress(visual, behavioral)
        
        # 6. 自信度检测
        confidence = self.detect_confidence(behavioral)
        
        state = StudentState(
            timestamp=time.time(),
            focus_level=focus,
            engagement=engagement,
            confusion=confusion,
            fatigue=fatigue,
            stress_level=stress,
            confidence=confidence,
        )
        
        return state
    
    def detect_fatigue(self, visual: Dict) -> float:
        """
        检测疲劳度
        
        指标：
        - EAR 降低（闭眼时间长）
        - 头部下垂（pitch 增大）
        - 打哈欠（mouth_opening 大）
        """
        fatigue_score = 0.0
        
        # 1. 眼睛闭合检测
        ear = visual.get('eye_aspect_ratio', 0.3)
        if ear < self.thresholds['low_ear']:
            fatigue_score += 0.4
        
        # 2. 头部姿态（低头）
        head_pose = visual.get('head_pose', {})
        pitch = head_pose.get('pitch', 0)
        if pitch > 20:  # 低头超过 20 度
            fatigue_score += 0.3
        
        # 3. 打哈欠
        mouth_opening = visual.get('mouth_opening', 0)
        if mouth_opening > self.thresholds['high_mouth'] * 2:
            fatigue_score += 0.3
        
        return min(1.0, fatigue_score)
    
    def detect_focus(self, visual: Dict, behavioral: Dict) -> float:
        """
        检测专注度
        
        指标：
        - Gaze 稳定在屏幕区域
        - 头部稳定
        - 交互连续
        """
        focus_score = 0.0
        
        # 1. Gaze 方向
        gaze = visual.get('gaze_direction', {})
        gaze_x = abs(gaze.get('gaze_x', 0))
        gaze_y = abs(gaze.get('gaze_y', 0))
        
        if gaze_x < 0.3 and gaze_y < 0.3:  # Gaze 集中在中央
            focus_score += 0.4
        
        # 2. 头部稳定性
        head_pose = visual.get('head_pose', {})
        roll = abs(head_pose.get('roll', 0))
        if roll < 10:  # 头部稳定
            focus_score += 0.3
        
        # 3. 交互频率
        interaction_rate = behavioral.get('interaction_rate', 0)
        if interaction_rate > 0.5:  # 频繁交互
            focus_score += 0.3
        
        return min(1.0, focus_score)
    
    def detect_confusion(self, visual: Dict, behavioral: Dict) -> float:
        """
        检测困惑度
        
        指标：
        - 眉毛倾斜（皱眉）
        - 响应时间长
        - 正确率低
        """
        confusion_score = 0.0
        
        # 1. 眉毛倾斜
        eyebrow_slope = visual.get('eyebrow_slope', 0)
        if eyebrow_slope > self.thresholds['high_eyebrow']:
            confusion_score += 0.4
        
        # 2. 响应时间长且犹豫
        avg_response = behavioral.get('avg_response_time', 0)
        response_std = behavioral.get('response_time_std', 0)
        
        if avg_response > self.thresholds['slow_response']:
            confusion_score += 0.3
        
        if response_std > 2.0:  # 响应时间波动大
            confusion_score += 0.2
        
        # 3. 正确率低
        accuracy = behavioral.get('accuracy', 0.5)
        if accuracy < self.thresholds['low_accuracy']:
            confusion_score += 0.1
        
        return min(1.0, confusion_score)
    
    def detect_engagement(self, behavioral: Dict) -> float:
        """检测参与度"""
        engagement_score = 0.0
        
        # 1. 交互频率
        interaction_rate = behavioral.get('interaction_rate', 0)
        engagement_score += min(0.4, interaction_rate * 0.4)
        
        # 2. 正确率（适度挑战）
        accuracy = behavioral.get('accuracy', 0.5)
        if 0.6 < accuracy < 0.9:  # 最佳挑战区
            engagement_score += 0.3
        elif accuracy > 0.9:  # 太简单，可能无聊
            engagement_score += 0.1
        else:  # 太难，可能挫败
            engagement_score += 0.15
        
        return min(1.0, engagement_score)
    
    def detect_stress(self, visual: Dict, behavioral: Dict) -> float:
        """检测压力水平"""
        stress_score = 0.0
        
        # 1. 眉毛紧张
        eyebrow_slope = visual.get('eyebrow_slope', 0)
        if eyebrow_slope > self.thresholds['high_eyebrow'] * 1.5:
            stress_score += 0.4
        
        # 2. 响应时间过长
        avg_response = behavioral.get('avg_response_time', 0)
        if avg_response > self.thresholds['slow_response'] * 2:
            stress_score += 0.3
        
        # 3. 正确率过低
        accuracy = behavioral.get('accuracy', 0.5)
        if accuracy < 0.3:
            stress_score += 0.3
        
        return min(1.0, stress_score)
    
    def detect_confidence(self, behavioral: Dict) -> float:
        """检测自信度"""
        confidence_score = 0.0
        
        # 1. 正确率高
        accuracy = behavioral.get('accuracy', 0.5)
        confidence_score += accuracy * 0.5
        
        # 2. 响应时间短且稳定
        avg_response = behavioral.get('avg_response_time', 0)
        response_std = behavioral.get('response_time_std', 0)
        
        if avg_response < 3.0:
            confidence_score += 0.2
        
        if response_std < 1.0:
            confidence_score += 0.2
        
        # 3. 交互流畅
        interaction_rate = behavioral.get('interaction_rate', 0)
        if interaction_rate > 1.0:
            confidence_score += 0.1
        
        return min(1.0, confidence_score)
    
    def ml_based_recognition(self, visual: Dict, behavioral: Dict) -> StudentState:
        """
        基于机器学习的状态识别（需要训练数据）
        
        这部分需要收集标注数据训练模型
        """
        # 特征工程
        feature_vector = self.extract_features(visual, behavioral)
        
        # 预测状态
        predictions = self.state_classifier.predict([feature_vector])[0]
        
        state = StudentState(
            timestamp=time.time(),
            focus_level=predictions[0],
            engagement=predictions[1],
            confusion=predictions[2],
            fatigue=predictions[3],
            stress_level=predictions[4],
            confidence=predictions[5],
        )
        
        return state
    
    def extract_features(self, visual: Dict, behavioral: Dict) -> np.ndarray:
        """特征提取（用于 ML 模型）"""
        features = [
            visual.get('eye_aspect_ratio', 0),
            visual.get('mouth_opening', 0),
            visual.get('eyebrow_slope', 0),
            visual.get('head_pose', {}).get('pitch', 0),
            visual.get('head_pose', {}).get('yaw', 0),
            visual.get('head_pose', {}).get('roll', 0),
            behavioral.get('accuracy', 0),
            behavioral.get('avg_response_time', 0),
            behavioral.get('response_time_std', 0),
            behavioral.get('interaction_rate', 0),
        ]
        return np.array(features)
    
    def train_state_classifier(self):
        """
        训练状态分类器（示例代码）
        
        实际使用时需要收集标注数据集
        """
        # TODO: 收集训练数据
        # X_train: 特征矩阵
        # y_train: 标注标签（专注度、困惑度等）
        
        # 使用多输出回归器
        classifier = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # classifier.fit(X_train, y_train)
        
        return classifier


# ==================== 神经调质映射器 ====================

class NeuromodulatorMapper:
    """
    学生状态 → 神经调质映射器
    
    核心理论：
    - 专注度高 → ACh↑
    - 参与度高 → DA↑
    - 困惑度适中 → NE↑（促进探索）
    - 压力过高 → 5-HT↓（需要稳定）
    """
    
    def __init__(self):
        # 映射权重（基于神经科学理论 + 实验校准）
        self.mapping_weights = {
            'focus_to_ACh': 0.6,
            'engagement_to_DA': 0.7,
            'confusion_to_NE': 0.5,
            'stress_to_5HT': -0.4,
            'confidence_to_DA': 0.3,
        }
    
    def map_to_neuromodulators(self, student_state: StudentState) -> NeuromodulatorState:
        """
        将学生状态映射到神经调质
        
        Args:
            student_state: 学生状态
            
        Returns:
            neuromod_state: 神经调质状态
        """
        
        # 基准浓度
        baseline = 0.5
        
        # 计算每种神经调质的浓度
        DA = baseline + \
             self.mapping_weights['engagement_to_DA'] * (student_state.engagement - baseline) + \
             self.mapping_weights['confidence_to_DA'] * (student_state.confidence - baseline)
        
        _5_HT = baseline + \
                self.mapping_weights['stress_to_5HT'] * (student_state.stress_level - baseline)
        
        NE = baseline + \
             self.mapping_weights['confusion_to_NE'] * (student_state.confusion - baseline)
        
        ACh = baseline + \
              self.mapping_weights['focus_to_ACh'] * (student_state.focus_level - baseline)
        
        # 限制范围
        DA = np.clip(DA, 0.0, 1.0)
        _5_HT = np.clip(_5_HT, 0.0, 1.0)
        NE = np.clip(NE, 0.0, 1.0)
        ACh = np.clip(ACh, 0.0, 1.0)
        
        neuromod_state = NeuromodulatorState(
            DA=DA,
            _5_HT=_5_HT,
            NE=NE,
            ACh=ACh,
        )
        
        return neuromod_state


# ==================== 教学策略生成器 ====================

class TeachingStrategyGenerator:
    """
    基于神经调质状态生成教学策略
    
    策略库：
    - 调整学习速率
    - 调整内容难度
    - 提供提示/反馈
    - 建议休息
    """
    
    def generate_strategy(self, neuromod_state: NeuromodulatorState) -> Dict:
        """
        生成教学策略
        
        Args:
            neuromod_state: 神经调质状态
            
        Returns:
            strategy: 教学策略字典
        """
        strategy = {
            'action': 'continue',
            'learning_rate_multiplier': 1.0,
            'difficulty_adjustment': 0,
            'feedback_type': 'none',
            'recommendation': '',
        }
        
        # 规则 1: 高 DA + 高 ACh → 最佳学习状态，增加难度
        if neuromod_state.DA > 0.7 and neuromod_state.ACh > 0.6:
            strategy['action'] = 'increase_difficulty'
            strategy['learning_rate_multiplier'] = 1.2
            strategy['difficulty_adjustment'] = 1
            strategy['feedback_type'] = 'encouragement'
            strategy['recommendation'] = '学生处于最佳学习状态，可以挑战更高难度'
        
        # 规则 2: 高 NE + 低 DA → 困惑/焦虑，需要帮助
        elif neuromod_state.NE > 0.7 and neuromod_state.DA < 0.4:
            strategy['action'] = 'provide_hint'
            strategy['learning_rate_multiplier'] = 0.8
            strategy['difficulty_adjustment'] = -1
            strategy['feedback_type'] = 'guidance'
            strategy['recommendation'] = '学生感到困惑，建议提供提示或降低难度'
        
        # 规则 3: 低 5-HT → 压力大，需要休息
        elif neuromod_state._5_HT < 0.3:
            strategy['action'] = 'suggest_break'
            strategy['learning_rate_multiplier'] = 0.5
            strategy['feedback_type'] = 'relaxation'
            strategy['recommendation'] = '学生压力过大，建议短暂休息'
        
        # 规则 4: 低 DA + 低 ACh → 动机不足，需要激励
        elif neuromod_state.DA < 0.4 and neuromod_state.ACh < 0.4:
            strategy['action'] = 'gamify_or_incentivize'
            strategy['learning_rate_multiplier'] = 1.0
            strategy['feedback_type'] = 'motivation'
            strategy['recommendation'] = '学生动机不足，建议游戏化或奖励激励'
        
        # 规则 5: 正常状态 → 继续
        else:
            strategy['action'] = 'continue'
            strategy['recommendation'] = '保持当前节奏'
        
        return strategy


# ==================== 主控制器 ====================

class IntelligentTutoringSystem:
    """
    智能辅导系统主控制器
    
    完整流程：
    数据采集 → 状态识别 → 神经调质映射 → 策略生成 → 执行
    """
    
    def __init__(self):
        self.sensor = MultiModalSensor()
        self.recognizer = StudentStateRecognizer(method='rule_based')
        self.mapper = NeuromodulatorMapper()
        self.strategy_generator = TeachingStrategyGenerator()
        
        # 历史记录
        self.state_history = []
    
    def run_cycle(self, camera_frame: np.ndarray, interaction_events: List[Dict]) -> Dict:
        """
        运行一个感知-决策周期
        
        Args:
            camera_frame: 摄像头帧
            interaction_events: 交互事件
            
        Returns:
            result: 包含所有中间结果和最终策略
        """
        
        # 1. 数据采集
        visual_features = self.sensor.capture_visual_data(camera_frame)
        behavioral_features = self.sensor.capture_behavioral_data(interaction_events)
        
        # 2. 状态识别
        student_state = self.recognizer.recognize_state(
            visual_features, 
            behavioral_features
        )
        
        # 3. 神经调质映射
        neuromod_state = self.mapper.map_to_neuromodulators(student_state)
        
        # 4. 策略生成
        teaching_strategy = self.strategy_generator.generate_strategy(neuromod_state)
        
        # 5. 记录历史
        self.state_history.append({
            'timestamp': time.time(),
            'student_state': student_state.to_dict(),
            'neuromod_state': neuromod_state.to_dict(),
            'strategy': teaching_strategy,
        })
        
        # 6. 返回结果
        result = {
            'student_state': student_state.to_dict(),
            'neuromod_state': neuromod_state.to_dict(),
            'teaching_strategy': teaching_strategy,
        }
        
        return result


# ==================== 使用示例 ====================

def demo_usage():
    """演示使用"""
    
    print("=" * 70)
    print("智能辅导系统演示")
    print("=" * 70)
    
    # 初始化系统
    system = IntelligentTutoringSystem()
    
    # 模拟数据（实际应用中使用真实摄像头和交互数据）
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_interactions = [
        {'type': 'answer', 'correct': True, 'response_time': 2.5},
        {'type': 'answer', 'correct': True, 'response_time': 3.1},
        {'type': 'answer', 'correct': False, 'response_time': 5.2},
    ]
    
    # 运行一个周期
    result = system.run_cycle(dummy_frame, dummy_interactions)
    
    # 输出结果
    print("\n📊 学生状态:")
    for key, value in result['student_state'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\n🧠 神经调质状态:")
    for key, value in result['neuromod_state'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\n🎯 教学策略:")
    print(f"  行动：{result['teaching_strategy']['action']}")
    print(f"  学习率调整：×{result['teaching_strategy']['learning_rate_multiplier']:.2f}")
    print(f"  难度调整：{result['teaching_strategy']['difficulty_adjustment']}")
    print(f"  建议：{result['teaching_strategy']['recommendation']}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == '__main__':
    demo_usage()
