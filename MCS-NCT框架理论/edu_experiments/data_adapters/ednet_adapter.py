"""
EdNet 知识追踪数据适配器
将 EdNet 学生答题序列转换为 MCS Solver 输入格式

数据源:
    - d:/data/ednet/KT1/u*.csv: 用户答题记录
    - d:/data/ednet/contents.csv: 题目信息

数据格式:
    KT1/u{id}.csv:
        - timestamp: 答题时间戳
        - question_id: 题目ID (q1, q2, ...)
        - bundle_id: 题目组ID
        - user_answer: 用户答案
        - elapsed_time: 答题耗时(ms)
        - answered_correctly: 是否正确(0/1)
        
    contents.csv:
        - question_id: 题目ID
        - bundle_id: 题目组ID
        - correct_answer: 正确答案
        - part: 部分(1-7)
        - tags: 标签
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edu_experiments.config import EDNET_DIR, D_MODEL, DEVICE


class EdNetAdapter(nn.Module):
    """
    EdNet 答题序列 → MCS Solver 输入格式适配器
    
    适配策略:
        - question_id → embedding
        - 特征拼接 [embedding, correct, elapsed_norm, part] → visual
        - 滑动窗口知识掌握度估计 → auditory
        - 窗口均值 → current_state
    """
    
    def __init__(
        self,
        d_model: int = D_MODEL,
        seq_len: int = 20,
        device: str = DEVICE,
        num_questions: int = 14000  # EdNet 约13169题
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.device = device
        self.num_questions = num_questions
        
        # 题目嵌入: question_id → D//2 维
        self.embed_dim = d_model // 2
        self.question_embed = nn.Embedding(num_questions, self.embed_dim)
        
        # 特征投影: [embed_dim + 4] → d_model
        # 4个额外特征: correct, elapsed_norm, part_norm, knowledge_est
        self.feature_proj = nn.Linear(self.embed_dim + 4, d_model)
        
        # 知识状态投影
        self.knowledge_proj = nn.Linear(d_model, d_model)
        
        # 状态投影
        self.state_proj = nn.Linear(d_model, d_model)
        
        # Xavier初始化
        nn.init.xavier_uniform_(self.question_embed.weight)
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.xavier_uniform_(self.knowledge_proj.weight)
        nn.init.xavier_uniform_(self.state_proj.weight)
        
        # 题目信息缓存
        self.contents_df = None
        self.question_to_idx = {}
        
        self.to(device)
        
    def _load_contents(self):
        """加载题目内容信息"""
        if self.contents_df is not None:
            return True
            
        try:
            import pandas as pd
            contents_path = EDNET_DIR / "contents.csv"
            if contents_path.exists():
                self.contents_df = pd.read_csv(str(contents_path))
                # 构建 question_id → index 映射
                for idx, qid in enumerate(self.contents_df['question_id'].unique()):
                    self.question_to_idx[qid] = idx % self.num_questions
                print(f"[EdNet] Loaded {len(self.question_to_idx)} questions from contents.csv")
                return True
        except Exception as e:
            print(f"[EdNet Warning] Failed to load contents.csv: {e}")
            
        return False
        
    def load_dataset(
        self,
        max_students: int = 300
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, float]]]:
        """
        加载学生答题序列
        
        Args:
            max_students: 最大学生数量
            
        Returns:
            (mcs_inputs_list, labels_dicts_list)
            labels_dict = {'accuracy': float, 'avg_time': float, 'total_questions': int}
        """
        try:
            import pandas as pd
        except ImportError:
            print("[EdNet Warning] pandas not installed, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        # 加载题目信息
        self._load_contents()
        
        kt1_dir = EDNET_DIR / "KT1"
        if not kt1_dir.exists():
            print(f"[EdNet Warning] {kt1_dir} not found, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        mcs_inputs = []
        labels_list = []
        loaded_students = 0
        
        # 遍历用户文件
        for user_file in sorted(kt1_dir.glob("u*.csv")):
            if loaded_students >= max_students:
                break
                
            try:
                user_df = pd.read_csv(str(user_file))
                
                if len(user_df) < self.seq_len:
                    continue  # 跳过答题数太少的用户
                    
                # 从单个学生生成多个样本（滑动窗口）
                samples = self.adapt_single(user_df)
                
                for sample in samples:
                    mcs_inputs.append(sample['mcs_input'])
                    labels_list.append(sample['labels'])
                    
                loaded_students += 1
                
            except Exception as e:
                continue
                
        if len(mcs_inputs) == 0:
            print("[EdNet Warning] No data loaded, returning synthetic data")
            return self._generate_synthetic_data(100)
            
        print(f"[EdNet] Loaded {len(mcs_inputs)} samples from {loaded_students} students")
        return mcs_inputs, labels_list
    
    def adapt_single(
        self,
        interaction_df
    ) -> List[Dict]:
        """
        单个学生的答题序列 → 多个MCS输入（滑动窗口）
        
        Args:
            interaction_df: 学生答题DataFrame
            
        Returns:
            List of {"mcs_input": {...}, "labels": {...}}
        """
        import pandas as pd
        
        results = []
        n_interactions = len(interaction_df)
        
        # 滑动窗口步长
        stride = self.seq_len // 2
        
        for start_idx in range(0, n_interactions - self.seq_len + 1, stride):
            window = interaction_df.iloc[start_idx:start_idx + self.seq_len]
            
            # 提取特征
            question_ids = []
            corrects = []
            elapsed_times = []
            parts = []
            
            for _, row in window.iterrows():
                qid = row['question_id']
                # 获取题目索引
                if qid in self.question_to_idx:
                    q_idx = self.question_to_idx[qid]
                else:
                    # 简单哈希
                    q_idx = hash(qid) % self.num_questions
                    
                question_ids.append(q_idx)
                corrects.append(float(row['answered_correctly']))
                
                # 归一化答题时间 (毫秒 → 秒，然后归一化到0-1)
                elapsed = row['elapsed_time'] / 1000.0  # 转秒
                elapsed_norm = min(elapsed / 300.0, 1.0)  # 5分钟为上限
                elapsed_times.append(elapsed_norm)
                
                # 获取题目部分 (1-7)
                if self.contents_df is not None and qid in self.contents_df['question_id'].values:
                    part = self.contents_df[self.contents_df['question_id'] == qid]['part'].iloc[0]
                else:
                    part = 4  # 默认中间值
                parts.append(part / 7.0)  # 归一化到0-1
            
            # 转为tensor
            q_ids = torch.tensor(question_ids, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                # 题目嵌入
                q_embed = self.question_embed(q_ids)  # (seq_len, embed_dim)
                
                # 组合特征
                corrects_t = torch.tensor(corrects, dtype=torch.float32, device=self.device).unsqueeze(-1)
                elapsed_t = torch.tensor(elapsed_times, dtype=torch.float32, device=self.device).unsqueeze(-1)
                parts_t = torch.tensor(parts, dtype=torch.float32, device=self.device).unsqueeze(-1)
                
                # 计算滑动知识掌握度
                knowledge_est = self._estimate_knowledge(corrects, elapsed_times)
                knowledge_t = torch.tensor(knowledge_est, dtype=torch.float32, device=self.device).unsqueeze(-1)
                
                # 拼接特征: [embed, correct, elapsed, part, knowledge]
                features = torch.cat([q_embed, corrects_t, elapsed_t, parts_t, knowledge_t], dim=-1)
                
                # 投影
                visual = self.feature_proj(features)  # (seq_len, d_model)
                
                # 知识状态序列 → auditory
                auditory = self.knowledge_proj(visual)  # (seq_len, d_model)
                
                # 均值 → state
                state = self.state_proj(visual.mean(dim=0, keepdim=True))  # (1, d_model)
            
            # 计算窗口标签
            accuracy = sum(corrects) / len(corrects)
            avg_time = sum(elapsed_times) / len(elapsed_times) * 300  # 还原到秒
            
            mcs_input = {
                "visual": visual.unsqueeze(0),       # [1, seq_len, D]
                "auditory": auditory.unsqueeze(0),   # [1, seq_len, D]
                "current_state": state               # [1, D]
            }
            
            labels = {
                'accuracy': accuracy,
                'avg_time': avg_time,
                'total_questions': self.seq_len,
                'knowledge_level': np.mean(knowledge_est)
            }
            
            results.append({
                "mcs_input": mcs_input,
                "labels": labels
            })
            
        return results
    
    def _estimate_knowledge(
        self,
        corrects: List[float],
        elapsed_times: List[float]
    ) -> List[float]:
        """
        滑动窗口估计知识掌握度
        
        使用指数移动平均结合正确率和答题时间
        """
        alpha = 0.3  # 平滑系数
        knowledge = []
        current_k = 0.5  # 初始知识水平
        
        for correct, elapsed in zip(corrects, elapsed_times):
            # 根据答题结果更新知识估计
            # 正确+快速 → 高掌握度提升
            # 错误+慢 → 掌握度下降
            if correct > 0.5:
                delta = 0.1 * (1 - elapsed)  # 答对越快，提升越多
            else:
                delta = -0.1 * (1 + elapsed)  # 答错越慢，下降越多
                
            current_k = alpha * (current_k + delta) + (1 - alpha) * current_k
            current_k = max(0, min(1, current_k))  # 限制在[0,1]
            knowledge.append(current_k)
            
        return knowledge
    
    def _generate_synthetic_data(
        self,
        n_samples: int
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, float]]]:
        """生成合成数据作为降级方案"""
        print(f"[EdNet] Generating {n_samples} synthetic samples")
        
        mcs_inputs = []
        labels_list = []
        
        for i in range(n_samples):
            # 模拟不同能力水平的学生
            ability = np.random.beta(2, 2)  # 能力分布
            
            # 生成答题序列
            question_ids = np.random.randint(0, self.num_questions, self.seq_len)
            
            # 正确率与能力相关
            corrects = []
            for _ in range(self.seq_len):
                diff = np.random.uniform(0.3, 0.7)  # 题目难度
                prob = 1 / (1 + np.exp(-(ability - diff) * 4))  # IRT模型
                corrects.append(1.0 if np.random.random() < prob else 0.0)
                
            # 答题时间与能力负相关
            elapsed_times = []
            for c in corrects:
                if c > 0.5:
                    # 答对：能力越高越快
                    time = np.random.exponential(60 / (ability + 0.5))
                else:
                    # 答错：通常更久
                    time = np.random.exponential(90 / (ability + 0.3))
                elapsed_times.append(min(time / 300.0, 1.0))
                
            # 随机part
            parts = [np.random.randint(1, 8) / 7.0 for _ in range(self.seq_len)]
            
            # 知识估计
            knowledge_est = self._estimate_knowledge(corrects, elapsed_times)
            
            # 构建特征
            q_ids = torch.tensor(question_ids, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                q_embed = self.question_embed(q_ids)
                
                corrects_t = torch.tensor(corrects, dtype=torch.float32, device=self.device).unsqueeze(-1)
                elapsed_t = torch.tensor(elapsed_times, dtype=torch.float32, device=self.device).unsqueeze(-1)
                parts_t = torch.tensor(parts, dtype=torch.float32, device=self.device).unsqueeze(-1)
                knowledge_t = torch.tensor(knowledge_est, dtype=torch.float32, device=self.device).unsqueeze(-1)
                
                features = torch.cat([q_embed, corrects_t, elapsed_t, parts_t, knowledge_t], dim=-1)
                
                visual = self.feature_proj(features)
                auditory = self.knowledge_proj(visual)
                state = self.state_proj(visual.mean(dim=0, keepdim=True))
            
            mcs_input = {
                "visual": visual.unsqueeze(0),
                "auditory": auditory.unsqueeze(0),
                "current_state": state
            }
            
            labels = {
                'accuracy': sum(corrects) / len(corrects),
                'avg_time': sum(elapsed_times) / len(elapsed_times) * 300,
                'total_questions': self.seq_len,
                'knowledge_level': np.mean(knowledge_est)
            }
            
            mcs_inputs.append(mcs_input)
            labels_list.append(labels)
            
        return mcs_inputs, labels_list
    
    def collate_batch(
        self,
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """将多个样本合并为批次"""
        return {
            "visual": torch.cat([b["visual"] for b in batch], dim=0),
            "auditory": torch.cat([b["auditory"] for b in batch], dim=0),
            "current_state": torch.cat([b["current_state"] for b in batch], dim=0)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("EdNet 知识追踪适配器测试")
    print("=" * 60)
    
    # 创建适配器
    adapter = EdNetAdapter(d_model=D_MODEL, seq_len=20, device=DEVICE)
    print(f"[Test] 适配器创建成功, device={DEVICE}, d_model={D_MODEL}")
    
    # 加载数据
    mcs_inputs, labels = adapter.load_dataset(max_students=10)
    
    print(f"\n[Test] 加载样本数: {len(mcs_inputs)}")
    
    # 统计标签分布
    accuracies = [l['accuracy'] for l in labels]
    knowledge_levels = [l['knowledge_level'] for l in labels]
    print(f"[Test] 正确率范围: [{min(accuracies):.2f}, {max(accuracies):.2f}]")
    print(f"[Test] 知识水平范围: [{min(knowledge_levels):.2f}, {max(knowledge_levels):.2f}]")
    
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
        
    print("\n" + "=" * 60)
    print("EdNet 适配器测试完成!")
    print("=" * 60)
