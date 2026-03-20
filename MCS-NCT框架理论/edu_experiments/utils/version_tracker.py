"""MCS-NCT 教育验证实验 - 版本追踪器

记录实验元数据：实验名称、版本、开始时间、超参数快照、结果指标、图表路径列表、脚本MD5。
"""
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class ExperimentTracker:
    """
    实验版本追踪器
    
    追踪实验的完整生命周期，包括：
    - 实验配置和超参数
    - 运行时间和状态
    - 结果指标
    - 生成的图表
    - 脚本版本（MD5 hash）
    
    Example:
        tracker = ExperimentTracker("exp_A", "v1")
        tracker.log_start({"lr": 0.001, "epochs": 100})
        # ... run experiment ...
        tracker.log_result({"accuracy": 0.95, "f1": 0.93})
        tracker.log_figure("results/exp_A/v1/figures/loss.png")
        tracker.save()
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        version: str = "v1",
        results_root: Optional[Path] = None,
        script_path: Optional[str] = None
    ):
        """
        初始化追踪器
        
        Args:
            experiment_name: 实验名称 (如 "exp_A")
            version: 版本号 (如 "v1")
            results_root: 结果根目录，默认使用 config 中的 RESULTS_ROOT
            script_path: 运行脚本路径，用于计算 MD5
        """
        self.experiment_name = experiment_name
        self.version = version
        self.script_path = script_path
        
        # 设置结果目录
        if results_root is None:
            from ..config import RESULTS_ROOT
            results_root = RESULTS_ROOT
        
        self.results_root = Path(results_root)
        self.experiment_dir = self.results_root / experiment_name / version
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        self.log_file = self.experiment_dir / "experiment_log.jsonl"
        
        # 实验记录
        self.record = {
            "experiment_name": experiment_name,
            "version": version,
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "status": "initialized",
            "hyperparameters": {},
            "metrics": {},
            "figures": [],
            "script_md5": None,
            "notes": []
        }
        
        # 计算脚本 MD5
        if script_path and os.path.exists(script_path):
            self.record["script_md5"] = self._compute_md5(script_path)
    
    def _compute_md5(self, file_path: str) -> str:
        """计算文件的 MD5 hash"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log_start(self, hyperparams: Dict[str, Any]) -> None:
        """
        记录实验开始
        
        Args:
            hyperparams: 超参数字典
        """
        self.record["start_time"] = self._get_timestamp()
        self.record["status"] = "running"
        self.record["hyperparameters"] = hyperparams
        
        print(f"[Tracker] Experiment '{self.experiment_name}' ({self.version}) started")
        print(f"[Tracker] Start time: {self.record['start_time']}")
        print(f"[Tracker] Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    
    def log_result(self, metrics: Dict[str, Any]) -> None:
        """
        记录实验结果指标
        
        Args:
            metrics: 结果指标字典
        """
        # 合并新的指标
        self.record["metrics"].update(metrics)
        
        print(f"[Tracker] Metrics logged: {json.dumps(metrics, indent=2)}")
    
    def log_figure(self, fig_path: Union[str, Path]) -> None:
        """
        记录生成的图表路径
        
        Args:
            fig_path: 图表文件路径
        """
        fig_path = str(fig_path)
        if fig_path not in self.record["figures"]:
            self.record["figures"].append(fig_path)
            print(f"[Tracker] Figure logged: {fig_path}")
    
    def log_note(self, note: str) -> None:
        """
        添加实验备注
        
        Args:
            note: 备注文本
        """
        timestamp = self._get_timestamp()
        self.record["notes"].append({
            "timestamp": timestamp,
            "note": note
        })
        print(f"[Tracker] Note added: {note}")
    
    def log_error(self, error_msg: str) -> None:
        """
        记录错误信息
        
        Args:
            error_msg: 错误信息
        """
        self.record["status"] = "failed"
        self.record["error"] = {
            "timestamp": self._get_timestamp(),
            "message": error_msg
        }
        print(f"[Tracker] ERROR: {error_msg}")
    
    def finish(self, status: str = "completed") -> None:
        """
        标记实验结束
        
        Args:
            status: 最终状态 ("completed", "failed", "cancelled")
        """
        self.record["end_time"] = self._get_timestamp()
        self.record["status"] = status
        
        # 计算运行时间
        if self.record["start_time"]:
            start = datetime.strptime(self.record["start_time"], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(self.record["end_time"], "%Y-%m-%d %H:%M:%S")
            self.record["duration_seconds"] = (end - start).total_seconds()
        
        print(f"[Tracker] Experiment finished with status: {status}")
        if self.record["duration_seconds"]:
            print(f"[Tracker] Duration: {self.record['duration_seconds']:.1f} seconds")
    
    def save(self) -> Path:
        """
        保存实验记录到 JSONL 文件（追加写入）
        
        Returns:
            日志文件路径
        """
        # 如果实验还在运行，先结束它
        if self.record["status"] == "running":
            self.finish()
        
        # 追加写入 JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.record, ensure_ascii=False) + "\n")
        
        print(f"[Tracker] Record saved to: {self.log_file}")
        return self.log_file
    
    def save_snapshot(self, filename: str = "snapshot.json") -> Path:
        """
        保存当前实验快照到独立的 JSON 文件
        
        Args:
            filename: 文件名
            
        Returns:
            快照文件路径
        """
        snapshot_path = self.experiment_dir / filename
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(self.record, f, ensure_ascii=False, indent=2)
        
        print(f"[Tracker] Snapshot saved to: {snapshot_path}")
        return snapshot_path
    
    def get_summary(self) -> str:
        """
        获取实验摘要
        
        Returns:
            格式化的摘要字符串
        """
        lines = [
            "=" * 60,
            f"Experiment: {self.experiment_name} ({self.version})",
            "=" * 60,
            f"Status: {self.record['status']}",
            f"Start: {self.record['start_time']}",
            f"End: {self.record['end_time']}",
            f"Duration: {self.record.get('duration_seconds', 'N/A')} seconds",
            "",
            "Hyperparameters:",
        ]
        
        for key, value in self.record["hyperparameters"].items():
            lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append("Metrics:")
        for key, value in self.record["metrics"].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append(f"Figures ({len(self.record['figures'])}):")
        for fig in self.record["figures"]:
            lines.append(f"  - {fig}")
        
        if self.record["script_md5"]:
            lines.append("")
            lines.append(f"Script MD5: {self.record['script_md5']}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    @classmethod
    def load_history(cls, experiment_name: str, version: str = "v1",
                     results_root: Optional[Path] = None) -> List[Dict]:
        """
        加载实验历史记录
        
        Args:
            experiment_name: 实验名称
            version: 版本号
            results_root: 结果根目录
            
        Returns:
            历史记录列表
        """
        if results_root is None:
            from ..config import RESULTS_ROOT
            results_root = RESULTS_ROOT
        
        log_file = Path(results_root) / experiment_name / version / "experiment_log.jsonl"
        
        if not log_file.exists():
            return []
        
        records = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        return records


def create_tracker(experiment_name: str, version: str = "v1", **kwargs) -> ExperimentTracker:
    """
    便捷函数：创建实验追踪器
    
    Args:
        experiment_name: 实验名称
        version: 版本号
        **kwargs: 传递给 ExperimentTracker 的其他参数
        
    Returns:
        ExperimentTracker 实例
    """
    return ExperimentTracker(experiment_name, version, **kwargs)


# === 导出的符号 ===
__all__ = ['ExperimentTracker', 'create_tracker']
