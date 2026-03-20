"""
CATS-NET 实验运行管理器
- 自动创建带时间戳的结果文件夹
- 版本控制（保持原始版本，生成 v2、v3...）
- 实验配置和结果归档
- 支持多次运行对比分析

作者：NeuroConscious Research Team
创建：2026-02-28
版本：v1.0.0
"""

import os
import sys
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import subprocess


class ExperimentRunner:
    """实验运行管理器"""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        
        # 获取当前版本号
        self.version = self._get_next_version()
        
        # 创建带时间戳的结果目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = self.base_dir / f"{experiment_name}_v{self.version}_{self.timestamp}"
        
        print("="*70)
        print(f"CATS-NET 实验运行管理器")
        print("="*70)
        print(f"实验名称：{experiment_name}")
        print(f"版本号：v{self.version}")
        print(f"时间戳：{self.timestamp}")
        print(f"结果目录：{self.result_dir}")
        print("="*70)
    
    def _get_next_version(self) -> int:
        """获取下一个版本号"""
        # 查找所有已有版本
        versions = []
        for item in self.base_dir.glob(f"{self.experiment_name}_v*"):
            try:
                # 提取版本号：experiment_name_v1_20260228_120000 -> v1
                version_str = item.name.split('_v')[1].split('_')[0]
                if version_str.isdigit():
                    versions.append(int(version_str))
            except (IndexError, ValueError):
                continue
        
        # 返回下一个版本号
        return max(versions, default=0) + 1
    
    def setup_result_directory(self) -> Path:
        """创建结果目录"""
        # 创建主目录
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.result_dir / "figures").mkdir(exist_ok=True)
        (self.result_dir / "reports").mkdir(exist_ok=True)
        (self.result_dir / "logs").mkdir(exist_ok=True)
        (self.result_dir / "config").mkdir(exist_ok=True)
        
        print(f"\n✅ 结果目录已创建：{self.result_dir}")
        print(f"   - figures/  : 图表文件")
        print(f"   - reports/  : JSON 报告")
        print(f"   - logs/     : 运行日志")
        print(f"   - config/   : 配置文件")
        
        return self.result_dir
    
    def save_experiment_config(self, config: Dict) -> Path:
        """保存实验配置"""
        config_path = self.result_dir / "config" / "experiment_config.json"
        
        # 添加元数据
        full_config = {
            "metadata": {
                "experiment_name": self.experiment_name,
                "version": f"v{self.version}",
                "timestamp": self.timestamp,
                "created_by": "ExperimentRunner v1.0.0",
                "python_version": sys.version,
                "working_directory": str(Path.cwd()),
            },
            "parameters": config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 实验配置已保存：{config_path}")
        return config_path
    
    def run_experiment(self, script_path: str, timeout: Optional[int] = None) -> int:
        """运行实验脚本
        
        Args:
            script_path: 脚本路径
            timeout: 超时时间（秒），None 表示不限制
            
        Returns:
            返回码（0 表示成功）
        """
        log_path = self.result_dir / "logs" / f"run_{self.timestamp}.log"
        
        print(f"\n{'='*70}")
        print(f"开始运行实验")
        print(f"脚本：{script_path}")
        print(f"日志：{log_path}")
        print(f"{'='*70}\n")
        
        # 构建命令（添加当前目录到 PYTHONPATH）
        env = os.environ.copy()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env['PYTHONPATH'] = current_dir + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONUTF8'] = '1'  # 强制使用 UTF-8 编码
        
        cmd = [sys.executable, script_path]
        
        # 重定向输出到日志文件
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # 写入头部信息
            log_file.write(f"Experiment: {self.experiment_name}\n")
            log_file.write(f"Version: v{self.version}\n")
            log_file.write(f"Timestamp: {self.timestamp}\n")
            log_file.write(f"Script: {script_path}\n")
            log_file.write(f"Python: {sys.version}\n")
            log_file.write(f"Working Dir: {current_dir}\n")
            log_file.write("="*70 + "\n\n")
            log_file.flush()
            
            # 运行进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env  # 使用包含 PYTHONPATH 的环境变量
            )
            
            # 实时输出并记录日志
            output_lines = []
            try:
                for line in process.stdout:
                    # 输出到控制台
                    print(line, end='')
                    # 记录到日志
                    log_file.write(line)
                    log_file.flush()
                    output_lines.append(line)
                
                process.wait(timeout=timeout)
                
            except subprocess.TimeoutExpired:
                process.kill()
                error_msg = f"\n❌ 实验超时（{timeout}秒）"
                print(error_msg)
                log_file.write(error_msg)
                return -1
            
            except Exception as e:
                error_msg = f"\n❌ 运行失败：{e}"
                print(error_msg)
                log_file.write(error_msg)
                return -1
        
        # 检查返回码
        if process.returncode == 0:
            print(f"\n{'='*70}")
            print(f"✅ 实验成功完成！")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"❌ 实验失败，返回码：{process.returncode}")
            print(f"{'='*70}")
        
        return process.returncode
    
    def move_output_files(self, file_patterns: list) -> Dict[str, Path]:
        """将生成的文件移动到结果目录
        
        Args:
            file_patterns: 文件模式列表，如 ['*.png', '*.json']
            
        Returns:
            移动的文件字典 {文件名：新路径}
        """
        moved_files = {}
        
        for pattern in file_patterns:
            # 在当前目录查找匹配的文件
            for file_path in Path('.').glob(pattern):
                if file_path.is_file():
                    # 确定目标目录
                    if file_path.suffix in ['.png', '.jpg', '.pdf']:
                        target_dir = self.result_dir / "figures"
                    elif file_path.suffix == '.json':
                        target_dir = self.result_dir / "reports"
                    else:
                        target_dir = self.result_dir
                    
                    # 移动文件
                    new_path = target_dir / file_path.name
                    shutil.move(str(file_path), str(new_path))
                    moved_files[file_path.name] = new_path
                    print(f"✓ 移动文件：{file_path.name} -> {new_path}")
        
        return moved_files
    
    def create_summary_report(self, results: Dict) -> Path:
        """创建总结报告"""
        summary_path = self.result_dir / "reports" / f"summary_{self.timestamp}.json"
        
        # 添加元数据
        full_report = {
            "metadata": {
                "experiment_name": self.experiment_name,
                "version": f"v{self.version}",
                "timestamp": self.timestamp,
                "duration_seconds": results.get('duration', 0),
                "status": "success" if results.get('return_code', -1) == 0 else "failed",
            },
            "results": results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 总结报告已保存：{summary_path}")
        return summary_path
    
    @classmethod
    def list_all_runs(cls, experiment_name: str, base_dir: str = "experiments") -> list:
        """列出某实验的所有历史运行记录"""
        base_path = Path(base_dir)
        runs = []
        
        for item in sorted(base_path.glob(f"{experiment_name}_v*")):
            if item.is_dir():
                # 解析版本和时间戳
                parts = item.name.split('_v')
                if len(parts) >= 2:
                    version_and_time = parts[1]
                    version = version_and_time.split('_')[0]
                    timestamp = '_'.join(version_and_time.split('_')[1:])
                    
                    runs.append({
                        'name': item.name,
                        'version': version,
                        'timestamp': timestamp,
                        'path': item,
                    })
        
        return runs


def main():
    """主函数：运行所有 CATS-NET 实验"""
    
    import time
    
    # ========== 实验 1: 猫识别对比 ==========
    print("\n" + "="*70)
    print("实验 1: 猫识别对比实验")
    print("="*70)
    
    runner1 = ExperimentRunner("cats_cat_recognition", base_dir="experiments/results")
    runner1.setup_result_directory()
    
    # 保存配置
    config1 = {
        "n_train_samples": 10,
        "n_test_samples": 50,
        "n_epochs": 50,
        "target_accuracy": 0.9,
        "use_small_config": True,
    }
    runner1.save_experiment_config(config1)
    
    # 运行实验
    start_time = time.time()
    return_code = runner1.run_experiment("experiments/run_cats_cat_recognition.py")
    duration = time.time() - start_time
    
    # 移动输出文件
    runner1.move_output_files(['*.png', '*.json'])
    
    # 创建总结
    runner1.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # ========== 实验 2: 概念形成 ==========
    print("\n" + "="*70)
    print("实验 2: 概念形成实验")
    print("="*70)
    
    runner2 = ExperimentRunner("concept_formation", base_dir="experiments/results")
    runner2.setup_result_directory()
    
    config2 = {
        "n_categories": 5,
        "samples_per_category": 20,
        "n_epochs": 5,
        "use_small_config": True,
    }
    runner2.save_experiment_config(config2)
    
    start_time = time.time()
    return_code = runner2.run_experiment("experiments/run_concept_formation.py")
    duration = time.time() - start_time
    
    runner2.move_output_files(['*.png', '*.json'])
    runner2.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # ========== 实验 3: 概念迁移 ==========
    print("\n" + "="*70)
    print("实验 3: 概念迁移实验")
    print("="*70)
    
    runner3 = ExperimentRunner("concept_transfer", base_dir="experiments/results")
    runner3.setup_result_directory()
    
    config3 = {
        "train_samples": 50,
        "test_samples": 30,
        "teacher_epochs": 20,
        "use_adversarial": True,
    }
    runner3.save_experiment_config(config3)
    
    start_time = time.time()
    return_code = runner3.run_experiment("experiments/run_concept_transfer.py")
    duration = time.time() - start_time
    
    runner3.move_output_files(['*.png', '*.json'])
    runner3.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # ========== 实验 4: 小样本学习对比 ==========
    print("\n" + "="*70)
    print("实验 4: 小样本学习对比实验")
    print("="*70)
    
    runner4 = ExperimentRunner("few_shot_learning", base_dir="experiments/results")
    runner4.setup_result_directory()
    
    config4 = {
        "sample_sizes": [5, 10, 20, 50, 100],
        "test_size": 100,
        "epochs_per_sample": 3,
    }
    runner4.save_experiment_config(config4)
    
    start_time = time.time()
    return_code = runner4.run_experiment("experiments/run_few_shot_learning.py")
    duration = time.time() - start_time
    
    runner4.move_output_files(['*.png', '*.json'])
    runner4.create_summary_report({
        'return_code': return_code,
        'duration': duration,
    })
    
    # ========== 总体总结 ==========
    print("\n" + "="*70)
    print("所有实验运行完成！")
    print("="*70)
    
    # 列出生成的结果目录
    results_base = Path("cats_nct/experiments/results")
    print(f"\n结果目录位置：{results_base.absolute()}")
    print("\n生成的运行记录:")
    
    experiments = [
        "cats_cat_recognition",
        "concept_formation",
        "concept_transfer",
        "few_shot_learning",
    ]
    
    for exp_name in experiments:
        runs = ExperimentRunner.list_all_runs(exp_name, base_dir=str(results_base))
        if runs:
            latest = runs[-1]
            print(f"\n  {exp_name}:")
            print(f"    最新版本：v{latest['version']} ({latest['timestamp']})")
            print(f"    路径：{latest['path']}")
    
    print(f"\n{'='*70}")
    print("✅ 实验管理器任务完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    # 设置 UTF-8 编码
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n实验管理器失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
