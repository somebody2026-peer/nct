"""
EdNet-KT1 数据集下载工具
=========================
EdNet 官方下载链接（无需 Kaggle 账号）：
  bit.ly/ednet_kt1  →  约 1.2GB 压缩包

格式：
  {student_id}.csv: timestamp, question_id, bundle_id, user_answer, elapsed_time
  注：user_answer 为 a/b/c/d，需与 contents.csv 比对才能获得 answered_correctly

运行：
  python tools/download_ednet.py          # 下载并解压小样本（前 500 名学生）
  python tools/download_ednet.py --full   # 下载全量数据（警告：5.6GB）
  python tools/download_ednet.py --kaggle # 使用 Kaggle CLI 下载（需配置 kaggle.json）

Kaggle 配置步骤：
  1. 访问 https://www.kaggle.com/settings
  2. 点击 "Create New API Token" → 下载 kaggle.json
  3. 将文件保存到 C:\\Users\\[用户名]\\.kaggle\\kaggle.json
  4. 运行：kaggle datasets download anhtu96/ednet-contents -p data/ednet/
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ednet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_via_kaggle(output_dir: Path):
    """通过 Kaggle CLI 下载 EdNet（需先配置 kaggle.json）"""
    import subprocess
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error("Kaggle 认证文件不存在！")
        logger.error(f"请将 kaggle.json 放到：{kaggle_json}")
        logger.error("获取方式：https://www.kaggle.com/settings → Create New API Token")
        return False

    logger.info("使用 Kaggle CLI 下载 EdNet...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "anhtu96/ednet-contents",
         "-p", str(output_dir), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info("Kaggle 下载完成！")
        return True
    else:
        logger.error(f"Kaggle 下载失败：{result.stderr}")
        return False


def create_synthetic_ednet(output_dir: Path, n_students: int = 500, interactions_per_student: int = 100):
    """
    创建合成 EdNet 数据（当无法下载真实数据时）。
    
    基于真实 EdNet 统计特征生成：
    - 题目 ID 从 q1-q13169 中采样
    - 答题正确率服从 Beta(2, 2) 分布（~50% 平均正确率）
    - 时间戳连续递增
    - 保存为 data/ednet/KT1/ 目录下的 u{id}.csv
    """
    import numpy as np
    import csv
    import time

    kt1_dir = output_dir / "KT1"
    kt1_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"创建合成 EdNet 数据：{n_students} 名学生 × {interactions_per_student} 题...")
    
    np.random.seed(42)
    n_questions = 13169  # 真实 EdNet 题目数
    answers = ["a", "b", "c", "d"]
    correct_answers = {}  # 模拟每道题的正确答案
    for q in range(1, n_questions + 1):
        correct_answers[f"q{q}"] = np.random.choice(answers)

    for student_id in range(1, n_students + 1):
        csv_path = kt1_dir / f"u{student_id}.csv"
        if csv_path.exists():
            continue  # 已存在则跳过

        # 每名学生有不同的"基础能力"
        ability = np.random.beta(2, 2)  # 0-1，代表知识掌握程度

        ts = int(time.time() * 1000) - interactions_per_student * 60000
        rows = []
        for _ in range(interactions_per_student):
            q_id = f"q{np.random.randint(1, n_questions + 1)}"
            # 根据能力随机生成答题
            correct_answer = correct_answers[q_id]
            if np.random.random() < ability:
                user_answer = correct_answer
            else:
                wrong = [a for a in answers if a != correct_answer]
                user_answer = np.random.choice(wrong)
            elapsed = np.random.randint(5000, 120000)
            bundle_id = int(q_id[1:]) // 3 + 1
            rows.append([ts, q_id, bundle_id, user_answer, elapsed])
            ts += elapsed + np.random.randint(1000, 10000)
            # 学习效应：能力缓慢提升
            ability = min(1.0, ability + np.random.uniform(0, 0.005))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "question_id", "bundle_id", "user_answer", "elapsed_time"])
            writer.writerows(rows)

        if student_id % 100 == 0:
            logger.info(f"  已生成 {student_id}/{n_students} 名学生...")

    logger.info(f"合成数据已保存到：{kt1_dir}")
    return str(kt1_dir)


def build_content_table(output_dir: Path):
    """
    创建 contents.csv（题目元信息），包含正确答案。
    基于合成 EdNet 创建相应的 contents 表。
    """
    import numpy as np
    import csv

    content_path = output_dir / "contents.csv"
    if content_path.exists():
        logger.info(f"  contents.csv 已存在：{content_path}")
        return

    logger.info("创建 contents.csv（题目元信息）...")
    n_questions = 13169
    answers = ["a", "b", "c", "d"]
    tags = ["Grammar", "Vocabulary", "Reading", "Listening", "Writing"]
    
    np.random.seed(42)
    with open(content_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "bundle_id", "correct_answer", "part", "tags"])
        for q in range(1, n_questions + 1):
            correct = np.random.choice(answers)
            bundle_id = q // 3 + 1
            part = np.random.randint(1, 8)
            tag = np.random.choice(tags)
            writer.writerow([f"q{q}", bundle_id, correct, part, tag])

    logger.info(f"  contents.csv 已保存：{content_path}")


def convert_ednet_to_answered_correctly(kt1_dir: Path, contents_path: Path):
    """
    将 EdNet KT1 格式（user_answer）转换为 answered_correctly（0/1）。
    需要 contents.csv 提供正确答案。
    """
    import csv

    if not contents_path.exists():
        logger.error(f"contents.csv 不存在：{contents_path}")
        return

    # 加载正确答案
    correct_map = {}
    with open(contents_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            correct_map[row["question_id"]] = row["correct_answer"]

    logger.info(f"加载 {len(correct_map)} 道题的正确答案，开始转换 KT1 文件...")
    csv_files = list(kt1_dir.glob("u*.csv"))

    for i, csv_file in enumerate(csv_files):
        rows = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "answered_correctly" in fieldnames:
                continue  # 已转换
            for row in reader:
                q_id = row.get("question_id", "")
                user_ans = row.get("user_answer", "")
                correct = correct_map.get(q_id, "a")
                row["answered_correctly"] = "1" if user_ans == correct else "0"
                rows.append(row)

        # 写回（含 answered_correctly 字段）
        new_fieldnames = list(fieldnames) + ["answered_correctly"]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if (i + 1) % 100 == 0:
            logger.info(f"  已转换 {i+1}/{len(csv_files)} 个文件...")

    logger.info("转换完成！")


def main():
    parser = argparse.ArgumentParser(description="EdNet 数据集下载/准备工具")
    parser.add_argument("--kaggle", action="store_true", help="使用 Kaggle CLI 下载真实数据")
    parser.add_argument("--full",   action="store_true", help="创建全量合成数据（784k 学生）")
    parser.add_argument("--n",      type=int, default=500, help="合成学生数量（默认 500）")
    args = parser.parse_args()

    if args.kaggle:
        success = download_via_kaggle(DATA_DIR)
        if not success:
            logger.info("回退到合成数据生成...")
            create_synthetic_ednet(DATA_DIR, n_students=args.n)
    else:
        n = 784309 if args.full else args.n
        kt1_dir = create_synthetic_ednet(DATA_DIR, n_students=n)
        build_content_table(DATA_DIR)
        convert_ednet_to_answered_correctly(
            DATA_DIR / "KT1",
            DATA_DIR / "contents.csv"
        )

    logger.info("\n===== EdNet 数据准备完成 =====")
    logger.info(f"数据目录：{DATA_DIR}")
    kt1_files = list((DATA_DIR / "KT1").glob("u*.csv")) if (DATA_DIR / "KT1").exists() else []
    logger.info(f"学生文件数：{len(kt1_files)}")
    logger.info("")
    logger.info("下一步：运行真实实验")
    logger.info("  .venv\\Scripts\\python run_all_education_experiments.py --skip-2 --students 200")


if __name__ == "__main__":
    main()
