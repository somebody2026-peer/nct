"""
FER2013 数据集下载工具（HuggingFace 镜像，无需 Kaggle 账号）
=============================================================
数据源：abhilash88/fer2013-enhanced（HuggingFace）
  - train: 25,117 张    validation: 5,380 张    test: 5,390 张
  - 7 类情绪：0=Angry 1=Disgust 2=Fear 3=Happy 4=Neutral 5=Sad 6=Surprise
  - pixels 字段：空格分隔的 48x48 灰度像素列表

运行：
  python tools/download_fer2013.py                 # 下载完整数据集
  python tools/download_fer2013.py --max 5000       # 快速测试
  python tools/download_fer2013.py --split train    # 展下载训练集
"""

import os
import sys
import argparse
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 自动使用国内镜像（如未设置 HF_ENDPOINT 则默认使用 hf-mirror.com）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[INFO] 使用 HuggingFace 国内镜像：{os.environ['HF_ENDPOINT']}")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "fer2013"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace 数据集配置
HF_REPO    = "abhilash88/fer2013-enhanced"
HF_SPLITS  = {"train": "Training", "validation": "PublicTest", "test": "PrivateTest"}
# 注意：该数据集中 emotion 6=Neutral，但原始 FER2013 中 6=Neutral，顺序不同
# 映射表：0=Angry,1=Disgust,2=Fear,3=Happy,4=Neutral,5=Sad,6=Surprise
# 原始 FER2013: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
EMOTION_REMAP = {0:0, 1:1, 2:2, 3:3, 4:6, 5:4, 6:5}  # HF索引 → 原始索引

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download(split: str = "train", max_samples: int = 25117):
    """
    从 HuggingFace abhilash88/fer2013-enhanced 下载数据并缓存为 CSV。

    直接利用 HuggingFace 中现成的 pixels 字段，无需图像解码转换。
    """
    csv_path   = DATA_DIR / "fer2013.csv"
    usage_tag  = HF_SPLITS.get(split, split)

    # 检查是否已下载该 split
    if csv_path.exists():
        import csv
        count = 0
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("Usage") == usage_tag:
                    count += 1
        if count > 100:
            logger.info(f"  {usage_tag} split 已存在 ({count} 条)，跳过下载。")
            return count

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请安装 datasets 库：pip install datasets huggingface_hub")
        sys.exit(1)

    logger.info(f"从 HuggingFace 下载 FER2013 ({HF_REPO}, split={split}, max={max_samples})…")
    ds = load_dataset(HF_REPO, split=split)
    n_total = len(ds)
    logger.info(f"  数据集大小：{n_total} 张，开始写入 CSV…")

    import csv as csv_module
    mode = "a" if csv_path.exists() else "w"
    written = 0
    with open(csv_path, mode, newline="") as f:
        writer = csv_module.writer(f)
        if mode == "w":
            writer.writerow(["emotion", "pixels", "Usage"])
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            # 情绪索引重映射（对齐原始 FER2013 理序）
            orig_label = EMOTION_REMAP.get(int(item["emotion"]), int(item["emotion"]))
            writer.writerow([orig_label, item["pixels"], usage_tag])
            written += 1
            if (i + 1) % 5000 == 0:
                logger.info(f"  进度：{i+1}/{min(max_samples, n_total)}…")

    logger.info(f"  完成！写入 {written} 条 → {csv_path}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Download FER2013 from HuggingFace")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split (default: train)")
    parser.add_argument("--max", type=int, default=25117,
                        help="Max samples (default: 25117 full train)")
    parser.add_argument("--all", action="store_true",
                        help="Download all splits (train+validation+test)")
    args = parser.parse_args()

    if args.all:
        counts = {}
        counts["train"]      = download("train",      25117)
        counts["validation"] = download("validation", 5380)
        counts["test"]       = download("test",        5390)
        total = sum(counts.values())
        print(f"\n下载完成：train={counts['train']}, val={counts['validation']}, test={counts['test']} (共 {total} 条)")
        print(f"数据位置：{DATA_DIR / 'fer2013.csv'}")
    else:
        n = download(args.split, args.max)
        print(f"\n下载完成：{n} 条 {args.split} 数据")
        print(f"数据位置：{DATA_DIR / 'fer2013.csv'}")


if __name__ == "__main__":
    main()
