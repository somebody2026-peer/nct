"""
MEMA EEG 数据集准备工具
=========================
MEMA (Multi-label EEG dataset for Mental Attention states)
西安交通大学，20 名受试者，3 类注意力状态

== 下载方式（需先完成） ==
  1. 打开百度网盘链接：https://pan.baidu.com/s/1ssvZWAI6gwV2ey0cRogDWg
  2. 提取码：2dg7
  3. 下载 "For-ML" 文件夹（.mat 格式，推荐）或 "For-DL" 文件夹
  4. 将 .mat 文件放到：data/mema/
     期望目录结构：
       data/mema/Subject_1.mat
       data/mema/Subject_2.mat
       ...
       data/mema/Subject_20.mat

== 本脚本功能 ==
  1. 验证 MEMA .mat 文件的格式与完整性
  2. 如果真实数据不可用，生成高质量合成 EEG 数据（可用于代码验证）
  3. 将 For-DL 格式转换为 For-ML 格式（如需要）

运行：
  python tools/prepare_mema.py --check   # 检查数据完整性
  python tools/prepare_mema.py --gen 20  # 生成 20 名受试者的合成数据
  python tools/prepare_mema.py --convert # 转换 For-DL → For-ML 格式
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mema"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def check_mema_data():
    """
    检查 MEMA .mat 文件的格式与内容。

    MEMA For-ML 格式：
      - 每个 .mat 文件对应一名受试者
      - data 字段：[n_channels, n_timepoints] 或 [n_trials, n_channels, n_timepoints]
      - labels：3 类（0=neutral, 1=relaxing, 2=concentrating）
      - 采样率：200 Hz
      - 32 个 EEG 通道
    """
    mat_files = sorted(DATA_DIR.rglob("*.mat"))
    if not mat_files:
        logger.warning(f"data/mema/ 目录下未找到 .mat 文件！")
        logger.warning("请先从百度网盘下载：https://pan.baidu.com/s/1ssvZWAI6gwV2ey0cRogDWg (密码: 2dg7)")
        return False

    logger.info(f"找到 {len(mat_files)} 个 .mat 文件")
    try:
        from scipy.io import loadmat
    except ImportError:
        logger.error("请安装 scipy：pip install scipy")
        return False

    ok_count = 0
    for i, mat_file in enumerate(mat_files[:5]):  # 只检查前 5 个
        try:
            mat = loadmat(str(mat_file))
            keys = [k for k in mat.keys() if not k.startswith("_")]
            logger.info(f"  [{i+1}] {mat_file.name}: 字段={keys}")
            for k in keys:
                if isinstance(mat[k], np.ndarray):
                    logger.info(f"        {k}: shape={mat[k].shape}, dtype={mat[k].dtype}")
            ok_count += 1
        except Exception as e:
            logger.error(f"  [{i+1}] {mat_file.name}: 读取失败 - {e}")

    logger.info(f"\n可读文件：{ok_count}/{min(5, len(mat_files))}")
    return ok_count > 0


def generate_synthetic_mema(n_subjects: int = 20, n_trials_per_subject: int = 12, fs: int = 200):
    """
    生成高质量合成 MEMA EEG 数据。

    每名受试者：
    - 12 次试验（4×neutral + 4×relaxing + 4×concentrating）
    - 每试验：32 通道，1200 个时间点（6 秒 × 200 Hz）
    - 频带特征由注意力状态决定（与神经科学文献一致）

    注意力状态 → 频带特征：
      neutral:      Alpha=中等, Beta=低, Theta=低
      relaxing:     Alpha=高,   Beta=极低, Theta=中
      concentrating: Beta=高,   Alpha=低,  Theta/Alpha 比高
    """
    try:
        from scipy.io import savemat
    except ImportError:
        logger.error("请安装 scipy：pip install scipy")
        return

    # 频带特征参数（基于 MEMA 论文图表）
    STATE_PARAMS = {
        0: {"theta": 0.4, "alpha": 0.5, "beta": 0.3, "name": "neutral"},
        1: {"theta": 0.5, "alpha": 0.8, "beta": 0.1, "name": "relaxing"},
        2: {"theta": 0.6, "alpha": 0.3, "beta": 0.7, "name": "concentrating"},
    }

    n_channels   = 32
    n_timepoints = 1200   # 6s × 200Hz
    n_classes    = 3
    n_reps       = n_trials_per_subject // n_classes

    def make_eeg_signal(state: int, t: np.ndarray) -> np.ndarray:
        """生成模拟 EEG 信号（32 通道）"""
        params  = STATE_PARAMS[state]
        signals = []
        for ch in range(n_channels):
            # 基础噪声
            sig = np.random.randn(len(t)) * 0.3
            # 叠加各频带成分
            for f_center, amp_key in [(6, "theta"), (10, "alpha"), (20, "beta")]:
                amp = params[amp_key] * (1 + 0.3 * np.random.randn())
                phase = np.random.uniform(0, 2 * np.pi)
                sig += amp * np.sin(2 * np.pi * f_center * t + phase)
            # 通道间微小差异
            sig *= (1 + 0.1 * np.random.randn())
            signals.append(sig)
        return np.array(signals, dtype=np.float32)  # [32, 1200]

    t = np.arange(n_timepoints) / fs

    logger.info(f"生成合成 MEMA 数据：{n_subjects} 名受试者 × {n_trials_per_subject} 次试验...")
    for subj in range(1, n_subjects + 1):
        np.random.seed(subj * 1000)

        all_trials = []
        all_labels = []

        for rep in range(n_reps):
            for state in range(n_classes):
                eeg = make_eeg_signal(state, t)  # [32, 1200]
                all_trials.append(eeg)
                all_labels.append(state)

        # 打乱顺序（模拟真实实验随机化）
        idx = np.random.permutation(len(all_trials))
        data_arr   = np.array(all_trials, dtype=np.float32)[idx]   # [12, 32, 1200]
        labels_arr = np.array(all_labels, dtype=np.float32)[idx]   # [12]

        mat_path = DATA_DIR / f"Subject_{subj}.mat"
        savemat(str(mat_path), {
            "data":   data_arr,
            "labels": labels_arr,
            "fs":     np.array([fs]),
            "info":   f"Synthetic MEMA subject {subj} (3-class attention EEG)",
        })

        if subj % 5 == 0:
            logger.info(f"  已生成 {subj}/{n_subjects} 名受试者...")

    logger.info(f"合成数据已保存到：{DATA_DIR}")
    logger.info("类别分布（每名受试者）：neutral=4, relaxing=4, concentrating=4")


def convert_for_dl_to_for_ml(for_dl_dir: Path):
    """
    将 MEMA For-DL 格式转换为 For-ML 格式。

    For-DL: [n_trials, n_channels, n_timepoints]，标签独立文件
    For-ML: Subject_N.mat 包含 data + labels 字段
    """
    try:
        from scipy.io import loadmat, savemat
    except ImportError:
        logger.error("请安装 scipy：pip install scipy")
        return

    if not for_dl_dir.exists():
        logger.error(f"For-DL 目录不存在：{for_dl_dir}")
        return

    mat_files = sorted(for_dl_dir.glob("*.mat"))
    logger.info(f"转换 {len(mat_files)} 个 For-DL 文件...")

    for mat_file in mat_files:
        mat = loadmat(str(mat_file))
        # For-DL 字段尝试
        data_key  = next((k for k in mat if not k.startswith("_") and
                         isinstance(mat[k], np.ndarray) and mat[k].ndim == 3), None)
        label_key = next((k for k in mat if "label" in k.lower()), None)

        if data_key is None:
            logger.warning(f"  {mat_file.name}: 无法识别 data 字段，跳过")
            continue

        out_path = DATA_DIR / mat_file.name
        savemat(str(out_path), {
            "data":   mat[data_key],
            "labels": mat[label_key] if label_key else np.zeros(mat[data_key].shape[0]),
        })
        logger.info(f"  {mat_file.name} → {out_path}")

    logger.info("转换完成！")


def main():
    parser = argparse.ArgumentParser(description="MEMA EEG 数据集准备工具")
    parser.add_argument("--check",   action="store_true", help="检查 .mat 文件格式")
    parser.add_argument("--gen",     type=int, metavar="N", help="生成 N 名受试者的合成数据（默认 20）")
    parser.add_argument("--convert", metavar="DIR", help="将 For-DL 格式目录转换为 For-ML 格式")
    parser.add_argument("--full",    action="store_true", help="生成完整合成数据集（20 受试者）")
    args = parser.parse_args()

    if args.check:
        has_data = check_mema_data()
        if not has_data:
            logger.info("\n建议：运行 --gen 20 生成合成数据进行代码验证")

    elif args.gen is not None or args.full:
        n = args.gen or 20
        generate_synthetic_mema(n_subjects=n)

    elif args.convert:
        convert_for_dl_to_for_ml(Path(args.convert))

    else:
        # 默认：检查然后生成
        logger.info("=== MEMA 数据准备 ===")
        logger.info(f"数据目录：{DATA_DIR}")
        logger.info("")
        logger.info("下载真实数据（百度网盘）：")
        logger.info("  链接：https://pan.baidu.com/s/1ssvZWAI6gwV2ey0cRogDWg")
        logger.info("  密码：2dg7")
        logger.info("  下载 For-ML 文件夹，将 Subject_*.mat 放入 data/mema/")
        logger.info("")
        has_real = check_mema_data()
        if not has_real:
            logger.info("\n未找到真实数据，生成高质量合成数据（用于代码验证）...")
            generate_synthetic_mema(n_subjects=20)
            logger.info("\n合成数据已就绪！运行 MEMA 实验：")
            logger.info("  .venv\\Scripts\\python experiments\\mema_nct_experiment.py")


if __name__ == "__main__":
    main()
