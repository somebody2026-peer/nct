"""
快速示例：从已有模型生成榜单提交文件
=========================================

这个脚本演示如何快速从训练好的 NCT 模型生成 Kaggle 提交文件。

运行示例:
    python examples/quickstart_leaderboard_submission.py --dataset mnist
    python examples/quickstart_leaderboard_submission.py --dataset cifar10
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def quick_mnist_submission(model_path: str = None):
    """快速生成 MNIST 提交文件"""
    
    print("=" * 60)
    print("🚀 MNIST 榜单提交快速示例")
    print("=" * 60)
    
    # 1. 检查模型文件
    if model_path is None:
        model_path = 'results/full_mnist_training/best_model.pt'
    
    model_file = project_root / model_path
    if not model_file.exists():
        print(f"⚠️  模型文件不存在：{model_file}")
        print(f"💡 请先运行训练:")
        print(f"   python experiments/run_full_mnist_training.py")
        return
    
    print(f"✅ 找到模型：{model_file}")
    
    # 2. 加载测试数据
    print("\n📥 加载 MNIST 测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root=project_root / 'data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"✅ 测试集：{len(test_dataset)} 张图像")
    
    # 3. 加载模型
    print("\n🔄 加载模型权重...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from nct_modules.nct_batched import BatchedNCTManager
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取配置
        config = checkpoint.get('config', {})
        n_classes = config.get('n_classes', 10)
        
        # 创建模型
        manager = BatchedNCTManager(
            n_classes=n_classes,
            n_heads=config.get('n_heads', 7),
            n_layers=config.get('n_layers', 4),
            d_model=config.get('d_model', 504)
        )
        model = manager.model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ 模型加载成功")
        if 'accuracy' in checkpoint:
            print(f"📈 验证集准确率：{checkpoint['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print(f"💡 请确保模型路径正确且文件格式有效")
        return
    
    # 4. 生成预测
    print("\n🔮 生成预测...")
    predictions = []
    image_id = 1
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Predicting"):
            data = data.to(device)
            
            # 推理
            outputs = model(data)
            
            # 获取预测
            if isinstance(outputs, tuple):
                pred = outputs[0].argmax(dim=1)
            else:
                pred = outputs.argmax(dim=1)
            
            # 收集结果
            for p in pred.cpu().numpy():
                predictions.append({
                    'ImageId': image_id,
                    'Label': int(p)
                })
                image_id += 1
    
    print(f"✅ 生成 {len(predictions)} 个预测")
    
    # 5. 保存提交文件
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = project_root / 'submissions' / f'mnist_quickstart_{timestamp}'
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    submission_file = submission_dir / 'mnist_submission.csv'
    df = pd.DataFrame(predictions)
    df.to_csv(submission_file, index=False)
    
    print(f"\n💾 提交文件已保存到:")
    print(f"   📁 {submission_file}")
    
    # 6. 显示统计信息
    print(f"\n📊 预测统计:")
    class_counts = df['Label'].value_counts().sort_index()
    for label, count in class_counts.items():
        bar = '█' * (count // 100)
        print(f"   类别 {label}: {count:4d} {bar}")
    
    # 7. 提供下一步指引
    print(f"\n{'='*60}")
    print(f"✅ 完成！下一步操作:")
    print(f"{'='*60}")
    print(f"1️⃣  访问 Kaggle: https://www.kaggle.com/competitions/digit-recognizer")
    print(f"2️⃣  点击 'Submit Prediction' 按钮")
    print(f"3️⃣  上传文件：{submission_file.name}")
    print(f"4️⃣  等待评分并查看排行榜!")
    print(f"\n💡 提示：可以修改模型参数或使用集成方法提高分数")
    
    return submission_file


def quick_cifar10_submission(model_path: str = None):
    """快速生成 CIFAR-10 提交文件"""
    
    print("=" * 60)
    print("🚀 CIFAR-10 榜单提交快速示例")
    print("=" * 60)
    
    # 1. 检查模型文件
    if model_path is None:
        model_path = 'results/cifar10/best_model.pt'
    
    model_file = project_root / model_path
    if not model_file.exists():
        print(f"⚠️  模型文件不存在：{model_file}")
        print(f"💡 请先运行训练:")
        print(f"   python experiments/run_cifar10_full.py --pretrained {model_path}")
        return
    
    print(f"✅ 找到模型：{model_file}")
    
    # 2. 加载测试数据
    print("\n📥 加载 CIFAR-10 测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(
        root=project_root / 'data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"✅ 测试集：{len(test_dataset)} 张图像")
    
    # 3. 加载模型
    print("\n🔄 加载模型权重...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from nct_modules.nct_cifar10 import NCTForCIFAR10
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取配置
        config = checkpoint.get('config', {})
        
        # 创建模型
        model = NCTForCIFAR10(
            d_model=config.get('d_model', 512),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            n_classes=10
        ).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ 模型加载成功")
        if 'accuracy' in checkpoint:
            print(f"📈 验证集准确率：{checkpoint['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return
    
    # 4. 生成预测
    print("\n🔮 生成预测...")
    predictions = []
    image_id = 1
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Predicting"):
            data = data.to(device)
            
            # 推理
            outputs = model(data)
            
            # 获取预测
            pred = outputs.argmax(dim=1)
            
            # 收集结果
            for p in pred.cpu().numpy():
                predictions.append({
                    'ImageId': image_id,
                    'Label': int(p)
                })
                image_id += 1
    
    print(f"✅ 生成 {len(predictions)} 个预测")
    
    # 5. 保存提交文件
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = project_root / 'submissions' / f'cifar10_quickstart_{timestamp}'
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    submission_file = submission_dir / 'cifar10_submission.csv'
    df = pd.DataFrame(predictions)
    df.to_csv(submission_file, index=False)
    
    print(f"\n💾 提交文件已保存到:")
    print(f"   📁 {submission_file}")
    
    # 6. 显示统计信息
    print(f"\n📊 预测统计:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    class_counts = df['Label'].value_counts().sort_index()
    
    for label, count in class_counts.items():
        bar = '█' * (count // 100)
        print(f"   {class_names[label]:10s}: {count:4d} {bar}")
    
    # 7. 提供下一步指引
    print(f"\n{'='*60}")
    print(f"✅ 完成！下一步操作:")
    print(f"{'='*60}")
    print(f"1️⃣  访问 Kaggle: https://www.kaggle.com/c/cifar-10")
    print(f"2️⃣  点击 'Submit Prediction' 按钮")
    print(f"3️⃣  上传文件：{submission_file.name}")
    print(f"4️⃣  等待评分并查看排行榜!")
    print(f"\n💡 提示：CIFAR-10 更具挑战性，建议使用迁移学习和数据增强")
    
    return submission_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='快速生成榜单提交文件')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['mnist', 'cifar10'],
                       help='选择数据集')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径 (可选，使用默认最佳模型)')
    
    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        quick_mnist_submission(args.model)
    elif args.dataset == 'cifar10':
        quick_cifar10_submission(args.model)


if __name__ == '__main__':
    main()
