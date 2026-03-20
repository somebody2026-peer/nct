"""检查 GPU 训练环境"""
import torch

print("="*70)
print("GPU 训练环境检查")
print("="*70)

print(f"\nPyTorch 版本：{torch.__version__}")
print(f"CUDA 可用：{torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本：{torch.version.cuda}")
    print(f"GPU 数量：{torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU 内存：{gpu_props.total_memory / 1e9:.2f} GB")
    print(f"多处理器数：{gpu_props.multi_processor_count}")
    
    # 推荐 batch_size
    if gpu_props.total_memory > 10e9:
        recommended_batch_size = 128
    elif gpu_props.total_memory > 6e9:
        recommended_batch_size = 64
    else:
        recommended_batch_size = 32
    
    print(f"\n推荐 batch_size: {recommended_batch_size}")
else:
    print("\n未检测到 GPU，将使用 CPU 训练")

print("\n" + "="*70)
