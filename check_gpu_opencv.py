"""
检查 GPU 和 OpenCV CUDA 支持状态
"""
import sys

print("=" * 60)
print("GPU 和 OpenCV CUDA 支持检查")
print("=" * 60)
print()

# 1. 检查 NVIDIA GPU
print("[1] 检查 NVIDIA GPU...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("[OK] 检测到 NVIDIA GPU")
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line and 'Driver' in line:
                print(f"     {line.strip()}")
            if 'CUDA Version' in line:
                print(f"     {line.strip()}")
            if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                print(f"     GPU: {line.strip()}")
    else:
        print("[X] nvidia-smi 命令执行失败")
except FileNotFoundError:
    print("[X] 未找到 nvidia-smi 命令")
except Exception as e:
    print(f"[X] 检查失败: {e}")

print()

# 2. 检查 OpenCV
print("[2] 检查 OpenCV...")
try:
    import cv2
    print(f"[OK] OpenCV 版本: {cv2.__version__}")
except Exception as e:
    print(f"[X] OpenCV 导入失败: {e}")
    sys.exit(1)

print()

# 3. 检查 CUDA 支持
print("[3] 检查 OpenCV CUDA 支持...")
try:
    if hasattr(cv2, 'cuda'):
        print("[OK] OpenCV 包含 CUDA 模块")
        
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"     CUDA 设备数量: {device_count}")
        
        if device_count > 0:
            print("[OK] GPU 可以被 OpenCV 使用！")
            print("     识别时将自动使用 GPU 加速")
        else:
            print("[X] GPU 无法被 OpenCV 使用")
            print("     原因: OpenCV 是 CPU-only 版本")
            print()
            print("     解决方案:")
            print("     1. 使用 conda 安装支持 CUDA 的 OpenCV:")
            print("        conda install -c conda-forge opencv")
            print("     2. 或者从源码编译支持 CUDA 的 OpenCV")
    else:
        print("[X] OpenCV 不包含 CUDA 模块")
        print("     需要安装支持 CUDA 的 OpenCV 版本")
except Exception as e:
    print(f"[X] CUDA 检查失败: {e}")

print()

# 4. 尝试创建 CUDA 对象
print("[4] 尝试创建 CUDA 对象...")
try:
    import numpy as np
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    if hasattr(cv2, 'cuda'):
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(test_image)
            downloaded = gpu_mat.download()
            print("[OK] 成功创建并上传/下载 GpuMat")
            print(f"     上传图像大小: {test_image.shape}")
            print(f"     下载图像大小: {downloaded.shape}")
            print()
            print("[OK] GPU 加速已就绪，识别时将使用 GPU！")
        except cv2.error as e:
            error_msg = str(e)
            if "No CUDA support" in error_msg or "without CUDA support" in error_msg:
                print("[X] GpuMat 操作失败: OpenCV 编译时未启用 CUDA 支持")
                print("     这是 CPU-only 版本的 OpenCV")
                print()
                print("     解决方案:")
                print("     conda install -c conda-forge opencv")
            else:
                print(f"[X] GpuMat 操作失败: {e}")
        except Exception as e:
            print(f"[X] GpuMat 操作失败: {e}")
except Exception as e:
    print(f"[X] CUDA 对象测试失败: {e}")

print()
print("=" * 60)
print("检查完成")
print("=" * 60)

