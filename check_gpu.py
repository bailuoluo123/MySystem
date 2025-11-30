"""
检测 GPU 模式模板识别是否可用
"""
import sys

print("=" * 60)
print("GPU 模式模板识别检测")
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

# 3. 检查 CUDA 模块
print("[3] 检查 OpenCV CUDA 模块...")
try:
    if hasattr(cv2, 'cuda'):
        print("[OK] OpenCV 包含 CUDA 模块")
        
        # 检查 CUDA 设备数量
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"     CUDA 设备数量: {device_count}")
            
            if device_count > 0:
                print("[OK] 检测到 CUDA 设备！")
                for i in range(device_count):
                    try:
                        device_info = cv2.cuda.getDevice(i)
                        print(f"     设备 {i}: {device_info}")
                    except:
                        print(f"     设备 {i}: 无法获取详细信息")
            else:
                print("[WARN] 未检测到 CUDA 设备")
        except Exception as e:
            print(f"[X] 获取 CUDA 设备数量失败: {e}")
            
    else:
        print("[X] OpenCV 不包含 CUDA 模块")
except Exception as e:
    print(f"[X] CUDA 模块检查失败: {e}")

print()

# 4. 尝试创建 CUDA 对象（实际测试）
print("[4] 测试 CUDA 对象创建...")
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
        except cv2.error as e:
            error_msg = str(e)
            if "No CUDA support" in error_msg or "without CUDA support" in error_msg:
                print("[X] GpuMat 操作失败: OpenCV 编译时未启用 CUDA 支持")
                print("     这是 CPU-only 版本的 OpenCV")
            else:
                print(f"[X] GpuMat 操作失败: {e}")
        except Exception as e:
            print(f"[X] GpuMat 操作失败: {e}")
except Exception as e:
    print(f"[X] CUDA 对象测试失败: {e}")

print()

# 5. 测试模板匹配（如果 CUDA 可用）
print("[5] 测试 CUDA 模板匹配...")
try:
    import numpy as np
    if hasattr(cv2, 'cuda'):
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count > 0:
            # 创建测试图像和模板
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            test_template = test_image[50:100, 50:100].copy()
            
            # 转换为灰度
            gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(test_template, cv2.COLOR_BGR2GRAY)
            
            try:
                # 尝试使用 CUDA 模板匹配
                gray_image_gpu = cv2.cuda_GpuMat()
                gray_image_gpu.upload(gray_image)
                
                template_gpu = cv2.cuda_GpuMat()
                template_gpu.upload(gray_template)
                
                matcher = cv2.cuda.createTemplateMatching(gray_image_gpu.type(), cv2.TM_CCOEFF_NORMED)
                result_gpu = matcher.match(gray_image_gpu, template_gpu)
                result = result_gpu.download()
                
                print("[OK] CUDA 模板匹配测试成功！")
                print(f"     结果矩阵大小: {result.shape}")
                print(f"     最大值: {result.max():.4f}")
                print("[结论] GPU 模式模板识别可用！")
            except Exception as e:
                print(f"[X] CUDA 模板匹配测试失败: {e}")
                print("[结论] GPU 模式模板识别不可用")
        else:
            print("[跳过] 未检测到 CUDA 设备，跳过 CUDA 模板匹配测试")
            print("[结论] GPU 模式模板识别不可用（无 CUDA 设备）")
    else:
        print("[跳过] OpenCV 不包含 CUDA 模块，跳过 CUDA 模板匹配测试")
        print("[结论] GPU 模式模板识别不可用（无 CUDA 模块）")
except Exception as e:
    print(f"[X] 模板匹配测试失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 6. 检查 auto_annotator 的 CUDA 状态
print("[6] 检查 auto_annotator CUDA 状态...")
try:
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.auto_annotator import AutoAnnotator
    
    annotator = AutoAnnotator()
    cuda_available = annotator.cuda_available
    compute_mode = annotator.get_compute_mode()
    
    print(f"     CUDA 可用: {cuda_available}")
    print(f"     计算模式: {compute_mode}")
    
    if cuda_available:
        print("[结论] auto_annotator 已启用 GPU 模式")
    else:
        print("[结论] auto_annotator 使用 CPU 模式")
except Exception as e:
    print(f"[X] 检查 auto_annotator 失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 7. 总结
print("=" * 60)
print("检测总结")
print("=" * 60)

try:
    if hasattr(cv2, 'cuda'):
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count > 0:
            print("[最终结论] [OK] GPU 模式模板识别可用！")
            print("           系统将自动使用 GPU 加速模板匹配")
        else:
            print("[最终结论] [X] GPU 模式模板识别不可用")
            print("           原因: OpenCV 无法访问 CUDA 设备")
            print("           系统将使用 CPU 多线程并行模式")
    else:
        print("[最终结论] [X] GPU 模式模板识别不可用")
        print("           原因: OpenCV 不支持 CUDA")
        print("           系统将使用 CPU 多线程并行模式")
except:
    print("[最终结论] [X] 无法确定 GPU 状态")
    print("           系统将使用 CPU 多线程并行模式")

print()
print("=" * 60)

