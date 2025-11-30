"""
自动标注算法模块 - 使用简单的图像识别方法进行自动标注
不使用 YOLO 模型，只使用基础的图像处理技术
"""
import cv2
import numpy as np
import os
import glob
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logger = logging.getLogger('labelImg')
from typing import List, Tuple, Optional, Union, Callable


def load_image_unicode(path: str) -> Optional[np.ndarray]:
    """兼容包含中文路径的图像读取。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.error(f"[自动标注] 读取图像失败: {path} -> {exc}")
        return None


class AutoAnnotator:
    """自动标注器，使用颜色分割和轮廓检测"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """初始化自动标注器"""
        # 默认参数（针对冒险岛游戏场景）
        self.min_contour_area = 100  # 最小轮廓面积
        self.max_contour_area = 50000  # 最大轮廓面积
        self.aspect_ratio_range = (0.3, 3.0)  # 宽高比范围
        
        # 模板目录
        if templates_dir is None:
            templates_dir = os.path.join(os.path.expanduser('~'), '.labelimg_templates')
        self.templates_dir = templates_dir
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # 模板匹配参数
        self.template_match_threshold = 0.7  # 模板匹配阈值
        
        # 特征匹配参数（ORB）
        self.use_feature_matching = False  # 是否使用特征匹配（更鲁棒但更慢）
        self.orb_match_ratio = 0.75  # ORB特征匹配比例阈值
        # 模板缓存：减少重复磁盘读取与灰度转换开销
        # key: 模板完整路径 -> value: {'mtime': float, 'img': np.ndarray, 'gray': np.ndarray}
        self._template_cache = {}

        # GPU (CUDA) 加速可用性检测
        self.cuda_available = self._init_cuda()

    def _init_cuda(self) -> bool:
        """检测并初始化 CUDA，如果存在 NVIDIA GPU 则开启加速。"""
        # 首先检查 OpenCV 是否有 CUDA 模块
        if not hasattr(cv2, "cuda"):
            logger.warning("[自动标注] OpenCV 未编译 CUDA 支持，将使用 CPU 模式")
            print("[自动标注] OpenCV 未编译 CUDA 支持，将使用 CPU 模式")
            print("提示：如需使用 GPU 加速，请安装支持 CUDA 的 OpenCV 版本")
            return False
        
        # 尝试多种方法检测 NVIDIA GPU
        has_nvidia_gpu = False
        gpu_name = None
        
        # 方法1: 尝试使用 nvidia-ml-py（如果已安装）
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count_nvml = pynvml.nvmlDeviceGetCount()
            if device_count_nvml > 0:
                has_nvidia_gpu = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                print(f"[自动标注] 通过 nvidia-ml-py 检测到 {device_count_nvml} 个 NVIDIA GPU: {gpu_name}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[自动标注] nvidia-ml-py 检测失败: {e}")
        
        # 方法2: 尝试使用 nvidia-smi 命令
        if not has_nvidia_gpu:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    has_nvidia_gpu = True
                    gpu_name = result.stdout.strip().split('\n')[0]
                    print(f"[自动标注] 通过 nvidia-smi 检测到 NVIDIA GPU: {gpu_name}")
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                pass
        
        # 检查 OpenCV 的 CUDA 设备数量
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"[自动标注] OpenCV CUDA 设备数量: {device_count}")
            
            # 如果 device_count 为 0，但检测到 NVIDIA GPU，尝试其他方法
            if device_count == 0:
                if has_nvidia_gpu:
                    logger.warning("[自动标注] 检测到 NVIDIA GPU，但 OpenCV 无法访问 CUDA 设备")
                    print(f"[自动标注] [WARN] 检测到 NVIDIA GPU ({gpu_name})，但 OpenCV 无法访问 CUDA 设备")
                    
                    # 尝试其他检测方法
                    try:
                        # 尝试检查 isCudaDevice
                        if hasattr(cv2.cuda, 'isCudaDevice'):
                            is_device = cv2.cuda.isCudaDevice(0)
                            print(f"[自动标注] cv2.cuda.isCudaDevice(0): {is_device}")
                            if is_device:
                                print("[自动标注] 尝试强制使用设备 0...")
                                cv2.cuda.setDevice(0)
                                # 再次检查
                                device_count = cv2.cuda.getCudaEnabledDeviceCount()
                                print(f"[自动标注] 重新检查 CUDA 设备数量: {device_count}")
                    except Exception as e:
                        print(f"[自动标注] 额外检测方法失败: {e}")
                    
                    if device_count == 0:
                        print("可能的原因：")
                        print("  1. OpenCV 安装的是 CPU-only 版本（虽然包含 cuda 模块但实际不支持）")
                        print("  2. CUDA 驱动版本与 OpenCV 编译时的 CUDA 版本不匹配")
                        print("  3. 需要安装支持 CUDA 的 OpenCV 版本")
                        print("解决方案：")
                        print("  - 方法1（推荐）: 使用 conda 安装支持 CUDA 的 OpenCV")
                        print("    conda install -c conda-forge opencv")
                        print("  - 方法2: 从源码编译支持 CUDA 的 OpenCV（复杂）")
                        print("  - 方法3: 使用预编译的 CUDA 版本（如果可用）")
                        print("当前将使用 CPU 多线程并行模式")
                        return False
                else:
                    logger.warning("[自动标注] 未检测到可用的 CUDA 设备，将使用 CPU 模式")
                    print("[自动标注] 未检测到可用的 CUDA 设备，将使用 CPU 模式")
                    return False
            
            # 设置使用第一个 GPU
            cv2.cuda.setDevice(0)
            
            # 尝试创建一个小的测试 GpuMat 来验证 GPU 是否真的可用
            try:
                test_mat = cv2.cuda_GpuMat()
                test_array = np.zeros((10, 10), dtype=np.uint8)
                test_mat.upload(test_array)
                test_mat.release()
                
                gpu_info = ""
                if gpu_name:
                    gpu_info = f" ({gpu_name})"
                logger.info(f"[自动标注] 检测到 {device_count} 个 CUDA 设备{gpu_info}，启用 GPU 加速模板匹配")
                print(f"[自动标注] [OK] GPU 加速已启用（检测到 {device_count} 个 CUDA 设备{gpu_info}）")
                return True
            except Exception as test_exc:
                logger.warning(f"[自动标注] GPU 测试失败，将使用 CPU 模式: {test_exc}")
                print(f"[自动标注] GPU 测试失败，将使用 CPU 模式: {test_exc}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as exc:
            logger.warning(f"[自动标注] 初始化 CUDA 失败，退回 CPU 匹配: {exc}")
            print(f"[自动标注] 初始化 CUDA 失败，退回 CPU 匹配: {exc}")
            import traceback
            traceback.print_exc()
        return False

    def get_compute_mode(self) -> str:
        """返回当前使用的计算后端描述（GPU / CPU）。"""
        return "GPU" if self.cuda_available else "CPU"

    def _load_template_cached(self, template_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        带缓存的模板加载函数：
        - 如果文件未变更，复用已加载的彩色图与灰度图
        - 如果文件有更新或未缓存，则重新读取并更新缓存
        """
        try:
            mtime = os.path.getmtime(template_path)
        except OSError:
            return None, None

        cached = self._template_cache.get(template_path)
        if cached and cached.get('mtime') == mtime:
            return cached.get('img'), cached.get('gray')

        img = load_image_unicode(template_path)
        if img is None:
            return None, None

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        self._template_cache[template_path] = {
            'mtime': mtime,
            'img': img,
            'gray': gray,
        }
        return img, gray
    
    def get_template_paths(self, template_name: Optional[Union[str, List[str]]] = None) -> List[str]:
        """根据模板名称或全部模板返回路径列表"""
        if template_name:
            if isinstance(template_name, str):
                template_names = [template_name]
            else:
                template_names = template_name
            
            template_paths: List[str] = []
            for name in template_names:
                template_path = os.path.join(self.templates_dir, name + '.png')
                if not os.path.exists(template_path):
                    found = False
                    for ext in ['.jpg', '.jpeg']:
                        alt_path = os.path.join(self.templates_dir, name + ext)
                        if os.path.exists(alt_path):
                            template_path = alt_path
                            found = True
                            break
                    if not found:
                        logger.warning(f"[自动标注] 模板不存在: {name}，跳过")
                        continue
                template_paths.append(template_path)
        else:
            patterns = [
                os.path.join(self.templates_dir, '*.png'),
                os.path.join(self.templates_dir, '*.jpg'),
                os.path.join(self.templates_dir, '*.jpeg')
            ]
            template_paths = []
            for pattern in patterns:
                template_paths.extend(glob.glob(pattern))
        return template_paths
        
    def detect_by_color(self, image: np.ndarray, 
                       color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
                       color_space: str = 'BGR') -> List[Tuple[int, int, int, int]]:
        """
        通过颜色分割检测目标
        :param image: 输入图像 (BGR格式)
        :param color_range: 颜色范围 ((min_b, min_g, min_r), (max_b, max_g, max_r))
        :param color_space: 颜色空间 ('BGR' 或 'HSV')
        :return: 检测框列表 [(x, y, w, h), ...]
        """
        if color_space == 'HSV':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, color_range[0], color_range[1])
        else:
            # BGR 颜色空间
            mask = cv2.inRange(image, color_range[0], color_range[1])
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                    boxes.append((x, y, w, h))
        
        return boxes
    
    def detect_by_template(self, image: np.ndarray,
                          template: np.ndarray,
                          threshold: float = 0.7,
                          gray_image: Optional[np.ndarray] = None,
                          gray_template: Optional[np.ndarray] = None,
                          gray_image_gpu: Optional['cv2.cuda_GpuMat'] = None
                          ) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        通过模板匹配检测目标（改进版：使用灰度匹配 + 图像预处理）
        :param image: 输入图像 (BGR格式)
        :param template: 模板图像 (BGR格式)
        :param threshold: 匹配阈值 (0-1)
        :return: (检测框列表, 最高匹配分数)
        """
        # 检查图像和模板大小
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            return [], 0.0
        
        # 转换为灰度图（灰度匹配更稳定，不受颜色变化影响）
        if gray_image is None:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

        if gray_template is None:
            if len(template.shape) == 3:
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                gray_template = template
        
        all_boxes: List[Tuple[int, int, int, int]] = []
        best_score = 0.0
        template_h, template_w = gray_template.shape

        # 特殊处理：如果模板和图像完全相同，直接返回匹配结果
        if gray_image.shape == gray_template.shape:
            diff = cv2.absdiff(gray_image, gray_template)
            diff_sum = np.sum(diff)
            if diff_sum == 0:
                # 图像完全相同，应该能匹配到
                logger.warning(f"[自动标注] 图像和模板完全相同，差异为0，应该能匹配")
                # 在整个图像中搜索模板（可能有多个）
                result = self._match_template(gray_image, gray_template, cv2.TM_CCOEFF_NORMED, gray_image_gpu)
                locations = np.where(result >= 0.99)  # 完全相同应该分数接近1.0
                for pt in zip(*locations[::-1]):
                    all_boxes.append((pt[0], pt[1], template_w, template_h))
                if all_boxes:
                    logger.info(f"[自动标注] 完全相同匹配成功，找到 {len(all_boxes)} 个目标")
                    return all_boxes, 1.0
                else:
                    logger.error(f"[自动标注] 错误：图像完全相同但匹配失败！这是算法bug！")

        # 注意：不进行直方图均衡化，因为这会改变图像特征，导致相同图片无法匹配
        # 如果图像对比度很低，可以考虑在保存模板时进行预处理
        
        # 首先尝试1:1匹配（如果模板和图像完全一样，应该能匹配到）
        # 使用最佳匹配方法
        method = cv2.TM_CCOEFF_NORMED  # 归一化相关系数（最常用，对光照变化鲁棒）
        
        # 先测试原始大小匹配
        result = self._match_template(gray_image, gray_template, method, gray_image_gpu)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
        
        logger.info(f"[自动标注] 1:1匹配分数: {max_val:.4f} (阈值: {threshold:.3f})")
        print(f"[DEBUG] 1:1匹配分数: {max_val:.4f} (阈值: {threshold:.3f})")
        
        # 如果1:1匹配成功，直接使用
        if max_val >= threshold:
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                all_boxes.append((pt[0], pt[1], template_w, template_h))
            msg = f"1:1匹配成功，找到 {len(all_boxes)} 个目标"
            logger.info(f"[自动标注] {msg}")
            print(f"[INFO] {msg}")
        
        # 如果1:1匹配失败，尝试多尺度匹配
        if not all_boxes:
            print(f"[DEBUG] 1:1匹配失败，尝试多尺度匹配...")
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]  # 多尺度范围
            
            for scale in scales:
                if scale == 1.0:
                    continue  # 已经尝试过了
                
                # 缩放模板
                new_w = int(template_w * scale)
                new_h = int(template_h * scale)
                if new_w < 10 or new_h < 10 or new_w > gray_image.shape[1] or new_h > gray_image.shape[0]:
                    continue
                
                scaled_template = cv2.resize(gray_template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 模板匹配
                result = self._match_template(gray_image, scaled_template, method, gray_image_gpu)
                
                # 获取最高匹配分数
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                
                # 查找所有匹配位置
                locations = np.where(result >= threshold)
                
                # 获取匹配位置
                for pt in zip(*locations[::-1]):
                    all_boxes.append((pt[0], pt[1], new_w, new_h))
                
                if max_val >= threshold:
                    print(f"[DEBUG] 尺度 {scale:.1f}x 匹配分数: {max_val:.4f}，找到匹配")
        
        # 记录最佳匹配分数
        if best_score > 0:
            print(f"[DEBUG] 模板匹配最高分数: {best_score:.3f} (阈值: {threshold:.3f})")
            if best_score < threshold:
                print(f"[WARN] 最高匹配分数 {best_score:.3f} 低于阈值 {threshold:.3f}")
                print(f"      → 建议降低阈值到 {max(0.2, best_score - 0.05):.2f} 或检查模板质量")
            
            # 如果分数非常低（<0.3），可能是图像读取或格式问题
            if best_score < 0.3:
                print(f"[ERROR] 匹配分数异常低！可能的原因：")
                print(f"      1. 图像格式不匹配（BGR vs RGB）")
                print(f"      2. 图像大小不匹配")
                print(f"      3. 模板保存时颜色转换错误")
                print(f"      图像大小: {image.shape}, 模板大小: {template.shape}")
                
                # 尝试直接比较（如果大小相同）
                if image.shape == template.shape:
                    diff = cv2.absdiff(image, template)
                    diff_sum = np.sum(diff)
                    print(f"      图像差异总和: {diff_sum} (如果为0则完全相同)")
                    if diff_sum == 0:
                        print(f"      [ERROR] 图像完全相同但匹配失败！这是算法bug！")
        
        # 非极大值抑制（去除重叠框）
        if all_boxes:
            all_boxes = self.non_max_suppression(all_boxes, overlap_threshold=0.3)
        
        return all_boxes, best_score

    def _match_template(self, gray_image: np.ndarray,
                        gray_template: np.ndarray,
                        method: int,
                        gray_image_gpu: Optional['cv2.cuda_GpuMat'] = None):
        """根据当前环境自动选择 GPU 或 CPU 的 matchTemplate"""
        if self.cuda_available and gray_image_gpu is not None:
            result = self._match_template_gpu(gray_image_gpu, gray_template, method)
            if result is not None:
                return result
            # 如果 GPU 匹配失败（可能因为尺寸或驱动问题），记录并回退到 CPU
            logger.warning("[自动标注] CUDA 匹配失败，切换回 CPU 模式")
            print("[自动标注] [WARN] CUDA 匹配失败，切换回 CPU 模式")
            self.cuda_available = False
        return cv2.matchTemplate(gray_image, gray_template, method)

    def _match_template_gpu(self,
                            gray_image_gpu: 'cv2.cuda_GpuMat',
                            gray_template: np.ndarray,
                            method: int):
        """使用 CUDA 执行模板匹配，返回结果矩阵（numpy）。"""
        try:
            template_gpu = cv2.cuda_GpuMat()
            template_gpu.upload(gray_template)
            matcher = cv2.cuda.createTemplateMatching(gray_image_gpu.type(), method)
            result_gpu = matcher.match(gray_image_gpu, template_gpu)
            result = result_gpu.download()
            template_gpu.release()
            result_gpu.release()
            return result
        except Exception as exc:
            logger.error(f"[自动标注] CUDA 模板匹配失败: {exc}")
            print(f"[自动标注] ❌ CUDA 模板匹配失败: {exc}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_by_contour(self, image: np.ndarray,
                          canny_low: int = 50,
                          canny_high: int = 150) -> List[Tuple[int, int, int, int]]:
        """
        通过边缘检测和轮廓查找检测目标
        :param image: 输入图像
        :param canny_low: Canny 边缘检测低阈值
        :param canny_high: Canny 边缘检测高阈值
        :return: 检测框列表 [(x, y, w, h), ...]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 边缘检测
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                    boxes.append((x, y, w, h))
        
        return boxes
    
    def detect_crocodiles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        专门检测鳄鱼（针对冒险岛游戏）
        使用绿色颜色分割 + 轮廓检测
        :param image: 输入图像
        :return: 检测框列表 [(x, y, w, h), ...]
        """
        # 转换到 HSV 颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 绿色范围（针对鳄鱼）
        # 可以根据实际游戏画面调整这些值
        lower_green = np.array([40, 50, 50])   # 绿色下限
        upper_green = np.array([80, 255, 255])  # 绿色上限
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # 鳄鱼通常面积在 500-20000 像素之间（根据实际调整）
            if 500 <= area <= 20000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                # 鳄鱼通常宽高比在 0.5-2.0 之间
                if 0.5 <= aspect_ratio <= 2.0:
                    boxes.append((x, y, w, h))
        
        # 非极大值抑制
        boxes = self.non_max_suppression(boxes, overlap_threshold=0.3)
        
        return boxes
    
    def non_max_suppression(self, boxes: List[Tuple[int, int, int, int]], 
                           overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """
        非极大值抑制，去除重叠的检测框
        :param boxes: 检测框列表 [(x, y, w, h), ...]
        :param overlap_threshold: 重叠阈值
        :return: 过滤后的检测框列表
        """
        if len(boxes) == 0:
            return []
        
        # 转换为 (x1, y1, x2, y2) 格式
        boxes_array = np.array([(x, y, x + w, y + h) for x, y, w, h in boxes], dtype=np.float32)
        
        # 计算面积
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
        
        # 按面积排序（从大到小）
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            # 计算重叠度
            xx1 = np.maximum(boxes_array[i, 0], boxes_array[indices[1:], 0])
            yy1 = np.maximum(boxes_array[i, 1], boxes_array[indices[1:], 1])
            xx2 = np.minimum(boxes_array[i, 2], boxes_array[indices[1:], 2])
            yy2 = np.minimum(boxes_array[i, 3], boxes_array[indices[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[indices[1:]]
            
            # 保留重叠度小于阈值的框
            indices = indices[1:][overlap <= overlap_threshold]
        
        return [boxes[i] for i in keep]
    
    def convert_to_yolo_format(self, boxes: List[Tuple[int, int, int, int]], 
                              img_width: int, img_height: int) -> List[Tuple[float, float, float, float]]:
        """
        将检测框转换为 YOLO 格式（归一化坐标）
        :param boxes: 检测框列表 [(x, y, w, h), ...] (像素坐标)
        :param img_width: 图像宽度
        :param img_height: 图像高度
        :return: YOLO 格式列表 [(center_x, center_y, width, height), ...] (归一化)
        """
        yolo_boxes = []
        for x, y, w, h in boxes:
            # 计算中心点
            center_x = (x + w / 2.0) / img_width
            center_y = (y + h / 2.0) / img_height
            # 归一化宽高
            norm_width = w / img_width
            norm_height = h / img_height
            
            # 确保坐标在 [0, 1] 范围内
            center_x = max(0.0, min(1.0, center_x))
            center_y = max(0.0, min(1.0, center_y))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))
            
            yolo_boxes.append((center_x, center_y, norm_width, norm_height))
        
        return yolo_boxes
    
    def auto_annotate(self, image_path: str, 
                     method: str = 'template',
                     class_id: int = 0,
                     template_name: Optional[str] = None,
                     threshold: Optional[float] = None,
                     progress_callback: Optional[Callable[[int, int, str], None]] = None
                     ) -> List[Tuple[int, float, float, float, float]]:
        """
        自动标注单张图片
        :param image_path: 图片路径
        :param method: 检测方法 ('crocodile', 'color', 'contour', 'template')
        :param class_id: 类别ID
        :param template_name: 模板名称（如果使用模板匹配）
        :param threshold: 匹配阈值（如果使用模板匹配）
        :return: YOLO 格式标签列表 [(class_id, center_x, center_y, width, height), ...]
        """
        # 读取图片（支持中文路径）
        image = load_image_unicode(image_path)
        if image is None:
            logger.error(f"[自动标注] 无法读取图片: {image_path}")
            return []
        
        img_height, img_width = image.shape[:2]
        
        # 根据方法选择检测算法
        if method == 'template':
            # 使用模板匹配，返回带标签信息的框
            labeled_boxes = self.detect_by_templates(
                image, template_name, threshold, progress_callback
            )
            boxes = [box for _, box in labeled_boxes]
        elif method == 'crocodile':
            boxes = self.detect_crocodiles(image)
        elif method == 'color':
            # 默认绿色检测
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            boxes = self.detect_by_color(image, (lower_green, upper_green), 'HSV')
        elif method == 'contour':
            boxes = self.detect_by_contour(image)
        else:
            boxes = []
        
        # 转换为 YOLO 格式
        yolo_boxes = self.convert_to_yolo_format(boxes, img_width, img_height)
        
        labels = []
        if method == 'template':
            for (template_label, _), (cx, cy, w, h) in zip(labeled_boxes, yolo_boxes):
                labels.append((template_label, class_id, cx, cy, w, h))
        else:
            labels = [(None, class_id, cx, cy, w, h) for cx, cy, w, h in yolo_boxes]
        
        return labels
    
    def detect_by_templates(self, image: np.ndarray, 
                           template_name: Optional[Union[str, List[str]]] = None,
                           threshold: Optional[float] = None,
                           progress_callback: Optional[Callable[[int, int, str], None]] = None
                           ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        使用所有保存的模板进行匹配检测
        :param image: 输入图像
        :param template_name: 指定模板名称（字符串或列表，如果为None，使用所有模板）
        :param threshold: 匹配阈值（如果为None，使用默认阈值）
        :return: 检测框列表 [(template_label, (x, y, w, h)), ...]
        """
        if threshold is None:
            threshold = self.template_match_threshold
        
        logger.info(f"[自动标注] 开始模板匹配，阈值: {threshold:.2f}, 图像大小: {image.shape}")
        print(f"[DEBUG] 开始模板匹配，阈值: {threshold:.2f}, 图像大小: {image.shape}")
        
        all_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []

        template_paths = self.get_template_paths(template_name)

        if not template_paths:
            print(f"[WARN] 未找到任何模板文件，模板目录: {self.templates_dir}")
            return []

        print(f"[DEBUG] 找到 {len(template_paths)} 个模板文件")

        # 预先计算整张图的灰度图，避免在每个模板里重复转换
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_image_gpu = None
        if self.cuda_available:
            try:
                gray_image_gpu = cv2.cuda_GpuMat()
                gray_image_gpu.upload(gray_image)
                print(f"[自动标注] [OK] 已上传图像到 GPU ({gray_image.shape[1]}x{gray_image.shape[0]})")
            except Exception as exc:
                logger.warning(f"[自动标注] 上传 CUDA 图像失败，回退 CPU: {exc}")
                print(f"[自动标注] [WARN] 上传 CUDA 图像失败，回退 CPU: {exc}")
                import traceback
                traceback.print_exc()
                gray_image_gpu = None
                self.cuda_available = False
        else:
            print(f"[自动标注] 使用 CPU 模式（CUDA 不可用）")

        total_templates = len(template_paths)
        
        # 使用多线程并行处理模板匹配（CPU 模式下显著提升速度）
        # 如果 GPU 可用，使用单线程（GPU 本身已并行）
        use_parallel = not self.cuda_available and total_templates > 1
        
        if use_parallel:
            import multiprocessing
            max_workers = min(multiprocessing.cpu_count(), total_templates, 8)  # 最多8个线程
            print(f"[自动标注] 使用 CPU 多线程并行处理（{max_workers} 个线程）")
            
            def process_template(template_path):
                """处理单个模板的匹配任务"""
                try:
                    # 使用带缓存的模板加载
                    template, gray_template = self._load_template_cached(template_path)
                    if template is None or gray_template is None:
                        return None, None, None
                    
                    # 检查模板大小
                    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                        return None, None, None
                    
                    # 模板匹配（CPU 模式，不使用 GPU）
                    boxes, best_score = self.detect_by_template(
                        image,
                        template,
                        threshold,
                        gray_image=gray_image,
                        gray_template=gray_template,
                        gray_image_gpu=None  # CPU 模式下不使用 GPU
                    )
                    
                    # 如果传统方法失败且阈值较低，尝试 ORB 特征匹配
                    if not boxes and threshold < 0.5:
                        orb_boxes = self.detect_by_template_orb(image, template)
                        if orb_boxes:
                            boxes = orb_boxes
                            if best_score < threshold:
                                best_score = threshold
                    
                    # 模板文件名处理
                    raw_label = os.path.splitext(os.path.basename(template_path))[0]
                    template_label = re.sub(r'_\d+$', '', raw_label)
                    
                    return template_label, boxes, best_score
                except Exception as e:
                    logger.error(f"[自动标注] 处理模板 {template_path} 时出错: {e}")
                    return None, None, None
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(process_template, tp): tp for tp in template_paths}
                completed = 0
                for future in as_completed(future_to_path):
                    completed += 1
                    template_path = future_to_path[future]
                    if progress_callback:
                        progress_callback(completed, total_templates, os.path.basename(template_path))
                    
                    try:
                        template_label, boxes, best_score = future.result()
                        if template_label is not None:
                            if boxes:
                                print(f"[INFO] 模板 {os.path.basename(template_path)} 匹配到 {len(boxes)} 个目标")
                                for box in boxes:
                                    all_boxes.append((template_label, box))
                            else:
                                print(f"[WARN] 模板 {os.path.basename(template_path)} 未匹配到目标")
                                if best_score and best_score > 0 and best_score < threshold:
                                    print(f"      → 最高匹配分数 {best_score:.3f} 低于阈值 {threshold:.2f}，建议降低阈值到 {max(0.3, best_score - 0.1):.2f}")
                    except Exception as e:
                        logger.error(f"[自动标注] 获取模板 {template_path} 结果时出错: {e}")
        else:
            # 单线程处理（GPU 模式或模板数量少时）
            for idx, template_path in enumerate(template_paths, start=1):
                if progress_callback:
                    progress_callback(idx, total_templates, os.path.basename(template_path))

                # 使用带缓存的模板加载，避免重复磁盘 IO 和灰度转换
                template, gray_template = self._load_template_cached(template_path)
                if template is None or gray_template is None:
                    print(f"[WARN] 无法加载模板: {template_path}")
                    continue

                logger.info(f"[自动标注] 加载模板: {os.path.basename(template_path)}, 大小: {template.shape}")
                print(f"[DEBUG] 加载模板: {os.path.basename(template_path)}, 大小: {template.shape}")

                # 检查模板大小
                if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                    msg = f"模板 {os.path.basename(template_path)} ({template.shape[1]}x{template.shape[0]}) 比图像 ({image.shape[1]}x{image.shape[0]}) 大，跳过"
                    logger.warning(f"[自动标注] {msg}")
                    print(f"[WARN] {msg}")
                    continue

                # 直接进入主模板匹配逻辑（内部包含 1:1 + 多尺度匹配），避免重复的 matchTemplate 计算
                boxes, best_score = self.detect_by_template(
                    image,
                    template,
                    threshold,
                    gray_image=gray_image,
                    gray_template=gray_template,
                    gray_image_gpu=gray_image_gpu
                )
                
                # 如果传统方法失败且阈值较低，尝试 ORB 特征匹配（相对较慢，只在必要时启用）
                if not boxes and threshold < 0.5:
                    print(f"[DEBUG] 传统模板匹配失败，尝试ORB特征匹配...")
                    orb_boxes = self.detect_by_template_orb(image, template)
                    if orb_boxes:
                        print(f"[INFO] ORB特征匹配成功找到 {len(orb_boxes)} 个目标")
                        boxes = orb_boxes
                        if best_score < threshold:
                            best_score = threshold  # ORB 成功时，将分数视为达到阈值
                # 模板文件名（不含扩展名）
                raw_label = os.path.splitext(os.path.basename(template_path))[0]
                # 逻辑名称：去掉末尾的 "_数字" 后缀，例如 "鳄鱼_1" 和 "鳄鱼_2" 都归为 "鳄鱼"
                template_label = re.sub(r'_\d+$', '', raw_label)
                if boxes:
                    print(f"[INFO] 模板 {os.path.basename(template_path)} 匹配到 {len(boxes)} 个目标")
                    for box in boxes:
                        all_boxes.append((template_label, box))
                else:
                    print(f"[WARN] 模板 {os.path.basename(template_path)} 未匹配到目标")
                    if best_score > 0:
                        if best_score < threshold:
                            print(f"      → 最高匹配分数 {best_score:.3f} 低于阈值 {threshold:.2f}，建议降低阈值到 {max(0.3, best_score - 0.1):.2f}")
                    else:
                        print("      → 未获得有效匹配分数，可能是模板与图像差异过大或需检查模板质量")
        
        # 非极大值抑制（去除重叠框）
        if all_boxes:
            print(f"[INFO] 匹配到 {len(all_boxes)} 个目标（抑制前）")
            boxes_only = [entry[1] for entry in all_boxes]
            filtered_boxes = self.non_max_suppression(boxes_only, overlap_threshold=0.3)
            print(f"[INFO] 抑制后剩余 {len(filtered_boxes)} 个目标")

            filtered_entries: List[Tuple[str, Tuple[int, int, int, int]]] = []
            used = set()
            for box in filtered_boxes:
                for idx, entry in enumerate(all_boxes):
                    if idx in used:
                        continue
                    if entry[1] == box:
                        filtered_entries.append((entry[0], box))
                        used.add(idx)
                        break
            all_boxes = filtered_entries
        else:
            print(f"[WARN] 所有模板都未匹配到目标")
        
        return all_boxes

