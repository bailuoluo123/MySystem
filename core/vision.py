import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import ctypes
from typing import Optional, Tuple

class VisionSystem:
    def __init__(self):
        self.model = None
        self.model_path = ""

    def load_yolo_model(self, model_path: str):
        """加载 YOLO 模型"""
        if self.model_path == model_path and self.model is not None:
            return True # 已经加载

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"[Vision] YOLO 模型已加载: {model_path}")
            return True
        except ImportError:
            print("[Vision Error] 未安装 ultralytics 库，无法使用 YOLO。请运行 pip install ultralytics")
            return False
        except Exception as e:
            print(f"[Vision Error] 模型加载失败: {e}")
            return False

    def detect_objects(self, img: np.ndarray, conf_threshold=0.5):
        """
        执行对象检测
        :return: results (YOLO result object), annotated_frame (画了框的图)
        """
        if self.model is None:
            return None, img

        # YOLOv8 推理
        results = self.model(img, conf=conf_threshold, verbose=False)[0]
        
        # 绘制结果
        annotated_frame = results.plot()
        return results, annotated_frame

    def parse_results(self, results) -> dict:
        """
        将 YOLO 结果解析为易读的字典格式
        :return: {
            'player': [{'box': [x1, y1, x2, y2], 'conf': 0.9, 'center': (x, y)}, ...],
            'rope': [...],
            ...
        }
        """
        data = {}
        if results is None: return data
        
        names = results.names
        boxes = results.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            if cls_name not in data:
                data[cls_name] = []
                
            data[cls_name].append({
                'box': xyxy,
                'conf': conf,
                'center': ((xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2)
            })
            
        return data

    def capture_window(self, hwnd: int) -> Optional[np.ndarray]:
        """
        截取指定窗口的画面
        :param hwnd: 窗口句柄
        :return: OpenCV 格式的图像 (BGR), 如果失败返回 None
        """
        try:
            # 获取窗口客户区尺寸 (不包含标题栏)
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width = right - left
            height = bottom - top
            
            if width <= 0 or height <= 0:
                return None

            # 创建设备上下文
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            # 创建位图对象
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)

            # 核心截图操作：PrintWindow (兼容后台截图)
            # PW_RENDERFULLCONTENT = 2 (Win8.1+ 支持更好的后台截图)
            result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
            
            if result != 1:
                # 如果 PrintWindow 失败，尝试 BitBlt (只能截前台或未被遮挡的窗口)
                saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

            # 转换为 numpy 数组
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (height, width, 4)
            
            # 释放资源
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            
            # 转为 BGR 格式 (去除 Alpha 通道)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        except Exception as e:
            print(f"[Vision Error] Capture failed for HWND {hwnd}: {e}")
            return None

    def find_template(self, haystack_img: np.ndarray, needle_img: np.ndarray, threshold=0.8):
        """
        模板匹配：在画面中寻找小图
        :return: (found, max_loc, max_val)
        """
        result = cv2.matchTemplate(haystack_img, needle_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            return True, max_loc, max_val
        return False, None, max_val

if __name__ == "__main__":
    import ctypes
    # 测试代码
    print("正在测试截图功能...")
    # 找一个窗口测试，这里尝试找 VMware
    hwnd = win32gui.FindWindow(None, "a - VMware Workstation") # 替换为你的实际标题
    if not hwnd:
        # 如果找不到，就找当前的前台窗口
        hwnd = win32gui.GetForegroundWindow()
        print(f"未找到指定窗口，使用当前前台窗口 HWND: {hwnd}")
    
    vision = VisionSystem()
    img = vision.capture_window(hwnd)
    
    if img is not None:
        print(f"截图成功! 尺寸: {img.shape}")
        cv2.imshow("Test Capture", img)
        print("按任意键关闭预览...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("截图失败")
