import win32gui
import win32process
import win32con
import re
import ctypes
from typing import List, Dict, Optional

class WindowManager:
    def __init__(self):
        self.user32 = ctypes.windll.user32

    @staticmethod
    def _enum_cb(hwnd, results):
        """回调函数：枚举所有可见窗口"""
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            results.append(hwnd)

    def get_all_windows(self) -> List[Dict]:
        """获取所有可见窗口的详细信息"""
        hwnds = []
        win32gui.EnumWindows(self._enum_cb, hwnds)
        windows = []
        for hwnd in hwnds:
            title = win32gui.GetWindowText(hwnd)
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                pid = 0
            
            # 获取窗口位置信息 (Left, Top, Right, Bottom)
            rect = win32gui.GetWindowRect(hwnd)
            
            windows.append({
                "hwnd": hwnd,
                "title": title,
                "pid": pid,
                "rect": rect
            })
        return windows

    def find_windows(self, pattern: str) -> List[Dict]:
        """
        根据标题正则查找窗口
        :param pattern: 窗口标题的正则表达式 (例如: "MapleStory", "VMware.*")
        :return: 匹配的窗口列表
        """
        all_windows = self.get_all_windows()
        matches = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            for win in all_windows:
                if regex.search(win['title']):
                    matches.append(win)
        except re.error:
            print(f"[Error] 无效的正则表达式: {pattern}")
        return matches

    def activate_window(self, hwnd: int) -> bool:
        """
        将指定窗口置于前台
        """
        try:
            # 恢复最小化的窗口
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            # 尝试置顶
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception as e:
            print(f"[Error] 无法激活窗口 {hwnd}: {e}")
            return False

    def get_screen_size(self):
        """获取主屏幕分辨率"""
        width = self.user32.GetSystemMetrics(0)
        height = self.user32.GetSystemMetrics(1)
        return width, height

    def calculate_grid_layout(self, num_windows: int):
        """
        根据窗口数量计算最佳行列数 (优先保证接近方形或横向铺满)
        :return: (rows, cols)
        """
        if num_windows <= 1: return 1, 1
        if num_windows <= 2: return 1, 2
        if num_windows <= 4: return 2, 2
        if num_windows <= 6: return 2, 3
        if num_windows <= 9: return 3, 3
        return 3, 4 # 更多窗口暂定

    def tile_windows(self, hwnds: List[int], aspect_ratio=4/3):
        """
        智能平铺窗口
        :param hwnds: 窗口句柄列表
        :param aspect_ratio: 期望的宽高比 (默认 4:3，适合冒险岛). 如果为 None，则保持每个窗口当前的比例
        """
        if not hwnds: return
        
        screen_w, screen_h = self.get_screen_size()
        # 预留任务栏高度 (估算 40px)
        screen_h -= 40 
        
        num = len(hwnds)
        rows, cols = self.calculate_grid_layout(num)
        
        # 计算每个网格单元的最大尺寸
        grid_w = screen_w // cols
        grid_h = screen_h // rows
        
        # 计算保持比例的实际窗口尺寸
        # 我们希望窗口尽可能大，但不能超过网格，且要保持比例
        # 尝试以宽为基准
        target_w = grid_w
        target_h = int(target_w / aspect_ratio)
        
        # 如果高度超出了网格高度，则以高为基准反推
        if target_h > grid_h:
            target_h = grid_h
            target_w = int(target_h * aspect_ratio)
            
        print(f"[Layout] Screen: {screen_w}x{screen_h} | Windows: {num} | Grid: {rows}x{cols}")
        print(f"[Size] Grid Cell: {grid_w}x{grid_h} | Final Window Size: {target_w}x{target_h}")

        for i, hwnd in enumerate(hwnds):
            row = i // cols
            col = i % cols
            
            # 计算居中位置 (如果网格比窗口大，就居中放置，或者左上角对齐)
            # 这里采用左上角对齐网格，视觉上更整齐
            x = col * grid_w
            y = row * grid_h
            
            try:
                # 移动并调整大小
                # 注意：MoveWindow 的宽高包含标题栏和边框
                win32gui.MoveWindow(hwnd, x, y, target_w, target_h, True)
                self.activate_window(hwnd) 
            except Exception as e:
                print(f"[Error] 无法移动窗口 {hwnd}: {e}")

if __name__ == "__main__":
    import time
    wm = WindowManager()
    print("="*50)
    print("正在扫描虚拟机窗口 (VMware)...")
    
    # 1. 扫描
    vm_wins = wm.find_windows("VMware Workstation")
    
    if not vm_wins:
        print("[Warning] 未找到包含 'VMware Workstation' 的窗口！")
    else:
        print(f"找到 {len(vm_wins)} 个虚拟机窗口，准备执行智能平铺...")
        vm_hwnds = [w['hwnd'] for w in vm_wins]
            
        # 2. 智能平铺测试
        # 假设冒险岛分辨率为 800x600 (4:3)
        # VMware 窗口包含标题栏，实际比例可能略有不同，但保持 4:3 通常能保证画面不被拉伸太严重
        wm.tile_windows(vm_hwnds, aspect_ratio=1.333)
        print("[Success] 智能平铺完成！")
        
    print("="*50)
