import time
import random
import win32gui
import win32api
import win32con
import ctypes
from typing import Optional

class InputDriver:
    """
    执行层：负责向游戏窗口发送键盘和鼠标指令。
    核心特性：后台发送（PostMessage）、拟人化随机延迟。
    """

    # 常用虚拟键码映射 (Virtual-Key Codes)
    VK_MAP = {
        'left': win32con.VK_LEFT,
        'right': win32con.VK_RIGHT,
        'up': win32con.VK_UP,
        'down': win32con.VK_DOWN,
        'space': win32con.VK_SPACE,
        'alt': win32con.VK_MENU,
        'ctrl': win32con.VK_CONTROL,
        'shift': win32con.VK_SHIFT,
        'enter': win32con.VK_RETURN,
        'esc': win32con.VK_ESCAPE,
        'f1': win32con.VK_F1,
        'f2': win32con.VK_F2,
        'delete': win32con.VK_DELETE,
        'ins': win32con.VK_INSERT,
        'end': win32con.VK_END,
        'home': win32con.VK_HOME,
        'pgup': win32con.VK_PRIOR,
        'pgdn': win32con.VK_NEXT,
        # 字母键 (A-Z)
        'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45, 'f': 0x46,
        'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A, 'k': 0x4B, 'l': 0x4C,
        'm': 0x4D, 'n': 0x4E, 'o': 0x4F, 'p': 0x50, 'q': 0x51, 'r': 0x52,
        's': 0x53, 't': 0x54, 'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58,
        'y': 0x59, 'z': 0x5A,
        # 数字键 (0-9)
        '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
        '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
    }

    def __init__(self):
        self.user32 = ctypes.windll.user32

    def _get_vk(self, key: str) -> int:
        """获取按键的虚拟键码"""
        key = key.lower()
        return self.VK_MAP.get(key, 0)

    def _random_sleep(self, min_ms=50, max_ms=120):
        """拟人化随机延迟"""
        duration = random.uniform(min_ms, max_ms) / 1000.0
        time.sleep(duration)

    def press_key(self, hwnd: int, key: str):
        """
        模拟一次按键点击 (按下 -> 随机延迟 -> 抬起)
        :param hwnd: 目标窗口句柄
        :param key: 按键名称 (如 'space', 'a', 'f1')
        """
        vk_code = self._get_vk(key)
        if not vk_code:
            print(f"[Input] 未知按键: {key}")
            return

        # 1. 按下 (WM_KEYDOWN)
        # lParam 构建: RepeatCount=1, ScanCode=0, Extended=0, Context=0, PrevState=0, Transition=0
        # 通常 PostMessage 不需要太复杂的 lParam，简单的 0 或 1 即可触发
        win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, vk_code, 0)
        
        # 2. 随机延迟 (模拟人类按键时长)
        self._random_sleep(60, 150)
        
        # 3. 抬起 (WM_KEYUP)
        # lParam 构建: Transition=1 (bit 31) -> 0xC0000000
        lparam_up = 0xC0000001 
        win32api.PostMessage(hwnd, win32con.WM_KEYUP, vk_code, lparam_up)
        
        # 4. 操作间隔延迟
        self._random_sleep(30, 80)

    def hold_key(self, hwnd: int, key: str, duration: float):
        """
        长按按键 (用于移动等)
        :param hwnd: 目标窗口句柄
        :param key: 按键名称
        :param duration: 持续时间 (秒)
        """
        vk_code = self._get_vk(key)
        if not vk_code: return

        # 按下
        win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, vk_code, 0)
        
        # 持续
        time.sleep(duration)
        
        # 抬起
        lparam_up = 0xC0000001
        win32api.PostMessage(hwnd, win32con.WM_KEYUP, vk_code, lparam_up)
        self._random_sleep(50, 100)

    def click_mouse(self, hwnd: int, x: int, y: int, button='left'):
        """
        后台鼠标点击
        :param hwnd: 目标窗口句柄
        :param x: 窗口内相对坐标 X
        :param y: 窗口内相对坐标 Y
        """
        # 构造坐标参数 (高位Y，低位X)
        lparam = win32api.MAKELONG(x, y)
        
        if button == 'left':
            msg_down = win32con.WM_LBUTTONDOWN
            msg_up = win32con.WM_LBUTTONUP
            wparam = win32con.MK_LBUTTON
        elif button == 'right':
            msg_down = win32con.WM_RBUTTONDOWN
            msg_up = win32con.WM_RBUTTONUP
            wparam = win32con.MK_RBUTTON
        else:
            return

        # 移动鼠标消息 (可选，有些游戏需要先检测到鼠标移动)
        # win32api.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
        
        # 按下
        win32api.PostMessage(hwnd, msg_down, wparam, lparam)
        self._random_sleep(40, 90)
        # 抬起
        win32api.PostMessage(hwnd, msg_up, 0, lparam)
        self._random_sleep(50, 100)

    def send_text(self, hwnd: int, text: str):
        """
        发送文本 (用于聊天或输入密码)
        注意：这通常使用 WM_CHAR 消息
        """
        for char in text:
            win32api.PostMessage(hwnd, win32con.WM_CHAR, ord(char), 0)
            self._random_sleep(10, 30)

# 测试代码
if __name__ == "__main__":
    import time
    driver = InputDriver()
    print("InputDriver 模块测试")
    print("请在 3 秒内将记事本窗口置顶...")
    time.sleep(3)
    
    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd)
    print(f"目标窗口: {title} ({hwnd})")
    
    print("发送文本...")
    driver.send_text(hwnd, "Hello MapleStory Bot!")
    driver.press_key(hwnd, "enter")
    
    print("测试按键 A...")
    driver.press_key(hwnd, "a")
