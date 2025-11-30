import time
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from .blackboard import Blackboard
from .input_driver import InputDriver
from .vision import VisionSystem
from .logic.rope_logic import RopeLogic

class LogicEngine(QObject):
    """
    逻辑引擎：系统的“大脑”。
    负责：观察(Vision) -> 更新黑板(Blackboard) -> 决策(Decision) -> 执行(Input)
    """
    log_signal = pyqtSignal(str) # 用于向 UI 发送日志

    def __init__(self, blackboard: Blackboard, input_driver: InputDriver, vision: VisionSystem):
        super().__init__()
        self.bb = blackboard
        self.input = input_driver
        self.vision = vision
        self.is_running = False
        self.thread = None
        
        # 初始化逻辑模块
        self.rope_logic = RopeLogic(blackboard, input_driver)
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()
        self.log_signal.emit("[Logic] 引擎已启动")
        
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        self.log_signal.emit("[Logic] 引擎已停止")
        
    def _loop(self):
        while self.is_running:
            try:
                # 1. 获取所有活跃角色
                roles = self.bb.get_all_roles()
                
                if not roles:
                    time.sleep(1)
                    continue
                
                for role in roles:
                    char_data = self.bb.get_character(role)
                    hwnd = char_data.get('hwnd')
                    if not hwnd: continue
                    
                    # --- 视觉感知 (Vision) ---
                    # 注意：频繁截图和推理可能会占用大量 CPU/GPU
                    # 建议根据实际情况调整频率，或者只在必要时检测
                    frame = self.vision.capture_window(hwnd)
                    if frame is None: continue
                    
                    # 运行 YOLO (不画图，只拿结果)
                    results, _ = self.vision.detect_objects(frame)
                    parsed_data = self.vision.parse_results(results)
                    
                    # 更新黑板
                    self.bb.update_vision(role, parsed_data)
                    
                    # --- 决策与执行 (Logic) ---
                    
                    # 策略 1: 爬绳子测试
                    # 如果视野里有绳子，就尝试去爬
                    if parsed_data.get('rope'):
                        if self.rope_logic.execute(role):
                            # self.log_signal.emit(f"[{role}] 正在执行爬绳逻辑")
                            pass
                    
                    # 策略 2: 简单的防发呆跳跃 (如果没有在爬绳子)
                    # current_time = time.time()
                    # last_action = char_data.get('last_action_time', 0)
                    # if current_time - last_action > 10.0:
                    #     self.input.press_key(hwnd, 'space')
                    #     self.bb.update_character(role, 'last_action_time', current_time)
                        
                # 3. 休息 (Tick Rate)
                time.sleep(0.1) # 10Hz
                
            except Exception as e:
                import traceback
                self.log_signal.emit(f"[Logic Error] {e}")
                print(traceback.format_exc())
                time.sleep(1)
