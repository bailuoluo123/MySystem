from typing import Dict, List, Any
import threading

class Blackboard:
    """
    黑板 (Blackboard)：全系统共享的数据中心。
    存储所有角色的状态、视觉感知结果、当前任务进度等。
    线程安全。
    """
    def __init__(self):
        self._lock = threading.Lock()
        
        # 角色状态 (Key: 'leader', 'member1'...)
        # Value: {'hwnd': 123, 'job': 'thief', 'hp': 100, 'pos': (0,0), 'status': 'idle'}
        self.characters: Dict[str, Dict[str, Any]] = {}
        
        # 全局游戏状态
        self.current_stage = 0
        self.is_fighting = False
        self.start_time = 0
        
        # 视觉感知结果 (最新一帧)
        # {'leader': {'mobs': [], 'ropes': [], 'player_pos': (x,y)}}
        self.vision_data: Dict[str, Any] = {}
        
    def register_character(self, role: str, hwnd: int, job: str = ""):
        """注册一个角色"""
        with self._lock:
            self.characters[role] = {
                'hwnd': hwnd,
                'job': job,
                'status': 'idle',
                'last_action_time': 0
            }
            
    def update_character(self, role: str, key: str, value: Any):
        """更新角色状态"""
        with self._lock:
            if role in self.characters:
                self.characters[role][key] = value
            
    def get_character(self, role: str) -> Dict[str, Any]:
        """获取角色状态副本"""
        with self._lock:
            return self.characters.get(role, {}).copy()
            
    def get_all_roles(self) -> List[str]:
        with self._lock:
            return list(self.characters.keys())
            
    def update_vision(self, role: str, data: Any):
        """更新视觉数据"""
        with self._lock:
            self.vision_data[role] = data
            
    def get_vision(self, role: str) -> Any:
        with self._lock:
            return self.vision_data.get(role, None)
