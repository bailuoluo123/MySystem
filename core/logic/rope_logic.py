from .base_logic import BaseLogic

class RopeLogic(BaseLogic):
    def execute(self, role: str) -> bool:
        vision = self.bb.get_vision(role)
        if not vision: return False
        
        players = vision.get('player', [])
        ropes = vision.get('rope', [])
        
        if not players or not ropes:
            return False
            
        me = players[0]
        mx, my = me['center']
        
        # 找到最近的绳子
        target_rope = None
        min_dist = 9999
        
        for rope in ropes:
            rx, ry = rope['center']
            dist = abs(rx - mx)
            if dist < min_dist:
                min_dist = dist
                target_rope = rope
                
        if not target_rope:
            return False
            
        rx, ry = target_rope['center']
        hwnd = self.bb.get_character(role).get('hwnd')
        if not hwnd: return False
        
        ALIGN_THRESHOLD = 20 # 像素宽容度
        
        if min_dist > ALIGN_THRESHOLD:
            if rx > mx:
                self.input.press_key(hwnd, 'right')
            else:
                self.input.press_key(hwnd, 'left')
            return True
        else:
            # 对齐了，爬
            self.input.press_key(hwnd, 'up')
            return True
