"""
标签编辑器 - 用于查看和编辑 YOLO 格式的标签文件
解决 LabelImg 右键菜单无法使用的问题
"""
import os
from typing import List, Tuple, Optional


class LabelEditor:
    """标签编辑器，用于读取、编辑和保存 YOLO 格式的标签文件"""
    
    def __init__(self, classes_file: Optional[str] = None):
        """
        初始化标签编辑器
        :param classes_file: classes.txt 文件路径，用于将类别ID转换为名称
        """
        self.classes = []
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
    
    def read_labels(self, label_file: str) -> List[Tuple[int, float, float, float, float]]:
        """
        读取标签文件
        :param label_file: .txt 标签文件路径
        :return: 标签列表，每个标签为 (class_id, center_x, center_y, width, height)
        """
        if not os.path.exists(label_file):
            return []
        
        labels = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((class_id, center_x, center_y, width, height))
                except (ValueError, IndexError):
                    continue
        
        return labels
    
    def save_labels(self, label_file: str, labels: List[Tuple[int, float, float, float, float]]):
        """
        保存标签文件
        :param label_file: .txt 标签文件路径
        :param labels: 标签列表
        """
        with open(label_file, 'w', encoding='utf-8') as f:
            for label in labels:
                class_id, cx, cy, w, h = label
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    def get_class_name(self, class_id: int) -> str:
        """
        获取类别名称
        :param class_id: 类别ID
        :return: 类别名称，如果不存在则返回 f"Class_{class_id}"
        """
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"Class_{class_id}"
    
    def delete_label(self, label_file: str, index: int) -> bool:
        """
        删除指定索引的标签
        :param label_file: 标签文件路径
        :param index: 要删除的标签索引
        :return: 是否成功删除
        """
        labels = self.read_labels(label_file)
        if 0 <= index < len(labels):
            labels.pop(index)
            self.save_labels(label_file, labels)
            return True
        return False
    
    def delete_labels(self, label_file: str, indices: List[int]) -> int:
        """
        删除多个标签
        :param label_file: 标签文件路径
        :param indices: 要删除的标签索引列表（从大到小排序，避免索引错位）
        :return: 成功删除的数量
        """
        labels = self.read_labels(label_file)
        # 从大到小排序，避免删除后索引变化
        indices = sorted(set(indices), reverse=True)
        deleted = 0
        for idx in indices:
            if 0 <= idx < len(labels):
                labels.pop(idx)
                deleted += 1
        
        if deleted > 0:
            self.save_labels(label_file, labels)
        return deleted

