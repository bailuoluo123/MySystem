import os
import shutil
import random
import yaml

class DatasetManager:
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        self.master_dir = os.path.join(base_dir, "master")
        self.ensure_master_structure()

    def ensure_master_structure(self):
        """确保主数据集目录结构存在"""
        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.master_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.master_dir, "labels", split), exist_ok=True)
        
        # 确保 classes.txt 存在 (这是类别的唯一真理来源)
        classes_file = os.path.join(self.master_dir, "classes.txt")
        if not os.path.exists(classes_file):
            # 默认创建一个空的，或者预设一些常用类别
            with open(classes_file, "w") as f:
                f.write("player\nmonster\nrope\nladder\nportal\n")

    def get_sessions(self):
        """获取所有未合并的采集会话"""
        sessions = []
        if not os.path.exists(self.base_dir): return []
        
        for d in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, d)
            # 排除 master 目录和非文件夹
            if os.path.isdir(path) and d != "master":
                # 统计图片数量
                images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
                # 统计已标注数量 (有对应txt文件)
                labeled = [f for f in images if os.path.exists(os.path.join(path, os.path.splitext(f)[0] + ".txt"))]
                
                sessions.append({
                    "name": d,
                    "path": path,
                    "total": len(images),
                    "labeled": len(labeled),
                    "count": len(images)
                })
        # 按时间倒序
        sessions.sort(key=lambda x: x['name'], reverse=True)
        return sessions

    def prepare_labelimg_classes(self, session_name):
        """在启动 LabelImg 前，将主数据集的 classes.txt 复制到会话目录"""
        # 这样能保证 LabelImg 读取到正确的类别 ID
        src = os.path.join(self.master_dir, "classes.txt")
        dst = os.path.join(self.base_dir, session_name, "classes.txt")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    def merge_session(self, session_name, val_ratio=0.2):
        """将会话数据合并到主数据集"""
        source_dir = os.path.join(self.base_dir, session_name)
        if not os.path.exists(source_dir): return False, "会话目录不存在"

        # 1. 找出所有已标注的图片 (必须有对应的 .txt)
        files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))]
        valid_pairs = []
        for img_file in files:
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            if os.path.exists(os.path.join(source_dir, txt_file)):
                valid_pairs.append((img_file, txt_file))
        
        if not valid_pairs:
            return False, "该会话中没有找到已标注的数据 (图片+TXT)"

        # 2. 随机打乱并分割
        random.shuffle(valid_pairs)
        split_idx = int(len(valid_pairs) * (1 - val_ratio))
        train_set = valid_pairs[:split_idx]
        val_set = valid_pairs[split_idx:]

        # 3. 复制文件
        def copy_files(pairs, split):
            for img, txt in pairs:
                # 构造唯一文件名，防止不同会话的文件名冲突
                # 策略：在文件名前加 session_name
                new_img_name = f"{session_name}_{img}"
                new_txt_name = f"{session_name}_{txt}"
                
                shutil.copy2(os.path.join(source_dir, img), 
                             os.path.join(self.master_dir, "images", split, new_img_name))
                shutil.copy2(os.path.join(source_dir, txt), 
                             os.path.join(self.master_dir, "labels", split, new_txt_name))

        copy_files(train_set, "train")
        copy_files(val_set, "val")

        # 4. 更新 data.yaml
        self.update_yaml()
        
        # 5. (可选) 可以在这里把源文件夹标记为“已合并”，或者直接删除
        # 为了安全起见，我们暂时保留源文件，但在文件夹名字上加个标记
        try:
            new_path = os.path.join(self.base_dir, session_name + "_MERGED")
            if not os.path.exists(new_path):
                os.rename(source_dir, new_path)
        except:
            pass # 重命名失败也不影响

        return True, f"成功合并 {len(valid_pairs)} 组数据 (训练集: {len(train_set)}, 验证集: {len(val_set)})"

    def update_yaml(self):
        """生成/更新 YOLO 训练所需的 data.yaml"""
        classes_file = os.path.join(self.master_dir, "classes.txt")
        with open(classes_file, 'r') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]

        # YOLOv8 推荐使用绝对路径
        data = {
            'path': os.path.abspath(self.master_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(names),
            'names': names
        }
        
        yaml_path = os.path.join(self.master_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)

    def save_classes(self, classes):
        """保存类别列表到 classes.txt"""
        classes_file = os.path.join(self.master_dir, "classes.txt")
        try:
            with open(classes_file, 'w') as f:
                for cls in classes:
                    if cls.strip():
                        f.write(f"{cls.strip()}\n")
            return True, "类别保存成功"
        except Exception as e:
            return False, f"保存失败: {str(e)}"

    def get_master_stats(self):
        """获取主数据集统计信息"""
        stats = {'train': 0, 'val': 0, 'classes': []}
        
        for split in ['train', 'val']:
            path = os.path.join(self.master_dir, "labels", split)
            if os.path.exists(path):
                stats[split] = len([f for f in os.listdir(path) if f.endswith('.txt')])
        
        classes_file = os.path.join(self.master_dir, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                stats['classes'] = [line.strip() for line in f.readlines() if line.strip()]
                
        return stats
