"""
LabelImg 增强补丁脚本
在 LabelImg 中添加自动标注功能
"""
import os
import shutil
import sys

def patch_labelimg():
    """为 LabelImg 添加自动标注功能"""
    # LabelImg 安装路径
    labelimg_path = os.path.dirname(__file__)
    # 查找 LabelImg 安装位置
    try:
        import labelImg
        labelimg_dir = os.path.dirname(labelImg.__file__)
        labelimg_file = os.path.join(labelimg_dir, 'labelImg.py')
    except ImportError:
        print("错误: 无法找到 LabelImg 安装位置")
        return False
    
    if not os.path.exists(labelimg_file):
        print(f"错误: 找不到 LabelImg 文件: {labelimg_file}")
        return False
    
    # 备份原文件
    backup_file = labelimg_file + '.backup'
    if not os.path.exists(backup_file):
        print(f"备份原文件到: {backup_file}")
        shutil.copy2(labelimg_file, backup_file)
    
    # 读取原文件
    with open(labelimg_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经打过补丁
    if 'auto_annotate' in content and 'from core.auto_annotator import AutoAnnotator' in content:
        print("LabelImg 已经包含自动标注功能，跳过补丁")
        return True
    
    # 添加导入语句（在文件开头附近）
    import_pattern = "from libs.ustr import ustr"
    import_insert = """from libs.ustr import ustr
# [Auto Annotate] 添加自动标注功能
import sys
import os
# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from core.auto_annotator import AutoAnnotator
    AUTO_ANNOTATE_AVAILABLE = True
except ImportError:
    AUTO_ANNOTATE_AVAILABLE = False
    print("[WARN] 自动标注功能不可用，请确保 core/auto_annotator.py 存在")
"""
    
    if import_pattern in content:
        content = content.replace(import_pattern, import_insert)
    else:
        # 如果找不到，在文件开头添加
        content = import_insert + "\n" + content
    
    # 在 MainWindow.__init__ 中添加自动标注器初始化
    init_pattern = "        self.lastLabel = None"
    init_insert = """        self.lastLabel = None
        # [Auto Annotate] 初始化自动标注器
        if AUTO_ANNOTATE_AVAILABLE:
            self.auto_annotator = AutoAnnotator()
        else:
            self.auto_annotator = None
"""
    
    if init_pattern in content:
        content = content.replace(init_pattern, init_insert)
    
    # 添加自动标注按钮（在 create 按钮之后）
    # 找到 create 按钮的定义
    create_action_pattern = "        create = action(get_str('crtBox'), self.create_shape,"
    auto_annotate_action = """        # [Auto Annotate] 自动标注按钮
        auto_annotate = action('自动标注', self.auto_annotate_image,
                              'Ctrl+Shift+A', 'new', '自动标注当前图片（使用简单图像识别）',
                              enabled=False)
"""
    
    if create_action_pattern in content:
        # 在 create 之后添加
        insert_pos = content.find(create_action_pattern)
        if insert_pos != -1:
            # 找到 create 定义的结束位置
            next_line = content.find('\n', insert_pos)
            content = content[:next_line+1] + auto_annotate_action + content[next_line+1:]
    
    # 将 auto_annotate 添加到工具栏
    beginner_pattern = "        self.actions.beginner = ("
    if beginner_pattern in content:
        # 在 create 之后添加 auto_annotate
        content = content.replace(
            "open, open_dir, change_save_dir, open_next_image, open_prev_image, verify, save, save_format, None, create, copy, delete, None,",
            "open, open_dir, change_save_dir, open_next_image, open_prev_image, verify, save, save_format, None, create, auto_annotate, copy, delete, None,"
        )
    
    # 添加自动标注方法
    method_insert = """
    # [Auto Annotate] 自动标注方法
    def auto_annotate_image(self):
        \"\"\"自动标注当前图片\"\"\"
        if not self.auto_annotator:
            QMessageBox.warning(self, '错误', '自动标注功能不可用')
            return
        
        if not self.file_path or not os.path.exists(self.file_path):
            QMessageBox.warning(self, '错误', '请先打开一张图片')
            return
        
        try:
            # 获取当前类别（如果有选中的标签）
            class_id = 0
            if self.label_list.count() > 0:
                # 尝试从当前选中的标签获取类别ID
                current_item = self.label_list.currentItem()
                if current_item:
                    # 从标签文本中提取类别ID（需要根据实际情况调整）
                    label_text = current_item.text()
                    # 查找类别ID
                    if hasattr(self, 'label_hist') and self.label_hist:
                        try:
                            class_id = self.label_hist.index(label_text.split()[0])
                        except (ValueError, IndexError):
                            pass
            
            # 执行自动标注
            labels = self.auto_annotator.auto_annotate(
                self.file_path, 
                method='crocodile',  # 默认使用鳄鱼检测
                class_id=class_id
            )
            
            if not labels:
                QMessageBox.information(self, '提示', '未检测到目标，请尝试调整检测参数')
                return
            
            # 将检测结果添加到画布
            img = cv2.imread(self.file_path)
            if img is None:
                QMessageBox.warning(self, '错误', '无法读取图片')
                return
            
            img_height, img_width = img.shape[:2]
            
            # 获取当前标签文本（如果没有，使用默认）
            label_text = '鳄鱼'  # 默认标签
            if hasattr(self, 'label_hist') and self.label_hist:
                label_text = self.label_hist[0] if self.label_hist else '鳄鱼'
            
            # 为每个检测框创建形状
            for _template_label, class_id, center_x, center_y, width, height in labels:
                # 转换为像素坐标
                x = (center_x - width / 2) * img_width
                y = (center_y - height / 2) * img_height
                w = width * img_width
                h = height * img_height
                
                # 创建矩形框
                shape = self.canvas.add_shape(
                    label=label_text,
                    shape_type='rectangle',
                    points=[(x, y), (x + w, y + h)]
                )
                self.add_label(shape)
            
            self.set_dirty()
            QMessageBox.information(self, '成功', f'已自动标注 {len(labels)} 个目标')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'自动标注失败: {str(e)}')
            import traceback
            traceback.print_exc()
"""
    
    # 在 create_shape 方法之后添加
    create_shape_end = "    def toggle_drawing_sensitive(self, drawing=True):"
    if create_shape_end in content:
        content = content.replace(create_shape_end, method_insert + "\n    " + create_shape_end)
    
    # 添加 cv2 导入（如果还没有）
    if 'import cv2' not in content:
        # 在文件开头添加
        content = "import cv2\n" + content
    
    # 保存修改后的文件
    with open(labelimg_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"成功为 LabelImg 添加自动标注功能: {labelimg_file}")
    return True

if __name__ == '__main__':
    if patch_labelimg():
        print("补丁应用成功！")
    else:
        print("补丁应用失败！")
        sys.exit(1)

