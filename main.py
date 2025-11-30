import sys
import os
import datetime
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QListWidget, QGroupBox, QLineEdit, QComboBox,
                             QTextEdit, QFormLayout, QSpinBox, QCheckBox, QFileDialog,
                             QDialog, QDialogButtonBox, QListWidgetItem, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread
from core.window_manager import WindowManager
from core.vision import VisionSystem
from core.dataset_manager import DatasetManager
from core.trainer import TrainingWorker
from core.input_driver import InputDriver
from core.blackboard import Blackboard
from core.logic_engine import LogicEngine
from core.label_editor import LabelEditor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("废弃都市KPQ智能指挥系统 - MapleStory Bot")
        
        # 初始化核心模块
        self.wm = WindowManager()
        self.vision = VisionSystem()
        self.dm = DatasetManager()
        self.input = InputDriver()
        self.bb = Blackboard()
        self.logic = LogicEngine(self.bb, self.input, self.vision)
        self.logic.log_signal.connect(self.log_area_append)
        
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_monitor_feed)

        self.collect_timer = QTimer()
        self.collect_timer.timeout.connect(self.collect_sample_image)
        self.is_collecting = False

        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # 左侧菜单
        self.menu_list = QListWidget()
        self.menu_list.setFixedWidth(200)
        self.menu_list.addItems([
            "📊 总控仪表盘 (Dashboard)",
            "👥 角色与窗口 (Characters)",
            "👁️ 视觉感知 (YOLO/OCR)",
            "💾 数据集管理 (Datasets)",
            "⚙️ 系统设置 (Settings)",
            "📝 运行日志 (Logs)"
        ])
        self.menu_list.currentRowChanged.connect(self.switch_tab)
        
        self.menu_list.setStyleSheet("""
            QListWidget { font-size: 14px; padding: 10px; }
            QListWidget::item { padding: 10px; height: 30px; }
            QListWidget::item:selected { background-color: #0078d7; color: white; }
        """)

        # 右侧内容区域
        self.stack = QStackedWidget()
        
        main_layout.addWidget(self.menu_list)
        main_layout.addWidget(self.stack)
        
        # 初始化数据结构
        self.role_combos = []
        self.role_keys = ['leader', 'member1', 'member2', 'member3', 'member4', 'member5']
        
        # 初始化各个页面
        self.init_dashboard_tab()
        self.init_character_tab()
        self.init_vision_tab()
        self.init_dataset_tab()
        self.init_settings_tab()
        self.init_logs_tab()
        
        # 默认选中第一页
        self.menu_list.setCurrentRow(0)

        QTimer.singleShot(1000, self.scan_game_windows)

    def log_area_append(self, msg):
        """线程安全的日志追加"""
        if hasattr(self, 'log_area'):
            self.log_area.append(msg)
            self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    # --- 页面 1: 仪表盘 ---
    def init_dashboard_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.status_label = QLabel("系统状态: 就绪 (Ready)")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: gray;")
        layout.addWidget(self.status_label)
        
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 启动自动化 (Start)")
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; font-size: 16px; height: 50px;")
        self.btn_start.clicked.connect(self.start_automation)
        
        self.btn_stop = QPushButton("⏹ 停止 (Stop)")
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white; font-size: 16px; height: 50px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_automation)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.stack.addWidget(tab)

    # --- 页面 2: 角色管理 ---
    def init_character_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        group = QGroupBox("多开窗口绑定 (Window Binding)")
        form_layout = QFormLayout()
        
        for role in self.role_keys:
            combo = QComboBox()
            combo.addItem("未绑定", None)
            combo.currentIndexChanged.connect(lambda idx, r=role, c=combo: self.on_role_changed(r, c))
            self.role_combos.append(combo)
            form_layout.addRow(f"角色 [{role}]:", combo)
            
        group.setLayout(form_layout)
        layout.addWidget(group)
        
        btn_scan = QPushButton("🔍 扫描游戏窗口")
        btn_scan.clicked.connect(self.scan_game_windows)
        layout.addWidget(btn_scan)
        
        btn_tile = QPushButton("🪟 平铺所有窗口")
        btn_tile.clicked.connect(self.tile_game_windows)
        layout.addWidget(btn_tile)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.stack.addWidget(tab)

    # --- 页面 3: 视觉感知 ---
    def init_vision_tab(self):
        tab = QWidget()
        main_h_layout = QHBoxLayout()
        
        # 左侧配置
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        
        # 监控源选择
        src_group = QGroupBox("监控源 (Source)")
        src_layout = QVBoxLayout()
        self.monitor_source_combo = QComboBox()
        self.monitor_source_combo.addItem("关闭监控", None)
        self.monitor_source_combo.currentIndexChanged.connect(self.toggle_monitoring)
        src_layout.addWidget(self.monitor_source_combo)
        src_group.setLayout(src_layout)
        config_layout.addWidget(src_group)
        
        # YOLO配置
        yolo_group = QGroupBox("YOLO 模型配置")
        yolo_layout = QFormLayout()
        self.model_path = QLineEdit("models/best.pt")
        self.yolo_conf_spin = QSpinBox()
        self.yolo_conf_spin.setRange(1, 100)
        self.yolo_conf_spin.setValue(60)
        self.yolo_conf_spin.setSuffix("%")
        yolo_layout.addRow("模型路径:", self.model_path)
        yolo_layout.addRow("置信度:", self.yolo_conf_spin)
        yolo_group.setLayout(yolo_layout)
        config_layout.addWidget(yolo_group)
        
        # 监控画面
        monitor_group = QGroupBox("实时画面 (Live Feed)")
        monitor_layout = QVBoxLayout()
        self.monitor_label = QLabel("等待视频源...")
        self.monitor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.monitor_label.setStyleSheet("background-color: black; color: white;")
        self.monitor_label.setMinimumSize(640, 360)
        monitor_layout.addWidget(self.monitor_label)
        monitor_group.setLayout(monitor_layout)
        
        # 数据采集
        collect_group = QGroupBox("数据采集 (Data Collection)")
        c_layout = QFormLayout()
        self.collect_interval = QSpinBox()
        self.collect_interval.setRange(100, 5000)
        self.collect_interval.setValue(1000)
        self.collect_interval.setSuffix(" ms")
        self.collect_btn = QPushButton("📷 开始自动采集样本")
        self.collect_btn.setCheckable(True)
        self.collect_btn.setStyleSheet("background-color: #17a2b8; color: white;")
        self.collect_btn.clicked.connect(self.toggle_collection)
        
        self.label_btn = QPushButton("🏷️ 启动标注工具 (LabelImg)")
        self.label_btn.clicked.connect(self.launch_labelimg)
        
        c_layout.addRow("采集间隔:", self.collect_interval)
        c_layout.addRow(self.collect_btn)
        c_layout.addRow(self.label_btn)
        collect_group.setLayout(c_layout)
        config_layout.addWidget(collect_group)
        
        config_layout.addStretch()
        config_widget.setLayout(config_layout)
        config_widget.setFixedWidth(350)
        
        main_h_layout.addWidget(config_widget)
        main_h_layout.addWidget(monitor_group)
        
        tab.setLayout(main_h_layout)
        self.stack.addWidget(tab)

    # --- 页面 4: 数据集管理 ---
    def init_dataset_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        master_group = QGroupBox("🏆 主数据集 (Master Dataset)")
        m_layout = QFormLayout()
        self.lbl_master_stats = QLabel("加载中...")
        self.lbl_master_classes = QLabel("加载中...")
        m_layout.addRow("样本数量:", self.lbl_master_stats)
        m_layout.addRow("包含类别:", self.lbl_master_classes)
        
        btn_edit_classes = QPushButton("✏️ 编辑类别 (Edit Classes)")
        btn_edit_classes.clicked.connect(self.edit_master_labels)
        m_layout.addRow(btn_edit_classes)
        
        master_group.setLayout(m_layout)
        layout.addWidget(master_group)
        
        session_group = QGroupBox("📦 待处理会话 (Raw Sessions)")
        s_layout = QVBoxLayout()
        
        self.session_list = QListWidget()
        s_layout.addWidget(self.session_list)
        
        btn_box = QHBoxLayout()
        btn_refresh = QPushButton("🔄 刷新列表")
        btn_refresh.clicked.connect(self.refresh_dataset_view)
        
        btn_label_session = QPushButton("🏷️ 标注选中会话")
        btn_label_session.clicked.connect(self.label_selected_session)
        
        btn_edit_labels = QPushButton("✏️ 编辑标签 (删除标签)")
        btn_edit_labels.setStyleSheet("background-color: #ffc107; color: black;")
        btn_edit_labels.clicked.connect(self.open_label_editor)
        
        btn_merge = QPushButton("📥 合并到主数据集")
        btn_merge.setStyleSheet("background-color: #28a745; color: white;")
        btn_merge.clicked.connect(self.merge_current_session)
        
        btn_box.addWidget(btn_refresh)
        btn_box.addWidget(btn_label_session)
        btn_box.addWidget(btn_edit_labels)
        btn_box.addWidget(btn_merge)
        s_layout.addLayout(btn_box)
        
        session_group.setLayout(s_layout)
        layout.addWidget(session_group)

        train_group = QGroupBox("🧠 模型训练 (Model Training)")
        t_layout = QFormLayout()
        
        h_params = QHBoxLayout()
        
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(50)
        self.spin_epochs.setPrefix("Epochs: ")
        
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(16)
        self.spin_batch.setPrefix("Batch: ")
        
        self.combo_model = QComboBox()
        self.combo_model.addItems(["n (Nano - 最快)", "s (Small - 均衡)", "m (Medium - 精准)"])
        
        h_params.addWidget(self.spin_epochs)
        h_params.addWidget(self.spin_batch)
        h_params.addWidget(self.combo_model)
        
        self.btn_train = QPushButton("🔥 开始训练 (Start Training)")
        self.btn_train.setStyleSheet("background-color: #fd7e14; color: white; font-weight: bold;")
        self.btn_train.setFixedHeight(40)
        self.btn_train.clicked.connect(self.start_training)
        
        t_layout.addRow(h_params)
        t_layout.addRow(self.btn_train)
        train_group.setLayout(t_layout)
        layout.addWidget(train_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.stack.addWidget(tab)

    # --- 页面 5: 系统设置 ---
    def init_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        group = QGroupBox("全局参数")
        form = QFormLayout()
        form.addRow("按键随机延迟 (ms):", QLineEdit("50-100"))
        form.addRow("鼠标移动轨迹:", QCheckBox("启用拟人化贝塞尔曲线"))
        form.addRow("窗口平铺模式:", QComboBox())
        group.setLayout(form)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.stack.addWidget(tab)

    # --- 页面 6: 日志 ---
    def init_logs_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        self.log_area.setText("[SYSTEM] 初始化完成...\n[SYSTEM] 等待用户配置窗口...")
        
        layout.addWidget(self.log_area)
        
        btn_test_log = QPushButton("测试日志写入")
        btn_test_log.clicked.connect(lambda: self.log_area.append("[INFO] 用户点击了测试按钮"))
        layout.addWidget(btn_test_log)
        
        tab.setLayout(layout)
        self.stack.addWidget(tab)

    def switch_tab(self, index):
        self.stack.setCurrentIndex(index)
        if index == 3: 
            self.refresh_dataset_view()

    # ---------------------------------------------------------
    # 业务逻辑
    # ---------------------------------------------------------

    def on_role_changed(self, role_key, combo):
        """当用户在下拉框中选择窗口时，更新 Blackboard"""
        hwnd = combo.currentData()
        if hwnd:
            self.bb.register_character(role_key, hwnd)
            self.log_area_append(f"[SYSTEM] 角色 {role_key} 已绑定到窗口 {hwnd}")
        else:
            # TODO: Unregister?
            pass

    def start_automation(self):
        """启动自动化"""
        self.logic.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("系统状态: 运行中 (Running)")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")

    def stop_automation(self):
        """停止自动化"""
        self.logic.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("系统状态: 已停止 (Stopped)")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")

    def scan_game_windows(self):
        self.log_area_append("[SYSTEM] 正在扫描游戏窗口...")
        windows = self.wm.find_windows(r"(MapleStory|VMware)")
        
        for combo in self.role_combos:
            current_data = combo.currentData()
            combo.clear()
            combo.addItem("未绑定", None)
            
            for win in windows:
                title = f"{win['title']} (PID: {win['pid']})"
                combo.addItem(title, win['hwnd'])
                
            if current_data:
                idx = combo.findData(current_data)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    
        self.monitor_source_combo.clear()
        self.monitor_source_combo.addItem("关闭监控", None)
        for win in windows:
            title = f"{win['title']} (PID: {win['pid']})"
            self.monitor_source_combo.addItem(title, win['hwnd'])
            
        self.log_area_append(f"[SYSTEM] 扫描完成，发现 {len(windows)} 个窗口")

    def toggle_monitoring(self, index):
        hwnd = self.monitor_source_combo.currentData()
        if hwnd:
            self.log_area_append(f"[VISION] 开始监控窗口 HWND: {hwnd}")
            self.monitor_timer.start(100) 
        else:
            self.log_area_append("[VISION] 停止监控")
            self.monitor_timer.stop()
            self.monitor_label.setText("监控已关闭")

    def update_monitor_feed(self):
        hwnd = self.monitor_source_combo.currentData()
        if not hwnd: return
        
        frame = self.vision.capture_window(hwnd)
        if frame is None: return
        
        self.current_raw_frame = frame.copy() 
        
        results, annotated_frame = self.vision.detect_objects(frame, conf_threshold=self.yolo_conf_spin.value()/100.0)
        
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.monitor_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.monitor_label.setPixmap(scaled_pixmap)

    def label_selected_session(self):
        item = self.session_list.currentItem()
        if not item: return
        
        session_name = item.data(Qt.ItemDataRole.UserRole)
        session_path = os.path.join("datasets", session_name)
        
        self.launch_labelimg_for_path(session_path)

    def open_dataset_folder(self):
        path = os.path.abspath("datasets")
        os.makedirs(path, exist_ok=True)
        os.startfile(path)

    def launch_labelimg(self):
        """启动 LabelImg，优先使用上次打开的目录"""
        # 读取本地配置文件中保存的上次目录
        config_file = os.path.join(os.path.dirname(__file__), '.labelimg_last_dir.txt')
        last_open_dir = None
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    last_open_dir = f.read().strip()
                    
                if last_open_dir and os.path.exists(last_open_dir):
                    self.log_area_append(f"[INFO] 使用上次打开的目录: {last_open_dir}")
                    self.launch_labelimg_for_path(last_open_dir)
                    return
                else:
                    if last_open_dir:
                        self.log_area_append(f"[WARN] 上次目录不存在: {last_open_dir}")
        except Exception as e:
            self.log_area_append(f"[WARN] 读取上次目录失败: {e}")
        
        # 如果没有上次目录，弹出选择窗口
        start_dir = os.path.abspath("datasets")
        target_dir = QFileDialog.getExistingDirectory(self, "选择要标注的图片文件夹", start_dir)
        
        if target_dir:
            self.launch_labelimg_for_path(target_dir)
        else:
            self.log_area_append("[INFO] 取消启动标注工具")

    def launch_labelimg_for_path(self, target_dir):
        try:
            import subprocess
            import shutil
            abs_path = os.path.abspath(target_dir)
            classes_path = os.path.abspath("datasets/master/classes.txt")
            
            # 保存当前目录到配置文件
            config_file = os.path.join(os.path.dirname(__file__), '.labelimg_last_dir.txt')
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(abs_path)
            except Exception as e:
                self.log_area_append(f"[WARN] 保存目录配置失败: {e}")
            
            self.log_area_append(f"[SYSTEM] 正在初始化标注文件...")
            files = [f for f in os.listdir(abs_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            count = 0
            for img_file in files:
                txt_file = os.path.splitext(img_file)[0] + ".txt"
                txt_path = os.path.join(abs_path, txt_file)
                if not os.path.exists(txt_path):
                    with open(txt_path, 'w'):
                        pass 
                    count += 1
            
            if count > 0:
                self.log_area_append(f"[SYSTEM] 已自动创建 {count} 个空标签文件")

            # 复制 classes.txt 到目标目录
            try:
                shutil.copy(classes_path, os.path.join(abs_path, "classes.txt"))
            except Exception as e:
                self.log_area_append(f"[WARN] 复制 classes.txt 失败: {e}")

            self.log_area_append(f"[SYSTEM] 正在启动 LabelImg -> {target_dir}")
            self.log_area_append("[TIP] 已移除高级模式限制，支持快捷键直接标注")
            
            # 修正启动参数: labelImg [image_dir] [class_file]
            subprocess.Popen(['labelImg', abs_path, classes_path], shell=True)
            
        except Exception as e:
            self.log_area_append(f"[ERROR] 启动 LabelImg 失败: {e}")

    def refresh_dataset_view(self):
        stats = self.dm.get_master_stats()
        self.lbl_master_stats.setText(f"训练集: {stats['train']} | 验证集: {stats['val']}")
        
        classes = stats.get('classes', [])
        if classes:
            self.lbl_master_classes.setText(", ".join(classes))
        else:
            self.lbl_master_classes.setText("(无类别)")
            
        self.session_list.clear()
        sessions = self.dm.get_sessions()
        for sess in sessions:
            display_name = f"{sess['name']} ({sess['count']} imgs)"
            if sess['merged']:
                display_name = "✅ " + display_name
            
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, sess['name'])
            self.session_list.addItem(item)

    def edit_master_labels(self):
        stats = self.dm.get_master_stats()
        current_classes = stats.get('classes', [])
        
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑类别 (Edit Classes)")
        dialog.resize(300, 400)
        
        layout = QVBoxLayout()
        
        info_label = QLabel("每行一个类别名称 (按顺序):\n(修改后请确保数据集标注一致)")
        layout.addWidget(info_label)
        
        text_edit = QTextEdit()
        text_edit.setPlainText("\n".join(current_classes))
        layout.addWidget(text_edit)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec():
            new_text = text_edit.toPlainText()
            new_classes = [line.strip() for line in new_text.split('\n') if line.strip()]
            
            success, msg = self.dm.save_classes(new_classes)
            if success:
                self.log_area_append(f"[SUCCESS] {msg}")
                self.refresh_dataset_view()
            else:
                self.log_area_append(f"[ERROR] {msg}")

    def merge_current_session(self):
        item = self.session_list.currentItem()
        if not item: 
            self.log_area_append("[WARN] 请先选择一个会话")
            return
            
        session_name = item.data(Qt.ItemDataRole.UserRole)
        if "_MERGED" in session_name:
            self.log_area_append("[WARN] 该会话似乎已经合并过了")
            
        success, msg = self.dm.merge_session(session_name)
        if success:
            self.log_area_append(f"[SUCCESS] {msg}")
            self.refresh_dataset_view()
        else:
            self.log_area_append(f"[ERROR] 合并失败: {msg}")

    def start_training(self):
        stats = self.dm.get_master_stats()
        if stats['train'] == 0:
            self.log_area_append("[ERROR] 训练集为空！请先合并一些数据。")
            return
            
        epochs = self.spin_epochs.value()
        batch = self.spin_batch.value()
        model_size = self.combo_model.currentText().split()[0] 
        data_yaml = os.path.abspath("datasets/master/data.yaml")
        
        self.btn_train.setEnabled(False)
        self.btn_train.setText("⏳ 训练中 (Training)...")
        
        self.train_thread = QThread()
        self.train_worker = TrainingWorker(data_yaml, model_size, epochs, batch)
        self.train_worker.moveToThread(self.train_thread)
        
        self.train_thread.started.connect(self.train_worker.run)
        self.train_worker.log_signal.connect(self.on_train_log)
        self.train_worker.finished_signal.connect(self.on_train_finished)
        self.train_worker.finished_signal.connect(self.train_thread.quit)
        self.train_worker.finished_signal.connect(self.train_worker.deleteLater)
        self.train_thread.finished.connect(self.train_thread.deleteLater)
        
        self.train_thread.start()

    def on_train_log(self, msg):
        self.log_area_append(msg)

    def on_train_finished(self, success, result):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("🔥 开始训练 (Start Training)")
        
        if success:
            self.log_area_append("[SUCCESS] 训练流程结束！")
            self.model_path.setText(result)
            self.log_area_append("[TIP] 已自动将新模型应用到视觉配置中。")
        else:
            self.log_area_append(f"[FAIL] 训练失败: {result}")

    def toggle_collection(self):
        if self.collect_btn.isChecked():
            if not self.monitor_source_combo.currentData():
                self.log_area_append("[WARN] 请先选择一个监控源！")
                self.collect_btn.setChecked(False)
                return
                
            interval = self.collect_interval.value()
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            win_title = self.monitor_source_combo.currentText()
            safe_title = "".join([c for c in win_title if c.isalnum() or c in (' ', '-', '_')]).strip()
            session_name = f"{timestamp}_{safe_title}"
            
            self.current_session_dir = os.path.join("datasets", session_name)
            os.makedirs(self.current_session_dir, exist_ok=True)
            
            self.collect_timer.start(interval)
            self.collect_btn.setText("⏹️ 停止采集")
            self.collect_btn.setStyleSheet("background-color: #dc3545; color: white;")
            self.log_area_append(f"[DATA] 开始采集 -> {session_name}")
            
        else:
            self.collect_timer.stop()
            self.collect_btn.setText("📷 开始自动采集样本")
            self.collect_btn.setStyleSheet("background-color: #17a2b8; color: white;")
            self.log_area_append("[DATA] 采集已停止")

    def collect_sample_image(self):
        if not hasattr(self, 'current_raw_frame') or self.current_raw_frame is None:
            return
            
        timestamp = datetime.datetime.now().strftime("%H%M%S_%f")
        filename = os.path.join(self.current_session_dir, f"img_{timestamp}.jpg")
        
        cv2.imwrite(filename, self.current_raw_frame)
        self.log_area_append(f"[DATA] Saved: {os.path.basename(filename)}")

    def test_window_activation(self, combo):
        hwnd = combo.currentData()
        if hwnd:
            success = self.wm.activate_window(hwnd)
            if success:
                self.log_area_append(f"[INFO] 已激活窗口 HWND: {hwnd}")
                self.log_area_append(f"[TEST] 发送跳跃指令 (Space)...")
                self.input.press_key(hwnd, 'space')
            else:
                self.log_area_append(f"[ERROR] 无法激活窗口 HWND: {hwnd}")
        else:
            self.log_area_append("[WARN] 请先选择一个窗口")

    def tile_game_windows(self):
        hwnds = []
        seen = set()
        
        for combo in self.role_combos:
            h = combo.currentData()
            if h and h not in seen:
                hwnds.append(h)
                seen.add(h)
        
        if not hwnds:
            self.log_area_append("[WARN] 没有绑定任何窗口，无法平铺")
            return
            
        self.log_area_append(f"[SYSTEM] 正在平铺 {len(hwnds)} 个窗口...")
        self.wm.tile_windows(hwnds, aspect_ratio=1.333)
        self.log_area_append("[SUCCESS] 平铺完成")

    def open_label_editor(self):
        """打开标签编辑器对话框"""
        # 让用户选择图片文件
        start_dir = os.path.abspath("datasets")
        img_file, _ = QFileDialog.getOpenFileName(
            self, 
            "选择要编辑标签的图片文件", 
            start_dir,
            "图片文件 (*.jpg *.png *.jpeg);;所有文件 (*.*)"
        )
        
        if not img_file:
            return
        
        # 获取对应的标签文件路径
        label_file = os.path.splitext(img_file)[0] + ".txt"
        classes_file = os.path.join(os.path.dirname(img_file), "classes.txt")
        
        # 如果当前目录没有 classes.txt，尝试从 master 目录获取
        if not os.path.exists(classes_file):
            classes_file = os.path.abspath("datasets/master/classes.txt")
        
        # 打开标签编辑对话框
        dialog = LabelEditorDialog(self, img_file, label_file, classes_file)
        if dialog.exec():
            self.log_area_append(f"[SUCCESS] 标签已保存: {os.path.basename(label_file)}")


class LabelEditorDialog(QDialog):
    """标签编辑对话框"""
    
    def __init__(self, parent, img_file: str, label_file: str, classes_file: str):
        super().__init__(parent)
        self.img_file = img_file
        self.label_file = label_file
        self.classes_file = classes_file
        self.label_editor = LabelEditor(classes_file)
        self.labels = []
        
        self.setWindowTitle(f"编辑标签 - {os.path.basename(img_file)}")
        self.resize(600, 500)
        
        self.init_ui()
        self.load_labels()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 文件信息
        info_label = QLabel(f"图片: {os.path.basename(self.img_file)}\n标签文件: {os.path.basename(self.label_file)}")
        info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # 标签列表
        list_label = QLabel("已标注的标签列表 (双击可查看详情):")
        layout.addWidget(list_label)
        
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.label_list)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        btn_delete = QPushButton("🗑️ 删除选中标签")
        btn_delete.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
        btn_delete.clicked.connect(self.delete_selected_labels)
        
        btn_refresh = QPushButton("🔄 刷新列表")
        btn_refresh.clicked.connect(self.load_labels)
        
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_refresh)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        # 对话框按钮
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
        self.setLayout(layout)
    
    def load_labels(self):
        """加载标签列表"""
        self.labels = self.label_editor.read_labels(self.label_file)
        self.label_list.clear()
        
        if not self.labels:
            self.label_list.addItem("(无标签)")
            return
        
        for idx, (class_id, cx, cy, w, h) in enumerate(self.labels):
            class_name = self.label_editor.get_class_name(class_id)
            # 将归一化坐标转换为像素坐标（假设图片尺寸，实际应该读取图片）
            item_text = f"[{idx}] {class_name} - 中心:({cx:.3f}, {cy:.3f}) 尺寸:({w:.3f}, {h:.3f})"
            self.label_list.addItem(item_text)
    
    def delete_selected_labels(self):
        """删除选中的标签"""
        selected_items = self.label_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的标签！")
            return
        
        # 获取选中的索引
        indices = []
        for item in selected_items:
            row = self.label_list.row(item)
            if row < len(self.labels):
                indices.append(row)
        
        if not indices:
            return
        
        # 确认删除
        count = len(indices)
        reply = QMessageBox.question(
            self, 
            "确认删除", 
            f"确定要删除选中的 {count} 个标签吗？\n此操作不可撤销！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 删除标签（从大到小排序）
            deleted = self.label_editor.delete_labels(self.label_file, indices)
            if deleted > 0:
                QMessageBox.information(self, "成功", f"已删除 {deleted} 个标签！")
                self.load_labels()
            else:
                QMessageBox.warning(self, "错误", "删除失败！")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())