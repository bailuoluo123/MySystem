import os
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from ultralytics import YOLO

class TrainingWorker(QObject):
    # 信号：进度更新(文本日志), 训练结束(成功/失败, 消息)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, data_yaml, model_size='n', epochs=50, batch_size=16):
        super().__init__()
        self.data_yaml = data_yaml
        self.model_size = model_size # n, s, m, l, x
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_running = False

    def run(self):
        self.is_running = True
        try:
            self.log_signal.emit(f"[TRAIN] 开始训练 YOLOv8{self.model_size} 模型...")
            self.log_signal.emit(f"[TRAIN] 数据集: {self.data_yaml}")
            self.log_signal.emit(f"[TRAIN] 参数: Epochs={self.epochs}, Batch={self.batch_size}")

            # 1. 加载预训练模型 (例如 yolov8n.pt)
            # 如果本地没有，ultralytics 会自动下载
            model_name = f"yolov8{self.model_size}.pt"
            model = YOLO(model_name)

            # 2. 开始训练
            # 注意：YOLO 的 train 方法是阻塞的，所以我们在独立线程运行
            # 我们无法直接获取实时进度条，但可以捕获最终结果
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=640,
                plots=True, # 生成训练曲线图
                project="models/runs", # 训练结果保存路径
                name="train_session", # 每次训练的子文件夹名
                exist_ok=True # 允许覆盖
            )

            # 3. 训练完成
            save_dir = str(results.save_dir)
            best_model = os.path.join(save_dir, "weights", "best.pt")
            
            self.log_signal.emit(f"[TRAIN] 训练完成！")
            self.log_signal.emit(f"[TRAIN] 最佳模型已保存至: {best_model}")
            
            # 4. 自动导出/复制到主目录方便使用
            import shutil
            target_path = "models/best_v8.pt"
            os.makedirs("models", exist_ok=True)
            shutil.copy2(best_model, target_path)
            self.log_signal.emit(f"[TRAIN] 已自动部署为: {target_path}")

            self.finished_signal.emit(True, target_path)

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.log_signal.emit(f"[ERROR] 训练出错:\n{err_msg}")
            self.finished_signal.emit(False, str(e))
        finally:
            self.is_running = False

class ModelTrainer:
    def __init__(self):
        self.worker = None
        self.thread = None

    def start_training(self, data_yaml, model_size, epochs, batch, on_log, on_finish):
        if self.worker and self.worker.is_running:
            return False, "训练正在进行中"

        self.thread = threading.Thread(target=self._run_worker, 
                                     args=(data_yaml, model_size, epochs, batch, on_log, on_finish))
        self.thread.daemon = True # 设置为守护线程，主程序关闭时自动结束
        self.thread.start()
        return True, "训练线程已启动"

    def _run_worker(self, data_yaml, model_size, epochs, batch, on_log, on_finish):
        # 这里我们不直接用 QThread，而是用简单的 Python Thread 配合回调
        # 注意：在 PyQt 中跨线程更新 UI 需要小心，最好用 Signal
        # 但为了简化架构，我们这里假设 on_log 是线程安全的 (或者在主线程处理)
        # 更好的做法是让 MainWindow 里的 QThread 来跑这个 Worker
        pass

# 修正：为了完美的 PyQt 集成，我们还是在 main.py 里直接使用 QThread
# 这个文件只保留 Worker 逻辑定义，供 main.py 导入
