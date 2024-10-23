# import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton, QMessageBox
# from PyQt5.QtCore import QThread, pyqtSignal
#
#
# class Worker(QThread):
#     progress_signal = pyqtSignal(int)
#     finished_signal = pyqtSignal(bool)
#
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.is_running = True
#
#     def run(self):
#         # 模拟检测任务
#         for i in range(101):
#             if not self.is_running:
#                 self.finished_signal.emit(False)
#                 return
#             self.progress_signal.emit(i)
#             self.msleep(50)  # 模拟任务耗时
#
#         self.finished_signal.emit(True)
#
#     def stop(self):
#         self.is_running = False
#
#
# class ProgressBarDemo(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.init_ui()
#
#     def init_ui(self):
#         self.setWindowTitle('Progress Bar Example')
#         self.setGeometry(100, 100, 400, 200)
#
#         layout = QVBoxLayout()
#
#         self.progress_bar = QProgressBar(self)
#         self.progress_bar.setMinimum(0)
#         self.progress_bar.setMaximum(100)
#         layout.addWidget(self.progress_bar)
#
#         self.start_button = QPushButton('Start Detection', self)
#         self.start_button.clicked.connect(self.start_detection)
#         layout.addWidget(self.start_button)
#
#         self.setLayout(layout)
#         self.thread = None
#
#     def start_detection(self):
#         if self.thread and self.thread.isRunning():
#             self.thread.stop()
#             self.start_button.setText('Start Detection')
#         else:
#             self.thread = Worker()
#             self.thread.progress_signal.connect(self.update_progress)
#             self.thread.finished_signal.connect(self.task_finished)
#             self.thread.start()
#             self.start_button.setText('Stop Detection')
#
#     def update_progress(self, value):
#         self.progress_bar.setValue(value)
#
#     def task_finished(self, success):
#         if success:
#             QMessageBox.information(self, '完成', '检测任务完成！')
#         else:
#             QMessageBox.warning(self, '中止', '检测任务被中止。')
#         self.start_button.setText('Start Detection')
#         self.thread = None
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = ProgressBarDemo()
#     window.show()
#     sys.exit(app.exec_())

