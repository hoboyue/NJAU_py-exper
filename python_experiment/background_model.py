import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk


class BackgroundModelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("背景建模工具")
        self.root.geometry("1000x700")

        self.video = None  # 保存视频对象
        self.last_frame = None  # 保存两帧差法的上一帧
        self.last_frames = [None, None]  # 保存三帧差法的前两帧

        self.canvas = None  # 用于显示视频的画布
        self.create_ui()

    def create_ui(self):
        """创建主界面"""
        # 创建功能按钮区域
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

        ttk.Style().configure("TButton", font=("Arial", 12))

        ttk.Button(button_frame, text="加载视频", command=self.load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="高斯混合模型", command=self.gaussian_mixture_modeling).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="KNN 背景建模", command=self.knn_background_modeling).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="两帧差法", command=self.two_frame_difference).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="三帧差法", command=self.three_frame_difference).pack(side=tk.LEFT, padx=5)

        # 视频显示区域
        self.video_frame = tk.Frame(self.root, width=800, height=600, bg="gray")
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 在视频显示区域创建一个 Canvas 用于显示视频
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 显示视频加载状态
        self.status_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.status_label.pack(side=tk.BOTTOM, pady=5)

    def load_video(self):
        """加载视频文件"""
        file_path = filedialog.askopenfilename(title="加载视频", filetypes=[("视频文件", "*.mp4 *.avi"), ("所有文件", "*.*")])
        if file_path:
            self.video = cv2.VideoCapture(file_path)
            self.status_label.config(text=f"已加载视频: {file_path}")
            messagebox.showinfo("加载成功", f"已加载视频: {file_path}")
        else:
            self.status_label.config(text="未加载视频")
            messagebox.showwarning("警告", "未选择视频文件")

    def display_frame(self, frame):
        """在 Canvas 上显示视频帧"""
        # 转换 BGR 到 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将 OpenCV 图像转换为 PIL 图像
        frame_image = Image.fromarray(frame_rgb)
        # 将 PIL 图像转换为 Tkinter 图像
        frame_tk = ImageTk.PhotoImage(frame_image)
        # 在 Canvas 上显示图像
        self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        self.canvas.image = frame_tk

    def process_video(self, process_frame_callback):
        """通用视频处理逻辑"""
        if self.video is None or not self.video.isOpened():
            messagebox.showwarning("警告", "请先加载视频！")
            return

        def update_frame():
            ret, frame = self.video.read()
            if not ret:
                self.status_label.config(text="视频播放完成")
                return

            # 调用具体的处理函数
            processed_frame = process_frame_callback(frame)

            # 显示处理后的帧
            self.display_frame(processed_frame)

            # 每 30 毫秒更新一次帧
            self.root.after(30, update_frame)

        update_frame()

    def gaussian_mixture_modeling(self):
        """高斯混合模型"""
        fgbg = cv2.createBackgroundSubtractorMOG2()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        def process_frame(frame):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 250:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame

        self.process_video(process_frame)

    def knn_background_modeling(self):
        """KNN 背景建模"""
        fgbg = cv2.createBackgroundSubtractorKNN()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        def process_frame(frame):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 250:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return frame

        self.process_video(process_frame)

    def two_frame_difference(self):
        """两帧差法"""
        self.last_frame = None

        def process_frame(frame):
            if self.last_frame is None:
                self.last_frame = frame
                return frame

            frame_delta = cv2.absdiff(self.last_frame, frame)
            self.last_frame = frame.copy()

            thresh = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 250:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame

        self.process_video(process_frame)

    def three_frame_difference(self):
        """三帧差法"""
        self.last_frames = [None, None]

        def process_frame(frame):
            if self.last_frames[0] is None:
                self.last_frames[0] = frame
                return frame
            if self.last_frames[1] is None:
                self.last_frames[1] = frame
                return frame

            frame_delta1 = cv2.absdiff(self.last_frames[0], self.last_frames[1])
            frame_delta2 = cv2.absdiff(self.last_frames[1], frame)
            self.last_frames[0], self.last_frames[1] = self.last_frames[1], frame

            thresh = cv2.bitwise_and(frame_delta1, frame_delta2)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 250:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame

        self.process_video(process_frame)


# 启动 Tkinter 应用
root = tk.Tk()
app = BackgroundModelingApp(root)
root.mainloop()
