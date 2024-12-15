import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# Harris 角点检测
def harris_corner_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)  # Harris 角点检测
    img_harris = img.copy()
    img_harris[dst > 0.01 * dst.max()] = [0, 0, 255]  # 用红色标记角点
    return img_harris

# SIFT 特征检测与描述
def sift_feature_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    sift = cv2.SIFT_create()  # 创建 SIFT 对象
    kp = sift.detect(gray, None)  # 检测特征点
    img_sift = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 绘制特征点
    kp, des = sift.compute(gray, kp)  # 计算特征点描述符
    return img_sift, kp, des

# 显示图像到 UI
def show_preview(image, title=""):
    """在UI上显示图像"""
    max_width, max_height = 600, 400
    height, width = image.shape[:2]

    # 计算缩放比例
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 转换为 RGB 或灰度模式
    if len(resized_image.shape) == 3:
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = resized_image

    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(pil_image)

    img_label.config(image=tk_image)
    img_label.image = tk_image
    img_label_title.config(text=title)

# Tkinter 主界面
def main_ui():
    global current_image
    current_image = None

    def open_image():
        """打开图片文件"""
        global current_image
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if not file_path:
            return
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法读取图片")
            current_image = img
            show_preview(img, "原始图像")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def apply_operation(operation):
        """应用操作"""
        global current_image
        if current_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return

        try:
            if operation == "Harris 角点检测":
                result = harris_corner_detection(current_image)
                show_preview(result, "Harris 角点检测结果")
            elif operation == "SIFT 特征检测":
                result, kp, des = sift_feature_detection(current_image)
                show_preview(result, "SIFT 特征检测结果")
                messagebox.showinfo("SIFT 特征信息", f"检测到 {len(kp)} 个特征点\n描述符形状：{des.shape}")
            else:
                raise ValueError("未知操作")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # 主窗口布局
    root = tk.Tk()
    root.title("图像操作工具")
    root.geometry("1000x700")

    # 功能按钮区域（顶部）
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

    ttk.Style().configure("TButton", font=("Arial", 12))

    ttk.Button(button_frame, text="打开图片", command=open_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Harris 角点检测", command=lambda: apply_operation("Harris 角点检测")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="SIFT 特征检测", command=lambda: apply_operation("SIFT 特征检测")).pack(side=tk.LEFT, padx=5)

    # 图片显示区域
    img_frame = tk.Frame(root, width=600, height=600, bg="gray")
    img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    global img_label, img_label_title
    img_label = tk.Label(img_frame)
    img_label.pack(fill=tk.BOTH, expand=True)

    img_label_title = tk.Label(root, text="", font=("Arial", 14))
    img_label_title.pack(side=tk.BOTTOM, pady=5)

    root.mainloop()

# 启动 UI
main_ui()
