import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

# 腐蚀操作
def apply_erosion(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

# 膨胀操作
def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

# 开运算（先腐蚀后膨胀）
def apply_opening(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 闭运算（先膨胀后腐蚀）
def apply_closing(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 梯度操作（膨胀 - 腐蚀）
def apply_gradient(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

# 礼帽操作（原图 - 开运算）
def apply_tophat(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# 黑帽操作（闭运算 - 原图）
def apply_blackhat(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

# Tkinter 主界面
def main_ui():
    global current_image, current_image_path
    current_image = None
    current_image_path = None

    def open_image():
        """打开图片文件"""
        global current_image, current_image_path
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if not file_path:
            return
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("无法读取图片")
            current_image_path = file_path
            current_image = img
            show_preview(img, "原始图像")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def save_image_ui():
        """保存图片"""
        global current_image
        if current_image is None:
            messagebox.showwarning("警告", "没有图片可以保存")
            return
        file_path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png",
                                                 filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
        if file_path:
            cv2.imwrite(file_path, current_image)
            messagebox.showinfo("成功", "图片已保存")

    def apply_morphology(operation, kernel_size=(5, 5), iterations=1):
        """应用形态学操作"""
        global current_image
        if current_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        try:
            if operation == "腐蚀":
                result = apply_erosion(current_image, kernel_size, iterations)
            elif operation == "膨胀":
                result = apply_dilation(current_image, kernel_size, iterations)
            elif operation == "开运算":
                result = apply_opening(current_image, kernel_size)
            elif operation == "闭运算":
                result = apply_closing(current_image, kernel_size)
            elif operation == "梯度":
                result = apply_gradient(current_image, kernel_size)
            elif operation == "礼帽":
                result = apply_tophat(current_image, kernel_size)
            elif operation == "黑帽":
                result = apply_blackhat(current_image, kernel_size)
            else:
                raise ValueError("未知形态学操作")
            show_preview(result, f"{operation} 操作结果")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def show_preview(image, title=""):
        """显示图片预览"""
        max_width, max_height = 800, 800  # 限制图片显示的最大宽度和高度
        height, width = image.shape[:2]

        # 计算缩放比例
        scale = min(max_width / width, max_height / height, 1.0)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 调整图片大小以适配显示区域
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 转换为RGB模式并显示在Tkinter中
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) if len(resized_image.shape) == 3 else resized_image
        pil_image = Image.fromarray(image_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)

        img_label.config(image=tk_image)
        img_label.image = tk_image
        img_label_title.config(text=title)

    # 主窗口布局调整
    root = tk.Tk()
    root.title("形态学操作工具")
    root.geometry("1200x900")

    # 功能按钮区域（顶部）
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

    ttk.Style().configure("TButton", font=("Arial", 12))

    ttk.Button(button_frame, text="打开图片", command=open_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="保存图片", command=save_image_ui).pack(side=tk.LEFT, padx=5)

    # 添加形态学操作按钮
    ttk.Button(button_frame, text="腐蚀", command=lambda: apply_morphology("腐蚀", (3, 3), 1)).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="膨胀", command=lambda: apply_morphology("膨胀", (3, 3), 1)).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="开运算", command=lambda: apply_morphology("开运算")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="闭运算", command=lambda: apply_morphology("闭运算")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="梯度", command=lambda: apply_morphology("梯度")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="礼帽", command=lambda: apply_morphology("礼帽")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="黑帽", command=lambda: apply_morphology("黑帽")).pack(side=tk.LEFT, padx=5)

    # 图片显示区域
    img_frame = tk.Frame(root, width=600, height=600, bg="gray")
    img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    img_label = tk.Label(img_frame)
    img_label.pack(fill=tk.BOTH, expand=True)

    img_label_title = tk.Label(root, text="", font=("Arial", 14))
    img_label_title.pack(side=tk.BOTTOM, pady=5)

    root.mainloop()

# 启动UI
main_ui()
