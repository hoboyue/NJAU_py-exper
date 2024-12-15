import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# Sobel算子边缘检测
def apply_sobel(img, combine=True):
    # 使用Sobel算子检测水平和垂直边缘
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向的Sobel算子
    sobelx = cv2.convertScaleAbs(sobelx)  # 转换为绝对值，转换成8位无符号整型
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向的Sobel算子
    sobely = cv2.convertScaleAbs(sobely)  # 转换为绝对值
    if combine:  # 如果需要合并水平和垂直边缘
        return cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)  # 合并两个方向的边缘
    else:
        return sobelx, sobely  # 返回水平和垂直边缘图像

# Scharr算子边缘检测
def apply_scharr(img):
    # 使用Scharr算子检测边缘
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # 水平方向的Scharr算子
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # 垂直方向的Scharr算子
    scharrx = cv2.convertScaleAbs(scharrx)  # 转换为绝对值
    scharry = cv2.convertScaleAbs(scharry)  # 转换为绝对值
    return cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)  # 合并两个方向的边缘

# Laplacian算子边缘检测
def apply_laplacian(img):
    # 使用Laplacian算子检测边缘
    laplacian = cv2.Laplacian(img, cv2.CV_64F)  # 计算Laplacian
    return cv2.convertScaleAbs(laplacian)  # 转换为绝对值并返回

# Canny边缘检测
def apply_canny(img, threshold1, threshold2):
    # 使用Canny算子进行边缘检测
    return cv2.Canny(img, threshold1, threshold2)  # 返回边缘检测结果

# Tkinter 主界面函数
def main_ui():
    # 全局变量定义
    global current_image, current_image_path
    current_image = None
    current_image_path = None

    # 打开图片函数
    def open_image():
        """打开图片文件"""
        global current_image, current_image_path
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if not file_path:  # 如果没有选择文件，则返回
            return
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # 读取图像
            if img is None:  # 检查图像是否读取成功
                raise ValueError("无法读取图片")
            current_image_path = file_path  # 保存当前图像路径
            current_image = img  # 保存当前图像数据
            show_preview(img, "原始图像")  # 显示原始图像
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    # 保存图片函数
    def save_image_ui():
        """保存图片"""
        global current_image
        if current_image is None:  # 检查当前是否有图像可保存
            messagebox.showwarning("警告", "没有图片可以保存")
            return
        file_path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png",
                                                 filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
        if file_path:  # 如果选择了文件路径，则保存图像
            cv2.imwrite(file_path, current_image)
            messagebox.showinfo("成功", "图片已保存")  # 显示保存成功信息

    # 处理边缘检测操作函数
    def process_edge_detection(method):
        """处理边缘检测操作"""
        global current_image
        if current_image is None:  # 检查是否加载了图像
            messagebox.showwarning("警告", "请先加载图片")
            return
        try:
            gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
            if method == "Sobel":  # 如果选择的是Sobel操作
                result = apply_sobel(gray_image)
            elif method == "Scharr":  # 如果选择的是Scharr操作
                result = apply_scharr(gray_image)
            elif method == "Laplacian":  # 如果选择的是Laplacian操作
                result = apply_laplacian(gray_image)
            else:
                raise ValueError("未知边缘检测操作")  # 抛出未知操作异常
            show_preview(result, f"{method} 操作结果")  # 显示操作结果
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    # 处理Canny边缘检测的函数
    def process_canny():
        """处理Canny边缘检测"""
        global current_image
        if current_image is None:  # 检查是否加载了图像
            messagebox.showwarning("警告", "请先加载图片")
            return
        try:
            threshold1 = int(canny_threshold1.get())  # 获取第一个阈值
            threshold2 = int(canny_threshold2.get())  # 获取第二个阈值
            gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
            result = apply_canny(gray_image, threshold1, threshold2)  # 进行Canny边缘检测
            show_preview(result, f"Canny 操作结果 (阈值1: {threshold1}, 阈值2: {threshold2})")  # 显示操作结果
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数阈值")  # 如果阈值无效则弹出错误
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 其他错误处理

    # 显示图片预览的函数
    def show_preview(image, title=""):
        """显示图片预览"""
        max_width, max_height = 400, 400  # 限制图片显示的最大宽度和高度
        height, width = image.shape[:2]

        # 计算缩放比例
        scale = min(max_width / width, max_height / height, 1.0)
        new_width = int(width * scale)  # 计算新宽度
        new_height = int(height * scale)  # 计算新高度

        # 调整图片大小以适配显示区域
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 转换为RGB模式并显示在Tkinter中
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) if len(resized_image.shape) == 3 else resized_image
        pil_image = Image.fromarray(image_rgb)  # 将OpenCV图像转换为PIL图像
        tk_image = ImageTk.PhotoImage(pil_image)  # 将PIL图像转换为Tkinter可用格式

        img_label.config(image=tk_image)  # 更新标签显示
        img_label.image = tk_image  # 保持对图像的引用，防止被垃圾回收
        img_label_title.config(text=title)  # 更新图像标题

    # 主窗口布局设置
    root = tk.Tk()
    root.title("边缘检测工具")  # 设置窗口标题
    root.geometry("900x700")  # 设置窗口大小

    # 功能按钮区域（顶部）
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

    ttk.Style().configure("TButton", font=("Arial", 12))  # 设置按钮样式

    # 添加功能按钮
    ttk.Button(button_frame, text="打开图片", command=open_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="保存图片", command=save_image_ui).pack(side=tk.LEFT, padx=5)

    # 添加边缘检测功能按钮
    ttk.Button(button_frame, text="Sobel", command=lambda: process_edge_detection("Sobel")).pack(side=tk.LEFT, padx=4)
    ttk.Button(button_frame, text="Scharr", command=lambda: process_edge_detection("Scharr")).pack(side=tk.LEFT, padx=4)
    ttk.Button(button_frame, text="Laplacian", command=lambda: process_edge_detection("Laplacian")).pack(side=tk.LEFT, padx=4)

    # Canny边缘检测配置
    canny_frame = tk.Frame(root)  # 创建Canny框架
    canny_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

    ttk.Label(canny_frame, text="Canny 阈值1:").pack(side=tk.LEFT, padx=5)  # 第一个阈值标签
    canny_threshold1 = ttk.Entry(canny_frame, width=10)  # 第一个阈值输入框
    canny_threshold1.insert(0, "50")  # 设置默认值
    canny_threshold1.pack(side=tk.LEFT, padx=5)

    ttk.Label(canny_frame, text="Canny 阈值2:").pack(side=tk.LEFT, padx=5)  # 第二个阈值标签
    canny_threshold2 = ttk.Entry(canny_frame, width=10)  # 第二个阈值输入框
    canny_threshold2.insert(0, "150")  # 设置默认值
    canny_threshold2.pack(side=tk.LEFT, padx=5)

    ttk.Button(canny_frame, text="应用 Canny边缘检测", command=process_canny).pack(side=tk.LEFT, padx=5)  # 应用Canny按钮

    # 图片显示区域
    img_frame = tk.Frame(root, width=600, height=600, bg="gray")  # 创建图像显示框架
    img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    img_label = tk.Label(img_frame)  # 创建图像标签
    img_label.pack(fill=tk.BOTH, expand=True)  # 填充显示区域

    img_label_title = tk.Label(root, text="", font=("Arial", 14))  # 创建标题标签
    img_label_title.pack(side=tk.BOTTOM, pady=5)  # 显示在底部

    root.mainloop()  # 启动主循环


# 启动UI
main_ui()  # 调用主UI函数来启动应用程序
