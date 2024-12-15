import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# 图像处理函数
def apply_threshold(img_gray, method):
    """应用阈值操作"""
    # 根据选择的阈值方法，对灰度图像进行处理
    if method == "BINARY":
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    elif method == "BINARY_INV":
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    elif method == "TRUNC":
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    elif method == "TOZERO":
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    elif method == "TOZERO_INV":
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
    else:
        raise ValueError("未知阈值操作")  # 如果方法未知，则抛出异常
    return thresh  # 返回处理后的阈值图像

def apply_filter(img, method):
    """应用滤波操作"""
    # 根据选择的滤波方法，进行相应处理
    if method == "均值滤波":
        return cv2.blur(img, (3, 3))  # 应用均值滤波
    elif method == "方框滤波":
        return cv2.boxFilter(img, -1, (3, 3), normalize=False)  # 应用方框滤波
    elif method == "高斯滤波":
        return cv2.GaussianBlur(img, (5, 5), 1)  # 应用高斯滤波
    elif method == "中值滤波":
        return cv2.medianBlur(img, 5)  # 应用中值滤波
    else:
        raise ValueError("未知滤波操作")  # 如果方法未知，则抛出异常

# 主UI
def main_ui():
    global current_image, current_image_path
    current_image = None  # 初始化当前图像变量
    current_image_path = None  # 初始化当前图像路径

    def open_image():
        """打开图片文件"""
        global current_image, current_image_path
        # 选择文件并读取图像
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if not file_path:  # 如果没有选择文件，则返回
            return
        try:
            img = cv2.imread(file_path)  # 读取图像文件
            if img is None:  # 检查图像是否读取成功
                raise ValueError("无法读取图片")
            current_image_path = file_path  # 保存当前图像路径
            current_image = img  # 保存当前图像数据
            show_preview(img, "原始图像")  # 显示原始图像
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def save_image_ui():
        """保存图片"""
        global current_image
        if current_image is None:  # 检查当前是否有图像可保存
            messagebox.showwarning("警告", "没有图片可以保存")
            return
        # 保存文件对话框
        file_path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png",
                                                 filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
        if file_path:
            cv2.imwrite(file_path, current_image)  # 保存图像
            messagebox.showinfo("成功", "图片已保存")  # 显示保存成功信息

    def process_threshold():
        """处理阈值操作"""
        global current_image
        if current_image is None:  # 检查是否已加载图像
            messagebox.showwarning("警告", "请先加载图片")
            return
        try:
            selected_method = threshold_combobox.get()  # 获取选择的阈值处理方法
            if not selected_method:  # 检查是否选择了方法
                messagebox.showwarning("警告", "请先选择阈值处理方法")
                return
            img_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            result = apply_threshold(img_gray, selected_method)  # 应用阈值处理
            show_preview(result, f"阈值处理 - {selected_method}")  # 显示处理结果
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def process_filter():
        """处理滤波操作"""
        global current_image
        if current_image is None:  # 检查是否已加载图像
            messagebox.showwarning("警告", "请先加载图片")
            return
        try:
            selected_method = filter_combobox.get()  # 获取选择的滤波方法
            if not selected_method:  # 检查是否选择了方法
                messagebox.showwarning("警告", "请先选择滤波方法")
                return
            result = apply_filter(current_image, selected_method)  # 应用滤波处理
            show_preview(result, f"滤波处理 - {selected_method}")  # 显示处理结果
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def show_preview(image, title=""):
        """显示图片预览"""
        max_width, max_height = 800, 800  # 限制图片显示的最大宽度和高度
        height, width = image.shape[:2]  # 获取图像的高度和宽度

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

    # 主窗口布局调整
    root = tk.Tk()  # 创建主窗口
    root.title("图像阈值与平滑处理")  # 设置窗口标题
    root.geometry("1350x900")  # 设置窗口大小

    # 功能按钮区域
    button_frame = tk.Frame(root)  # 创建按钮框架
    button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

    # 设置按钮和标签的样式
    ttk.Style().configure("TButton", font=("Arial", 12))
    ttk.Style().configure("TLabel", font=("Arial", 12))

    ttk.Button(button_frame, text="打开图片", command=open_image).pack(side=tk.LEFT, padx=5)  # 打开图片按钮
    ttk.Button(button_frame, text="保存图片", command=save_image_ui).pack(side=tk.LEFT, padx=5)  # 保存图片按钮

    # 添加阈值处理下拉菜单
    ttk.Label(button_frame, text="灰度图操作:").pack(side=tk.LEFT, padx=5)  # 标签
    threshold_methods = ['BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']  # 阈值处理方法
    threshold_combobox = ttk.Combobox(button_frame, values=threshold_methods, state="readonly", width=15)  # 下拉菜单
    threshold_combobox.set("选择操作")  # 默认提示文本
    threshold_combobox.pack(side=tk.LEFT, padx=5)  # 显示下拉菜单
    ttk.Button(button_frame, text="应用", command=process_threshold).pack(side=tk.LEFT, padx=5)  # 应用阈值处理按钮

    # 添加滤波操作下拉菜单
    ttk.Label(button_frame, text="图像平滑:").pack(side=tk.LEFT, padx=5)  # 标签
    filter_methods = ['均值滤波', '方框滤波', '高斯滤波', '中值滤波']  # 滤波方法
    filter_combobox = ttk.Combobox(button_frame, values=filter_methods, state="readonly", width=15)  # 下拉菜单
    filter_combobox.set("选择操作")  # 默认提示文本
    filter_combobox.pack(side=tk.LEFT, padx=5)  # 显示下拉菜单
    ttk.Button(button_frame, text="应用", command=process_filter).pack(side=tk.LEFT, padx=5)  # 应用滤波处理按钮

    # 图片显示区域
    img_frame = tk.Frame(root, width=600, height=600, bg="gray")  # 创建图像显示框架
    img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    img_label = tk.Label(img_frame)  # 创建图像标签
    img_label.pack(fill=tk.BOTH, expand=True)  # 填充整个显示区域

    img_label_title = tk.Label(root, text="", font=("Arial", 14))  # 创建标题标签
    img_label_title.pack(side=tk.BOTTOM, pady=5)  # 显示在底部

    root.mainloop()  # 启动主循环


# 启动UI
main_ui()  # 调用主UI函数
