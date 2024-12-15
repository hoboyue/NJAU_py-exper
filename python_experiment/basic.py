import tkinter as tk  # 导入Tkinter库，用于创建图形用户界面
from tkinter import filedialog, messagebox  # 导入文件对话框和消息框，用于打开文件和显示消息
from tkinter import ttk  # 导入ttk模块，用于更现代的样式组件
import cv2  # 导入OpenCV库，用于图像处理
from PIL import Image, ImageTk  # 导入PIL库，用于图像处理和与Tkinter兼容的图像显示
import numpy as np  # 导入NumPy库，用于处理图像数据

# 用来显示的全局变量
current_image = None  # 当前显示的图像
current_image_path = None  # 当前图像的文件路径
fusion_image1 = None  # 第一张用于融合的图像
fusion_image2 = None  # 第二张用于融合的图像

# 读取图片
def read_image(image_path, mode=cv2.IMREAD_COLOR):
    img = cv2.imread(image_path, mode)  # 使用OpenCV读取图像
    if img is None:  # 如果读取失败
        raise ValueError(f"无法读取图片：{image_path}")  # 抛出错误
    return img  # 返回图像数据

# 保存图片
def save_image(image_path, img):
    cv2.imwrite(image_path, img)  # 使用OpenCV保存图像到指定路径

# 添加边框
def add_border(image, top, bottom, left, right, border_type, value=0):
    bordered_image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type)  # 给图像加边框
    return bordered_image  # 返回加了边框的图像

# 调整亮度与对比度
def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    blank = np.zeros_like(image)  # 创建一个与图像大小相同的黑色图像
    blank[:, :] = brightness  # 将黑色图像设置为指定的亮度值
    adjusted = cv2.addWeighted(image, contrast, blank, 0, 0)  # 调整图像亮度与对比度
    return adjusted  # 返回调整后的图像

# 彩色图片直方图均衡化
def equalize_color_histogram(image):
    b, g, r = cv2.split(image)  # 分离图像的蓝色、绿色和红色通道
    b_eq = cv2.equalizeHist(b)  # 对蓝色通道进行直方图均衡化
    g_eq = cv2.equalizeHist(g)  # 对绿色通道进行直方图均衡化
    r_eq = cv2.equalizeHist(r)  # 对红色通道进行直方图均衡化
    equalized_image = cv2.merge((b_eq, g_eq, r_eq))  # 合并均衡化后的三个通道
    return equalized_image  # 返回均衡化后的图像

# 缩放图片
def resize_image(img, width=None, height=None, fx=1, fy=1):
    if width and height:  # 如果给定了宽度和高度
        resized = cv2.resize(img, (width, height))  # 按指定的宽高进行缩放
    else:
        resized = cv2.resize(img, (0, 0), fx=fx, fy=fy)  # 按比例缩放
    return resized  # 返回缩放后的图像

# 图像融合
def blend_images(image1, image2, alpha=0.5, beta=0.5):
    """对两张图片进行融合"""
    if image1.shape != image2.shape:  # 如果两张图像的尺寸不同
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))  # 调整第二张图片大小
    return cv2.addWeighted(image1, alpha, image2, beta, 0)  # 融合两张图片

# 播放视频
def play_video(video_path):
    """播放视频并正确关闭窗口"""
    cap = cv2.VideoCapture(video_path)  # 使用OpenCV打开视频文件
    if not cap.isOpened():  # 如果视频无法打开
        raise ValueError(f"无法打开视频：{video_path}")  # 抛出错误
    while True:
        ret, frame = cap.read()  # 读取视频帧
        if not ret:  # 如果读取失败，视频结束
            break
        cv2.imshow('播放视频', frame)  # 显示当前帧
        if cv2.waitKey(30) & 0xFF == 27:  # 如果按下ESC键退出
            break
    cap.release()  # 释放视频捕获对象
    cv2.destroyAllWindows()  # 销毁所有OpenCV窗口

# Tkinter 主界面
def main_ui():
    def open_image():
        global current_image, current_image_path
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.jpg *.png *.bmp")])  # 打开文件对话框选择图片
        if not file_path:  # 如果没有选择文件
            return
        try:
            img = read_image(file_path)  # 读取图片
            current_image = img  # 更新当前图像
            current_image_path = file_path  # 更新当前图片路径
            show_preview(img, "原始图像")  # 显示图像预览
        except Exception as e:  # 捕获异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def save_image_ui():
        global current_image
        if current_image is None:  # 如果没有加载图片
            messagebox.showwarning("警告", "没有图片可以保存")  # 弹出警告
            return
        file_path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png",  # 打开文件对话框保存图片
                                                 filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
        if file_path:
            save_image(file_path, current_image)  # 保存当前图片
            messagebox.showinfo("成功", "图片已保存")  # 弹出保存成功消息

    def process_image(action):
        global current_image
        if current_image is None:  # 如果没有加载图片
            messagebox.showwarning("警告", "请先加载图片")  # 弹出警告
            return
        try:
            if action == "灰度化":
                processed_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
            elif action == "调整亮度/对比度":
                processed_image = adjust_brightness_contrast(current_image, brightness=50, contrast=1.2)  # 调整亮度与对比度
            elif action == "直方图均衡化":
                processed_image = equalize_color_histogram(current_image)  # 直方图均衡化
            elif action == "缩放":
                processed_image = resize_image(current_image, fx=0.5, fy=0.5)  # 缩放图像
            else:
                raise ValueError("未知操作")  # 如果未知操作类型
            current_image = processed_image  # 更新当前图像
            show_preview(processed_image, f"{action} 后的图像")  # 显示处理后的图像
        except Exception as e:  # 捕获异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def resize_image_ui():
        global current_image
        if current_image is None:  # 如果没有加载图片
            messagebox.showwarning("警告", "请先加载图片")  # 弹出警告
            return
        try:
            # 获取用户输入的缩放比例
            x_scale = float(x_scale_entry.get())  # 获取水平缩放比例
            y_scale = float(y_scale_entry.get())  # 获取垂直缩放比例
            processed_image = resize_image(current_image, fx=x_scale, fy=y_scale)  # 缩放图像
            current_image = processed_image  # 更新当前图像
            show_preview(processed_image, "缩放后的图像")  # 显示缩放后的图像
        except ValueError:
            messagebox.showerror("错误", "请输入有效的缩放比例")  # 弹出错误信息
        except Exception as e:  # 捕获其他异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def add_border_ui():
        global current_image
        if current_image is None:  # 如果没有加载图片
            messagebox.showwarning("警告", "请先加载图片")  # 弹出警告
            return
        try:
            border_type = BORDER_TYPES[border_combobox.get()]  # 获取选中的边框类型
            top, bottom, left, right = 50, 50, 50, 50  # 设置边框的大小
            value = (0, 255, 0) if border_type == cv2.BORDER_CONSTANT else None  # 如果是常量边框，设置颜色
            processed_image = add_border(current_image, top, bottom, left, right, border_type, value=value)  # 添加边框
            current_image = processed_image  # 更新当前图像
            show_preview(processed_image, f"边框填充 ({border_combobox.get()})")  # 显示添加边框后的图像
        except Exception as e:  # 捕获异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def open_fusion_image1():
        global fusion_image1
        file_path = filedialog.askopenfilename(title="选择第一张图片", filetypes=[("Image Files", "*.jpg *.png *.bmp")])  # 选择第一张图像
        if not file_path:  # 如果没有选择文件
            return
        try:
            fusion_image1 = read_image(file_path)
            show_preview(fusion_image1, "融合图像1")  # 显示第一张融合图像的预览
        except Exception as e:  # 捕获异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def open_fusion_image2():
        global fusion_image2
        file_path = filedialog.askopenfilename(title="选择第二张图片",
                                               filetypes=[("Image Files", "*.jpg *.png *.bmp")])  # 选择第二张图像
        if not file_path:  # 如果没有选择文件
            return
        try:
            fusion_image2 = read_image(file_path)  # 读取第二张图像
            show_preview(fusion_image2, "融合图像2")  # 显示第二张融合图像的预览
        except Exception as e:  # 捕获异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def blend_images_ui():
        global fusion_image1, fusion_image2, current_image
        if fusion_image1 is None or fusion_image2 is None:  # 如果两张融合图像没有加载
            messagebox.showwarning("警告", "请先加载两张图片进行融合")  # 弹出警告
            return
        try:
            alpha = alpha_var.get()  # 获取融合权重（alpha值）
            beta = 1.0 - alpha  # 计算另一张图像的权重（beta）
            fused_image = blend_images(fusion_image1, fusion_image2, alpha, beta)  # 对两张图像进行融合
            current_image = fused_image  # 更新当前图像为融合后的图像
            show_preview(fused_image, f"融合图像 (alpha={alpha}, beta={beta})")  # 显示融合后的图像
        except Exception as e:  # 捕获异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def open_video():
        """打开并播放视频"""
        file_path = filedialog.askopenfilename(title="选择视频", filetypes=[("Video Files", "*.mp4 *.avi")])  # 选择视频文件
        if not file_path:  # 如果没有选择文件
            return
        try:
            play_video(file_path)  # 播放视频
        except Exception as e:  # 捕获异常
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    def show_preview(image, title=""):
        max_width, max_height = 800, 800  # 设置图像预览的最大宽度和高度
        height, width = image.shape[:2]  # 获取图像的高度和宽度
        scale = min(max_width / width, max_height / height, 1.0)  # 根据图像尺寸计算缩放比例
        new_width = int(width * scale)  # 计算新的宽度
        new_height = int(height * scale)  # 计算新的高度
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)  # 缩放图像
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) if len(
            resized_image.shape) == 3 else resized_image  # 转换为RGB模式，以便在Tkinter中显示
        pil_image = Image.fromarray(image_rgb)  # 将NumPy数组转换为PIL图像
        tk_image = ImageTk.PhotoImage(pil_image)  # 将PIL图像转换为Tkinter兼容的图像
        img_label.config(image=tk_image)  # 设置标签显示图像
        img_label.image = tk_image  # 保持对图像的引用，以防被垃圾回收
        img_label_title.config(text=title)  # 更新标题标签显示当前图像的描述

    # 主窗口布局调整
    root = tk.Tk()  # 创建Tkinter的主窗口
    root.title("图像和视频操作工具")  # 设置窗口标题
    root.geometry("1300x900")  # 设置窗口大小
    style = ttk.Style()  # 创建ttk样式对象
    style.configure("TButton", font=("Arial", 14))  # 设置按钮的字体大小
    style.configure("TLabel", font=("Arial", 14))  # 设置标签的字体大小

    # 第一行按钮区域
    row1_frame = tk.Frame(root)  # 创建一个Frame容器，放置第一行按钮
    row1_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)  # 将容器放置在顶部，并设置边距

    ttk.Button(row1_frame, text="打开图片", command=open_image).pack(pady=5, side=tk.LEFT, padx=10)  # 创建并放置“打开图片”按钮
    ttk.Button(row1_frame, text="保存图片", command=save_image_ui).pack(pady=5, side=tk.LEFT, padx=10)  # 创建并放置“保存图片”按钮
    ttk.Button(row1_frame, text="调整亮度/对比度", command=lambda: process_image("调整亮度/对比度")).pack(pady=5,
                                                                                                          side=tk.LEFT,
                                                                                                          padx=10)  # 创建并放置“调整亮度/对比度”按钮
    ttk.Button(row1_frame, text="直方图均衡化", command=lambda: process_image("直方图均衡化")).pack(pady=5,
                                                                                                    side=tk.LEFT,
                                                                                                    padx=10)  # 创建并放置“直方图均衡化”按钮
    ttk.Button(row1_frame, text="播放视频", command=open_video).pack(pady=5, side=tk.LEFT, padx=10)  # 创建并放置“播放视频”按钮

    # 边框类型选择
    BORDER_TYPES = {  # 定义可选的边框类型
        "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
        "BORDER_REFLECT": cv2.BORDER_REFLECT,
        "BORDER_REFLECT_101": cv2.BORDER_REFLECT_101,
        "BORDER_WRAP": cv2.BORDER_WRAP,
        "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
    }

    ttk.Label(row1_frame, text="选择边框类型:").pack(side=tk.LEFT, padx=5)  # 标签提示用户选择边框类型
    border_combobox = ttk.Combobox(row1_frame, values=list(BORDER_TYPES.keys()), state="readonly")  # 创建下拉框用于选择边框类型
    border_combobox.set("BORDER_CONSTANT")  # 设置默认选择“BORDER_CONSTANT”
    border_combobox.pack(side=tk.LEFT, padx=5)  # 放置下拉框
    ttk.Button(row1_frame, text="添加边框", command=add_border_ui).pack(pady=5, side=tk.LEFT, padx=10)  # 创建并放置“添加边框”按钮

    # 第二行按钮区域
    row2_frame = tk.Frame(root)  # 创建一个Frame容器，放置第二行按钮
    row2_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)  # 将容器放置在顶部，并设置边距

    ttk.Button(row2_frame, text="加载融合图像1", command=open_fusion_image1).pack(pady=5, side=tk.LEFT,
                                                                                  padx=10)  # 创建并放置“加载融合图像1”按钮
    ttk.Button(row2_frame, text="加载融合图像2", command=open_fusion_image2).pack(pady=5, side=tk.LEFT,
                                                                                  padx=10)  # 创建并放置“加载融合图像2”按钮
    ttk.Label(row2_frame, text="融合权重(alpha):").pack(side=tk.LEFT, padx=5)  # 标签提示用户选择融合权重
    alpha_var = tk.DoubleVar(value=0.5)  # 创建一个变量，用于控制alpha值，默认为0.5
    ttk.Scale(row2_frame, from_=0.0, to=1.0, variable=alpha_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT,
                                                                                                        padx=5)  # 创建并放置一个滑动条用于调整alpha值
    ttk.Button(row2_frame, text="融合图像", command=blend_images_ui).pack(pady=5, side=tk.LEFT,
                                                                          padx=10)  # 创建并放置“融合图像”按钮

    # 添加缩放比例输入框
    ttk.Label(row2_frame, text="水平缩放比例:").pack(side=tk.LEFT, padx=5)  # 标签提示用户输入水平缩放比例
    x_scale_entry = ttk.Entry(row2_frame, width=5)  # 创建输入框用于输入水平缩放比例
    x_scale_entry.insert(0, "1.0")  # 设置默认值为1.0
    x_scale_entry.pack(side=tk.LEFT, padx=5)

    ttk.Label(row2_frame, text="垂直缩放比例:").pack(side=tk.LEFT, padx=5)  # 标签提示用户输入垂直缩放比例
    y_scale_entry = ttk.Entry(row2_frame, width=5)  # 创建输入框用于输入垂直缩放比例
    y_scale_entry.insert(0, "1.0")  # 设置默认值为1.0
    y_scale_entry.pack(side=tk.LEFT, padx=5)

    ttk.Button(row2_frame, text="缩放", command=resize_image_ui).pack(pady=5, side=tk.LEFT, padx=10)  # 创建并放置“缩放”按钮

    # 图片显示区域
    img_frame = tk.Frame(root, width=400, height=400, bg="gray")  # 创建一个Frame容器，用于显示图像
    img_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.BOTH, expand=True)  # 放置在窗口底部，并设置填充

    img_label = tk.Label(img_frame)  # 创建一个标签用于显示图像
    img_label.pack(fill=tk.BOTH, expand=True)  # 将标签放入frame中并允许扩展

    img_label_title = tk.Label(root, text="", font=("Arial", 16))  # 创建一个标题标签
    img_label_title.pack(side=tk.BOTTOM, pady=5)  # 将标题标签放置在窗口底部，并设置上下边距

    root.mainloop()  # 启动Tkinter的主事件循环，开始运行GUI

# 启动UI
main_ui()  # 调用主界面函数，启动图像和视频操作工具的UI
