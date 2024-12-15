import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# 模板匹配方法映射
MATCH_METHODS = {
    "TM_SQDIFF": cv2.TM_SQDIFF,
    "TM_CCORR": cv2.TM_CCORR,
    "TM_CCOEFF": cv2.TM_CCOEFF,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
}

# 图像金字塔操作函数
def pyramid_operations(img, operation):
    # 根据选择的操作类型执行金字塔上采样或下采样
    if operation == "PyrUp":
        return cv2.pyrUp(img)  # 执行金字塔上采样
    elif operation == "PyrDown":
        return cv2.pyrDown(img)  # 执行金字塔下采样
    else:
        raise ValueError("未知金字塔操作")  # 抛出未知操作异常

# 轮廓检测函数
def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 应用二值化阈值处理
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 查找图像轮廓
    return contours, hierarchy, thresh  # 返回轮廓和层次结构

# 绘制轮廓函数
def draw_contours(img, contours, index=-1):
    # 在图像上绘制轮廓，默认为绘制所有轮廓
    return cv2.drawContours(img.copy(), contours, index, (0, 0, 255), 2)  # 用红色绘制轮廓

# 模板匹配函数
def template_matching(img, template, method=cv2.TM_CCOEFF_NORMED):
    # 使用指定的方法执行模板匹配
    return cv2.matchTemplate(img, template, method)  # 返回匹配结果

# 多模板匹配函数
def multi_template_matching(img, template, threshold=0.8):
    # 获取模板的高度和宽度
    h, w = template.shape[:2]
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)  # 执行模板匹配
    loc = np.where(res >= threshold)  # 查找符合阈值的坐标
    matches = []  # 存储匹配结果
    for pt in zip(*loc[::-1]):
        # 保存每个匹配的左上角和右下角的坐标
        matches.append((pt, (pt[0] + w, pt[1] + h)))
    return matches  # 返回所有匹配的区域

# Tkinter 主界面函数
def main_ui():
    # 全局变量定义
    global current_image, current_image_path, template_image, selected_method
    current_image = None
    template_image = None
    current_image_path = None
    selected_method = "TM_CCOEFF_NORMED"  # 默认匹配方法

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

    # 打开模板函数
    def open_template():
        """打开模板文件"""
        global template_image
        file_path = filedialog.askopenfilename(title="选择模板", filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if not file_path:  # 如果没有选择文件，则返回
            return
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度模板图像
            if img is None:  # 检查模板是否读取成功
                raise ValueError("无法读取模板")
            template_image = img  # 保存模板图像
            show_preview(img, "模板图像")  # 显示模板图像
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

    # 应用操作函数
    def apply_operation(operation):
        """应用图像操作"""
        global current_image, template_image
        if current_image is None:  # 检查是否加载了图像
            messagebox.showwarning("警告", "请先加载图片")
            return
        if operation == "Multi Template Matching" and template_image is None:  # 检查多模板匹配是否加载了模板
            messagebox.showwarning("警告", "请先加载模板")
            return
        try:
            if operation in ["PyrUp", "PyrDown"]:  # 如果操作是金字塔上采样或下采样
                result = pyramid_operations(current_image, operation)
                show_preview(result, f"{operation} 操作结果")  # 显示操作结果
            elif operation == "Contours":  # 如果操作是轮廓检测
                contours, hierarchy, thresh = find_contours(current_image)
                result = draw_contours(current_image, contours)  # 绘制轮廓
                show_preview(result, f"{operation} 操作结果")  # 显示操作结果
            elif operation == "Multi Template Matching":  # 如果操作是多模板匹配
                threshold = threshold_var.get()  # 获取阈值
                matches = multi_template_matching(current_image, template_image, threshold)  # 进行多模板匹配
                show_multi_template_matching_on_main_page(matches)  # 在主页面显示匹配结果
            else:
                raise ValueError("未知操作")  # 抛出未知操作异常
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    # 应用模板匹配函数
    def apply_template_matching():
        """应用模板匹配"""
        global current_image, template_image, selected_method
        if current_image is None:  # 检查是否加载了图像
            messagebox.showwarning("警告", "请先加载图片")
            return
        if template_image is None:  # 检查是否加载了模板
            messagebox.showwarning("警告", "请先加载模板")
            return
        try:
            gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # 将当前图像转换为灰度图像
            method = MATCH_METHODS[selected_method]  # 获取选择的模板匹配方法
            match_result = template_matching(gray_image, template_image, method)  # 进行模板匹配
            show_template_matching_on_main_page(match_result, method)  # 显示匹配结果
        except Exception as e:
            messagebox.showerror("错误", str(e))  # 弹出错误信息

    # 在主页面显示模板匹配结果的函数
    def show_template_matching_on_main_page(match_result, method):
        """在主页面显示模板匹配结果"""
        global current_image, template_image

        h, w = template_image.shape[:2]  # 获取模板的高度和宽度
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)  # 获取匹配结果的最小值和最大值及其位置

        # 根据匹配方法选择匹配点
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  # 如果是平方差匹配方法
            top_left = min_loc  # 选择最小值位置
        else:
            top_left = max_loc  # 选择最大值位置

        bottom_right = (top_left[0] + w, top_left[1] + h)  # 计算矩形的右下角坐标

        result_image = current_image.copy()  # 复制当前图像
        cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 255), 2)  # 在匹配区域绘制矩形框
        show_preview(result_image, f"模板匹配结果 ({selected_method})")  # 显示带有匹配矩形的图像

    # 在主页面显示多模板匹配结果的函数
    def show_multi_template_matching_on_main_page(matches):
        """在主页面显示多模板匹配结果"""
        global current_image, template_image

        result_image = current_image.copy()  # 复制当前图像
        for top_left, bottom_right in matches:
            cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)  # 在每个匹配区域绘制矩形框

        show_preview(result_image, "多模板匹配结果")  # 显示匹配结果图像

    # 显示图像预览函数
    def show_preview(image, title=""):
        """显示图片预览"""
        max_width, max_height = 400, 400  # 限制图片显示的最大宽度和高度
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

    # 主窗口布局设置
    root = tk.Tk()
    root.title("图像操作工具")  # 设置窗口标题
    root.geometry("1000x700")  # 设置窗口大小

    # 功能按钮区域（顶部）
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

    ttk.Style().configure("TButton", font=("Arial", 12))  # 设置按钮样式

    # 按钮控件的创建和布局
    ttk.Button(button_frame, text="打开图片", command=open_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="加载模板", command=open_template).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="保存图片", command=save_image_ui).pack(side=tk.LEFT, padx=5)

    # 添加功能按钮，用于金字塔操作和轮廓检测
    ttk.Button(button_frame, text="PyrUp", command=lambda: apply_operation("PyrUp")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="PyrDown", command=lambda: apply_operation("PyrDown")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Contours", command=lambda: apply_operation("Contours")).pack(side=tk.LEFT, padx=5)

    # 模板匹配方法选择
    method_label = ttk.Label(button_frame, text="模板匹配方法:")
    method_label.pack(side=tk.LEFT, padx=5)

    method_combobox = ttk.Combobox(button_frame, values=list(MATCH_METHODS.keys()), state="readonly")  # 创建下拉选择框
    method_combobox.set(selected_method)  # 设置默认值
    method_combobox.pack(side=tk.LEFT, padx=5)

    def update_method(event):
        global selected_method  # 使用 global 声明全局变量
        selected_method = method_combobox.get()  # 更新选定的模板匹配方法

    method_combobox.bind("<<ComboboxSelected>>", update_method)  # 当下拉框更改时更新方法

    ttk.Button(button_frame, text="模板匹配", command=apply_template_matching).pack(side=tk.LEFT, padx=5)  # 模板匹配按钮

    # 阈值输入框
    threshold_label = ttk.Label(button_frame, text="阈值:")
    threshold_label.pack(side=tk.LEFT, padx=5)

    threshold_var = tk.DoubleVar(value=0.8)  # 默认阈值设置为0.8
    threshold_entry = ttk.Entry(button_frame, textvariable=threshold_var, width=5)  # 创建阈值输入框
    threshold_entry.pack(side=tk.LEFT, padx=5)

    ttk.Button(button_frame, text="多模板匹配", command=lambda: apply_operation("Multi Template Matching")).pack(
        side=tk.LEFT, padx=5)  # 多模板匹配按钮

    # 图片显示区域
    img_frame = tk.Frame(root, width=600, height=600, bg="gray")  # 创建图像显示框架
    img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    img_label = tk.Label(img_frame)  # 创建图像标签
    img_label.pack(fill=tk.BOTH, expand=True)  # 填充整个显示区域

    img_label_title = tk.Label(root, text="", font=("Arial", 14))  # 创建标题标签
    img_label_title.pack(side=tk.BOTTOM, pady=5)  # 显示在底部

    root.mainloop()  # 启动主循环


# 启动UI
main_ui()  # 调用主UI函数来启动应用程序
