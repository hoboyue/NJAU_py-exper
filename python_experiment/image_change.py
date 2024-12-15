import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk


def calc_gray_hist(img):
    """计算灰度图直方图，设置背景为白色并标注坐标轴"""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_img = np.full((320, 400, 3), 255, dtype=np.uint8)  # 创建白色背景图像

    # 绘制坐标轴
    cv2.line(hist_img, (50, 10), (50, 290), (0, 0, 0), 2)  # y 轴
    cv2.line(hist_img, (50, 290), (390, 290), (0, 0, 0), 2)  # x 轴

    # 标注 y 轴刻度
    for i in range(0, 301, 50):
        cv2.line(hist_img, (45, 290 - i), (50, 290 - i), (0, 0, 0), 1)
        cv2.putText(hist_img, str(i), (5, 295 - i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 标注 x 轴刻度
    for i in range(0, 256, 50):
        x = int(50 + i * (340 / 256))
        cv2.line(hist_img, (x, 290), (x, 295), (0, 0, 0), 1)
        cv2.putText(hist_img, str(i), (x - 10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 归一化直方图高度
    cv2.normalize(hist, hist, 0, 280, cv2.NORM_MINMAX)

    # 绘制直方图
    for i in range(1, 256):
        value1 = int(hist[i - 1].item())
        value2 = int(hist[i].item())
        x1 = int(50 + (i - 1) * (340 / 256))
        x2 = int(50 + i * (340 / 256))
        cv2.line(hist_img, (x1, 290 - value1), (x2, 290 - value2), (0, 0, 0), 1)
    return hist_img


def calc_color_hist(img):
    """计算三通道直方图，设置背景为白色并标注坐标轴"""
    hist_img = np.full((320, 400, 3), 255, dtype=np.uint8)  # 创建白色背景图像
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 280, cv2.NORM_MINMAX)

        # 绘制直方图
        for j in range(1, 256):
            value1 = int(hist[j - 1].item())
            value2 = int(hist[j].item())
            x1 = int(50 + (j - 1) * (340 / 256))
            x2 = int(50 + j * (340 / 256))
            cv2.line(hist_img,
                     (x1, 290 - value1), (x2, 290 - value2),
                     (255, 0, 0) if col == 'b' else (0, 255, 0) if col == 'g' else (0, 0, 255), 1)

    # 绘制坐标轴
    cv2.line(hist_img, (50, 10), (50, 290), (0, 0, 0), 2)  # y 轴
    cv2.line(hist_img, (50, 290), (390, 290), (0, 0, 0), 2)  # x 轴

    # 标注 y 轴刻度
    for i in range(0, 301, 50):
        cv2.line(hist_img, (45, 290 - i), (50, 290 - i), (0, 0, 0), 1)
        cv2.putText(hist_img, str(i), (5, 295 - i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 标注 x 轴刻度
    for i in range(0, 256, 50):
        x = int(50 + i * (340 / 256))
        cv2.line(hist_img, (x, 290), (x, 295), (0, 0, 0), 1)
        cv2.putText(hist_img, str(i), (x - 10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return hist_img

def apply_clahe(img):
    """应用 CLAHE 并返回结果"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)
    return result


def fourier_transform(img):
    """傅里叶变换并返回频谱图"""
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude_spectrum


def low_pass_filter(img):
    """低通滤波并返回结果"""
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # 创建低通滤波掩码
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 应用掩码
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

def high_pass_filter(img):
    """高通滤波并返回结果"""
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # 创建高通滤波掩码
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 应用掩码
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

# 显示图像到 UI
def show_preview(image, title=""):
    """在UI上显示图像"""
    max_width, max_height = 400, 400
    height, width = image.shape[:2]

    # 计算缩放比例
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 转换为 RGB 或灰度模式
    if len(resized_image.shape) == 3:  # 彩色图像
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    else:  # 灰度图像
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
            if operation == "灰度直方图":
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                result = calc_gray_hist(gray)
                show_preview(result, "灰度直方图")
            elif operation == "三通道直方图":
                result = calc_color_hist(current_image)
                show_preview(result, "三通道直方图")
            elif operation == "CLAHE":
                result = apply_clahe(current_image)
                show_preview(result, "CLAHE 结果")
            elif operation == "傅里叶变换":
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                result = fourier_transform(gray)
                show_preview(result, "傅里叶变换频谱图")
            elif operation == "低通滤波":
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                result = low_pass_filter(gray)
                show_preview(result, "低通滤波结果")
            elif operation == "高通滤波":
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                result = high_pass_filter(gray)
                show_preview(result, "高通滤波结果")
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
    ttk.Button(button_frame, text="灰度直方图", command=lambda: apply_operation("灰度直方图")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="三通道直方图", command=lambda: apply_operation("三通道直方图")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="CLAHE", command=lambda: apply_operation("CLAHE")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="傅里叶变换", command=lambda: apply_operation("傅里叶变换")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="低通滤波", command=lambda: apply_operation("低通滤波")).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="高通滤波", command=lambda: apply_operation("高通滤波")).pack(side=tk.LEFT, padx=5)

    # 图片显示区域
    img_frame = tk.Frame(root, width=600, height=600, bg="gray")
    img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    global img_label, img_label_title
    img_label = tk.Label(img_frame)
    img_label.pack(fill=tk.BOTH, expand=True)

    img_label_title = tk.Label(root, text="", font=("Arial", 14))
    img_label_title.pack(side=tk.BOTTOM, pady=5)

    root.mainloop()


# 启动UI
main_ui()
