import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import imutils


# 显示图像到 Tkinter UI
def show_preview(image, title=""):
    max_width, max_height = 600, 400
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    if len(resized_image.shape) == 3:  # 彩色图像
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    else:  # 灰度图像
        image_rgb = resized_image
    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(pil_image)
    img_label.config(image=tk_image)
    img_label.image = tk_image
    img_label_title.config(text=title)


# 图像预处理（灰度化、模糊化、边缘检测）
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return edged


# 四点透视变换
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


# 点排序
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# 检测答题卡的被涂区域
def detect_answers(image):
    # 自适应二值化
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 检测轮廓
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # 答题卡的区域筛选
    question_cnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if 20 <= w <= 50 and 20 <= h <= 50 and 0.9 <= aspect_ratio <= 1.1:
            question_cnts.append(c)

    # 根据位置排序
    question_cnts = sorted(question_cnts, key=lambda x: cv2.boundingRect(x)[1])

    answers = []
    for i in range(0, len(question_cnts), 5):
        row = sorted(question_cnts[i:i + 5], key=lambda x: cv2.boundingRect(x)[0])
        answers.append(row)

    result = []
    for row in answers:
        bubbled = None
        for j, c in enumerate(row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        result.append(bubbled[1])

    return result


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

    def process_image():
        """处理图像并检测答题卡答案"""
        global current_image
        if current_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        try:
            # 图像预处理
            edged = preprocess_image(current_image)

            # 检测并提取答题卡
            cnts = imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
            doc_cnt = None
            for c in sorted(cnts, key=cv2.contourArea, reverse=True):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    doc_cnt = approx
                    break

            if doc_cnt is None:
                raise ValueError("无法找到答题卡区域")

            warped = four_point_transform(cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY), doc_cnt.reshape(4, 2))
            show_preview(warped, "透视变换后的答题卡")

            # 检测答案
            answers = detect_answers(warped)
            result_label.config(text=f"检测到的答案: {answers}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # 创建主窗口
    root = tk.Tk()
    root.title("答题卡识别工具")
    root.geometry("1000x700")

    # 顶部按钮区域
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

    ttk.Style().configure("TButton", font=("Arial", 12))

    ttk.Button(button_frame, text="打开图片", command=open_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="处理图片", command=process_image).pack(side=tk.LEFT, padx=5)

    # 图像显示区域
    img_frame = tk.Frame(root, width=600, height=600, bg="gray")
    img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    global img_label, img_label_title, result_label
    img_label = tk.Label(img_frame)
    img_label.pack(fill=tk.BOTH, expand=True)

    img_label_title = tk.Label(root, text="", font=("Arial", 14))
    img_label_title.pack(side=tk.BOTTOM, pady=5)

    result_label = tk.Label(root, text="", font=("Arial", 14))
    result_label.pack(side=tk.BOTTOM, pady=5)

    root.mainloop()

# 启动 Tkinter UI
main_ui()
