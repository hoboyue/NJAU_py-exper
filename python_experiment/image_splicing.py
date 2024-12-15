import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class PanoramaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("全景拼接工具")
        self.root.geometry("1000x800")
        self.image_left = None
        self.image_right = None
        self.ratio = 2  # 缩放比例

        # 界面布局
        self.create_widgets()

    def create_widgets(self):
        # 按钮区
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, padx=20, pady=10, fill=tk.X)

        ttk.Style().configure("TButton", font=("Arial", 12))
        ttk.Button(button_frame, text="打开左图", command=self.load_left_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="打开右图", command=self.load_right_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="拼接图片", command=self.stitch_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存拼接结果", command=self.save_image).pack(side=tk.LEFT, padx=5)

        # 图像显示区
        self.img_frame = tk.Frame(self.root, width=800, height=600, bg="gray")
        self.img_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.left_label = tk.Label(self.img_frame, text="左图", bg="white", relief="solid")
        self.left_label.place(x=50, y=50, width=400, height=400)

        self.right_label = tk.Label(self.img_frame, text="右图", bg="white", relief="solid")
        self.right_label.place(x=550, y=50, width=400, height=400)

        self.result_label = tk.Label(self.img_frame, text="拼接结果", bg="white", relief="solid")
        self.result_label.place(x=300, y=500, width=400, height=200)

    def load_left_image(self):
        file_path = filedialog.askopenfilename(title="选择左图", filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if file_path:
            self.image_left = cv2.imread(file_path)
            self.display_image(self.image_left, self.left_label)
        else:
            messagebox.showwarning("警告", "未选择任何图片！")

    def load_right_image(self):
        file_path = filedialog.askopenfilename(title="选择右图", filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if file_path:
            self.image_right = cv2.imread(file_path)
            self.display_image(self.image_right, self.right_label)
        else:
            messagebox.showwarning("警告", "未选择任何图片！")

    def stitch_images(self):
        if self.image_left is None or self.image_right is None:
            messagebox.showwarning("警告", "请先加载两张图片！")
            return

        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch([self.image_left, self.image_right])

        if status == cv2.Stitcher_OK:
            # 黑边裁剪处理
            stitched = cv2.copyMakeBorder(pano, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            mask = np.zeros(thresh.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(cnts[0])
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            minRect = mask.copy()
            sub = mask.copy()

            while cv2.countNonZero(sub) > 0:
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)

            cnts = cv2.findContours(minRect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            (x, y, w, h) = cv2.boundingRect(cnts[0])
            self.panoramic_image = stitched[y:y + h, x:x + w]

            self.display_image(self.panoramic_image, self.result_label)
        else:
            messagebox.showerror("错误", "拼接失败，可能是特征点不足！")

    def save_image(self):
        if hasattr(self, 'panoramic_image') and self.panoramic_image is not None:
            file_path = filedialog.asksaveasfilename(title="保存拼接结果", filetypes=[("Image Files", "*.jpg *.png *.bmp")],
                                                     defaultextension=".jpg")
            if file_path:
                cv2.imwrite(file_path, self.panoramic_image)
                messagebox.showinfo("成功", f"拼接结果已保存到：{file_path}")
        else:
            messagebox.showwarning("警告", "没有拼接结果可保存！")

    def display_image(self, image, label):
        max_width, max_height = 400, 400
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk


root = tk.Tk()
app = PanoramaApp(root)
root.mainloop()
