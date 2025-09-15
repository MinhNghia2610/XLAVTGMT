# Edge Detection: Sobel, Canny, Laplacian

## 📌 Giới thiệu
Trong xử lý ảnh và thị giác máy tính, **phát hiện biên (edge detection)** là một bước quan trọng để xác định ranh giới và hình dạng của đối tượng trong ảnh.  
Trong dự án này, chúng ta so sánh ba thuật toán phổ biến: **Sobel, Canny và Laplacian**.

---

## 📖 1. Kiến thức nền tảng

### 🔹 Sobel
- Toán tử đạo hàm bậc nhất, dùng để tính **gradient** theo trục X và Y.
- Dùng hai kernel: SobelX, SobelY.
- **Ưu điểm**: Đơn giản, dễ cài đặt.  
- **Nhược điểm**: Nhạy với nhiễu, biên dày.  

### 🔹 Canny
- Thuật toán nhiều bước:
  1. Làm mờ ảnh bằng Gaussian.
  2. Tính gradient (Sobel).
  3. Non-maximum suppression (làm mảnh biên).
  4. Double threshold (phân loại biên mạnh/yếu).
  5. Edge tracking by hysteresis (liên kết biên).
- **Ưu điểm**: Biên sắc nét, liên tục, loại bỏ nhiễu tốt.  
- **Nhược điểm**: Tính toán phức tạp, có nhiều tham số.  

### 🔹 Laplacian
- Toán tử đạo hàm bậc hai.  
- Kernel phổ biến:  

[ 0 -1  0 ]
[-1  4 -1 ]
[ 0 -1  0 ]


- **Ưu điểm**: Phát hiện biên theo mọi hướng.  
- **Nhược điểm**: Rất nhạy với nhiễu, thường cần làm mờ ảnh trước.  

---

## ⚖️ 2. So sánh ba thuật toán

| Thuật toán    |     Đặc trưng          |        Biên thu được           | Độ nhạy nhiễu  |
|---------------|------------------------|--------------------------------|----------------|
| **Sobel**     | Gradient bậc nhất      | Biên dày, không mảnh           | Trung bình     |
| **Canny**     | Chuỗi xử lý nhiều bước | Biên mảnh, chính xác, liên tục | Rất tốt        |
| **Laplacian** | Gradient bậc hai       | Biên rõ theo mọi hướng         | Nhạy cao       |

---

## 🛠️ 3. Cài đặt & Code minh họa

### Yêu cầu
- Python 3.x  
- OpenCV (`pip install opencv-python`)  
- Matplotlib (`pip install matplotlib`)  

### Code

```python
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh xám
img = cv2.imread('lena.png', 0)

# Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# Canny
canny = cv2.Canny(img, 100, 200)

# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Hiển thị kết quả
titles = ['Original', 'Sobel', 'Canny', 'Laplacian']
images = [img, sobel, canny, laplacian]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
