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
import numpy as np

# ==============================
# ĐỀ TÀI: NHẬN DIỆN VẬT THỂ BẰNG BIÊN,
#          PHÁT HIỆN CHUYỂN ĐỘNG BẰNG
#          CÁC THUẬT TOÁN SOBEL, CANNY, LAPLACIAN
# ==============================

# Mở camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Khung hình trước để phát hiện chuyển động
previous_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ======================
    # 1️⃣ Phát hiện biên bằng Sobel
    # ======================
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)

    # ======================
    # 2️⃣ Phát hiện biên bằng Canny
    # ======================
    v = np.median(blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    canny = cv2.Canny(blur, lower, upper)

    # ======================
    # 3️⃣ Phát hiện biên bằng Laplacian
    # ======================
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # ======================
    # 4️⃣ Phát hiện chuyển động
    # ======================
    motion_mask = np.zeros_like(gray)
    if previous_frame is not None:
        diff = cv2.absdiff(previous_frame, blur)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Làm mượt vùng chuyển động
        motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        # Vẽ khung chữ nhật quanh vùng chuyển động
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 1000:  # chỉ vẽ nếu vùng đủ lớn
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Chuyen dong", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    previous_frame = blur.copy()

    # ======================
    # Hiển thị kết quả
    # ======================
    sobel_c = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    canny_c = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    laplacian_c = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    motion_c = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    # Gộp các khung hình
    top_row = np.hstack((sobel_c, canny_c, laplacian_c))
    bottom_row = np.hstack((motion_c, frame, np.zeros_like(frame)))
    combined = np.vstack((top_row, bottom_row))

    combined_resized = cv2.resize(combined, (1200, 700))
    cv2.imshow("NHAN DIEN VAT THE & PHAT HIEN CHUYEN DONG (Sobel - Canny - Laplacian)", combined_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
