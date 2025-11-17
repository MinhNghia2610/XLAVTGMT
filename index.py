from ultralytics import YOLO
import cv2
import numpy as np

# ===========================
# LOAD YOLO MODEL
# ===========================
model = YOLO("yolov8n.pt")

# ===========================
# CAMERA
# ===========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ===========================
# Kích thước layout gốc
# ===========================
YOLO_W, YOLO_H = 1920, 720       # YOLO nổi bật trên
ALG_W, ALG_H = 640, 360          # Sobel, Canny, Laplacian dưới

# Lấy tỉ lệ màn hình máy
screen_w, screen_h = 1366, 768   # ví dụ màn hình 1366x768, bạn có thể thay theo máy

# ===========================
# Scale ratio để vừa màn hình
# ===========================
layout_w = YOLO_W
layout_h = YOLO_H + ALG_H
scale_ratio = min(screen_w / layout_w, screen_h / layout_h)

final_w = int(layout_w * scale_ratio)
final_h = int(layout_h * scale_ratio)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===========================
    # YOLO DETECTION
    # ===========================
    results = model(frame, stream=True)
    yolo_frame = frame.copy()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls].upper()

            cv2.rectangle(yolo_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(yolo_frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Resize YOLO panel lên 1920x720 gốc
    yolo_panel = cv2.resize(yolo_frame, (YOLO_W, YOLO_H))

    # ===========================
    # EDGE DETECTION
    # ===========================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    sobel = cv2.resize(sobel, (ALG_W, ALG_H))

    canny = cv2.Canny(gray, 100, 200)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    canny = cv2.resize(canny, (ALG_W, ALG_H))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    lap = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    lap = cv2.resize(lap, (ALG_W, ALG_H))

    # Gộp 3 panel dưới
    bottom_row = np.hstack((sobel, canny, lap))  # width=1920, height=360

    # Gộp YOLO + bottom row
    full_layout = np.vstack((yolo_panel, bottom_row))  # width=1920, height=1080

    # ===========================
    # Scale toàn bộ layout xuống màn hình
    # ===========================
    final_layout = cv2.resize(full_layout, (final_w, final_h))

    cv2.imshow("YOLO + Sobel + Canny + Laplacian", final_layout)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
