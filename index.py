from ultralytics import YOLO
import cv2

# =====================================================
# LOAD MODEL YOLO (COCO 80 classes)
# YOLOv8n là bản nhẹ, chạy nhanh, phù hợp laptop
# =====================================================
model = YOLO("yolov8n.pt")  # sẽ tự tải nếu chưa có

# =====================================================
# MỞ CAMERA
# =====================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =================================================
    # NHẬN DIỆN TRỰC TIẾP VỚI YOLO
    # =================================================
    results = model(frame, stream=True)

    # =================================================
    # VẼ KẾT QUẢ LÊN MÀN HÌNH
    # =================================================
    for r in results:
        for box in r.boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = box.xyxy[0]

            # Độ chính xác
            conf = float(box.conf[0])

            # Loại đối tượng (person, dog, cat, car,...)
            cls = int(box.cls[0])
            label = model.names[cls].upper()

            # Vẽ khung
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # Vẽ nhãn
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # =================================================
    # HIỂN THỊ CAMERA
    # =================================================
    cv2.imshow("NHAN DIEN DA DOI TUONG - YOLOv8", frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
   