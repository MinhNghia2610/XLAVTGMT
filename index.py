import cv2
import numpy as np

# ==============================
# NHẬN DIỆN NGƯỜI BẰNG HOG + SVM
# ==============================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Mở camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize cho nhanh hơn
    resize_frame = cv2.resize(frame, (640, 480))

    # ======================
    # 1️⃣ Nhận diện người
    # ======================
    (rects, weights) = hog.detectMultiScale(resize_frame,
                                            winStride=(8, 8),
                                            padding=(8, 8),
                                            scale=1.05)

    # ======================
    # Vẽ bounding box
    # ======================
    for (x, y, w, h) in rects:
        cv2.rectangle(resize_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(resize_frame, "Human", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ======================
    # Nếu không phát hiện người → đánh nhãn Object
    # ======================
    if len(rects) == 0:
        cv2.putText(resize_frame, "Object detected (not human)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("NHAN DIEN NGUOI / VAT THE - HOG + SVM", resize_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
