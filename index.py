import cv2
import numpy as np

# Mở camera (0 = camera mặc định)
cap = cv2.VideoCapture(0)

# Đặt kích thước camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Sobel ---
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)

    # --- Scharr (nâng cấp Sobel) ---
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr = cv2.magnitude(scharrx, scharry)
    scharr = cv2.convertScaleAbs(scharr)

    # --- Canny (Adaptive threshold) ---
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    canny = cv2.Canny(gray, lower, upper)

    # --- Laplacian of Gaussian ---
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Chuẩn hóa tất cả ảnh thành 3 kênh để ghép cùng nhau
    gray_c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    sobel_c = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    scharr_c = cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
    canny_c = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    laplacian_c = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    # Ghép các ảnh lại (2x3 grid)
    top_row = np.hstack((gray_c, sobel_c, scharr_c))
    bottom_row = np.hstack((canny_c, laplacian_c, frame))
    combined = np.vstack((top_row, bottom_row))

    # Resize về đúng 800x600 để không tràn màn hình
    combined_resized = cv2.resize(combined, (800, 600))

    # Hiển thị
    cv2.imshow("Edge Detection Comparison", combined_resized)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
