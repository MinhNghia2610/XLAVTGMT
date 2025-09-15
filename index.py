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
titles = ['Ảnh Gốc', 'Sobel', 'Canny', 'Laplacian']
images = [img, sobel, canny, laplacian]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
