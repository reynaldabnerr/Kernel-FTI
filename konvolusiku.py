import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi konvolusi manual
def konvolusiku(image, kernel):
    # Mendapatkan dimensi gambar dan kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Menentukan jarak padding (untuk menjaga dimensi gambar)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Membuat image yang sudah dipadding
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Menyiapkan array untuk menyimpan hasil konvolusi
    output = np.zeros_like(image)

    # Melakukan konvolusi
    for i in range(image_height):
        for j in range(image_width):
            # Ekstrak sub-gambar (window) yang akan dikonvolusi
            window = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Lakukan perkalian elemenwise dan jumlahkan hasilnya
            output[i, j] = np.sum(window * kernel)

    return output

# Membaca gambar Lenna.png
image = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)

# Mendefinisikan Outline Detection Kernel (misalnya Sobel)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Menggunakan fungsi konvolusi
output = konvolusiku(image, kernel)

# Menampilkan gambar asli dan hasilnya
plt.figure(figsize=(10, 5))

# Gambar asli
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Gambar hasil filter
plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title("Outline Detection")
plt.axis('off')

plt.show()