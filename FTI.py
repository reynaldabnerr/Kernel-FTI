import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)  # Shift zero frequency component to center

# Calculate the magnitude spectrum
magnitude_spectrum = np.abs(fshift)
log_magnitude_spectrum = np.log(magnitude_spectrum + 1)  # Log for better visibility

# Process the Fourier Transformed Image (For example: apply low-pass filter)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create a mask with a circular low-pass filter in the frequency domain
mask = np.ones((rows, cols), np.uint8)
r = 30  # Radius of the low-pass filter (you can adjust this value)
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2
mask[mask_area] = 0  # Set mask to 0 inside the circle

# Apply the mask in the frequency domain
fshift_processed = fshift * mask

# Inverse Fourier Transform (IFT) to get the processed image in the spatial domain
f_ishift = np.fft.ifftshift(fshift_processed)
processed_image = np.fft.ifft2(f_ishift)
processed_image = np.abs(processed_image)

# Plotting the images
plt.figure(figsize=(12, 6))

# Original Image (Spatial Domain)
plt.subplot(1, 3, 1)
plt.title("Original Image (Spatial)")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Fourier Transformed Image (Frequency Domain)
plt.subplot(1, 3, 2)
plt.title("Fourier Transformed Image (Frequency)")
plt.imshow(log_magnitude_spectrum, cmap='gray')
plt.axis('off')

# Processed Image (Spatial Domain after IFT)
plt.subplot(1, 3, 3)
plt.title("Processed Image (Spatial)")
plt.imshow(processed_image, cmap='gray')
plt.axis('off')

plt.show()