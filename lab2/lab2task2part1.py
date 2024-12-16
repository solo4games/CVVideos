import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функция для создания идеального низкочастотного фильтра
def ideal_lowpass_filter(shape, cutoff):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance <= cutoff:
                mask[i, j] = 1
    return mask

# Функция для применения идеального низкочастотного фильтра
def apply_ideal_lowpass_filter(image, cutoff):
    # Преобразование изображения в частотную область
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # Создание маски фильтра
    mask = ideal_lowpass_filter(image.shape, cutoff)

    # Применение фильтра
    filtered_dft_shifted = dft_shifted * mask[:, :, np.newaxis]

    # Обратное преобразование в пространственную область
    dft_inv_shifted = np.fft.ifftshift(filtered_dft_shifted)
    image_back = cv2.idft(dft_inv_shifted)
    image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

    return image_back

# Загрузка изображения в оттенках серого
image = cv2.imread('imageKitty.jpg', cv2.IMREAD_GRAYSCALE)

# Применение идеального низкочастотного фильтра
cutoff = 50  # Пороговое значение для фильтрации
ideal_filtered_image = apply_ideal_lowpass_filter(image, cutoff)

# Отображение оригинального и фильтрованного изображения
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(ideal_filtered_image, cmap='gray')
plt.title(f'Ideal Lowpass Filter (cutoff={cutoff})')

plt.show()