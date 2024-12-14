import cv2
import numpy as np

# Собственная реализация пороговой сегментации
def thresholding(image, threshold):
    # Создание пустой матрицы для результата
    binary_image = np.zeros_like(image)

    # Назначение значений пикселям на основе порога
    binary_image[image > threshold] = 255
    return binary_image

# Загрузка изображения в оттенках серого
image = cv2.imread('imageKitty.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка успешности загрузки изображения
if image is None:
    print("Ошибка загрузки изображения.")
    exit()

# Пороговое значение
threshold_value = 127

# Применение пороговой сегментации
binary_image = thresholding(image, threshold_value)

# Отображение исходного и сегментированного изображения
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', binary_image)

# Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()
