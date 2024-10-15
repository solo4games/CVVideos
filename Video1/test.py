import cv2
import numpy as np


def manual_threshold(image, threshold_value):
    """
    Функция для применения пороговой обработки к изображению вручную.

    Параметры:
    - image: входное изображение (numpy массив)
    - threshold_value: пороговое значение

    Возвращает:
    - бинарное изображение
    """
    # Получаем размеры изображения
    height, width = image.shape

    # Создаем пустой массив для хранения порогового изображения
    thresholded_image = np.zeros((height, width), dtype=np.uint8)

    # Проходим по каждому пикселю
    for i in range(height):
        for j in range(width):
            # Если значение пикселя больше порога, делаем его белым (255), иначе — черным (0)
            if image[i, j] > threshold_value:
                thresholded_image[i, j] = 255
            else:
                thresholded_image[i, j] = 0

    return thresholded_image


# Загрузка изображения в градациях серого
image = cv2.imread('imageKitty.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка успешности загрузки изображения
if image is None:
    print("Ошибка загрузки изображения.")
    exit()

# Устанавливаем пороговое значение
threshold_value = 127

# Применение пороговой обработки вручную
thresholded_image = manual_threshold(image, threshold_value)

# Отображение исходного и обработанного изображений
cv2.imshow('Original Image', image)
cv2.imshow('Manual Thresholded Image', thresholded_image)

# Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()