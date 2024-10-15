import cv2
import numpy as np


def manual_adaptive_threshold(image, block_size, C):
    """
    Функция для применения адаптивной пороговой обработки вручную.

    Параметры:
    - image: входное изображение (numpy массив)
    - block_size: размер блока для локальной пороговой обработки (должен быть нечетным)
    - C: константа, которая вычитается из среднего значения

    Возвращает:
    - бинарное изображение
    """
    # Проверяем, что размер блока нечетный
    if block_size % 2 == 0:
        raise ValueError("block_size должен быть нечетным числом")

    # Получаем размеры изображения
    height, width = image.shape

    # Создаем пустое изображение для результата
    thresholded_image = np.zeros((height, width), dtype=np.uint8)

    # Определяем смещение, чтобы центрировать блок вокруг каждого пикселя
    half_block = block_size // 2

    # Проходим по каждому пикселю
    for i in range(height):
        for j in range(width):
            # Определяем границы окна (учитывая граничные пиксели)
            x1 = max(0, i - half_block)
            x2 = min(height, i + half_block + 1)
            y1 = max(0, j - half_block)
            y2 = min(width, j + half_block + 1)

            # Извлекаем локальный блок
            local_block = image[x1:x2, y1:y2]

            # Вычисляем среднее значение в блоке
            local_mean = np.mean(local_block)

            # Применяем порог: если значение пикселя больше среднего минус C, делаем его белым
            if image[i, j] > (local_mean - C):
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

# Параметры адаптивной пороговой обработки
block_size = 11  # Размер блока (должен быть нечетным)
C = 2  # Константа, которая вычитается из среднего значения

# Применение адаптивной пороговой обработки вручную
adaptive_thresholded_image = manual_adaptive_threshold(image, block_size, C)

# Отображение исходного и обработанного изображений
cv2.imshow('Original Image', image)
cv2.imshow('Manual Adaptive Thresholded Image', adaptive_thresholded_image)

# Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()