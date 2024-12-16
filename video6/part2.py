import numpy as np
import cv2

# =============== Алгоритм 2: Построение карты глубины вручную ===============

def compute_disparity_map(img_left, img_right, block_size=5, disparities=16):
    """
    Вычисляет карту глубины с использованием метода блокового сопоставления.

    :param img_left: numpy.ndarray, левое изображение в градациях серого.
    :param img_right: numpy.ndarray, правое изображение в градациях серого.
    :param block_size: int, размер блока для сравнения (должен быть нечетным).
    :param disparities: int, максимальное смещение (глубина).
    :return: numpy.ndarray, карта глубины (размер совпадает с исходными изображениями).
    """
    height, width = img_left.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)

    half_block = block_size // 2

    # Перебор всех пикселей в изображении
    for y in range(half_block, height - half_block):
        for x in range(half_block, width - half_block):
            min_diff = float('inf')
            best_offset = 0

            # Извлечение блока из левого изображения
            block_left = img_left[y - half_block:y + half_block + 1, x - half_block:x + half_block + 1]

            for d in range(disparities):
                if x - d - half_block < 0:  # Проверка на границы
                    continue

                # Извлечение блока из правого изображения
                block_right = img_right[y - half_block:y + half_block + 1, x - d - half_block:x - d + half_block + 1]

                # Сравнение блоков с использованием суммы абсолютных различий (SAD)
                diff = np.sum(np.abs(block_left - block_right))

                if diff < min_diff:
                    min_diff = diff
                    best_offset = d

            # Запись значения несоответствия в карту глубины
            disparity_map[y, x] = best_offset

    return disparity_map

# Пример использования
img_left = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)
disparity_map = compute_disparity_map(img_left, img_right)
cv2.imshow('Disparity Map', (disparity_map / 16 * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
