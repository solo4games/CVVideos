import cv2
import numpy as np


def adaptive_threshold_custom(image, block_size=11, C=2):
    """
    Реализация адаптивной пороговой обработки для выделения переднего и заднего фона.

    :param image: Входное изображение в оттенках серого
    :param block_size: Размер блока, в котором вычисляется пороговое значение (должно быть нечетным)
    :param C: Константа, которая вычитается из среднего значения (регулирует контраст)
    :return: Изображение с белым передним планом и черным задним фоном
    """
    # Проверка на корректность параметров
    if block_size % 2 == 0 or block_size <= 1:
        raise ValueError("Размер блока должен быть нечетным и больше 1.")

    # Получаем размеры изображения
    rows, cols = image.shape

    # Создаем пустое изображение для результата
    result_image = np.zeros_like(image)

    # Отступы от краев изображения (радиус вокруг центрального пикселя окна)
    offset = block_size // 2

    # Проход по изображению, вычисление среднего значения для каждого блока
    for i in range(offset, rows - offset):
        for j in range(offset, cols - offset):
            # Извлекаем локальный блок вокруг текущего пикселя
            block = image[i - offset:i + offset + 1, j - offset:j + offset + 1]

            # Вычисляем среднее значение блока
            mean_value = np.mean(block)

            # Применяем пороговое значение: если пиксель больше среднего - передний план (255), иначе - задний (0)
            if image[i, j] > (mean_value - C):
                result_image[i, j] = 255  # Передний план закрашиваем белым (255)
            else:
                result_image[i, j] = 0  # Задний фон закрашиваем черным (0)

    return result_image


# Загрузка изображения в оттенках серого
image = cv2.imread('anime_background_image.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка, что изображение было загружено
if image is None:
    raise ValueError("Изображение не найдено!")

# Применение адаптивной пороговой обработки для выделения переднего и заднего фона
result_image = adaptive_threshold_custom(image, block_size=11, C=2)

cv2.imwrite('segmented_foreground_background.png', result_image)

# Визуализация оригинального и обработанного изображения
cv2.imshow("Original Image", image)
cv2.imshow("Segmented Image (Foreground/Background)", result_image)

# Ожидание нажатия любой клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()