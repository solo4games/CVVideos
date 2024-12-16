import cv2
import numpy as np


# Функция для вычисления гистограммы
def calculate_histogram(image):
    hist = np.zeros(256)
    for pixel in image.ravel():
        hist[pixel] += 1
    return hist


# Функция бинаризации по методу Оцу
def otsu_threshold(image):
    # Вычисление гистограммы
    hist = calculate_histogram(image)

    total_pixels = image.shape[0] * image.shape[1]
    sum_all = np.dot(np.arange(256), hist)  # Сумма всех интенсивностей умноженных на их количество
    sum_background, weight_background, max_variance, threshold = 0, 0, 0, 0
    weight_foreground = total_pixels  # Изначально весь вес на переднем плане (все пиксели)

    for t in range(256):
        #weight_background увеличивается на количество пикселей с интенсивностью t.
        weight_background += hist[t]
        #Если фоновая масса равна 0, цикл переходит к следующему значению порога.
        if weight_background == 0:
            continue

        #weight_foreground уменьшается на количество пикселей с интенсивностью t.
        weight_foreground -= hist[t]
        #Если количество пикселей переднего плана становится равно 0, алгоритм завершает выполнение.
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground

        # Межклассовая дисперсия
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # Поиск максимальной дисперсии и порога
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    # Применение порога к изображению
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binary_image, threshold


# Загрузка изображения в оттенках серого
image = cv2.imread('imageKitty.jpg', cv2.IMREAD_GRAYSCALE)

# Применение нашей функции бинаризации методом Оцу
binary_custom, custom_threshold = otsu_threshold(image)

# Применение встроенной функции OpenCV для бинаризации методом Оцу
_, binary_opencv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Отображение оригинального изображения, нашей бинаризации и бинаризации OpenCV
cv2.imshow('Original Image', image)
cv2.imshow(f'Custom Otsu Binarization (T={custom_threshold})', binary_custom)
cv2.imshow('OpenCV Otsu Binarization', binary_opencv)

# Ожидание нажатия любой клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()