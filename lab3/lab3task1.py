import cv2
import numpy as np


# Функция для выделения границ методом Канни вручную
def custom_canny(image, low_threshold, high_threshold):
    # 1. Размытие изображения методом Гаусса для подавления шума
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # 2. Вычисление градиента методом Собеля
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Вычисление модуля градиента и угла направления
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Перевод угла из радиан в градусы
    angle = np.abs(angle)

    # 4. Не максимальное подавление (Non-Maximum Suppression)
    nms = np.zeros_like(magnitude, dtype=np.uint8)
    angle = angle % 180  # Ограничиваем углы в диапазоне [0, 180]

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            try:
                q, r = 255, 255  # Пиксели для сравнения
                # Определение направления градиента и сравнение соседних пикселей
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    nms[i, j] = magnitude[i, j]
                else:
                    nms[i, j] = 0

            except IndexError as e:
                pass

    # 5. Двойная пороговая обработка
    strong = 255
    weak = 50
    result = np.zeros_like(nms)
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    # 6. Отслеживание по гистерезису
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if result[i, j] == weak:
                if ((result[i + 1, j - 1] == strong) or (result[i + 1, j] == strong) or (result[i + 1, j + 1] == strong)
                        or (result[i, j - 1] == strong) or (result[i, j + 1] == strong)
                        or (result[i - 1, j - 1] == strong) or (result[i - 1, j] == strong) or (
                                result[i - 1, j + 1] == strong)):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result


# Загрузка изображения в оттенках серого
image = cv2.imread('imageKitty.jpg', cv2.IMREAD_GRAYSCALE)

# Применение собственной функции Канни
custom_canny_edges = custom_canny(image, 50, 150)

# Применение встроенной функции Канни OpenCV
opencv_canny_edges = cv2.Canny(image, 50, 150)

# Отображение результатов
cv2.imshow('Original Image', image)
cv2.imshow('Custom Canny Edges', custom_canny_edges)
cv2.imshow('OpenCV Canny Edges', opencv_canny_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()