import cv2
import numpy as np

# Функция для вычисления градиента методом Превитта вручную
def apply_prewitt(image):
    # Операторы Превитта для горизонтального и вертикального направлений
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

    kernel_y = np.array([[ 1,  1,  1],
                         [ 0,  0,  0],
                         [-1, -1, -1]])

    # Применение свёртки для нахождения градиентов по x и y
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    # Вычисление градиента по модулю
    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))

    # Нормализация к диапазону [0, 255]
    grad = cv2.convertScaleAbs(grad)

    return grad

# Функция усредняющего сглаживания
def apply_mean_blur(image, kernel_size=(5, 5)):
    return cv2.blur(image, kernel_size)

# Функция медианной фильтрации
def apply_median_blur(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# Функция вычисления градиента методом Собеля
def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.convertScaleAbs(sobel)
    return sobel

# Функция вычисления Лапласиана
def apply_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

# Функция повышения резкости изображения на основе Лапласиана
def apply_sharpen(image):
    laplacian = apply_laplacian(image)
    sharpened_image = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)
    return sharpened_image

# Загрузка изображения в оттенках серого
image = cv2.imread('imageKitty.jpg', cv2.IMREAD_GRAYSCALE)

# Применение всех функций
prewitt_image = apply_prewitt(image)
#mean_blurred_image = apply_mean_blur(image)
#median_blurred_image = apply_median_blur(image)
#sobel_image = apply_sobel(image)
#laplacian_image = apply_laplacian(image)
#sharpened_image = apply_sharpen(image)

# Отображение результатов
cv2.imshow('Original Image', image)
cv2.imshow('Prewitt (Manual)', prewitt_image)
#cv2.imshow('Mean Blur', mean_blurred_image)
#cv2.imshow('Median Blur', median_blurred_image)
#cv2.imshow('Sobel', sobel_image)
#cv2.imshow('Laplacian', laplacian_image)
#cv2.imshow('Sharpened Image', sharpened_image)
# Применение встроенного метода Превитта в OpenCV (cv2.Sobel)
# Использование Собеля с ksize=3 эквивалентно оператору Превитта
prewitt_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
prewitt_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
prewitt_opencv = np.sqrt(prewitt_x**2 + prewitt_y**2)

# Нормализация встроенного метода к диапазону [0, 255]
prewitt_opencv = cv2.convertScaleAbs(prewitt_opencv)

cv2.imshow('Prewitt (OpenCV)', prewitt_opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()