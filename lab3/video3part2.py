import cv2
import numpy as np

# Собственная реализация выделения границ на основе оператора Собеля
def sobel_edge_detection(image):
    # Определение операторов Собеля
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Применение фильтров Собеля
    grad_x = cv2.filter2D(image, -1, sobel_x)
    grad_y = cv2.filter2D(image, -1, sobel_y)

    # Вычисление общей амплитуды градиента
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.uint8(magnitude / magnitude.max() * 255)  # Нормализация

    return magnitude

# Загрузка изображения в оттенках серого
image = cv2.imread('imageKitty.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка успешности загрузки изображения
if image is None:
    print("Ошибка загрузки изображения.")
    exit()

# Применение алгоритма
edges = sobel_edge_detection(image)

# Отображение исходного и сегментированного изображения
cv2.imshow('Original Image', image)
cv2.imshow('Edge Detected Image', edges)

# Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()