import cv2
import numpy as np
import math

# Инициализация переменных для овала
center = (0, 0)
axes = (0, 0)
drawing_ellipse = False
start_point = (0, 0)

# Функция для рисования овала
def draw_ellipse(img, center, axes):
    # Разворачиваем углы от 0 до 360 градусов
    for angle in range(0, 360, 1):
        theta = np.radians(angle)
        x = int(center[0] + axes[0] * math.cos(theta))
        y = int(center[1] + axes[1] * math.sin(theta))
        
        # Проверяем, что точка лежит внутри изображения
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = (0, 0, 0)  # Черный цвет для рисования пикселя

# Функция обратного вызова для обработки событий мыши
def my_mouse_callback(event, x, y, flags, param):
    global center, axes, drawing_ellipse, start_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # Начало рисования эллипса
        drawing_ellipse = True
        start_point = (x, y)
        center = (x, y)
        axes = (0, 0)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_ellipse:
            # Обновляем радиусы эллипса в зависимости от текущей позиции мыши
            axes = (abs(x - start_point[0]), abs(y - start_point[1]))

    elif event == cv2.EVENT_LBUTTONUP:
        # Завершаем рисование эллипса
        drawing_ellipse = False
        axes = (abs(x - start_point[0]), abs(y - start_point[1]))
        draw_ellipse(param, center, axes)

# Главная функция
def main():
    global center, axes, drawing_ellipse
    
    # Создание изображения
    image = 255 * np.ones((400, 400, 3), np.uint8)  # Белый фон
    temp = image.copy()

    cv2.namedWindow('Ellipse Example')
    cv2.setMouseCallback('Ellipse Example', my_mouse_callback, image)
    
    while True:
        temp = image.copy()
        if drawing_ellipse:
            draw_ellipse(temp, center, axes)
        
        cv2.imshow('Ellipse Example', temp)
        
        if cv2.waitKey(15) == 27:  # Нажатие ESC для выхода
            break

    cv2.destroyAllWindows()

if name == "main":
    main()
