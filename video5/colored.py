import cv2
import numpy as np


def enhanced_color_filter(frame, target_color, tolerance):
    # Преобразование изображения в HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Нижняя и верхняя границы цвета
    lower_bound = np.maximum(0, target_color - tolerance)
    upper_bound = np.minimum(255, target_color + tolerance)

    # Создание маски
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # Устранение шумов с помощью размытия
    mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0)

    # Применение морфологических операций
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_blurred, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    return mask_cleaned


# Захват видео
cap = cv2.VideoCapture(0)

# Целевой цвет в HSV (например, красный)
target_color = np.array([0, 120, 120])  # Пример HSV для красного
tolerance = np.array([10, 50, 50])  # Допустимое отклонение по HSV

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Улучшенный цветовой фильтр
    mask = enhanced_color_filter(frame, target_color, tolerance)

    # Выделение объекта на исходном изображении
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Визуализация
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered Object", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()