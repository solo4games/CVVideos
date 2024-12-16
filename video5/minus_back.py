import cv2
import numpy as np


def subtract_background(frame, background):
    # Преобразование кадров в градации серого
    gray_frame = np.mean(frame, axis=2)
    gray_background = np.mean(background, axis=2)

    # Вычитание фона
    diff = np.abs(gray_frame - gray_background)

    # Пороговое значение
    threshold = 50
    binary_diff = (diff > threshold).astype(np.uint8) * 255

    return binary_diff


# Захват видео
cap = cv2.VideoCapture(0)

# Получение фона (фиксированный кадр)
_, background_frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка текущего кадра
    result = subtract_background(frame, background_frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Foreground", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()