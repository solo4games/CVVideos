import cv2
import numpy as np


# Функция для вычисления оптического потока Лукаса-Канаде
def lucas_kanade_optical_flow(prev_img, next_img, points, window_size=15):
    optical_flow = []
    half_window = window_size // 2

    # Градиенты изображения
    Ix = cv2.Sobel(prev_img, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(prev_img, cv2.CV_64F, 0, 1, ksize=5)
    It = next_img.astype(np.float32) - prev_img.astype(np.float32)

    for point in points:
        x, y = point.ravel()

        # Ограничение области вокруг точки
        x_start, x_end = int(x - half_window), int(x + half_window + 1)
        y_start, y_end = int(y - half_window), int(y + half_window + 1)

        # Извлечение областей для вычислений
        Ix_window = Ix[y_start:y_end, x_start:x_end].flatten()
        Iy_window = Iy[y_start:y_end, x_start:x_end].flatten()
        It_window = It[y_start:y_end, x_start:x_end].flatten()

        # Построение матриц для решения системы уравнений
        A = np.vstack((Ix_window, Iy_window)).T
        b = -It_window

        # Решение системы с методом наименьших квадратов
        nu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Добавляем результат смещения для точки
        optical_flow.append(nu)

    return np.array(optical_flow)


# Захват видео
cap = cv2.VideoCapture('background_video.mp4')

# Используем Subtractor для выделения переднего плана
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Чтение первого кадра
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Инициализация начальных точек для отслеживания (например, точки по углам с помощью Shi-Tomasi)
p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Настройки для отображения
color = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем текущий кадр в градации серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Выделение переднего плана
    fgmask = fgbg.apply(frame)

    # Оптический поток Лукаса-Канаде (реализован вручную)
    if p0 is not None:
        flow = lucas_kanade_optical_flow(prev_gray, gray_frame, p0)

        # Обновление позиций точек
        for i, (new_point, old_point) in enumerate(zip(flow, p0)):
            a, b = old_point.ravel()
            c, d = (old_point + new_point).ravel()

            # Отображение движения с помощью линий и кругов
            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(c), int(d)), 5, color[i].tolist(), -1)

        # Обновление предыдущего изображения и точек для следующего кадра
        prev_gray = gray_frame.copy()
        p0 = p0 + flow.reshape(-1, 1, 2)

    # Отображение результатов
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()