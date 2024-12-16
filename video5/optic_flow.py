import cv2
import numpy as np


def lucas_kanade_optical_flow_optimized(prev_frame, curr_frame, window_size=5):
    # Преобразование кадров в оттенки серого
    prev_gray = np.mean(prev_frame, axis=2).astype(np.float32)
    curr_gray = np.mean(curr_frame, axis=2).astype(np.float32)

    # Градиенты по осям X и Y
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = curr_gray - prev_gray

    # Половина размера окна
    half_window = window_size // 2

    # Создание сетки координат
    h, w = prev_gray.shape
    flow = np.zeros((h, w, 2))

    # Векторизованный расчет
    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            # Извлечение окон градиентов
            Ix_window = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
            Iy_window = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
            It_window = It[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()

            # Матрицы A и b для системы уравнений
            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            # Решение методом наименьших квадратов
            if np.linalg.det(A.T @ A) > 1e-2:  # Проверка на невыраженность
                flow_vector = np.linalg.inv(A.T @ A) @ (A.T @ b)
                flow[y, x] = flow_vector

    return flow


def visualize_optical_flow_bw(frame, flow, step=10):
    h, w = frame.shape[:2]
    flow_vis = np.zeros_like(frame, dtype=np.uint8)

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            magnitude = np.sqrt(dx ** 2 + dy ** 2)
            if magnitude > 1:  # Порог движения
                end_x, end_y = int(x + dx), int(y + dy)
                cv2.line(flow_vis, (x, y), (end_x, end_y), 255, 1)
                cv2.circle(flow_vis, (end_x, end_y), 1, 255, -1)

    return flow_vis


# Захват видео
cap = cv2.VideoCapture(0)

# Уменьшение разрешения для ускорения
scale = 0.5

# Считывание первого кадра
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Уменьшение разрешения
    curr_frame = cv2.resize(curr_frame, (0, 0), fx=scale, fy=scale)

    # Вычисление оптического потока Лукаса-Канаде
    flow = lucas_kanade_optical_flow_optimized(prev_frame, curr_frame, window_size=5)

    # Визуализация в черно-белых тонах
    vis_frame = visualize_optical_flow_bw(curr_frame[:, :, 0], flow, step=15)

    cv2.imshow("Optical Flow (Black & White)", vis_frame)

    # Переход к следующему кадру
    prev_frame = curr_frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()