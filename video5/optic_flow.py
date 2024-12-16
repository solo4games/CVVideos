import cv2
import numpy as np


def lucas_kanade_optical_flow_optimized(prev_frame, curr_frame, window_size=5):
    """
    Реализация метода Лукаса-Канаде для вычисления оптического потока.

    :param prev_frame: Предыдущий кадр (RGB или BGR).
    :param curr_frame: Текущий кадр (RGB или BGR).
    :param window_size: Размер окна для локальных расчетов градиентов.
    :return: Поле оптического потока (двумерный массив векторов).
    """
    # Преобразование кадров в оттенки серого, чтобы снизить размерность данных.
    prev_gray = np.mean(prev_frame, axis=2).astype(np.float32)
    curr_gray = np.mean(curr_frame, axis=2).astype(np.float32)

    # Вычисление градиентов изображения по x, y и по времени (It).
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)  # Градиент по x.
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)  # Градиент по y.
    It = curr_gray - prev_gray  # Градиент по времени.

    # Половина размера окна — для определения границ выборки пикселей.
    half_window = window_size // 2

    # Определяем размер изображения и создаем пустое поле для хранения векторов движения.
    h, w = prev_gray.shape
    flow = np.zeros((h, w, 2))

    # Цикл по каждому пикселю в пределах допустимого окна
    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            # Извлечение локальных окон для градиентов.
            Ix_window = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
            Iy_window = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
            It_window = It[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()

            # Построение матриц A и b для системы уравнений.
            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            # Проверка, чтобы матрица A^T A была невырожденной, и решение системы уравнений.
            if np.linalg.det(A.T @ A) > 1e-2:
                flow_vector = np.linalg.inv(A.T @ A) @ (A.T @ b)
                flow[y, x] = flow_vector

    return flow


def visualize_optical_flow_bw(frame, flow, step=10):
    """
    Визуализация оптического потока в черно-белых тонах.

    :param frame: Текущий кадр (в оттенках серого).
    :param flow: Поле оптического потока (двумерный массив векторов).
    :param step: Шаг для прореживания стрелок (уменьшение их количества на изображении).
    :return: Чёрно-белое изображение с отображением движения в виде стрелок.
    """
    # Получение размеров изображения.
    h, w = frame.shape
    # Создание пустого черно-белого изображения.
    flow_vis = np.zeros_like(frame, dtype=np.uint8)

    # Цикл через каждые `step` пикселей для прореживания.
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]  # Вектор движения в данной точке.
            magnitude = np.sqrt(dx ** 2 + dy ** 2)  # Модуль вектора движения.
            if magnitude > 1:  # Отображать только значительные движения.
                end_x, end_y = int(x + dx), int(y + dy)  # Конечная точка стрелки.
                cv2.line(flow_vis, (x, y), (end_x, end_y), 255, 1)  # Рисование линии.
                cv2.circle(flow_vis, (end_x, end_y), 1, 255, -1)  # Рисование точки.

    return flow_vis


# Захват видео с камеры.
cap = cv2.VideoCapture(0)

# Уменьшение разрешения кадров для повышения скорости вычислений.
scale = 0.5

# Считывание первого кадра.
ret, prev_frame = cap.read()
# Уменьшение разрешения кадра (масштабирование).
prev_frame = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)

while True:
    # Считывание текущего кадра.
    ret, curr_frame = cap.read()
    if not ret:  # Если не удалось считать кадр, прерываем цикл.
        break

    # Уменьшение разрешения текущего кадра.
    curr_frame = cv2.resize(curr_frame, (0, 0), fx=scale, fy=scale)

    # Вычисление оптического потока методом Лукаса-Канаде.
    flow = lucas_kanade_optical_flow_optimized(prev_frame, curr_frame, window_size=5)

    # Визуализация оптического потока в черно-белых тонах.
    vis_frame = visualize_optical_flow_bw(curr_frame[:, :, 0], flow, step=15)

    # Отображение результата.
    cv2.imshow("Optical Flow (Black & White)", vis_frame)

    # Переход к следующему кадру.
    prev_frame = curr_frame.copy()

    # Прерывание работы по нажатию клавиши 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов.
cap.release()
cv2.destroyAllWindows()
