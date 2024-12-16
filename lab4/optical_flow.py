import cv2
import numpy as np

class FireOpticalFlowDetector:
    def __init__(self, initial_frame, threshold=10):
        """
        Инициализация детектора с начальным кадром и порогом чувствительности.

        Параметры:
            initial_frame (np.ndarray): Первый кадр видео для начальной настройки.
            threshold (int): Порог чувствительности для обнаружения движения (по умолчанию 10).
        """
        # Конвертируем первый кадр в оттенки серого для анализа изменений
        self.prev_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        # Устанавливаем порог для чувствительности к движению
        self.threshold = threshold

    def preprocess_fire_color_mask(self, frame):
        """
        Создает цветовую маску для выделения областей с оттенками огня (оранжевый/красный).

        Параметры:
            frame (np.ndarray): Кадр видео для обработки.

        Возвращает:
            np.ndarray: Цветовая маска огня.
        """
        # Преобразуем кадр в цветовое пространство HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Устанавливаем диапазон оранжево-красных оттенков для выделения огня
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([35, 255, 255])
        fire_color_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Применяем морфологическое закрытие для улучшения маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_color_mask = cv2.morphologyEx(fire_color_mask, cv2.MORPH_CLOSE, kernel)

        return fire_color_mask

    def calculate_optical_flow(self, next_frame):
        """
        Рассчитывает оптический поток между текущим и предыдущим кадрами методом Лукаса-Канаде.

        Параметры:
            next_frame (np.ndarray): Текущий кадр видео.

        Возвращает:
            tuple: Горизонтальная (u) и вертикальная (v) компоненты оптического потока.
        """
        # Конвертируем текущий кадр в оттенки серого
        curr_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Вычисляем градиенты по x, y и временной градиент
        Ix = cv2.Sobel(self.prev_frame, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(self.prev_frame, cv2.CV_64F, 0, 1, ksize=3)
        It = curr_frame - self.prev_frame

        # Инициализируем массивы для хранения компонентов оптического потока
        u = np.zeros(curr_frame.shape)
        v = np.zeros(curr_frame.shape)

        # Устанавливаем размер окна для локального анализа
        window_size = 5
        half_win = window_size // 2

        # Вычисляем оптический поток для каждого пикселя
        for y in range(half_win, curr_frame.shape[0] - half_win):
            for x in range(half_win, curr_frame.shape[1] - half_win):
                # Извлекаем локальные окна
                Ix_window = Ix[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1].flatten()
                Iy_window = Iy[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1].flatten()
                It_window = It[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1].flatten()

                # Формируем матрицу A и вектор B
                A = np.vstack((Ix_window, Iy_window)).T
                B = -It_window

                # Решаем систему уравнений методом наименьших квадратов
                nu = np.linalg.pinv(A.T @ A) @ A.T @ B
                u[y, x] = nu[0]  # Горизонтальный компонент
                v[y, x] = nu[1]  # Вертикальный компонент

        # Обновляем предыдущий кадр для следующего шага
        self.prev_frame = curr_frame

        return u, v

    def detect_fire_regions(self, flow_u, flow_v, fire_color_mask):
        """
        Определяет регионы огня, объединяя данные о движении и цветовую маску.

        Параметры:
            flow_u (np.ndarray): Горизонтальная компонента оптического потока.
            flow_v (np.ndarray): Вертикальная компонента оптического потока.
            fire_color_mask (np.ndarray): Цветовая маска огня.

        Возвращает:
            np.ndarray: Маска регионов огня.
        """
        # Вычисляем величину оптического потока
        flow_magnitude = np.sqrt(flow_u**2 + flow_v**2)

        # Объединяем данные оптического потока и цветовой маски
        fire_mask = ((flow_magnitude > self.threshold) & (fire_color_mask > 0)).astype(np.uint8) * 255

        # Применяем морфологическую дилатацию для улучшения маски
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)

        return fire_mask

    def draw_fire_regions(self, frame, fire_mask):
        """
        Рисует прямоугольники вокруг областей, соответствующих маске огня.

        Параметры:
            frame (np.ndarray): Исходный кадр видео.
            fire_mask (np.ndarray): Маска регионов огня.

        Возвращает:
            np.ndarray: Кадр с нарисованными прямоугольниками.
        """
        # Находим контуры в маске
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Рисуем прямоугольники вокруг найденных контуров
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)  # Вычисляем координаты прямоугольника
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Рисуем красный прямоугольник

        return frame

# Пример использования
cap = cv2.VideoCapture('fire.mp4')  # Замените на путь к вашему видео

# Считываем первый кадр для инициализации детектора
ret, first_frame = cap.read()
detector = FireOpticalFlowDetector(first_frame)

while cap.isOpened():
    ret, frame = cap.read()  # Считываем новый кадр
    if not ret:
        break  # Прерываем, если кадры закончились

    # Создаем маску цветовых областей огня
    fire_color_mask = detector.preprocess_fire_color_mask(frame)

    # Рассчитываем оптический поток
    flow_u, flow_v = detector.calculate_optical_flow(frame)

    # Определяем области огня
    fire_mask = detector.detect_fire_regions(flow_u, flow_v, fire_color_mask)

    # Рисуем области огня на кадре
    output_frame = detector.draw_fire_regions(frame, fire_mask)

    # Отображаем результат
    cv2.imshow("Fire Detection via Optical Flow", output_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break  # Прерывание при нажатии 'q'

# Завершаем захват и закрываем окна
cap.release()
cv2.destroyAllWindows()