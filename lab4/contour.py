import cv2
import numpy as np


class FireContourDetectorCustom:
    def __init__(self, min_contour_area=500, kernel_size=5):
        """
        Инициализирует параметры для обнаружения контуров огня.
        :param min_contour_area: Минимальная площадь контура для учета.
        :param kernel_size: Размер ядра для морфологических операций.
        """
        self.min_contour_area = min_contour_area  # Устанавливаем минимальную площадь контура для фильтрации
        self.kernel_size = kernel_size  # Задаём размер ядра для морфологических операций

    def preprocess_fire_color_mask(self, frame):
        """
        Создаёт цветовую маску для выделения регионов огня на основе HSV.
        :param frame: Входное изображение.
        :return: Цветовая маска.
        """
        # Переводим изображение из цветовой модели BGR в HSV для работы с цветами
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Задаём диапазон цветов, соответствующих огню, в HSV
        lower_orange = np.array([0, 50, 100])  # Нижняя граница цвета (оттенок, насыщенность, яркость)
        upper_orange = np.array([40, 255, 255])  # Верхняя граница цвета

        # Создаём бинарную маску для выделения областей, попадающих в заданный диапазон
        fire_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Создаём ядро для морфологических операций
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))

        # Применяем морфологическую операцию замыкания для удаления мелких отверстий в маске
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        return fire_mask

    def find_fire_contours(self, mask):
        """
        Реализует метод нахождения контуров без использования встроенных функций OpenCV.
        :param mask: Бинарная маска для нахождения контуров.
        :return: Список контуров, где каждый контур представлен списком координат.
        """
        contours = []  # Список для хранения контуров
        visited = np.zeros_like(mask, dtype=bool)  # Маска для отслеживания посещённых пикселей

        # Основной алгоритм обхода
        def trace_contour(start_point):
            """
            Обходит границу объекта, начиная с заданной точки.
            :param start_point: Начальная точка объекта.
            :return: Контур в виде списка координат.
            """
            contour = []  # Список для хранения точек текущего контура
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Направления движения: вверх, вправо, вниз, влево
            current_point = start_point  # Устанавливаем начальную точку
            contour.append(current_point)  # Добавляем начальную точку в контур
            visited[current_point] = True  # Помечаем начальную точку как посещённую

            while True:
                for d in directions:
                    # Рассчитываем координаты следующей точки
                    next_point = (current_point[0] + d[0], current_point[1] + d[1])

                    # Проверяем границы изображения и статус пикселя
                    if (0 <= next_point[0] < mask.shape[0] and
                            0 <= next_point[1] < mask.shape[1] and
                            mask[next_point] > 0 and not visited[next_point]):
                        contour.append(next_point)  # Добавляем точку в контур
                        visited[next_point] = True  # Помечаем точку как посещённую
                        current_point = next_point  # Обновляем текущую точку
                        break
                else:
                    break  # Если не удалось найти соседнюю точку, завершаем обход
            return contour

        # Перебираем все пиксели изображения
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] > 0 and not visited[y, x]:  # Если найден непосещённый пиксель объекта
                    contour = trace_contour((y, x))  # Обход границы объекта
                    if len(contour) >= self.min_contour_area:  # Фильтруем по минимальной площади
                        contours.append(contour)

        return contours

    def draw_fire_contours(self, frame, contours):
        """
        Рисует контуры на изображении.
        :param frame: Исходное изображение.
        :param contours: Контуры для отображения.
        :return: Изображение с нанесением контуров.
        """
        for contour in contours:
            for pt in contour:  # Проходимся по каждой точке контура
                frame[pt[0], pt[1]] = (0, 0, 255)  # Закрашиваем точку красным цветом
        return frame

# Пример использования
if __name__ == "__main__":
    image_path = "4.jpg"  # Укажите путь к изображению
    frame = cv2.imread(image_path)  # Считываем изображение

    # Создаём объект класса FireContourDetectorCustom
    contour_detector = FireContourDetectorCustom(min_contour_area=100, kernel_size=5)

    fire_mask = contour_detector.preprocess_fire_color_mask(frame)  # Генерируем маску огня
    fire_contours = contour_detector.find_fire_contours(fire_mask)  # Находим контуры огня
    result_frame = contour_detector.draw_fire_contours(frame, fire_contours)  # Рисуем контуры на изображении

    # Отображаем результат
    cv2.imshow("Custom Fire Contour Detection", result_frame)
    cv2.waitKey(0)  # Ожидаем нажатия клавиши для закрытия окна
    cv2.destroyAllWindows()  # Закрываем все окна