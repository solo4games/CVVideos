import os
import cv2
import numpy as np


class FireDetector:
    def __init__(self, image_path):
        """
        Инициализация объекта FireDetector, загрузка изображения и его предварительная обработка.

        Параметры:
            image_path (str): Путь к изображению, на котором будет выполняться детекция пожара.
        """
        # Сохраняем путь к изображению
        self.image_path = image_path

        # Загружаем изображение в цветном формате
        self.image_color = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.image_color is None:
            # Если изображение не найдено или не удалось загрузить, выводим сообщение об ошибке и завершаем программу
            print("Ошибка: изображение не найдено или не удалось загрузить.")
            exit()

        # Преобразуем изображение в оттенки серого для дальнейшей обработки
        self.image_gray = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2GRAY)

    def detect_fire_canny(self, low_threshold=50, high_threshold=150):
        """
        Реализация алгоритма Canny для выделения контуров, связанных с очагами пожара.

        Параметры:
            low_threshold (int): Нижний порог для градиента.
            high_threshold (int): Верхний порог для градиента.

        Возвращает:
            np.ndarray: Бинарное изображение с контурами пожара.
        """
        # Преобразуем изображение в HSV для фильтрации по цвету
        hsv_image = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2HSV)

        # Задаем диапазон теплых оттенков (оранжевый/красный) для выделения пожара
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        fire_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

        # Применяем Canny для выделения контуров только в области пожара
        edges = cv2.Canny(fire_mask, low_threshold, high_threshold)

        print("Контуры Canny, связанные с огнем, обнаружены.")
        return edges

    def draw_edges(self, edges):
        """
        Отрисовывает контуры на оригинальном изображении.

        Параметры:
            edges (np.ndarray): Бинарное изображение с выделенными контурами.

        Возвращает:
            np.ndarray: Изображение с наложенными контурами.
        """
        # Копируем оригинальное изображение
        output_image = self.image_color.copy()

        # Накладываем контуры на изображение (красный цвет для выделения пожара)
        output_image[edges > 0] = [0, 0, 255]
        return output_image

    def show_images(self, *images_titles):
        """
        Отображает изображения в отдельных окнах.

        Параметры:
            images_titles (tuple): Список пар (изображение, название окна).
        """
        for img, title in images_titles:
            cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Путь к изображению
image_path = '2.jpg'  # Укажите путь к изображению

# Запуск алгоритма
detector = FireDetector(image_path)
fire_edges = detector.detect_fire_canny()  # Обнаружение контуров с помощью Canny
output_image_canny = detector.draw_edges(fire_edges)  # Наложение контуров на оригинальное изображение

# Отображение результатов
detector.show_images(
    (detector.image_color, "Original Image"),
    (fire_edges, "Canny Edges"),
    (output_image_canny, "Canny Fire Contours")
)