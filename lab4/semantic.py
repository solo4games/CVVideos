import cv2
import numpy as np


class FireSemanticSegmentation:
    def __init__(self, threshold=0.3, kernel_size=7):
        """
        Инициализирует параметры для сегментации огня.
        :param threshold: Порог уверенности для текстурной карты.
        :param kernel_size: Размер ядра для морфологических операций.
        """
        self.threshold = threshold  # Порог уверенности для маски
        self.kernel_size = kernel_size  # Размер ядра для морфологических операций

    def preprocess_fire_color_mask(self, frame):
        """
        Создаёт цветовую маску для выделения регионов огня на основе HSV.
        :param frame: Входное изображение.
        :return: Цветовая маска.
        """
        # Конвертируем изображение в цветовое пространство HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Устанавливаем расширенный диапазон для цветов огня (более широкий оранжево-красный)
        lower_orange = np.array([0, 50, 100])
        upper_orange = np.array([40, 255, 255])

        # Создаём маску на основе расширенного диапазона
        fire_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        return fire_mask  # Возвращаем цветовую маску

    def calculate_segmentation_map(self, frame):
        """
        Создаёт текстурную карту на основе анализа градиентов.
        :param frame: Входное изображение.
        :return: Текстурная карта.
        """
        # Преобразуем изображение в оттенки серого
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Рассчитываем горизонтальный градиент (резкие изменения по x)
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=5)

        # Рассчитываем вертикальный градиент (резкие изменения по y)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=5)

        # Вычисляем величину градиента
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Нормализуем величину градиента в диапазон [0, 1]
        normalized_magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        # Создаём более чувствительную бинарную карту на основе порога
        segmentation_map = (normalized_magnitude > self.threshold).astype(np.uint8) * 255

        return segmentation_map  # Возвращаем карту сегментации

    def combine_segmentation_and_color(self, fire_color_mask, segmentation_map):
        """
        Объединяет цветовую маску и текстурную карту для улучшенной сегментации.
        :param fire_color_mask: Маска на основе цвета.
        :param segmentation_map: Карта на основе текстур.
        :return: Итоговая сегментация.
        """
        # Объединяем маску цвета и карту сегментации (логическое "И")
        combined_mask = cv2.bitwise_and(fire_color_mask, segmentation_map)

        # Создаём ядро для морфологических операций
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))

        # Выполняем морфологическое "расширение" и "закрытие" для выделения областей огня
        processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.dilate(processed_mask, kernel, iterations=2)

        return processed_mask  # Возвращаем итоговую маску

    def draw_segmentation(self, frame, mask):
        """
        Наносит сегментированные области на изображение.
        :param frame: Исходное изображение.
        :param mask: Итоговая маска сегментации.
        :return: Изображение с нанесением сегментации.
        """
        # Находим контуры на итоговой маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Проходим по каждому найденному контуру
        for contour in contours:
            # Вычисляем прямоугольник, ограничивающий контур
            x, y, w, h = cv2.boundingRect(contour)

            # Рисуем прямоугольник на исходном изображении
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return frame  # Возвращаем изображение с прямоугольниками


# Пример использования
if __name__ == "__main__":
    # Загружаем изображение с огнём
    image_path = "2.jpg"  # Укажите путь к изображению
    frame = cv2.imread(image_path)  # Читаем изображение

    # Инициализируем сегментатор
    segmentator = FireSemanticSegmentation(threshold=0.3, kernel_size=7)

    # Создаём цветовую маску
    fire_color_mask = segmentator.preprocess_fire_color_mask(frame)

    # Создаём карту сегментации на основе текстур
    segmentation_map = segmentator.calculate_segmentation_map(frame)

    # Объединяем цветовую маску и карту сегментации
    final_mask = segmentator.combine_segmentation_and_color(fire_color_mask, segmentation_map)

    # Отображаем сегментированные области на изображении
    result_frame = segmentator.draw_segmentation(frame, final_mask)

    # Отображаем результат
    cv2.imshow("Semantic Segmentation of Fire", result_frame)  # Показываем результат
    cv2.waitKey(0)  # Ждём нажатия клавиши
    cv2.destroyAllWindows()  # Закрываем все окна