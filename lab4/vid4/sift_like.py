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

    def detect_fire_keypoints(self):
        """
        Выделяет ключевые точки, связанные с потенциальными очагами пожара, на основе градиентов и цветового анализа.

        Возвращает:
            list: Список ключевых точек (cv2.KeyPoint), представляющих возможные очаги пожара.
        """
        # Применяем фильтр Гаусса для сглаживания изображения и уменьшения шума
        blurred = cv2.GaussianBlur(self.image_gray, (5, 5), 1.4)

        # Рассчитываем градиенты по осям X и Y
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Рассчитываем величину градиента и углы
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_angle = np.arctan2(grad_y, grad_x)

        # Применяем порог для выделения областей с сильными градиентами
        _, thresholded = cv2.threshold(grad_magnitude, 60, 255, cv2.THRESH_BINARY)
        thresholded = np.uint8(thresholded)

        # Применяем алгоритм Харриса для выделения углов
        harris_response = cv2.cornerHarris(thresholded, blockSize=2, ksize=3, k=0.04)
        harris_response = cv2.dilate(harris_response, None)  # Увеличиваем углы для лучшей визуализации

        # Отбираем углы, превышающие заданный порог, и преобразуем их в ключевые точки
        keypoints = np.argwhere(harris_response > 0.02 * harris_response.max())
        keypoints = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in keypoints]

        # Преобразуем изображение в цветовое пространство HSV для фильтрации по цвету
        hsv_image = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2HSV)

        # Задаем диапазон теплых оттенков (оранжевый/красный) для выделения пожара
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        fire_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

        # Фильтруем ключевые точки, оставляя только те, которые соответствуют маске пожара
        fire_keypoints = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])  # Координаты ключевой точки
            if fire_mask[y, x] > 0:  # Проверяем, находится ли точка в области пожара
                fire_keypoints.append(kp)

        # Выводим количество найденных ключевых точек
        print("Найдено ключевых точек, связанных с огнем:", len(fire_keypoints))
        return fire_keypoints

    def draw_keypoints(self, keypoints):
        """
        Отрисовывает найденные ключевые точки на изображении.

        Параметры:
            keypoints (list): Список ключевых точек (cv2.KeyPoint), которые нужно отобразить.

        Возвращает:
            np.ndarray: Изображение с нанесенными ключевыми точками.
        """
        # Рисуем ключевые точки на копии оригинального изображения
        output_image = cv2.drawKeypoints(
            self.image_color,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return output_image

    def show_images(self, output_image):
        """
        Отображает оригинальное изображение и изображение с выделенными ключевыми точками.

        Параметры:
            output_image (np.ndarray): Изображение с нанесенными ключевыми точками.
        """
        # Отображение исходного изображения
        cv2.imshow("Original Image", self.image_color)
        # Отображение изображения с ключевыми точками
        cv2.imshow("Detected Fire Keypoints", output_image)
        # Ожидание нажатия клавиши для закрытия окон
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Путь к изображению
image_path = '2.jpg'  # Укажите путь к изображению

# Запуск алгоритма
detector = FireDetector(image_path)
keypoints = detector.detect_fire_keypoints()  # Обнаружение ключевых точек
output_image = detector.draw_keypoints(keypoints)  # Отрисовка ключевых точек
detector.show_images(output_image)  # Отображение результатов