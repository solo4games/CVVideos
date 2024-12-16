import os
import numpy as np
import cv2
import random
from scipy.ndimage import label

# Определение класса FireDetectorKMeans для детектирования пожара с использованием алгоритма k-средних
class FireDetectorKMeans:
    def __init__(self, k=3, max_iters=10, tolerance=1e-4):
        """
                Инициализация объекта FireDetectorKMeans.

                Args:
                    k (int): Количество кластеров. По умолчанию 3.
                    max_iters (int): Максимальное количество итераций. По умолчанию 10.
                    tolerance (float): Допустимое изменение центроидов для сходимости. По умолчанию 1e-4.
        """
        # Инициализация параметров алгоритма:
        # k - количество кластеров, max_iters - максимальное число итераций, tolerance - допуск для сходимости
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def initialize_centroids(self, data):
        """
                Инициализация центроидов путем случайного выбора точек из данных.

                Args:
                    data (np.ndarray): Массив данных, представляющий пиксели изображения.

                Returns:
                    np.ndarray: Массив начальных центроидов.
        """
        # Инициализация центроидов случайным выбором k пикселей из данных
        centroids = random.sample(list(data), self.k)
        return np.array(centroids, dtype=np.float32)

    def assign_clusters(self, data, centroids):
        """
                Присвоение пикселей кластерам на основе минимального евклидова расстояния до центроидов.

                Args:
                    data (np.ndarray): Массив данных, представляющий пиксели изображения.
                    centroids (np.ndarray): Массив текущих центроидов.

                Returns:
                    list: Список кластеров, где каждый элемент содержит индексы пикселей, принадлежащих кластеру.
        """
        # Присваиваем каждому пикселю ближайший кластер на основе евклидова расстояния до центроидов
        clusters = [[] for _ in range(self.k)]
        for idx, pixel in enumerate(data):
            # Рассчитываем расстояния от текущего пикселя до всех центроидов
            distances = [np.linalg.norm(pixel - centroid) for centroid in centroids]
            # Находим индекс ближайшего центроида
            cluster_idx = np.argmin(distances)
            # Добавляем индекс пикселя в соответствующий кластер
            clusters[cluster_idx].append(idx)
        return clusters

    def update_centroids(self, data, clusters):
        """
                Пересчет центроидов как среднего значения всех пикселей в каждом кластере.

                Args:
                    data (np.ndarray): Массив данных, представляющий пиксели изображения.
                    clusters (list): Список кластеров с индексами пикселей.

                Returns:
                    np.ndarray: Обновленные центроиды.
        """
        # Пересчитываем центроиды, беря среднее значение пикселей в каждом кластере
        centroids = []
        for cluster in clusters:
            # Проверка, что кластер не пустой, и вычисление нового центроида
            if cluster:
                centroids.append(np.mean(data[cluster], axis=0))
            else:
                # Если кластер пустой, выбираем случайный пиксель как новый центроид
                centroids.append(random.choice(data))
        return np.array(centroids, dtype=np.float32)

    def fit(self, image):
        """
                Применение алгоритма k-средних к изображению для сегментации.

                Args:
                    image (np.ndarray): Исходное изображение в формате RGB.

                Returns:
                    tuple: Сегментированное изображение, центроиды и кластеры.
         """
        # Применяем алгоритм k-средних к изображению
        # Преобразуем изображение в одномерный массив пикселей
        data = image.reshape(-1, 3).astype(np.float32)
        # Инициализируем центроиды
        centroids = self.initialize_centroids(data)

        # Повторяем до достижения максимального числа итераций или сходимости
        for i in range(self.max_iters):
            # Присваиваем пиксели кластерам
            clusters = self.assign_clusters(data, centroids)
            # Пересчитываем центроиды
            new_centroids = self.update_centroids(data, clusters)
            # Проверяем сходимость центроидов
            if np.all(np.abs(new_centroids - centroids) < self.tolerance):
                break
            centroids = new_centroids
            print(f"End on - {self.max_iters}, Now on - {i}")

        # Формируем сегментированное изображение, присваивая каждому пикселю цвет его центроида
        segmented_image = np.zeros_like(data)
        for cluster_idx, cluster in enumerate(clusters):
            for pixel_idx in cluster:
                # Назначаем пикселям в каждом кластере цвет соответствующего центроида
                segmented_image[pixel_idx] = centroids[cluster_idx]

        # Возвращаем изображение в исходной форме
        segmented_image = segmented_image.reshape(image.shape)
        return segmented_image, centroids, clusters

    def detect_fire_areas(self, image, centroids, clusters):
        """
                Обнаружение областей, соответствующих цветам пожара, и выделение их на изображении.

                Args:
                    image (np.ndarray): Исходное изображение в формате RGB.
                    centroids (np.ndarray): Центроиды кластеров.
                    clusters (list): Список кластеров с индексами пикселей.

                Returns:
                    np.ndarray: Изображение с выделенными областями пожара.
        """
        # Определяем кластер, наиболее похожий на цвет пожара
        fire_cluster_idx = np.argmax([np.mean(centroid[0]) for centroid in centroids])
        # Получаем индексы пикселей, принадлежащих "огненному" кластеру
        fire_pixels = clusters[fire_cluster_idx]
        fire_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Преобразуем индексы пикселей в координаты и отмечаем их на маске
        fire_coords = np.unravel_index(fire_pixels, image.shape[:2])
        fire_mask[fire_coords] = 1

        # Находим связные области на маске "пожара"
        labeled_array, num_features = label(fire_mask)
        # Копируем исходное изображение для нанесения прямоугольников
        output_image = image.copy()
        for i in range(1, num_features + 1):
            # Находим координаты текущей области, если количество пикселей больше порога
            y, x = np.where(labeled_array == i)
            if len(y) > 50:  # Порог для минимального числа пикселей в зоне пожара
                min_x, max_x = np.min(x), np.max(x)
                min_y, max_y = np.min(y), np.max(y)
                # Рисуем прямоугольник на скоплениях "огненных" пикселей
                cv2.rectangle(output_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

        return output_image

    def save_image(self, image, output_path="fire_detection_output.jpg"):
        """
                Сохранение изображения с выделенными зонами пожара.

                Args:
                    image (np.ndarray): Изображение с выделенными зонами пожара.
                    output_path (str): Путь для сохранения изображения. По умолчанию "fire_detection_output.jpg".
        """
        # Сохраняем изображение с выделенными зонами "пожара"
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Изображение сохранено по пути: {output_path}")


# Основной блок обработки изображения
image_path = 'neuro/dataset/test/2.jpg'  # Укажите путь к изображению

if not os.path.exists(image_path):
    # Проверка существования пути к изображению
    print(f"Ошибка: Изображение не найдено по пути '{image_path}'. Проверьте корректность пути.")
else:
    image = cv2.imread(image_path)
    if image is None:
        # Обработка ошибки загрузки изображения
        print(f"Ошибка при загрузке изображения. Проверьте файл '{image_path}'.")
    else:
        # Преобразование изображения в цветовую модель RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Создание экземпляра класса для детектирования пожара
        fire_detector = FireDetectorKMeans(k=3)
        # Применение алгоритма k-средних для сегментации
        segmented_image, centroids, clusters = fire_detector.fit(image_rgb)

        # Определение и выделение областей с огнем
        output_image = fire_detector.detect_fire_areas(image_rgb, centroids, clusters)
        # Сохранение изображения с выделенными зонами пожара
        fire_detector.save_image(output_image, "fire_detection_output.jpg")
