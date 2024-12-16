import cv2
import numpy as np


def initialize_centroids(image, k):
    """
    Инициализация центроидов случайным образом.

    Параметры:
    - image: входное изображение в виде массива (каждый пиксель - это вектор RGB)
    - k: количество кластеров

    Возвращает:
    - массив начальных центроидов (случайно выбранных пикселей изображения)
    """
    pixels = image.reshape(-1, 3)  # Преобразуем изображение в плоский массив пикселей (RGB)
    centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]  # Случайные пиксели как начальные центроиды
    return centroids


def assign_clusters(image, centroids):
    """
    Присваивание каждого пикселя к ближайшему кластеру на основе евклидова расстояния.

    Параметры:
    - image: входное изображение в виде массива
    - centroids: массив текущих центроидов

    Возвращает:
    - массив индексов кластеров для каждого пикселя
    """
    pixels = image.reshape(-1, 3)  # Преобразуем изображение в плоский массив пикселей (RGB)

    # Вычисляем евклидово расстояние до каждого центроида
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)

    # Присваиваем каждый пиксель ближайшему кластеру (индекс минимального расстояния)
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments


def update_centroids(image, cluster_assignments, k):
    """
    Пересчёт центроидов как средних значений всех пикселей в каждом кластере.

    Параметры:
    - image: входное изображение в виде массива
    - cluster_assignments: массив индексов кластеров для каждого пикселя
    - k: количество кластеров

    Возвращает:
    - обновленные центроиды
    """
    pixels = image.reshape(-1, 3)  # Преобразуем изображение в плоский массив пикселей (RGB)
    new_centroids = np.zeros((k, 3), dtype=np.float32)

    for i in range(k):
        # Вычисляем среднее значение для всех пикселей, отнесённых к кластеру i
        cluster_pixels = pixels[cluster_assignments == i]
        if len(cluster_pixels) > 0:
            new_centroids[i] = np.mean(cluster_pixels, axis=0)

    return new_centroids


def k_means_segmentation(image, k, max_iters=10):
    """
    Метод k-средних для сегментации изображения.

    Параметры:
    - image: входное изображение в виде массива
    - k: количество кластеров
    - max_iters: максимальное количество итераций

    Возвращает:
    - сегментированное изображение
    """
    # Инициализация центроидов
    centroids = initialize_centroids(image, k)

    for _ in range(max_iters):
        # Шаг 1: Присваиваем каждый пиксель ближайшему кластеру
        cluster_assignments = assign_clusters(image, centroids)

        # Шаг 2: Обновляем центроиды
        new_centroids = update_centroids(image, cluster_assignments, k)

        # Проверяем сходимость (если центроиды не меняются, выходим)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Создаём сегментированное изображение на основе кластеров
    segmented_image = centroids[cluster_assignments].reshape(image.shape).astype(np.uint8)
    return segmented_image


def segment_image_opencv(image, k):
    # Преобразование изображения в 2D массив пикселей
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Критерий завершения (точность, количество итераций)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Применение K-средних из OpenCV
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Замена каждого пикселя на цвет его центроида
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


# Загрузка изображения
image = cv2.imread('imageKitty.jpg')

# Проверка успешности загрузки изображения
if image is None:
    print("Ошибка загрузки изображения.")
    exit()

# Количество кластеров
k = 4  # Например, сегментация на 4 области

# Применение метода k-средних для сегментации изображения
segmented_image = k_means_segmentation(image, k)

# Применение встроенной функции OpenCV для сегментации
segmented_image_opencv = segment_image_opencv(image, k)

# Визуализация результата
cv2.imshow("Segmented Image (OpenCV K-Means)", segmented_image_opencv)

# Отображение исходного и сегментированного изображений
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)

# Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()