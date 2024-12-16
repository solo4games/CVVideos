import numpy as np
import cv2

# =============== Алгоритм 3: Реконструкция 3D-точек вручную ===============

def triangulate_points(pts1, pts2, P1, P2):
    """
    Выполняет триангуляцию для восстановления 3D-точек из двух видов сцены.

    :param pts1: numpy.ndarray, координаты точек на первом изображении (Nx2).
    :param pts2: numpy.ndarray, координаты точек на втором изображении (Nx2).
    :param P1: numpy.ndarray, матрица камеры для первого вида (3x4).
    :param P2: numpy.ndarray, матрица камеры для второго вида (3x4).
    :return: numpy.ndarray, массив реконструированных 3D-точек (Nx3).
    """
    points_3D = []
    for pt1, pt2 in zip(pts1, pts2):
        # Формируем матрицу A для каждого набора точек
        A = [
            pt1[0] * P1[2, :] - P1[0, :],
            pt1[1] * P1[2, :] - P1[1, :],
            pt2[0] * P2[2, :] - P2[0, :],
            pt2[1] * P2[2, :] - P2[1, :]
        ]
        A = np.array(A)

        # Применяем SVD для нахождения решения
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # Последняя строка Vt
        points_3D.append(X[:3] / X[3])  # Преобразование в декартовы координаты

    return np.array(points_3D)

# Пример использования
P1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)  # Матрица камеры 1
P2 = np.array([[1, 0, 0, -10],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)  # Матрица камеры 2
pts1 = np.array([[100, 200], [150, 250], [300, 400]], dtype=np.float32)
pts2 = np.array([[105, 205], [155, 255], [305, 405]], dtype=np.float32)
points_3D = triangulate_points(pts1, pts2, P1, P2)
print("Реконструированные 3D-точки:\n", points_3D)
