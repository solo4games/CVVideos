import numpy as np
import cv2

# =============== Алгоритм 1: Вычисление фундаментальной матрицы вручную ===============

def compute_fundamental_matrix(pts1, pts2):
    """
    Вычисляет фундаментальную матрицу с использованием метода наименьших квадратов и SVD.

    :param pts1: numpy.ndarray, массив координат точек на первом изображении (Nx2).
    :param pts2: numpy.ndarray, массив координат точек на втором изображении (Nx2).
    :return: numpy.ndarray, фундаментальная матрица (3x3).
    """
    # Построение матрицы A из точек
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    A = np.array(A)

    # Применение SVD к A для нахождения фундаментальной матрицы
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)  # Последняя строка Vt дает решение

    # Приведение F к рангу 2 (обнуление наименьшего сингулярного значения)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Обнуление последнего значения
    F = np.dot(U, np.dot(np.diag(S), Vt))

    return F

# Пример использования
pts1 = np.array([[100, 150], [200, 250], [300, 350], [400, 450]], dtype=np.float32)
pts2 = np.array([[110, 160], [210, 260], [310, 360], [410, 460]], dtype=np.float32)
F = compute_fundamental_matrix(pts1, pts2)
print("Фундаментальная матрица:\n", F)
