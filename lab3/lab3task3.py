import cv2
import numpy as np


class ImageManipulator:
    @staticmethod
    def draw_dotted_grid(image):
        """
        Отобразить белую пунктирную сетку на изображении.

        :param image: Входное изображение
        :return: Изображение с сеткой
        """
        # Проход по всем пикселям изображения
        for i in range(image.shape[0]):  # по строкам
            for j in range(image.shape[1]):  # по столбцам
                # Проверка условий для отображения белой точки
                if (i % 20 == 10 and j % 2 == 1) or (j % 50 == 25 and i % 2 == 1):
                    # Устанавливаем цвет пикселя в белый (255, 255, 255)
                    image[i, j] = [255, 255, 255]
        return image

    @staticmethod
    def replace_green_with_red(image):
        """
        Заменить зеленые пиксели на ярко-красные.

        :param image: Входное изображение
        :return: Изображение с замененными пикселями
        """
        # Проход по всем пикселям изображения
        for i in range(image.shape[0]):  # по строкам
            for j in range(image.shape[1]):  # по столбцам
                pixel = image[i, j]
                # Проверяем условие, чтобы заменить зеленые пиксели на красные
                if pixel[1] > 64 and pixel[0] < pixel[1] - 10 and pixel[2] < pixel[1] - 10:
                    # Заменяем зеленые пиксели на красные
                    image[i, j] = [0, 0, 255]  # Устанавливаем цвет в ярко-красный
        return image

    @staticmethod
    def create_image(rows, cols, img_type):
        """
        Создать матрицу изображения.

        :param rows: Количество строк
        :param cols: Количество столбцов
        :param img_type: Тип матрицы
        :return: Созданная матрица
        """
        # Создаем матрицу указанного размера и типа
        return np.zeros((rows, cols, img_type), dtype=np.uint8)

    @staticmethod
    def adjust_brightness(image, factor):
        """
        Копирование матрицы с изменением яркости.

        :param image: Входное изображение
        :param factor: Фактор изменения яркости
        :return: Изображение с измененной яркостью
        """
        # Преобразование изображения в формат float для корректного изменения яркости
        adjusted_image = np.zeros_like(image, dtype=np.float32)
        image = image.astype(np.float32)

        # Умножаем на коэффициент яркости
        adjusted_image = cv2.multiply(image, factor)

        # Преобразование обратно в 8-битный формат (без переполнения)
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        return adjusted_image

    @staticmethod
    def get_region_of_interest(image, x, y, width, height):
        """
        Копирование региона интереса (ROI) с помощью координат и размеров.

        :param image: Входное изображение
        :param x: Координата x верхнего левого угла ROI
        :param y: Координата y верхнего левого угла ROI
        :param width: Ширина региона интереса
        :param height: Высота региона интереса
        :return: Копия региона
        """
        # Используем срезы для выделения региона интереса
        return image[y:y + height, x:x + width]

    @staticmethod
    def add_images(img1, img2):
        """
        Сложение двух изображений.

        :param img1: Первое изображение
        :param img2: Второе изображение
        :return: Результат сложения изображений
        """
        # Суммируем два изображения (размеры изображений должны совпадать)
        return cv2.add(img1, img2)


# Пример использования:

    # Загрузка изображения
image = cv2.imread('frog.jpg')

    # Добавление сетки на изображение
grid_image = ImageManipulator.draw_dotted_grid(image.copy())
cv2.imshow('Grid Image', grid_image)

    # Замена зеленого на красный
red_image = ImageManipulator.replace_green_with_red(image.copy())
cv2.imshow('Red Image', red_image)

    # Регулировка яркости
bright_image = ImageManipulator.adjust_brightness(image.copy(), 1.5)
cv2.imshow('Bright Image', bright_image)
    # Определение области интереса (ROI)
x, y, width, height = 50, 50, 500, 500  # Пример ROI (координаты и размеры)
roi_image = ImageManipulator.get_region_of_interest(image, x, y, width, height)
cv2.imshow('ROI Image', roi_image)

    # Сложение двух изображений
added_image = ImageManipulator.add_images(image, image)
cv2.imshow('Added Image', added_image)

cv2.waitKey(0)
cv2.destroyAllWindows()