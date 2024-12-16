import cv2
import numpy as np

# Глобальные переменные
box = (0, 0, 0, 0)  # Координаты для первого изображения
box1 = (0, 0, 0, 0)  # Координаты для второго изображения
drawing_box = False
drawing_box1 = False


# Функция для рисования прямоугольника
def draw_box(img, box):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)


# Функция для первого изображения
def my_mouse_callback_img1(event, x, y, flags, param):
    global box, drawing_box
    if event == cv2.EVENT_MOUSEMOVE and drawing_box:
        box = (box[0], box[1], x - box[0], y - box[1])
    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing_box = True
        box = (x, y, 0, 0)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        if box[2] < 0:
            box = (box[0] + box[2], box[1], -box[2], box[3])
        if box[3] < 0:
            box = (box[0], box[1] + box[3], box[2], -box[3])
        draw_box(param, box)


# Функция для второго изображения
def my_mouse_callback_img2(event, x, y, flags, param):
    global box1, drawing_box1
    if event == cv2.EVENT_MOUSEMOVE and drawing_box1:
        box1 = (box1[0], box1[1], x - box1[0], y - box1[1])
    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing_box1 = True
        box1 = (x, y, 0, 0)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box1 = False
        if box1[2] < 0:
            box1 = (box1[0] + box1[2], box1[1], -box1[2], box1[3])
        if box1[3] < 0:
            box1 = (box1[0], box1[1] + box1[3], box1[2], -box1[3])
        draw_box(param, box1)


def main():
    global box, box1

    # Чтение изображений
    image1 = cv2.imread("gojo.jpg")
    image2 = cv2.imread("sukuna.jpg")

    # Проверка на наличие изображений
    if image1 is None or image2 is None:
        print("Ошибка при загрузке изображений")
        return

    temp1 = image1.copy()

    # Окно для первого изображения
    cv2.namedWindow("Task3Img1")
    cv2.setMouseCallback("Task3Img1", my_mouse_callback_img1, temp1)

    while True:
        temp1 = image1.copy()
        if drawing_box:
            draw_box(temp1, box)
        cv2.imshow("Task3Img1", temp1)

        key = cv2.waitKey(15)
        if key == 13:  # Если нажать Enter, окно закрывается и переходим к следующему изображению
            cv2.destroyWindow("Task3Img1")
            break

    temp2 = image2.copy()

    # Окно для второго изображения
    cv2.namedWindow("Task3Img2")
    cv2.setMouseCallback("Task3Img2", my_mouse_callback_img2, temp2)

    while True:
        temp2 = image2.copy()
        if drawing_box1:
            draw_box(temp2, box1)
        cv2.imshow("Task3Img2", temp2)

        key = cv2.waitKey(15)
        if key == 27:  # Если нажать ESC, программа завершит выбор
            cv2.destroyWindow("Task3Img2")
            break

    # Обрезаем выбранные области и выполняем склейку
    part1 = image1[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    part2 = image2[box1[1]:box1[1] + box1[3], box1[0]:box1[0] + box1[2]]

    # Проверяем, чтобы выбранные области были корректны
    if part1.size == 0 or part2.size == 0:
        print("Некорректные области для обрезки")
        return

    # Склейка двух изображений по ширине
    rows = max(part1.shape[0], part2.shape[0])  # Количество строк для склеенного изображения
    cols = part1.shape[1] + part2.shape[1]  # Количество столбцов для склеенного изображения
    result = np.zeros((rows, cols, 3), dtype=np.uint8)

    result[0:part1.shape[0], 0:part1.shape[1]] = part1
    result[0:part2.shape[0], part1.shape[1]:part1.shape[1] + part2.shape[1]] = part2

    # Сохранение результата
    cv2.imwrite("image_result.jpg", result)
    print("Результат сохранен как 'image_result.jpg'")


main()