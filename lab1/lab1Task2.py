import cv2
import numpy as np

# Инициализация переменных для прямоугольника
box = (0, 0, 0, 0)
drawing_box = False


# Функция для рисования прямоугольника
def draw_box(img, box):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 0), 2)


# Функция обратного вызова для обработки событий мыши
def my_mouse_callback(event, x, y, flags, param):
    global box, drawing_box

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing_box:
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


# Главная функция
def main():
    global box

    # Загрузка изображения
    image = cv2.imread("gojo.jpg", 1)

    if image is None:
        print("Ошибка: не удалось загрузить изображение.")
        return

    box = (-1, -1, 0, 0)
    img2 = image.copy()

    cv2.namedWindow("Box Example")
    cv2.setMouseCallback("Box Example", my_mouse_callback, image)

    while True:
        img2 = image.copy()
        if drawing_box:
            draw_box(img2, box)

        cv2.imshow("Box Example", img2)

        if cv2.waitKey(15) == 27:  # Выход по нажатию ESC
            break

    # Выделение части изображения, ограниченной прямоугольником
    if box[2] > 0 and box[3] > 0:
        part = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        cv2.imwrite("image2.jpg", part)

    cv2.destroyWindow("Box Example")


main()