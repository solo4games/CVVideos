import cv2
import numpy as np
import random

def run_normal_bayes_classifier():
    """
        Реализует процесс создания, обучения и тестирования NormalBayesClassifier на случайно сгенерированных данных.

        Шаги работы функции:
        1. Генерация тренировочных данных:
           - Создаются 6 образцов, каждый из которых состоит из двух признаков (float32).
           - Признаки генерируются случайным образом в диапазоне [0, 10].
           - Метки классов для каждого образца также генерируются случайным образом (значения: 1, 2 или 3).
        2. Создание и обучение классификатора:
           - Используется NormalBayesClassifier из OpenCV.
           - Классификатор обучается на сгенерированных тренировочных данных.
        3. Классификация новых данных:
           - Генерируется один новый образец для тестирования, также случайным образом.
           - Классификатор делает предсказание для тестового образца.
        4. Вывод результатов:
           - Отображаются сгенерированные тестовые данные.
           - Возвращается предсказанный класс для тестового образца.

        Возвращаемое значение:
            int: Предсказанный класс для тестового образца (1, 2 или 3).
    """

    # Создаем тренировочные данные (например, 6 образцов с 2 признаками)
    train_data = np.zeros((6, 2), dtype=np.float32)
    for i in range(6):
        features = np.array([random.random() * 10, random.random() * 10], dtype=np.float32)
        train_data[i] = features

    # Метки классов для каждого образца (рандомно выбираем от 1 до 3)
    responses = np.zeros((6, 1), dtype=np.int32)
    for i in range(6):
        label = random.randint(1, 3)  # генерируем случайные метки классов (1, 2 или 3)
        responses[i] = label

    # Создаем и обучаем NormalBayesClassifier
    bayes_classifier = cv2.ml.NormalBayesClassifier_create()
    bayes_classifier.train(train_data, cv2.ml.ROW_SAMPLE, responses)

    # Данные для классификации (также случайные)
    test_data = np.array([[random.random() * 10, random.random() * 10]], dtype=np.float32)

    # Предсказание
    retval, predicted_class = bayes_classifier.predict(test_data)

    # Выводим сгенерированные данные для тестирования
    print(f"Тестовые данные: [{test_data[0][0]}, {test_data[0][1]}]")

    # Возвращаем предсказанный класс
    return predicted_class[0, 0]

# Запуск классификатора и вывод результата
predicted_class = run_normal_bayes_classifier()
print(f"Предсказанный класс: {predicted_class}")