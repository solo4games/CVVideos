import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # Для работы с моделями нейронных сетей
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Слои нейросети
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # Для обработки изображений
from sklearn.utils import class_weight  # Для вычисления весов классов
import matplotlib.pyplot as plt  # Для построения графиков обучения

class FireClassifier:
    """
    Класс FireClassifier для классификации изображений на наличие или отсутствие пожара.
    Использует сверточную нейронную сеть (Convolutional Neural Network) для бинарной классификации.

    Методы:
    - init: Инициализация модели, либо загрузка сохраненной, либо создание новой.
    - load_images: Загрузка и предобработка изображений из указанной директории.
    - train: Обучение модели на тренировочных данных с использованием градиентного спуска.
    - predict_fire: Предсказание наличия пожара на изображении.

    Алгоритм обучения:
    Используется метод градиентного спуска с функцией потерь binary_crossentropy и оптимизатором adam.
    """

    def __init__(self, model_path=None):
        """
        Инициализация классификатора. Если указан путь к сохраненной модели, загружает её,
        иначе создаёт новую модель сверточной нейронной сети.

        Аргументы:
        - model_path: str или None. Путь к сохраненной модели.
        """
        # Если указан путь и модель существует, загружаем её
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print("Модель загружена из", model_path)
        else:
            # Создаём архитектуру сверточной нейронной сети
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Первый сверточный слой
                MaxPooling2D(pool_size=(2, 2)),  # Первый слой подвыборки (пулинг)
                Conv2D(64, (3, 3), activation='relu'),  # Второй сверточный слой
                MaxPooling2D(pool_size=(2, 2)),  # Второй слой подвыборки
                Conv2D(128, (3, 3), activation='relu'),  # Третий сверточный слой
                MaxPooling2D(pool_size=(2, 2)),  # Третий слой подвыборки
                Flatten(),  # Преобразование данных в одномерный массив
                Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами
                Dropout(0.5),  # Dropout для предотвращения переобучения
                Dense(1, activation='sigmoid')  # Выходной слой с сигмоидальной активацией (бинарная классификация)
            ])
            # Компиляция модели с функцией потерь и оптимизатором
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def load_images(self, directory, target_size=(128, 128)):
        """
        Загрузка изображений из указанной директории и их предобработка.

        Аргументы:
        - directory: str. Путь к директории с изображениями.
        - target_size: tuple. Размер изображений для изменения масштаба.

        Возвращает:
        - images: np.array. Массив обработанных изображений.
        - labels: np.array. Массив меток классов (1 - пожар, 0 - отсутствие пожара).
        """
        images, labels = [], []  # Списки для изображений и меток
        count_fire, count_no_fire = 0, 0  # Счётчики для классов

        # Проходим по всем файлам в указанной директории
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)  # Полный путь к файлу
                try:
                    # Загружаем изображение и изменяем его размер
                    img = load_img(file_path, target_size=target_size)
                    img_array = img_to_array(img) / 255.0  # Нормализация изображения
                    images.append(img_array)  # Добавляем изображение в список

                    # Определяем метку на основе имени папки
                    label = 0 if 'no_fire' in root.lower() else 1
                    labels.append(label)
                    # Увеличиваем счётчики классов
                    if label == 1:
                        count_fire += 1
                    else:
                        count_no_fire += 1
                except Exception:
                    pass  # Игнорируем ошибки при загрузке

        # Предупреждаем, если классы в датасете несбалансированы
        if abs(count_fire - count_no_fire) > 0.1 * (count_fire + count_no_fire):
            print("Внимание: Датасет несбалансирован. Классы имеют различное количество изображений.")

        return np.array(images), np.array(labels)

    def train(self, train_path, val_path, save_path='fire_classifier_model.h5', epochs=10):
        """
        Обучение модели на тренировочных данных и валидация.

        Аргументы:
        - train_path: str. Путь к тренировочным данным.
        - val_path: str. Путь к валидационным данным.
        - save_path: str. Путь для сохранения обученной модели.
        - epochs: int. Количество эпох обучения.
        """
        # Загружаем тренировочные и валидационные данные
        train_images, train_labels = self.load_images(train_path)
        val_images, val_labels = self.load_images(val_path)

        # Рассчитываем веса классов для сбалансированного обучения
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = dict(enumerate(class_weights))

        # Обучение модели с использованием градиентного спуска
        history = self.model.fit(
            train_images, train_labels,  # Тренировочные данные
            epochs=epochs,  # Количество эпох
            batch_size=32,  # Размер батча
            validation_data=(val_images, val_labels),  # Валидационные данные
            class_weight=class_weights  # Веса классов
        )

        # Сохранение обученной модели
        self.model.save(save_path)
        print(f"Модель сохранена по пути: {save_path}")

        # Построение графиков потерь и точности
        plt.figure(figsize=(12, 4))

        # График потерь
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Потери на обучении')
        plt.plot(history.history['val_loss'], label='Потери на валидации')
        plt.legend()
        plt.title("График потерь")

        # График точности
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Точность на обучении')
        plt.plot(history.history['val_accuracy'], label='Точность на валидации')
        plt.legend()
        plt.title("График точности")

        plt.show()

    def predict_fire(self, image_path, threshold=0.5):
        """
        Предсказание наличия пожара на изображении.

        Аргументы:
        - image_path: str. Путь к изображению для предсказания.
        - threshold: float. Порог для классификации (по умолчанию 0.5).

        Возвращает:
        - str: Результат классификации ("Пожар обнаружен!" или "Пожар не обнаружен.").
        """
        # Загрузка и предобработка изображения
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0  # Нормализация изображения
        img_array = np.expand_dims(img_array, axis=0)  # Добавление измерения для батча

        # Предсказание класса
        prediction = self.model.predict(img_array)
        print("Предсказание:", prediction[0][0])

        # Возвращаем результат на основе порога
        return "Пожар обнаружен!" if prediction[0][0] > threshold else "Пожар не обнаружен."

# Создание и обучение классификатора
classifier = FireClassifier()
train_path = 'dataset/train'  # Путь к тренировочным данным
val_path = 'dataset/validation'  # Путь к валидационным данным

# Обучаем модель и сохраняем её
# classifier.train(train_path, val_path, save_path='fire_classifier_model.h5', epochs=2)

# Пример использования сохранённой модели для предсказания
loaded_classifier = FireClassifier(model_path='fire_classifier_model.h5')
result = loaded_classifier.predict_fire('dataset/test/1.jpg', threshold=0.5)
print(result)