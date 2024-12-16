import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121  # Предобученная модель DenseNet121
from tensorflow.keras.layers import Dense, Flatten  # Полносвязные слои и слой выравнивания
from tensorflow.keras.models import Model, load_model  # Для создания и загрузки моделей
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # Для обработки изображений

class FireClassifierDenseNet:
    """
    Класс FireClassifierDenseNet для классификации изображений на наличие или отсутствие пожара.
    Использует предобученную модель DenseNet121 для извлечения признаков.

    Методы:
    - init: Инициализация модели или загрузка сохранённой модели.
    - load_images: Загрузка и предобработка изображений из указанной директории.
    - train: Обучение модели на тренировочных данных.
    - predict_fire: Предсказание наличия пожара на изображении.
    """

    def __init__(self, model_path=None, input_shape=(128, 128, 3), num_classes=1):
        """
        Инициализация классификатора. Если указан путь к сохранённой модели, загружает её,
        иначе создаёт новую модель на основе предобученной DenseNet121.

        Аргументы:
        - model_path: str или None. Путь к сохранённой модели.
        - input_shape: tuple. Размер входных изображений.
        - num_classes: int. Количество классов (по умолчанию 1 - бинарная классификация).
        """
        # Если путь к модели указан и файл существует, загружаем модель
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)  # Загружаем модель из файла
            print("Модель загружена из", model_path)
        else:
            # Загружаем предобученную DenseNet121 без верхних слоёв
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
            # Замораживаем слои базовой модели, чтобы они не обновлялись при обучении
            for layer in base_model.layers:
                layer.trainable = False

            # Добавляем дополнительные слои поверх DenseNet
            x = base_model.output  # Выход базовой модели
            x = Flatten()(x)  # Преобразуем данные в одномерный массив
            x = Dense(128, activation='relu')(x)  # Полносвязный слой с 128 нейронами
            predictions = Dense(num_classes, activation='sigmoid')(x)  # Выходной слой для бинарной классификации

            # Создаём модель, объединяя базовую модель и добавленные слои
            self.model = Model(inputs=base_model.input, outputs=predictions)
            # Компилируем модель с функцией потерь и оптимизатором
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def load_images(self, directory, target_size=(128, 128)):
        """
        Загрузка и предобработка изображений из указанной директории.

        Аргументы:
        - directory: str. Путь к директории с изображениями.
        - target_size: tuple. Размер изображений для изменения масштаба.

        Возвращает:
        - images: np.array. Массив обработанных изображений.
        - labels: np.array. Массив меток классов (1 - пожар, 0 - отсутствие пожара).
        """
        images, labels = [], []  # Списки для изображений и их меток

        # Проходим по всем файлам в директории
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)  # Полный путь к файлу
                try:
                    # Загружаем изображение и изменяем его размер
                    img = load_img(file_path, target_size=target_size)
                    img_array = img_to_array(img) / 255.0  # Преобразуем изображение в массив и нормализуем
                    images.append(img_array)  # Добавляем изображение в список

                    # Определяем метку на основе структуры папок
                    label = 0 if 'no_fire' in root.lower() else 1
                    labels.append(label)
                except Exception:
                    pass  # Игнорируем файлы, которые не удалось загрузить

        return np.array(images), np.array(labels)  # Возвращаем массивы изображений и меток

    def train(self, train_path, val_path, save_path='fire_classifier_densenet_model.h5'):
        """
        Обучение модели на тренировочных данных.

        Аргументы:
        - train_path: str. Путь к тренировочным данным.
        - val_path: str. Путь к валидационным данным.
        - save_path: str. Путь для сохранения обученной модели.
        """
        # Загружаем тренировочные данные
        train_images, train_labels = self.load_images(train_path)
        # Загружаем валидационные данные
        val_images, val_labels = self.load_images(val_path)

        # Обучаем модель с указанием тренировочных и валидационных данных
        self.model.fit(
            train_images, train_labels,  # Данные для обучения
            epochs=2,  # Количество эпох
            batch_size=32,  # Размер батча
            validation_data=(val_images, val_labels)  # Данные для валидации
        )

        # Сохраняем обученную модель в файл
        self.model.save(save_path)
        print(f"Модель сохранена по пути: {save_path}")

    def predict_fire(self, image_path):
        """
        Предсказание наличия пожара на изображении.

        Аргументы:
        - image_path: str. Путь к изображению для предсказания.

        Возвращает:
        - str: Результат предсказания ("Пожар обнаружен!" или "Пожар не обнаружен.").
        """
        # Загружаем изображение и изменяем его размер
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0  # Преобразуем изображение в массив и нормализуем
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем ось для батча

        # Получаем предсказание модели
        prediction = self.model.predict(img_array)
        # Возвращаем результат на основе порога 0.5
        return "Пожар обнаружен!" if prediction[0][0] > 0.5 else "Пожар не обнаружен."


# Создание и обучение классификатора
classifier = FireClassifierDenseNet()  # Создаём экземпляр классификатора
train_path = 'dataset/train'  # Путь к тренировочным данным
val_path = 'dataset/validation'  # Путь к валидационным данным

# Обучаем модель и сохраняем её
# classifier.train(train_path, val_path, save_path='fire_classifier_densenet_model.h5')

# Пример использования сохранённой модели для предсказания
loaded_classifier = FireClassifierDenseNet(model_path='fire_classifier_densenet_model.h5')  # Загружаем сохранённую модель
result = loaded_classifier.predict_fire('dataset/test/2.jpg')  # Предсказываем по тестовому изображению
print(result)  # Выводим результат предсказания