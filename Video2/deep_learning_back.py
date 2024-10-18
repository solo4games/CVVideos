
import cv2
import numpy as np
import tensorflow.compat.v1 as tf  # Включение совместимости с TF 1.x
tf.disable_v2_behavior()

# Функция для загрузки графа модели
def load_frozen_model(pb_file):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_file, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
    return detection_graph

# Функция для сегментации изображения
def segment_foreground_background(image_path, graph):
    # Загрузка изображения
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Изменение размера изображения для модели
    img_resized = cv2.resize(img_rgb, (513, 513))  # Размер, который использует DeepLab
    img_input = np.expand_dims(img_resized, axis=0)

    # Получение необходимых операций из графа
    input_tensor = graph.get_tensor_by_name('ImageTensor:0')
    output_tensor = graph.get_tensor_by_name('SemanticPredictions:0')

    with tf.Session(graph=graph) as sess:
        # Выполнение предсказания
        seg_map = sess.run(output_tensor, feed_dict={input_tensor: img_input})

    seg_map = seg_map[0]  # Убираем пакетную размерность

    # Маска переднего плана
    foreground_mask = (seg_map > 0).astype(np.uint8) * 255

    # Визуализация переднего и заднего плана
    cv2.imshow("Original Image", img)
    cv2.imshow("Foreground Mask", foreground_mask)
    cv2.imshow("Background Mask", 255 - foreground_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Загрузка модели
frozen_graph_path = 'frozen_inference_graph.pb'
graph = load_frozen_model(frozen_graph_path)

# Запуск сегментации
segment_foreground_background('anime_background_image.jpg', graph)
