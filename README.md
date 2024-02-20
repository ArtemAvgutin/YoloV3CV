
# Project Yolo V3
<p align="center">
   <img src="https://media.giphy.com/media/VYPDYUBR9bGEIYtz5s/giphy.gif?cid=ecf05e47pdggmnkmdmfzgcp00hq8cg5nmxrnb2w5m8e8c36t&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="300">
</p>

## About (Eng)
* Target:
Create a neural network to detect objects in an image. This is a training project for working with computer vision and tensorflow libraries.
* Result:
A neural network was created that was capable of identifying 80 classes in an image with good accuracy (~85%) and working quite quickly.

## About (Rus)
* Цель:
Создать нейронную сеть для детектирования объектов на изображении. Это тренировочный проект для работы с компьютерным зрением и библиотеками tensorflow.
* Результат:
Была создана нейронная сеть, способная определять 80 классов на изображении с хорошей точностью (~ 85%) и работать достаточно быстро.

## Пример работы/Example of work
Входное изображение/Input image:
![ML photo](https://github.com/InfinityBlazze/YoloV3CV/assets/131138862/1d175c38-92e4-4561-8719-96d558a5bba9)

Обработанное изображение/Output image:
![image](https://github.com/InfinityBlazze/YoloV3CV/assets/131138862/68e92188-53e1-4587-beb8-37539a55ff56)


YOLO – это передовая сеть для распознавания объектов (object detection)
* YOLO может обнаруживать сразу несколько объектов, предсказывать классы и идентифицировать объекты на изображении.
* В YOLO, распознавание объектов было реализовано как задача регрессии к раздельным ограничивающим рамкам, с которыми связаны вероятности принадлежности к разным классам.
* Поскольку YOLO смотрит на изображение только один раз, плавающее окно – это неправильный подход. Вместо этого, все изображение разбивается с помощью сетки на ячейки размером 𝑆∗𝑆. После этого каждая ячейка отвечает за предсказание нескольких вещей
* В YOLO для предсказания содержащих рамок используются якорные рамки (anchor boxes). 
Якоря были рассчитаны на датасете COCO с помощью k-means кластеризации. Всего сеть распознает 80 классов объектов
* YOLO использует 53 CNN слоя (darknet-53) и соединены с ещё 53-мя слоями. В сумме 106 слоев, где 3 слоя отвечают за обнаружение (82,94,106)
* Версия 3 включает в себя несколько важных элементов. А именно: residual blocks (остаточные блоки), skip connections (пропуск соединений), и up-sampling (повышение частоты дискретизации). За каждым сверточным слоем (CNN) следует слой batch normalization (пакетная нормализации), Leaky ReLu (функция активации релу).

Для создания модели YOLOv3 (You Only Look Once Version 3) ключевые компоненты включают в себя:
* Реализацию на Python, используя фреймворк глубокого обучения TensorFlow.
* Реализацию алгоритма якорных рамок и подавления не-максимумов.
* Берём готовые веса DarkNet из оригинальной тренированной модели.
* Подачу изображения на вход и выход обработанного изображения.

Дополнительная информация:
* На вход сети подаются изображения формы (n, 416, 416, 3), где n-кол-во изображений,  416- ширина и высота изображений, 3 -  кол-во каналов цветов (rgb). Размер изображений можно менять, но он должен делиться на 32 без остатка (608, 1024 и т.д)
* Images of the form (n, 416, 416, 3) are supplied to the network input, where n is the number of images, 416 is the width and height of the images, 3 is the number of color channels (rgb). The size of the images can be changed, but it must be divisible by 32 without a remainder (608, 1024, etc.)
* Пример работы якорей для предсказания
* Example of how anchors work for prediction:

![image](https://github.com/InfinityBlazze/YoloV3CV/assets/131138862/21d56247-c8f2-4940-99ed-b4f7f6874f71)

* Далее сеть проходит по изображению и создает по 3 рамки на каждый объект. Испуользуя функцию nonMaximumSuppression мы отбрасываем рамки с наименьшим процентом вероятности.
* Next, the network passes through the image and creates 3 frames for each object. Using the nonMaximumSuppression function, we discard frames with the lowest percentage of probability.

![image](https://github.com/InfinityBlazze/YoloV3CV/assets/131138862/2069f460-e1c0-4ddc-9a39-d5f1a35f051a)
    
* На выходе мы получаем обработанное изображение с рамкой, уверенностью (в %) и названием объекта.
* At the output we get a processed image with a frame, confidence (in%) and the name of the object.

# Some info about project
  YOLO is an advanced network for object detection
* YOLO can detect multiple objects at once, predict classes and identify objects in an image.
* In YOLO, object recognition was implemented as a regression problem to separate bounding boxes with associated probabilities of membership in different classes.
* Since YOLO only looks at an image once, a floating window is the wrong approach. Instead, the entire image is divided using a grid into cells of size 𝑆∗𝑆. After that, each cell is responsible for predicting several things
* YOLO uses anchor boxes to predict containing boxes.
Anchors were calculated on the COCO dataset using k-means clustering. In total, the network recognizes 80 object classes
* YOLO uses 53 CNN layers (darknet-53) and is connected to another 53 layers. A total of 106 layers, where 3 layers are responsible for detection (82,94,106)
*Version 3 includes several important elements. Namely: residual blocks, skip connections, and up-sampling. Each convolutional layer (CNN) is followed by a batch normalization layer, Leaky ReLu (relu activation function).

To create a YOLOv3 (You Only Look Once Version 3) model, key components include:
* Implementation in Python using the TensorFlow deep learning framework.
* Implementation of the algorithm for anchor frames and suppression of non-maxima.
* We take ready-made DarkNet weights from the original trained model.
* Feeding the image to the input and output of the processed image.

