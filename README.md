<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="200"/>
</div>
</p>

<div id="badges" align="center">
  <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  <img src="https://img.shields.io/badge/YouTube-red?style=for-the-badge&logo=youtube&logoColor=blue" alt="VK Badge"/>

</div>

<p align="center">
   <img src="https://media.giphy.com/media/VYPDYUBR9bGEIYtz5s/giphy.gif?cid=ecf05e47pdggmnkmdmfzgcp00hq8cg5nmxrnb2w5m8e8c36t&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="100">
</p>

## About (Eng)

To create the YOLOv3 (You Only Look Once Version 3) model, various tools and technologies are used. The key components include:
* The model is implemented in Python using the TensorFlow deep learning framework.
* Data: Pretrained DarkNet weights are used from the original trained model.
* The architecture of YOLOv3 is a neural network based on the concept of single-pass object detection.

The model consists of several blocks:
* Darknet layer, which is a convolutional neural network based on the Darknet-53 architecture.
* Three YoloConv blocks, each performing multiple convolutions to extract features from different scales of the image.
* Three YoloOutput blocks, generating output data for each YoloConv block. The Lambda block utilizes the yolo_boxes() and nonMaximumSuppression() functions, which take the output data output_0, output_1, and output_2 from the YoloOutput blocks, convert them into bounding boxes, and perform non-maximum suppression to obtain the final model predictions. If the training parameter is True, the function returns a model to be used for training. If training is False, the function returns a model for making predictions on new data.

Methods: 
* Object detection: YOLOv3 utilizes grid cells to divide the image and predict bounding boxes and class probabilities for objects in each cell.
* Non-max suppression (NMS): This method is used to reduce redundant detections by removing unnecessary boxes with low confidences and overlaps. By using these tools and methods, YOLOv3 achieves good accuracy and speed in real-time object detection tasks on photos and video material.
* The network recognizes 80 classes.
* Object detection layers output predictions for object detection and classification.
*  Upsampling blocks that combine layers.

## About(Rus)

Для создания модели YOLOv3 (You Only Look Once Version 3) используются различные инструменты и технологии. Ключевые компоненты включают в себя:
* Реализацию на Python, используя фреймворк глубокого обучения TensorFlow.
* Взяты готовые веса DarkNet из оригинальной тренированной модели.
* Архитектура YOLOv3 представляет собой нейронную сеть, основанную на концепции однопроходного обнаружения объектов.

Модель состоит из нескольких блоков:
* Слой Darknet, который представляет собой сверточную нейронную сеть, основанную на архитектуре Darknet-53.
* Три блока YoloConv, каждый из которых выполняет несколько сверток с целью извлечения признаков из различных масштабов изображения.
* Три блока YoloOutput, генерирующих выходные данные для каждого блока YoloConv.
* В блоке Lambda используются функции yolo_boxes() и nonMaximumSuppression(), которые принимают выходные данные output_0, output_1 и output_2 из блоков YoloOutput, преобразуют их в ящики (bounding boxes) и выполняют подавление немаксимальных значений для получения окончательных предсказаний модели.
* Если параметр training равен True, то функция возвращает модель, которая будет использоваться в процессе обучения. Если training равен False, то функция возвращает модель, которая будет использоваться для выполнения предсказаний на новых данных.

Методы:
* Обнаружение объектов: YOLOv3 использует грид-ячейки для разбиения изображения и прогнозирования ограничивающих рамок и вероятностей классов для объектов в каждой ячейке.
* Non-max suppression (NMS): Этот метод используется для уменьшения повторяющихся обнаружений путем удаления лишних рамок с низкими уверенностями и перекрытиями.
* Благодаря использованию этих инструментов и методов, YOLOv3 достигает хорошей точности и скорости на задачах обнаружения объектов в реальном времени на фото и видео материале.
* Сеть распознает 80 классов.
* Слои обнаружения объектов, которые выводят прогнозы по обнаружению и классификации объектов.
* Повышающие блоки (upsampling), объединяющие слои.



