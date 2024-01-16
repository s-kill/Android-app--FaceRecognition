El objetivo de este proyecto es generar una versión de Tiny-YOLO v3 que funcione en teléfonos (Android o IoS). En el caso de Android, se recomienda usar tensorflow en Java, ya que es la opción más sencilla. Dado que la red ya está entrenada, simplemente debe ser cargada desde la aplicación en Android (no es necesario entrenar nada). El sistema final debe mostrarse funcionando en un teléfono real, detectando los objetos capturados por la cámara trasera. Se usará como base un código que permite usar otro tipo de detector (SSD-MobileNet), y deberá ser modificado de modo que pueda usar Tiny-YOLO v3. Finalmente, se debe comparar la performance de las dos redes.
Referencias:
- YOLO: Real-Time Object Detection. https://pjreddie.com/darknet/yolo/
- Jonathan Huang et al. Speed/accuracy trade-offs for modern convolutional object detectors. https://arxiv.org/pdf/1611.10012.pdf
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
