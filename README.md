# gaussian-yolov3-detection-caffe
1. 实现caffe-gaussian-yolov3博客地址：https://blog.csdn.net/weixin_38715903/article/details/103695844

2. 实现caffe-gaussian-yolov3的时候
  gaussian_yolo_layer.h和gaussian_yolo_layer.cpp是caffe-yolov3中yolo.cpp和yolo.h的替换
  
3.检测一个文件夹下面的多张图片(有txt标注)，并计算map，修改caffe-yolov3中./detectnet/detectnet.cpp
  detectnet.cpp则是我修改过的代码
