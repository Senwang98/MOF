#### DeepSort-YOLOv4

------

github:https://github.com/LeonLok/Deep-SORT-YOLOv4

作者使用的是YOLOv3,链接：https://github.com/Qidian213/deep_sort_yolov3

直接使用YOLOv3/4的检测器，demo.py中利用pre-trained model直接检测，demo.py我修改成对目录下所有视频都进行检测处理，处理的结果包含：每一个行人检测区域resize成(128, 256)并保存，跟踪结果处理成tracking_result.csv的格式，另一种格式是为了适配dtp的格式，myvideo_detection.csv就是这样的一个作用。dtp_box_statistics.npy保存的是encider/GRU-1所需要的数据，size=(n, 8, 25)。