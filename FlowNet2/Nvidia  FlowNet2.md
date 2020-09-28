#### Nvidia / FlowNet2

------

作用：根据图片计算出水平与垂直方向上的光流图。

主要使用的文件是main.py，读取本地的图片，保存处理后的结果并按照格式命名好。

启动命令：

```bash
python main.py --inference 
--model FlowNet2  
--save_flow --save /home/wangsen/flownet2_testpic/
 --inference_dataset ImagesFromFolder --inference_dataset_root /home/wangsen/MOF/Deep-SORT-YOLOv4-			 master/tensorflow1.14/deep-sort-yolov4-low-confidence-track-filtering/crop_image/  
--resume /home/wangsen/flownet2_testpic/FlowNet2_checkpoint.pth.tar
 其中--model表示调用的模型名称，--save_flow表示保存，--save表示保存的路径，--inference_dataset ImagesFromFolder表示从本地文件夹读数据，--inference_dataset_root表示需要处理的图片路径，--resume给出模型的路径。
 
 出于调试的方便，我在main.py中设置上述参数的默认值，所以现在的启动命令为：
 python main.py --inference --model FlowNet2 --save_flow
```

