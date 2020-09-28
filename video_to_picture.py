import pandas as pd
import numpy as np
from cv2 import cv2
import os
val_feature = pd.read_csv('./sted_feature/outputs/sted/test_yolo_fold_1.csv')

# 保存所有的城市路径
val_feature_video_path = {}
tmp_filename = list(val_feature['filename'])
tmp_vid = list(val_feature['vid'])

val_feature_video_path['filename'] = []
val_feature_video_path['vid'] = []


if len(tmp_filename) >= 2:
    val_feature_video_path['filename'].append(tmp_filename[0]+'/'+tmp_vid[0])
    for i in range(1, len(tmp_filename)):
        pre = tmp_filename[i - 1] + '/' + tmp_vid[i - 1]
        cur = tmp_filename[i] + '/' + tmp_vid[i]
        if pre == cur:
            pass
        else:
            val_feature_video_path['filename'].append(tmp_filename[i]+'/'+tmp_vid[i])

print(val_feature_video_path['filename'])
# 保存
# pd.DataFrame(val_feature_video_path).to_csv('./data/val_feature_video_path.csv')

save_path = '/home/wangsen/ws/video2frame/'
# 读取本地视频，保存成一帧帧图片
for i in val_feature_video_path['filename']:
    city = i.split('/')[0]
    video_number = i.split('/')[1]
    print(city)
    print(video_number)

    city_dir = ''.join([
        save_path,
        city,
    ])
    if not os.path.exists(city_dir):
        os.mkdir(city_dir)
    video_dir = ''.join([city_dir, '/', video_number])
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    tmp_save_path = '/home/wangsen/ws/video2frame/' + i + '/'
    video_path = '/home/wangsen/ws/citywalks/clips/' + i
    cap = cv2.VideoCapture(video_path)

    n = 0
    while (1):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('', frame)
            cv2.waitKey(1)
            cv2.imwrite(tmp_save_path + str(n) + '.png', frame)
            n = n + 1
            print("Processing: ", i, " ---- ", n)
        else:
            break
    cap.release()