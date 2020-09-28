"""
    测试用
"""

import pandas as pd
import numpy as np
from cv2 import cv2
import os

# train_boxes = np.load('./model/predictions_rn18_flow_css_9stack_jaad_fold_3pretrained-False_disp.npy')
# for i in range(0, 50):
#     print(i, ': ', train_boxes[i])
# pkl = pd.read_pickle('./data/myvideo_cv.pkl')
# print('CV ', pkl.shape)
# pkl = pd.read_pickle('./data/myvideo_train_yolo.pkl')
# print('Train ', pkl.shape)
# pkl = pd.read_pickle('./data/myvideo_val_yolo.pkl')
# print('Val ', pkl.shape)

val_feature = pd.read_csv('./data/val.csv')

# 保存所有的城市路径
val_feature_video_path = {}
tmp_filename = list(val_feature['filename'])
val_feature_video_path['filename'] = []
if len(tmp_filename) >= 2:
    val_feature_video_path['filename'].append(tmp_filename[0])
    for i in range(1, len(tmp_filename)):
        if tmp_filename[i - 1] == tmp_filename[i]:
            pass
        else:
            val_feature_video_path['filename'].append(tmp_filename[i])
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
            cv2.waitKey(30)
            cv2.imwrite(tmp_save_path + str(n) + '.png', frame)
            n = n + 1
            print("Processing: ", i, " ---- ", n)
        else:
            break
    cap.release()