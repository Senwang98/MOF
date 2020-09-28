import pandas as pd
import os
import cv2
from math import floor
import numpy as np
import argparse
import processing_utils as utils
import csv
parser = argparse.ArgumentParser()
parser.add_argument('--detector',
                    '-d',
                    help="Use detections from 'yolo' or 'faster-rcnn'",
                    type=str,
                    default='yolo')
args = parser.parse_args()

# 读取跟踪结果csv
TRACK_PATH = '../../DeepSort_YOLOv4/clip_data/myvideo_yolo_detection.csv'
SAVE_PATH = '../data/'

# 设置过去以及预测帧数
VELOCITY_FRAMES = 5
MIN_LENGTH_PAST = 30
MIN_LENGTH_FUTURE = 60
MIN_DETECTION_LENGTH = MIN_LENGTH_PAST + MIN_LENGTH_FUTURE

boxes = pd.read_csv(TRACK_PATH)

boxes['Labeled'] = 0

# BB format: top left (x,y), bottom right (x,y)
boxes['Mid_x'] = (boxes['bb1'] + boxes['bb3']) / 2
boxes['Mid_y'] = (boxes['bb2'] + boxes['bb4']) / 2
boxes['Height'] = boxes['bb4'] - boxes['bb2']
boxes['Width'] = boxes['bb3'] - boxes['bb1']

boxes['Past_x'] = 0
boxes['Past_y'] = 0
boxes['Past_x'] = boxes['Past_x'].astype(object)
boxes['Past_y'] = boxes['Past_y'].astype(object)

print(boxes.shape)

# boxes = boxes[boxes['Height'] > 50]
boxes = boxes.sort_values(by=['filename', 'track', 'frame_num'])
boxes = boxes.reset_index()
del boxes['index']

################################
###  detection_length从0开始  ###
################################
boxes['Labeled'] = np.where(
    boxes['detection_length'] >= MIN_DETECTION_LENGTH - 1, 1,
    0)  # 注意这里的90 - 1 的含义
boxes['Labeled'] = boxes['Labeled'].shift(-MIN_LENGTH_FUTURE)

past_x_names = []
for past in range(MIN_LENGTH_PAST, 0, -1):
    boxes['prev_x' + str(past)] = boxes['Mid_x'].shift(past)
    past_x_names.append('prev_x' + str(past))
past_y_names = []
for past in range(MIN_LENGTH_PAST, 0, -1):
    boxes['prev_y' + str(past)] = boxes['Mid_y'].shift(past)
    past_y_names.append('prev_y' + str(past))
boxes['Past_x'] = boxes[past_x_names].values.tolist()
boxes['Past_y'] = boxes[past_y_names].values.tolist()

# 计算过去n帧的平均速度
boxes['Velocity_x'] = boxes['Past_x'].apply(
    lambda x: utils.mean_velocity(x,
                                  len(x) - VELOCITY_FRAMES))
boxes['Velocity_y'] = boxes['Past_y'].apply(
    lambda y: utils.mean_velocity(y,
                                  len(y) - VELOCITY_FRAMES))
# 计算匀速运动坐标序列
boxes['Predicted_x_seq'] = boxes.apply(lambda x: utils.get_seq_preds(
    x['Mid_x'], x['Velocity_x'], MIN_LENGTH_FUTURE),
                                       axis=1)
boxes['Predicted_y_seq'] = boxes.apply(lambda x: utils.get_seq_preds(
    x['Mid_y'], x['Velocity_y'], MIN_LENGTH_FUTURE),
                                       axis=1)

print('Storing centroids. This make take a few minutes.')
# 增加条目
for i in range(1, 61):
    boxes['x1_' + str(i)] = 0
    boxes['y1_' + str(i)] = 0
    boxes['x2_' + str(i)] = 0
    boxes['y2_' + str(i)] = 0

print('Add items OK!')

cnt = 0
for i in range(0, len(boxes['filename'])):
    # for i in range(0, 10000):
    if boxes['Labeled'][i] == 1:
        cnt += 1
        # print(i)
        for j in range(0, 60):
            boxes['x1_' + str(j + 1)][
                i] = boxes['Predicted_x_seq'][i][j] - boxes['Width'][i] / 2
            boxes['y1_' + str(j + 1)][
                i] = boxes['Predicted_y_seq'][i][j] - boxes['Height'][i] / 2
            boxes['x2_' + str(j + 1)][
                i] = boxes['Predicted_x_seq'][i][j] + boxes['Width'][i] / 2
            boxes['y2_' + str(j + 1)][
                i] = boxes['Predicted_y_seq'][i][j] + boxes['Height'][i] / 2
        if cnt % 100 == 0:
            print('Calculate 60: ', i, ' / ', cnt)
            # break

print('Saving')
boxes = boxes.dropna(subset=['filename'], axis=0)
# del boxes['bb1']
# del boxes['bb2']
# del boxes['bb3']
# del boxes['bb4']
# del boxes['track']
# del boxes['detection_length']
# del boxes['Height']
# del boxes['Labeled']
# del boxes['Mid_x']
# del boxes['Mid_y']
# del boxes['Width']
# del boxes['Velocity_x']
# del boxes['Velocity_y']
# del boxes['Predicted_x_seq']
# del boxes['Predicted_y_seq']

# 数据集的分组，train、val、test
# Fold 1
Train_cities1 = [
    'BARCELONA', 'BRNO', 'ERFURT', 'KAUNAS', 'LEIPZIG', 'NUREMBERG', 'PALMA',
    'PRAGUE', 'TALLINN', 'TARTU', 'VILNIUS', 'WEIMAR'
]
Validation_cities1 = [
    'DRESDEN', 'HELSINKI', 'PADUA', 'POZNAN', 'VERONA', 'WARSAW'
]
Test_cities1 = ['KRAKOW', 'RIGA', 'WROCLAW']

# Fold 2
Train_cities2 = [
    'BARCELONA', 'DRESDEN', 'ERFURT', 'HELSINKI', 'KRAKOW', 'LEIPZIG', 'PADUA',
    'PALMA', 'POZNAN', 'RIGA', 'TALLINN', 'VERONA', 'VILNIUS', 'WARSAW',
    'WROCLAW'
]
Validation_cities2 = ['KAUNAS', 'PRAGUE', 'WEIMAR']
Test_cities2 = ['BRNO', 'NUREMBERG', 'TARTU']

# Fold 3
Train_cities3 = [
    'BRNO', 'DRESDEN', 'HELSINKI', 'KAUNAS', 'KRAKOW', 'NUREMBERG', 'PADUA',
    'POZNAN', 'PRAGUE', 'RIGA', 'TARTU', 'VERONA', 'WARSAW', 'WEIMAR',
    'WROCLAW'
]
Validation_cities3 = ['BARCELONA', 'LEIPZIG', 'TALLINN']
Test_cities3 = ['ERFURT', 'PALMA', 'VILNIUS']

f1 = open('../../outputs/constant_velocity/test_yolo_fold_1.csv', 'w')
csv_writer1 = csv.writer(f1)
f2 = open('../../outputs/constant_velocity/test_yolo_fold_2.csv', 'w')
csv_writer2 = csv.writer(f2)
f3 = open('../../outputs/constant_velocity/test_yolo_fold_3.csv', 'w')
csv_writer3 = csv.writer(f3)

header = ["vid", "filename", "frame_num"
          ]  # "Predicted_x_seq","Predicted_y_seq", "width", "height"
for i in range(1, 61):
    header.append("x1_" + str(i))
    header.append("y1_" + str(i))
    header.append("x2_" + str(i))
    header.append("y2_" + str(i))
csv_writer1.writerow(header)
csv_writer2.writerow(header)
csv_writer3.writerow(header)

cnt = 0
for i in range(0, len(boxes['filename'])):
    if boxes['Labeled'][i] == 1:
        cnt += 1
        city = boxes['filename'][i].split('/')[0]
        video = boxes['filename'][i].split('/')[1]
        tmp = []
        tmp.append(video)
        tmp.append(city)
        tmp.append(boxes['frame_num'][i])
        # tmp.append(boxes['Width'][i])
        # tmp.append(boxes['Height'][i])
        # tmp.append(boxes['Predicted_x_seq'][i])
        # tmp.append(boxes['Predicted_y_seq'][i])
        for j in range(1, 61):
            tmp.append(boxes['x1_' + str(j)][i])
            tmp.append(boxes['y1_' + str(j)][i])
            tmp.append(boxes['x2_' + str(j)][i])
            tmp.append(boxes['y2_' + str(j)][i])
        if city in Test_cities1:
            csv_writer1.writerow(tmp)
        if city in Test_cities2:
            csv_writer2.writerow(tmp)
        if city in Test_cities3:
            csv_writer3.writerow(tmp)
        if cnt % 100 == 0:
            print('CSV writing: ', i, ' / ', cnt)

f1.close()
f2.close()
f3.close()

del boxes['Labeled']
# boxes.to_csv('../outputs/constant_velocity/yolo_test.csv')
print('Done.')
