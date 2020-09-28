import pandas as pd
import os
import cv2
from math import floor
import numpy as np
import argparse
import csv
parser = argparse.ArgumentParser()
parser.add_argument('--detector',
                    '-d',
                    help="Use detections from 'yolo' or 'faster-rcnn'",
                    type=str,
                    default='yolo')
args = parser.parse_args()

# 读取跟踪结果csv
# TRACK_PATH = '../../DeepSort_YOLOv4/clip_data/myvideo_yolo_detection.csv'
TRACK_PATH = '../../DeepSort_YOLOv4/clip_data/myvideo_yolo_detection.csv'

SAVE_PATH = '../data/'

# 设置过去以及预测帧数
# How far back and how far foward to use as features/predict respectively
MIN_LENGTH_PAST = 30
MIN_LENGTH_FUTURE = 60
MIN_DETECTION_LENGTH = MIN_LENGTH_PAST + MIN_LENGTH_FUTURE

boxes = pd.read_csv(TRACK_PATH)

# boxes['Requires_features'] = 0
boxes['Labeled'] = 0

# BB format: top left (x,y), bottom right (x,y)
boxes['Mid_x'] = (boxes['bb1'] + boxes['bb3']) / 2
boxes['Mid_y'] = (boxes['bb2'] + boxes['bb4']) / 2
boxes['Height'] = boxes['bb4'] - boxes['bb2']
boxes['Width'] = boxes['bb3'] - boxes['bb1']

# boxes['Past_x'] = 0
# boxes['Past_y'] = 0
# boxes['Future_x'] = 0
# boxes['Future_y'] = 0
# boxes['Past_x'] = boxes['Past_x'].astype(object)
# boxes['Past_y'] = boxes['Past_y'].astype(object)
# boxes['Future_x'] = boxes['Future_x'].astype(object)
# boxes['Future_y'] = boxes['Future_y'].astype(object)

print(boxes.shape)
# Remove small pedesboxes['Requires_features'] = 0
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

# features = boxes[boxes['Labeled'] == 1]
# print("label的个数为：", features.shape)

print('Storing centroids. This make take a few minutes.')

# 增加条目
for i in range(1, 61):
    boxes['x1_' + str(i)] = 0
    # boxes['x1_'+ str(i)] = boxes['x1_'+ str(i)].astype(object)
    boxes['y1_' + str(i)] = 0
    # boxes['y1_'+ str(i)] = boxes['y1_'+ str(i)].astype(object)
    boxes['x2_' + str(i)] = 0
    # boxes['x2_'+ str(i)] = boxes['x2_'+ str(i)].astype(object)
    boxes['y2_' + str(i)] = 0
    # boxes['y2_'+ str(i)] = boxes['y2_'+ str(i)].astype(object)

print('Add items OK!')

bb1 = boxes['Mid_x'] - boxes['Width'] / 2
bb2 = boxes['Mid_y'] - boxes['Height'] / 2
bb3 = boxes['Mid_x'] + boxes['Width'] / 2
bb4 = boxes['Mid_y'] + boxes['Height'] / 2

cnt = 0
for i in range(0, len(boxes['filename']) - 59):
    if boxes['Labeled'][i] == 1:
        cnt += 1
        # print(i)
        for j in range(1, 61):
            boxes['x1_' + str(j)][i] = bb1[i + j]
            boxes['y1_' + str(j)][i] = bb2[i + j]
            boxes['x2_' + str(j)][i] = bb3[i + j]
            boxes['y2_' + str(j)][i] = bb4[i + j]
        if cnt % 1000 == 0:
            print(i, ' / ', cnt)
            # break

print('Saving')
boxes = boxes.dropna(subset=['filename'], axis=0)
del boxes['bb1']
del boxes['bb2']
del boxes['bb3']
del boxes['bb4']
del boxes['track']
del boxes['detection_length']
del boxes['Height']
# del boxes['Labeled']
del boxes['Mid_x']
del boxes['Mid_y']
del boxes['Width']

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

f1 = open('../../outputs/ground_truth/test_yolo_fold_1.csv', 'w')
csv_writer1 = csv.writer(f1)
f2 = open('../../outputs/ground_truth/test_yolo_fold_2.csv', 'w')
csv_writer2 = csv.writer(f2)
f3 = open('../../outputs/ground_truth/test_yolo_fold_3.csv', 'w')
csv_writer3 = csv.writer(f3)

header = ["vid", "filename", "frame_num"]
for i in range(1, 61):
    header.append("x1_" + str(i))
    header.append("y1_" + str(i))
    header.append("x2_" + str(i))
    header.append("y2_" + str(i))
csv_writer1.writerow(header)
csv_writer2.writerow(header)
csv_writer3.writerow(header)

print(boxes['filename'])

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
        if cnt % 1000 == 0:
            print(i, ' / ', cnt)

f1.close()
f2.close()
f3.close()

del boxes['Labeled']
boxes.to_csv('../outputs/ground_truth/yolo_test.csv')

# if args.detector == 'yolo':
#     print('ok yolo')
#     boxes.to_pickle(SAVE_PATH + 'myvideo_location_features_yolo.pkl')
# else:
#     print('ok rcnn')
#     boxes.to_pickle(SAVE_PATH + 'bdd_10k_location_features_faster-rcnn.pkl')

print('Done.')
