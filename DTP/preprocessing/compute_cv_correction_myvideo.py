'''
Before running this file bounding boxes must be processed using
process_bounding_boxes_bdd.py

This file takes the processed bounding boxes and does the following:
    1. Computes the pedestrians velocity and prints the error under the assumption
    that the pedestrian will maintain their veloicty
    2. Computes E_x and E_y. This is the value to be predicted by the model using
    the constant veloicty correction term. C_x = 1 - E_x
    3. Splits the data training and validation
'''
import pandas as pd
import numpy as np
from scipy.spatial import distance
import math
# from scipy.misc import imresize
import scipy.ndimage
import sys
import processing_utils as utils
import argparse


PATH = '../data/'
MIN_LENGTH_PAST = 30
MIN_LENGTH_FUTURE = 60
VELOCITY_FRAMES = 5

parser = argparse.ArgumentParser()
parser.add_argument('--detector', '-d',
                    help="Use detections from 'yolo' or 'faster-rcnn'", type=str, default='yolo')
args = parser.parse_args()

if args.detector == 'yolo':
    features = pd.read_pickle(PATH + 'myvideo_location_features_yolo.pkl')
else:
    features = pd.read_pickle(
        PATH + 'bdd_10k_location_features_faster-rcnn.pkl')


print(features.shape)
features = features[features['Labeled'] == 1]
print(features.shape)

# 设置终点为最后一个预测点
features['Final_x'] = features['Future_x'].apply(lambda x: x[-1])
features['Final_y'] = features['Future_y'].apply(lambda y: y[-1])

##################### Compute velocity and make predictions #####################
print('Computing velocity...')

# 计算过去n帧的平均速度
features['Velocity_x'] = features['Past_x'].apply(
    lambda x: utils.mean_velocity(x, len(x) - VELOCITY_FRAMES))
features['Velocity_y'] = features['Past_y'].apply(
    lambda y: utils.mean_velocity(y, len(y) - VELOCITY_FRAMES))

# 计算匀速运动终点
features['Predicted_x'] = features['Mid_x'] + \
    (MIN_LENGTH_FUTURE * features['Velocity_x'])
features['Predicted_y'] = features['Mid_y'] + \
    (MIN_LENGTH_FUTURE * features['Velocity_y'])

print('Getting predictions...')

# 计算匀速运动坐标序列
features['Predicted_x_seq'] = features.apply(
    lambda x: utils.get_seq_preds(x['Mid_x'], x['Velocity_x'], MIN_LENGTH_FUTURE), axis=1)
features['Predicted_y_seq'] = features.apply(
    lambda x: utils.get_seq_preds(x['Mid_y'], x['Velocity_y'], MIN_LENGTH_FUTURE), axis=1)

# 作为训练的ground truth，其代表匀速运动与真实值的误差序列
features['E_x'] = features.apply(lambda x: (
    x['Future_x'] - x['Predicted_x_seq']), axis=1)
features['E_y'] = features.apply(lambda y: (
    y['Future_y'] - y['Predicted_y_seq']), axis=1)

##################### Get errors #####################

print('Computing errors...')
features['MSE_15'] = features.apply(
    lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 15), axis=1)
features['MSE_10'] = features.apply(
    lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 10), axis=1)
features['MSE_5'] = features.apply(
    lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 5), axis=1)

features['EPE_15'] = (
    ((features['Predicted_x'] - features['Final_x']) * (features['Predicted_x'] - features['Final_x'])) + (
        (features['Predicted_y'] - features['Final_y']) * (features['Predicted_y'] - features['Final_y'])))
features['EPE_15'] = features['EPE_15'].apply(lambda x: math.sqrt(x))

features = features.reset_index()
del features['index']
print(features.shape)

pd.DataFrame(features).to_csv('../data/feature.csv')

train = features[0:int(len(features) * 0.8)]
val = features[int(len(features) * 0.8):]
pd.DataFrame(train).to_csv('../data/train.csv')
pd.DataFrame(val).to_csv('../data/val.csv')

print('Constant velocity val set EPE           :', round(val['EPE_15'].mean(), 0))
print('Constant velocity val set MSE@15        :', round(val['MSE_15'].mean(), 0))
print('Constant velocity val set MSE@10        :', round(val['MSE_10'].mean(), 0))
print('Constant velocity val set MSE@5         :', round(val['MSE_5'].mean(), 0))

assert len(train) + len(val) == len(features)
features.to_pickle(PATH + 'myvideo_cv.pkl')

train = train.reset_index()
val = val.reset_index()
del train['index']
del val['index']

print('Saving...')
if args.detector == 'yolo':
    train.to_pickle(PATH + 'myvideo_train_yolo.pkl')
    val.to_pickle(PATH + 'myvideo_val_yolo.pkl')
else:
    train.to_pickle(PATH + 'myvideo_train_faster-rcnn.pkl')
    val.to_pickle(PATH + 'myvideo_val_faster-rcnn.pkl')

print('Done.')

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

filename = features['filename'].values.tolist()

train1 = pd.DataFrame()
train2 = pd.DataFrame()
train3 = pd.DataFrame()

val1 = pd.DataFrame()
val2 = pd.DataFrame()
val3 = pd.DataFrame()

test1 = pd.DataFrame()
test2 = pd.DataFrame()
test3 = pd.DataFrame()

for i in range(len(features['filename'])):
    #if i >= 100: break
    city_name = features['filename'][i].split('/')[0]
    # print(city_name)
    series = features.iloc[i]
    if city_name in Train_cities1:
        train1 = train1.append(series)
    if city_name in Train_cities2:
        train2 = train2.append(series)
    if city_name in Train_cities3:
        train3 = train3.append(series)

    if city_name in Validation_cities1:
        val1 = val1.append(series)
    if city_name in Validation_cities2:
        val2 = val2.append(series)
    if city_name in Validation_cities3:
        val3 = val3.append(series)

    if city_name in Test_cities1:
        test1 = test1.append(series)
    if city_name in Test_cities2:
        test2 = test2.append(series)
    if city_name in Test_cities3:
        test3 = test3.append(series)
    if i % 100 == 0:
        print('Processing: ', i)

train1.to_pickle(PATH + 'train1_myvideo_location_features_yolo.pkl')
train2.to_pickle(PATH + 'train2_myvideo_location_features_yolo.pkl')
train3.to_pickle(PATH + 'train3_myvideo_location_features_yolo.pkl')

val1.to_pickle(PATH + 'val1_myvideo_location_features_yolo.pkl')
val2.to_pickle(PATH + 'val2_myvideo_location_features_yolo.pkl')
val3.to_pickle(PATH + 'val3_myvideo_location_features_yolo.pkl')

test1.to_pickle(PATH + 'test1_myvideo_location_features_yolo.pkl')
test2.to_pickle(PATH + 'test2_myvideo_location_features_yolo.pkl')
test3.to_pickle(PATH + 'test3_myvideo_location_features_yolo.pkl')
