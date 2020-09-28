import pandas as pd
import numpy as np
from scipy.spatial import distance
import math
import scipy.ndimage
import sys
import processing_utils as utils
import argparse

PATH = './data_inference/'
MIN_LENGTH_PAST = 30
MIN_LENGTH_FUTURE = 0
VELOCITY_FRAMES = 5

parser = argparse.ArgumentParser()
parser.add_argument('--detector',
                    '-d',
                    help="Use detections from 'yolo' or 'faster-rcnn'",
                    type=str,
                    default='yolo')
args = parser.parse_args()

features = pd.read_pickle(PATH + 'myvideo_location_features_yolo.pkl')
print('原始数据: ', features.shape)
features = features[features['Labeled'] == 1]
print('label = 1: ', features.shape)

# 设置终点为最后一个预测点
# features['Final_x'] = features['Future_x'].apply(lambda x: x[-1])
# features['Final_y'] = features['Future_y'].apply(lambda y: y[-1])

##################### Compute velocity and make predictions #####################
print('Computing velocity...')
features['Velocity_x'] = features['Past_x'].apply(
    lambda x: utils.mean_velocity(x,
                                  len(x) - VELOCITY_FRAMES))
features['Velocity_y'] = features['Past_y'].apply(
    lambda y: utils.mean_velocity(y,
                                  len(y) - VELOCITY_FRAMES))
# 计算匀速运动终点
# features['Predicted_x'] = features['Mid_x'] + \
#     (MIN_LENGTH_FUTURE * features['Velocity_x'])
# features['Predicted_y'] = features['Mid_y'] + \
#     (MIN_LENGTH_FUTURE * features['Velocity_y'])

print('Getting predictions...')
# 计算匀速运动坐标序列
features['Predicted_x_seq'] = features.apply(
    lambda x: utils.get_seq_preds(x['Mid_x'], x['Velocity_x'], 60), axis=1)
features['Predicted_y_seq'] = features.apply(
    lambda x: utils.get_seq_preds(x['Mid_y'], x['Velocity_y'], 60), axis=1)

# 作为训练的ground truth，其代表匀速运动与真实值的误差序列
# features['E_x'] = features.apply(lambda x: (
#     x['Future_x'] - x['Predicted_x_seq']), axis=1)
# features['E_y'] = features.apply(lambda y: (
#     y['Future_y'] - y['Predicted_y_seq']), axis=1)

##################### Get errors #####################
print('Computing errors...')
# features['MSE_15'] = features.apply(
#     lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 15), axis=1)
# features['MSE_10'] = features.apply(
#     lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 10), axis=1)
# features['MSE_5'] = features.apply(
#     lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 5), axis=1)

# features['EPE_15'] = (
#     ((features['Predicted_x'] - features['Final_x']) * (features['Predicted_x'] - features['Final_x'])) + (
#         (features['Predicted_y'] - features['Final_y']) * (features['Predicted_y'] - features['Final_y'])))
# features['EPE_15'] = features['EPE_15'].apply(lambda x: math.sqrt(x))

features = features.reset_index()
del features['index']
# pd.DataFrame(features).to_csv(PATH + '/feature.csv')
# train = features[0:int(len(features) * 0.8)]
val = features[int(len(features) * 0.8):int(len(features) * 0.8) + 1]
# pd.DataFrame(train).to_csv(PATH + '/train.csv')
pd.DataFrame(val).to_csv(PATH + '/val.csv')

# print('Constant velocity val set EPE           :',
#       round(val['EPE_15'].mean(), 0))
# print('Constant velocity val set MSE@15        :',
#       round(val['MSE_15'].mean(), 0))
# print('Constant velocity val set MSE@10        :',
#       round(val['MSE_10'].mean(), 0))
# print('Constant velocity val set MSE@5         :',
#       round(val['MSE_5'].mean(), 0))

# assert len(train) + len(val) == len(features)
# features.to_pickle(PATH + 'myvideo_cv.pkl')

# train = train.reset_index()
# val = val.reset_index()
# del train['index']
# del val['index']

print('Saving...')
# train.to_pickle(PATH + 'myvideo_train_yolo.pkl')
val.to_pickle(PATH + 'myvideo_val_yolo_0.pkl')
print('Done.')
'''
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

SAVE_PATH = PATH
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
    if i >= 1000: break
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
    if i % 1 == 0:
        print('Processing: ', i)

train1 = train1.reset_index()
del train1['index']
train2 = train2.reset_index()
del train2['index']
train3 = train3.reset_index()
del train3['index']
val1 = val1.reset_index()
del val1['index']
val2 = val2.reset_index()
del val2['index']
val3 = val3.reset_index()
del val3['index']
test1 = test1.reset_index()
del test1['index']
test2 = test2.reset_index()
del test2['index']
test3 = test3.reset_index()
del test3['index']

train1.to_pickle(SAVE_PATH + 'train1_myvideo_location_features_yolo.pkl')
train2.to_pickle(SAVE_PATH + 'train2_myvideo_location_features_yolo.pkl')
train3.to_pickle(SAVE_PATH + 'train3_myvideo_location_features_yolo.pkl')

val1.to_pickle(SAVE_PATH + 'val1_myvideo_location_features_yolo.pkl')
val2.to_pickle(SAVE_PATH + 'val2_myvideo_location_features_yolo.pkl')
val3.to_pickle(SAVE_PATH + 'val3_myvideo_location_features_yolo.pkl')

test1.to_pickle(SAVE_PATH + 'test1_myvideo_location_features_yolo.pkl')
test2.to_pickle(SAVE_PATH + 'test2_myvideo_location_features_yolo.pkl')
test3.to_pickle(SAVE_PATH + 'test3_myvideo_location_features_yolo.pkl') 
'''