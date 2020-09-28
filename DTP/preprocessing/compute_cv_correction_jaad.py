'''
Before running this file bounding boxes must be processed using
process_bounding_boxes_jaad.py

This file takes the processed bounding boxes and does the following:
    1. Computes the pedestrians velocity and prints the error under the assumption
    that the pedestrian will maintain their veloicty
    2. Computes E_x and E_y. This is the value to be predicted by the model using
    the constant veloicty correction term. C_x = 1 - E_x
    3. Splits the data training and testing
    4. Splits the training data into 5 folds
'''
import pandas as pd
import numpy as np
from scipy.spatial import distance
import math
from scipy.misc import imresize
import scipy.ndimage
import sys
import processing_utils as utils

PATH = '../data/'
MIN_LENGTH_PAST = 10
MIN_LENGTH_FUTURE = 15
VELOCITY_FRAMES = 5
VELOCITY_FRAMES = MIN_LENGTH_PAST - VELOCITY_FRAMES

# 只保留满足跟踪条件的帧
features = pd.read_pickle(PATH + 'jaad_location_features.pkl')
features = features[features['Labeled'] == 1]

##################### Compute velocity and make predictions #####################
# 对于当前帧而言，终点同样也关心，该条目为单一值
features['Final_x'] = features['Future_x'].apply(lambda x: x[-1])
features['Final_y'] = features['Future_y'].apply(lambda y: y[-1])

# VELOCITY_FRAMES ~ len(x)之间的平均速度，该条目为单一值
features['Velocity_x'] = features['Past_x'].apply(lambda x: utils.mean_velocity(x, VELOCITY_FRAMES))
features['Velocity_y'] = features['Past_y'].apply(lambda y: utils.mean_velocity(y, VELOCITY_FRAMES))

# 计算 MIN_LENGTH_FUTURE 之后匀速运动的位置，该条目为单一值
# 该值将会作为ResNet的初始值，模型初始权重（“赢在起跑线”）
features['Predicted_x'] = features['Mid_x'] + (MIN_LENGTH_FUTURE * features['Velocity_x'])
features['Predicted_y'] = features['Mid_y'] + (MIN_LENGTH_FUTURE * features['Velocity_y'])

# 产生 MIN_LENGTH_FUTURE 长度的未来位置序列，该值为序列
features['Predicted_x_seq'] = features.apply(
    lambda x: utils.get_seq_preds(x['Mid_x'], x['Velocity_x'], MIN_LENGTH_FUTURE), axis=1)
features['Predicted_y_seq'] = features.apply(
    lambda x: utils.get_seq_preds(x['Mid_y'], x['Velocity_y'], MIN_LENGTH_FUTURE), axis=1)

# 当前帧之后的15帧的gt与CV之间的差值，该值为序列
features['E_x'] = features.apply(lambda x: (x['Future_x'] - x['Predicted_x_seq']), axis=1)
features['E_y'] = features.apply(lambda y: (y['Future_y'] - y['Predicted_y_seq']), axis=1)

##################### Get errors #####################
features['MSE'] = features.apply(
    lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 15), axis=1)
features['MSE_10'] = features.apply(
    lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 10), axis=1)
features['MSE_5'] = features.apply(
    lambda x: utils.calc_mse(x['Predicted_x_seq'], x['Future_x'], x['Predicted_y_seq'], x['Future_y'], 5), axis=1)

# 终点误差（边缘放置误差？约等于FDE）
features['EPE'] = (
        ((features['Predicted_x'] - features['Final_x']) * (features['Predicted_x'] - features['Final_x'])) + (
        (features['Predicted_y'] - features['Final_y']) * (features['Predicted_y'] - features['Final_y'])))
features['EPE'] = features['EPE'].apply(lambda x: math.sqrt(x))

##################### Pre-process filenames #####################
features['Video'] = features['Video'].apply(lambda x: x.split('.')[0])
# video0001 -> video_0001
# print('print(features[Video])', features['Video'])
features['Video'] = features['Video'].apply(lambda x: x[0:5] + '_' + x[5:])
# print('print(features[Video])', features['Video'])

# print('print(features[Frame])', features['Frame'])
features['Frame'] = features['Frame'].astype(int).astype(str).apply(lambda x: x.zfill(4))
# print('print(features[Frame])', features['Frame'])

# 变成 ‘video_0001/frame_0199_ped_1.png’ 格式
features['Filename'] = features['Video'] + '/frame_' + features['Frame'] + '_ped_' + features['Pedestrian'].astype(
    str) + '.png'
# print('print(features[Filename])', features['Filename'])

features = features.reset_index()
del features['index']
features.to_pickle(PATH + 'jaad_cv.pkl')

##################### Split into train and test. Compute error #####################
train = features[features['Video'] < 'video_0250.mp4']
test = features[features['Video'] >= 'video_0250.mp4']
print('train size = ', train.shape)
print('test size = ', test.shape)

test = test.reset_index()
del test['index']
print('train size = ', train.shape)
print('test size = ', test.shape)

print('Constant velocity test set EPE               :', round(test['EPE'].mean(), 1))
print('Constant velocity test set MSE@15            :', round(test['MSE'].mean(), 0))
print('Constant velocity test set MSE@10            :', round(test['MSE_10'].mean(), 0))
print('Constant velocity test set MSE@5             :', round(test['MSE_5'].mean(), 0))

##################### Split into 5 folds #####################

# 200 - 250
train_1 = train[train['Video'] < 'video_0200.mp4']
val_1 = train[train['Video'] >= 'video_0200.mp4']
assert len(train_1) + len(val_1) == len(train)
assert (len(set(train_1.Video.unique()).intersection(set(val_1.Video.unique()))) == 0)

# 150 - 200
train_2 = train[(train['Video'] <= 'video_0150.mp4') | (train['Video'] > 'video_0200.mp4')]
val_2 = train[(train['Video'] > 'video_0150.mp4') & (train['Video'] <= 'video_0200.mp4')]
assert len(train_2) + len(val_2) == len(train)
assert (len(set(train_2.Video.unique()).intersection(set(val_2.Video.unique()))) == 0)

# 100 - 150
train_3 = train[(train['Video'] <= 'video_0100.mp4') | (train['Video'] > 'video_0150.mp4')]
val_3 = train[(train['Video'] > 'video_0100.mp4') & (train['Video'] <= 'video_0150.mp4')]
assert len(train_3) + len(val_3) == len(train)
assert (len(set(train_3.Video.unique()).intersection(set(val_3.Video.unique()))) == 0)

# 50 - 100
train_4 = train[(train['Video'] <= 'video_0050.mp4') | (train['Video'] > 'video_0100.mp4')]
val_4 = train[(train['Video'] > 'video_0050.mp4') & (train['Video'] <= 'video_0100.mp4')]
assert len(train_4) + len(val_4) == len(train)
assert (len(set(train_4.Video.unique()).intersection(set(val_4.Video.unique()))) == 0)

# 0 - 50
train_5 = train[train['Video'] > 'video_0050.mp4']
val_5 = train[train['Video'] <= 'video_0050.mp4']
assert len(train_5) + len(val_5) == len(train)
assert (len(set(train_5.Video.unique()).intersection(set(val_5.Video.unique()))) == 0)

train_1 = train_1.reset_index()
val_1 = val_1.reset_index()
del train_1['index']
del val_1['index']

train_2 = train_2.reset_index()
val_2 = val_2.reset_index()
del train_2['index']
del val_2['index']

train_3 = train_3.reset_index()
val_3 = val_3.reset_index()
del train_3['index']
del val_3['index']

train_4 = train_4.reset_index()
val_4 = val_4.reset_index()
del train_4['index']
del val_4['index']

train_5 = train_5.reset_index()
val_5 = val_5.reset_index()
del train_5['index']
del val_5['index']

print('train_1 size = ', train_1.shape)
print('val_1 size = ', val_1.shape)
print('train_2 size = ', train_2.shape)
print('val_2 size = ', val_2.shape)
print('train_3 size = ', train_3.shape)
print('val_3 size = ', val_3.shape)
print('train_4 size = ', train_4.shape)
print('val_4 size = ', val_4.shape)
print('train_5 size = ', train_5.shape)
print('val_5 size = ', val_5.shape)

train_1.to_pickle(PATH + 'jaad_cv_train_1.pkl')
val_1.to_pickle(PATH + 'jaad_cv_val_1.pkl')
train_2.to_pickle(PATH + 'jaad_cv_train_2.pkl')
val_2.to_pickle(PATH + 'jaad_cv_val_2.pkl')
train_3.to_pickle(PATH + 'jaad_cv_train_3.pkl')
val_3.to_pickle(PATH + 'jaad_cv_val_3.pkl')
train_4.to_pickle(PATH + 'jaad_cv_train_4.pkl')
val_4.to_pickle(PATH + 'jaad_cv_val_4.pkl')
train_5.to_pickle(PATH + 'jaad_cv_train_5.pkl')
val_5.to_pickle(PATH + 'jaad_cv_val_5.pkl')
test.to_pickle(PATH + 'jaad_cv_test.pkl')
