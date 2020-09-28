from cv2 import cv2
import numpy as np
import pandas as pd
import os

pred = pd.read_csv('./sted_feature/outputs/sted/test_yolo_fold_1.csv')
gt = pd.read_csv('./sted_feature/outputs/ground_truth/test_yolo_fold_1.csv')

path_save_frame = './sted_result/frame/'
path_save_video = './sted_result/video/'

# for index in range(len(pred['filename'])):
for index in range(1000):
    city = str(pred['filename'][index])
    video = str(pred['vid'][index])
    frame_num = str(pred['frame_num'][index])

    pred_x1 = pred['x1_60'][index]
    pred_y1 = pred['y1_60'][index]
    pred_x2 = pred['x2_60'][index]
    pred_y2 = pred['y2_60'][index]
    pred_center_x60, pred_center_y60 = int((pred_x1 + pred_x2) / 2), int(
        (pred_y1 + pred_y2) / 2)

    gt_x1 = gt['x1_60'][index]
    gt_y1 = gt['y1_60'][index]
    gt_x2 = gt['x2_60'][index]
    gt_y2 = gt['y2_60'][index]
    gt_center_x60, gt_center_y60 = int((gt_x1 + gt_x2) / 2), int(
        (gt_y1 + gt_y2) / 2)

    x1 = pred['x1_1'][index]
    y1 = pred['y1_1'][index]
    x2 = pred['x2_1'][index]
    y2 = pred['y2_1'][index]
    pred_center_x1, pred_center_y1 = int((x1 + x2) / 2), int((y1 + y2) / 2)

    x1 = gt['x1_1'][index]
    y1 = gt['y1_1'][index]
    x2 = gt['x2_1'][index]
    y2 = gt['y2_1'][index]
    gt_center_x1, gt_center_y1 = int((x1 + x2) / 2), int((y1 + y2) / 2)

    img = cv2.imread('/home/wangsen/ws/video2frame/' + city + '/' + video +
                     '/' + frame_num + '.png')
    cv2.circle(img, (pred_center_x1, pred_center_y1), 5, (0, 255, 0), -1)
    # cv2.circle(img, (gt_center_x1, gt_center_y1), 5, (255, 0, 0), -1)

    cv2.line(img, (pred_center_x1, pred_center_y1),
             (pred_center_x60, pred_center_y60), (0, 255, 0), 1)
    cv2.line(img, (gt_center_x1, gt_center_y1), (gt_center_x60, gt_center_y60),
             (255, 0, 0), 1)

    # 预测
    cv2.line(img, (pred_x1, pred_y1), (pred_x2, pred_y1), (0, 255, 0), 3)
    cv2.line(img, (pred_x1, pred_y1), (pred_x1, pred_y2), (0, 255, 0), 3)
    cv2.line(img, (pred_x2, pred_y1), (pred_x2, pred_y2), (0, 255, 0), 3)
    cv2.line(img, (pred_x1, pred_y2), (pred_x2, pred_y2), (0, 255, 0), 3)

    # gt
    cv2.line(img, (gt_x1, gt_y1), (gt_x2, gt_y1), (255, 0, 0), 3)
    cv2.line(img, (gt_x1, gt_y1), (gt_x1, gt_y2), (255, 0, 0), 3)
    cv2.line(img, (gt_x2, gt_y1), (gt_x2, gt_y2), (255, 0, 0), 3)
    cv2.line(img, (gt_x1, gt_y2), (gt_x2, gt_y2), (255, 0, 0), 3)

    # cv2.putText(img, str(index), (pred_x1, pred_y1 + 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # cv2.putText(img, str(index), (gt_x1, gt_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.75, (0, 0, 255), 2)

    cv2.imshow('', img)
    cv2.waitKey(1)
    print('Processing draw: ', city, '/', video, ' --- [ ', index, ' / ',
          len(pred['filename']), ' ]')

    # frame_city
    city_dir_1 = ''.join([path_save_frame, city])
    if not os.path.exists(city_dir_1):
        os.mkdir(city_dir_1)
    # frame_video
    video_dir_1 = ''.join([city_dir_1, '/', video])
    if not os.path.exists(video_dir_1):
        os.mkdir(video_dir_1)

    cv2.imwrite(
        '/home/wangsen/ws/video2frame/' + city + '/' + video + '/' +
        frame_num + '.png', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def read_video(path, res):
    file_list = os.listdir(path)
    for filename in file_list:
        tmp_path = os.path.join(path, filename)
        if tmp_path[-4:] == '.mp4':
            res.append(tmp_path)
        elif os.path.isdir(tmp_path):
            # print("目录：", filename)
            read_video(tmp_path, res)


'''
    根据上面图片结果
    保存视频
'''
root_path = '/home/wangsen/ws/video2frame/'
result_path = []
read_video(root_path, result_path)
for i in result_path:
    city = i.split('/')[-2]
    video = i.split('/')[-1]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video_city
    city_dir_2 = ''.join([path_save_video, city])
    if not os.path.exists(city_dir_2):
        os.mkdir(city_dir_2)
    # # video_video
    # video_dir_2 = ''.join([city_dir_2, '/', video])
    # if not os.path.exists(video_dir_2):
    #     os.mkdir(video_dir_2)

    out = cv2.VideoWriter(city_dir_2 + '/' + city + '_' + video[0:-4] + '.avi',
                          fourcc, 30, (1280, 720))

    for idx in range(600):
        img = cv2.imread('/home/wangsen/ws/video2frame/' + city + '/' + video +
                         '/' + str(idx) + '.png')
        out.write(img)
        cv2.imshow('', img)
        cv2.waitKey(1)
        print('Processing: ', city, '/', video, ' --- ', idx)
    out.release()
    # pd.DataFrame(val_feature_video_path).to_csv('./data/val_feature_video_path.csv')
