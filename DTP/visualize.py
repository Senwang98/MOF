from cv2 import cv2
import numpy as np
import pandas as pd
import os

gt = np.load('./yolo_targets_rn50_flow_css_9stack_myvideo__shuffled_disp.npy')
gt = gt.reshape(-1, 60, 2)
# print(gt.shape)
# print(gt[0])
# print(gt[0][0])

# pred = np.load(
#     './yolo_predictions_rn50_flow_css_9stack_myvideo_training_proportion_100_shuffled_disp.npy'
# )
pred = np.load('./val_prediction.npy')
pred = pred.reshape(-1, 60, 2)
# print(pred.shape)
# print(pred[0])

val_feature = pd.read_csv('./data/val.csv')

# 保存所有的城市路径
# val_feature_video_path = {}
# tmp_filename = list(val_feature['filename'])
# val_feature_video_path['filename'] = []
# if len(tmp_filename) >=2:
#     val_feature_video_path['filename'].append(tmp_filename[0])
#     for i in range(1,len(tmp_filename)):
#         if tmp_filename[i-1] == tmp_filename[i]:
#             pass
#         else:
#             val_feature_video_path['filename'].append(tmp_filename[i])

# pd.DataFrame(val_feature_video_path).to_csv('./data/val_feature_video_path.csv')

path_save_frame = './result/frame/'
path_save_video = './result/video/'
for idx in range(len(val_feature)):
    index = idx
    city = str(val_feature['filename'][index]).split('/')[0]
    video = str(val_feature['filename'][index]).split('/')[1]
    # print(val_feature['filename'][index])
    # print(val_feature['frame_num'][index])
    # print(val_feature['track'][index])
    # print(val_feature['Predicted_x_seq'][index])
    # print(val_feature['Predicted_y_seq'][index])
    # print(val_feature['Mid_x'][index])
    # print(val_feature['Mid_y'][index])

    frame_num = str(val_feature['frame_num'][index])
    mid_x, mid_y = int(val_feature['Mid_x'][index]), int(
        val_feature['Mid_y'][index])
    cv_x = (val_feature['Predicted_x_seq'][index])
    cv_y = (val_feature['Predicted_y_seq'][index])
    cv_x, cv_y = cv_x[1:-1], cv_y[1:-1]
    cv_x, cv_y = cv_x.split(), cv_y.split()
    # print(len(cv_x))

    img = cv2.imread('/home/wangsen/ws/video2frame/' + city + '/' + video +
                     '/' + frame_num + '.png')

    cv2.circle(img, (mid_x, mid_y), 5, (0, 0, 255), -1)
    pre_gt_x, pre_gt_y = int(mid_x), int(mid_y)
    pre_pred_x, pre_pred_y = int(mid_x), int(mid_y)

    for i in range(0, len(cv_x), 1):
        tmpx = eval(cv_x[i])
        tmpy = eval(cv_y[i])
        gt_x = int(tmpx + gt[0][i][0])
        gt_y = int(tmpy + gt[0][i][1])
        pred_x = int(tmpx + pred[0][i][0])
        pred_y = int(tmpy + pred[0][i][1])
        cv2.line(img, (pre_gt_x, pre_gt_y), (gt_x, gt_y), (0, 255, 0), 3)
        cv2.line(img, (pre_pred_x, pre_pred_y), (pred_x, pred_y), (255, 0, 0),
                 3)
        pre_gt_x, pre_gt_y = gt_x, gt_y
        pre_pred_x, pre_pred_y = pred_x, pred_y
        # cv2.circle(img, (gt_x, gt_y), 3, (0, 255, 0), -1)  # 绿色gt
        # cv2.circle(img, (pred_x, pred_y), 3, (255, 0, 0), -1)  # 蓝色预测
        if i == len(cv_yolo_predictions_rn50_flow_css_9stack_myvideo_training_proportion_100_shuffled_dispx) - 1:
            # gt
            cv2.line(img, (gt_x, gt_y), (gt_x - 15, gt_y - 15), (0, 255, 0), 3)
            cv2.line(img, (gt_x, gt_y), (gt_x - 15, gt_y + 15), (0, 255, 0), 3)
            cv2.line(img, (gt_x, gt_y), (gt_x + 15, gt_y - 15), (0, 255, 0), 3)
            cv2.line(img, (gt_x, gt_y), (gt_x + 15, gt_y + 15), (0, 255, 0), 3)

            # pred
            cv2.line(img, (pred_x, pred_y), (pred_x - 15, pred_y - 15),
                     (255, 0, 0), 3)
            cv2.line(img, (pred_x, pred_y), (pred_x - 15, pred_y + 15),
                     (255, 0, 0), 3)
            cv2.line(img, (pred_x, pred_y), (pred_x + 15, pred_y - 15),
                     (255, 0, 0), 3)
            cv2.line(img, (pred_x, pred_y), (pred_x + 15, pred_y + 15),
                     (255, 0, 0), 3)

    cv2.imshow('', img)
    cv2.waitKey(30)
    print('Processing draw: ', city, '/', video, ' --- [ ', idx, ' / ',
          len(val_feature), ' ]')
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
        print('Processing: ', city, '/', video, ' --- ', idx)
    out.release()