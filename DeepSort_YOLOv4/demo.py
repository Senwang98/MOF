#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

warnings.filterwarnings('ignore')

"""
    整体的时间复杂度与空间复杂度并没有做优化，所有citywalks数据集需要在内存8G以上的pc上运行
    TODO：一边跟踪一边保存数据，但是npy文件是否支持追加暂不清楚。但一定程度上将大大减少空间复杂度。
"""

# 数据集的分组，train、val、test
# Fold 1
Train_cities1 = ['BARCELONA', 'BRNO', 'ERFURT', 'KAUNAS', 'LEIPZIG',
                 'NUREMBERG', 'PALMA', 'PRAGUE', 'TALLINN', 'TARTU', 'VILNIUS', 'WEIMAR']
Validation_cities1 = ['DRESDEN', 'HELSINKI',
                      'PADUA', 'POZNAN', 'VERONA', 'WARSAW']
Test_cities1 = ['KRAKOW', 'RIGA', 'WROCLAW']

# Fold 2
Train_cities2 = ['BARCELONA', 'DRESDEN', 'ERFURT', 'HELSINKI', 'KRAKOW', 'LEIPZIG',
                 'PADUA', 'PALMA', 'POZNAN', 'RIGA', 'TALLINN', 'VERONA', 'VILNIUS', 'WARSAW', 'WROCLAW']
Validation_cities2 = ['KAUNAS', 'PRAGUE', 'WEIMAR']
Test_cities2 = ['BRNO', 'NUREMBERG', 'TARTU']

# Fold 3
Train_cities3 = ['BRNO', 'DRESDEN', 'HELSINKI', 'KAUNAS', 'KRAKOW', 'NUREMBERG',
                 'PADUA', 'POZNAN', 'PRAGUE', 'RIGA', 'TARTU', 'VERONA', 'WARSAW', 'WEIMAR', 'WROCLAW']
Validation_cities3 = ['BARCELONA', 'LEIPZIG', 'TALLINN']
Test_cities3 = ['ERFURT', 'PALMA', 'VILNIUS']


def read_video(path, res):
    """
        读取文件夹下的所有文件，此处仅仅考虑了只有视频的情况，TODO：后期可以修改适配视频类型
    """
    file_list = os.listdir(path)
    for filename in file_list:
        tmp_path = os.path.join(path, filename)
        if os.path.isdir(tmp_path):
            # print("目录：", filename)
            read_video(tmp_path, res)
        else:
            res.append(tmp_path)
            # print("普通文件", filename)


def train_val_test_box(result):
    """
        跟踪数据分割与保存
    """
    train_box1, train_box2, train_box3 = [], [], []
    val_box1, val_box2, val_box3 = [], [], []
    test_box1, test_box2, test_box3 = [], [], []

    # 所有数据保存（暂不需要）
    # dtp_box_statistics = []

    for i in range(len(result)):
        if result[i][9] == '1':
            track_object = []
            for j in range(24, -1, -1):
                tmp = []
                x = eval(result[i - j][4])
                y = eval(result[i - j][5])
                w = eval(result[i - j][6])
                h = eval(result[i - j][7])
                vx = (eval(result[i - j][4]) - eval(result[i - j - 5][4])) / 5
                vy = (eval(result[i - j][5]) - eval(result[i - j - 5][5])) / 5
                # 这里我计算是相邻帧的位置变化，而不是5帧平均变化
                delta_w = (eval(result[i - j][6]) - eval(result[i - j - 1][6]))
                delta_h = (eval(result[i - j][7]) - eval(result[i - j - 1][7]))
                tmp.append(x)
                tmp.append(y)
                tmp.append(w)
                tmp.append(h)
                tmp.append(vx)
                tmp.append(vy)
                tmp.append(delta_w)
                tmp.append(delta_h)
                track_object.append(tmp)
            # 数据 ”对号入座“
            # train
            if result[i][1] in Train_cities1:
                train_box1.append(track_object)

            if result[i][1] in Train_cities2:
                train_box2.append(track_object)

            if result[i][1] in Train_cities3:
                train_box3.append(track_object)

            # val
            if result[i][1] in Validation_cities1:
                val_box1.append(track_object)

            if result[i][1] in Validation_cities2:
                val_box2.append(track_object)

            if result[i][1] in Validation_cities3:
                val_box3.append(track_object)

            # test
            if result[i][1] in Test_cities1:
                test_box1.append(track_object)

            if result[i][1] in Test_cities2:
                test_box2.append(track_object)

            if result[i][1] in Test_cities3:
                test_box3.append(track_object)

            #dtp_box_statistics.append(track_object)

    # (n, 25, 8) -> (n, 8, 25)
    # # ans = np.array(dtp_box_statistics).transpose(0, 2, 1)
    train1 = np.array(train_box1).transpose(0, 2, 1)
    train2 = np.array(train_box2).transpose(0, 2, 1)
    train3 = np.array(train_box3).transpose(0, 2, 1)

    val1 = np.array(val_box1).transpose(0, 2, 1)
    val2 = np.array(val_box2).transpose(0, 2, 1)
    val3 = np.array(val_box3).transpose(0, 2, 1)

    test1 = np.array(test_box1).transpose(0, 2, 1)
    test2 = np.array(test_box2).transpose(0, 2, 1)
    test3 = np.array(test_box3).transpose(0, 2, 1)

    # 所有numpy数据保存本地
    # np.save('./dtp_box_statistics.npy', ans)
    np.save('./clip_data/fold_1_train_dtp_box_statistics.npy', train1)
    np.save('./clip_data/fold_2_train_dtp_box_statistics.npy', train2)
    np.save('./clip_data/fold_3_train_dtp_box_statistics.npy', train3)
    np.save('./clip_data/fold_1_val_dtp_box_statistics.npy', val1)
    np.save('./clip_data/fold_2_val_dtp_box_statistics.npy', val2)
    np.save('./clip_data/fold_3_val_dtp_box_statistics.npy', val3)
    np.save('./clip_data/fold_1_test_dtp_box_statistics.npy', test1)
    np.save('./clip_data/fold_2_test_dtp_box_statistics.npy', test2)
    np.save('./clip_data/fold_3_test_dtp_box_statistics.npy', test3)

    # 输出size确认
    print("All box.npy size print: ")
    # print("All data size: ", ans.shape)
    print("train1 box data size: ", train1.shape)
    print("train2 box data size: ", train2.shape)
    print("train3 box data size: ", train3.shape)
    print("val1 box data size: ", val1.shape)
    print("val2 box data size: ", val2.shape)
    print("val3 box data size: ", val3.shape)
    print("test1 box data size: ", test1.shape)
    print("test2 box data size: ", test2.shape)
    print("test3 box data size: ", test3.shape)


def train_val_test_label(result):
    """
        STED 训练时的标签，每一帧4个数，预测60帧，size = (n, 240)
        根据跟踪结果保存STED输出的结果作为label
    """
    train_label1, train_label2, train_label3 = [], [], []
    val_label1, val_label2, val_label3 = [], [], []
    test_label1, test_label2, test_label3 = [], [], []

    for i in range(len(result)):
        # 如果track_label = 1, 则向后检索60帧, 处理成label
        if result[i][9] == '1':
            # 计算后面60帧与当前的所有变化误差
            tmp_output = []
            for j in range(1, 61):
                xi, xj = eval(result[i][4]), eval(result[i + j][4])
                yi, yj = eval(result[i][5]), eval(result[i + j][5])
                wi, wj = eval(result[i][6]), eval(result[i + j][6])
                hi, hj = eval(result[i][7]), eval(result[i + j][7])
                delta_x, delta_y, delta_w, delta_h = xj-xi, yj-yi, wj-wi, hj-hi
                tmp_output.append(delta_x)
                tmp_output.append(delta_y)
                tmp_output.append(delta_w)
                tmp_output.append(delta_h)

            # train
            if result[i][1] in Train_cities1:
                train_label1.append(tmp_output)

            if result[i][1] in Train_cities2:
                train_label2.append(tmp_output)

            if result[i][1] in Train_cities3:
                train_label3.append(tmp_output)

            # val
            if result[i][1] in Validation_cities1:
                val_label1.append(tmp_output)

            if result[i][1] in Validation_cities2:
                val_label2.append(tmp_output)

            if result[i][1] in Validation_cities3:
                val_label3.append(tmp_output)

            # test
            if result[i][1] in Test_cities1:
                test_label1.append(tmp_output)

            if result[i][1] in Test_cities2:
                test_label2.append(tmp_output)

            if result[i][1] in Test_cities3:
                test_label3.append(tmp_output)

    # list -> numpy.ndarray
    train1 = np.array(train_label1)
    train2 = np.array(train_label2)
    train3 = np.array(train_label3)
    val1 = np.array(val_label1)
    val2 = np.array(val_label2)
    val3 = np.array(val_label3)
    test1 = np.array(test_label1)
    test2 = np.array(test_label2)
    test3 = np.array(test_label3)

    # 所有numpy数据保存本地
    np.save('./clip_data/fold_1_train_dtp_targets.npy', train1)
    np.save('./clip_data/fold_2_train_dtp_targets.npy', train2)
    np.save('./clip_data/fold_3_train_dtp_targets.npy', train3)
    np.save('./clip_data/fold_1_val_dtp_targets.npy', val1)
    np.save('./clip_data/fold_2_val_dtp_targets.npy', val2)
    np.save('./clip_data/fold_3_val_dtp_targets.npy', val3)
    np.save('./clip_data/fold_1_test_dtp_targets.npy', test1)
    np.save('./clip_data/fold_2_test_dtp_targets.npy', test2)
    np.save('./clip_data/fold_3_test_dtp_targets.npy', test3)

    # 输出size确认
    print("All label.npy size print: ")
    print("train1 label data size: ", train1.shape)
    print("train2 label data size: ", train2.shape)
    print("train3 label data size: ", train3.shape)
    print("val1 label data size: ", val1.shape)
    print("val2 label data size: ", val2.shape)
    print("val3 label data size: ", val3.shape)
    print("test1 label data size: ", test1.shape)
    print("test2 label data size: ", test2.shape)
    print("test3 label data size: ", test3.shape)


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    frame_wid = 1280
    frame_hei = 720

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False

    # 存放所有视频数据的上级目录
    #root_video_path = '/home/yhzn/ws/clip/'
    root_video_path = '/home/yhzn/ws/citywalks/'
    all_video_path = []
    read_video(root_video_path, all_video_path)

    break_flag = 0
    result = []
    myvideo_result = []
    print("Video number = ", len(all_video_path))
    just_video_cnt = 0

    # 依次便利所有视频的路径并读取，each_video代表的是str类型路径
    for each_video in all_video_path:

        tracker = Tracker(metric)

        file_path = each_video
        # 将城市名与视频编号识别出来
        video_name_split = file_path.split('/')
        city_name = video_name_split[-2]
        video_number = video_name_split[-1]
        print(city_name, video_number)

        if asyncVideo_flag:
            video_capture = VideoCaptureAsync(file_path)
        else:
            video_capture = cv2.VideoCapture(file_path)

        if asyncVideo_flag:
            video_capture.start()

        if writeVideo_flag:
            if asyncVideo_flag:
                w = int(video_capture.cap.get(3))
                h = int(video_capture.cap.get(4))
            else:
                w = int(video_capture.get(3))
                h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
            frame_index = -1

        fps = 0.0
        fps_imutils = imutils.video.FPS().start()

        frame_num = 0

        # 确认保存路径的存在
        city_dir = ''.join(['/home/yhzn/ws/crop_image/', city_name])
        if not os.path.exists(city_dir):
            os.mkdir(city_dir)
        video_dir = ''.join(
            ['/home/yhzn/ws/crop_image/', city_name, '/', video_number])
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)

        # 计数功能，仅仅为了显示处理进度用
        just_frame_cnt = 0
        just_video_cnt += 1
        while True:
            ret, frame = video_capture.read()
            if ret != True:
                break

            t1 = time.time()

            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxes, confidence, classes = yolo.detect_image(image)

            features = encoder(frame, boxes)
            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.cls for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # 检测框是否显示
            # for det in detections:
            #     bbox = det.to_tlbr()
            #     if show_detections and len(classes) > 0:
            #         det_cls = det.cls
            #         score = "%.2f" % (det.confidence * 100) + "%"
            #         cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
            #                     1e-3 * frame.shape[0], (0, 0, 255), 1)
            #         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            # 仅仅使用了跟踪框
            tmp_frame = frame.copy()

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()

                # print("ord1: ", bbox[0], bbox[1], bbox[2], bbox[3])
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[0] >= frame_wid:
                    bbox[0] = frame_wid - 1
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[1] >= frame_hei:
                    bbox[1] = frame_hei - 1
                if bbox[2] < 0:
                    bbox[2] = 0
                if bbox[2] >= frame_wid:
                    bbox[2] = frame_wid - 1
                if bbox[3] < 0:
                    bbox[3] = 0
                if bbox[3] >= frame_hei:
                    bbox[3] = frame_hei - 1
                # print("ord2: ", bbox[0], bbox[1], bbox[2], bbox[3])

                # if int(bbox[0]) < 0 or int(bbox[1]) < 0 or int(bbox[2]) < 0 or int(bbox[3]) < 0:
                #     continue
                
                # 行人图像的裁剪
                crop_image = frame[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])]  # 裁剪
                crop_image = cv2.resize(crop_image, (128, 256))
                cv2.imwrite(video_dir + '/frame_' + str(frame_num).zfill(4) + '_ped_' + str(track.track_id) + '.png',
                            crop_image)

                # Average detection confidence
                adc = "%.2f" % (track.adc * 100) + "%"
                cv2.rectangle(tmp_frame, (int(bbox[0]), int(
                    bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(tmp_frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1e-3 * frame.shape[0], (0, 0, 255), 1)

                if not show_detections:
                    track_cls = track.cls
                    cv2.putText(tmp_frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0],
                                (0, 255, 0),
                                1)
                    cv2.putText(tmp_frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                                1e-3 * frame.shape[0], (0, 255, 0), 1)

                cx = int((int(bbox[0]) + int(bbox[2])) / 2)
                cy = int((int(bbox[1]) + int(bbox[3])) / 2)
                w = int(int(bbox[2]) - int(bbox[0]))
                h = int(int(bbox[3]) - int(bbox[1]))
                
                

                # 下面所有的append操作是为了保存所有跟踪到的结果，方便后需存放npy文件
                tmp_result = []
                tmp_result.append(video_number)
                tmp_result.append(city_name)
                tmp_result.append(str(frame_num))
                tmp_result.append(str(track.track_id))
                tmp_result.append(str(cx))
                tmp_result.append(str(cy))
                tmp_result.append(str(w))
                tmp_result.append(str(h))
                tmp_result.append('0')
                tmp_result.append('0')
                tmp_result.append('0')
                if h>50:
                    result.append(tmp_result)

                # my video detection
                # 这个文件是保存成DTP项目所需要的格式，与上面的保存大致类似，测试用
                tmp_myvideo = []
                tmp_myvideo.append(city_name + '/' + video_number)
                tmp_myvideo.append(str(frame_num))
                tmp_myvideo.append(int(bbox[0]))
                tmp_myvideo.append(int(bbox[1]))
                tmp_myvideo.append(int(bbox[2]))
                tmp_myvideo.append(int(bbox[3]))
                # track
                tmp_myvideo.append(str(track.track_id))
                # detection length
                tmp_myvideo.append(str(0))
                # Height
                tmp_myvideo.append(str(h))
                if h>50:
                    myvideo_result.append(tmp_myvideo)

            frame_num += 1

            # cv2.imshow('', tmp_frame)

            if writeVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1

            fps_imutils.update()

            # 添加一些提示信息
            just_frame_cnt += 1
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("[%d / %d]" %
                  (just_video_cnt, len(all_video_path)), end=' ')
            print("[frame: %d] [fps: %f]" %
                  (just_frame_cnt, (fps + (1. / (time.time() - t1))) / 2))

            # 显示FPS
            # if not asyncVideo_flag:
            #     fps = (fps + (1. / (time.time() - t1))) / 2
            #     print("FPS = %f" % (fps))

            # Press Q to stop!
            #if cv2.waitKey(1) & 0xFF == ord('q'):
             #   break_flag = 1
              #  break
        #if break_flag == 1:
         #   break

    '''
        带label的跟踪结果保存
    '''
    f = open('./clip_data/tracking_result.csv', 'w')
    csv_writer = csv.writer(f)
    # 写入csv第一行
    csv_writer.writerow(["vid", "filename", "frame_num", "track", "cx", "cy", "w", "h", "track_length", "labeled",
                         "requires_features"])
    # 按照行人序号、城市名的顺序排序，帧数自动排序了
    result.sort(key=lambda x: (x[1], eval(x[3])))

    # 处理track条目
    for i in range(len(result)):
        if i == 0:
            pass
        else:
            if result[i][3] == result[i - 1][3] and eval(result[i][2]) == eval(result[i - 1][2]) + 1:
                result[i][8] = str(eval(result[i - 1][8]) + 1)
            else:
                result[i][8] = '0'

    # 处理labeled and require_feature条目
    for i in range(len(result)):
        # 不满足过去30帧、并且存在未来60帧的不进行预测，因为数据帧不足
        if i <= 28 or i >= len(result) - 60:
            pass
            # csv_writer.writerow(result[i])
        else:
            track_index_now = eval(result[i][8])
            track_index_pre29 = eval(result[i - 29][8])
            track_index_post60 = eval(result[i + 60][8])
            if result[i][3] == result[i - 29][3] and result[i][3] == result[i + 60][
                    3] and track_index_now == track_index_pre29 + 29 and track_index_now == track_index_post60 - 60:
                result[i][9] = '1'
                result[i][10] = '1'
                for j in range(i - 29, i):
                    result[j][10] = '1'
            # csv_writer.writerow(result[i])

    for i in range(len(result)):
        csv_writer.writerow(result[i])

    # 写DTP的tracking result csv
    train_file1 = open('./clip_data/myvideo_yolo_detection_train1.csv', 'w')
    train_file2 = open('./clip_data/myvideo_yolo_detection_train2.csv', 'w')
    train_file3 = open('./clip_data/myvideo_yolo_detection_train3.csv', 'w')
    val_file1 = open('./clip_data/myvideo_yolo_detection_val1.csv', 'w')
    val_file2 = open('./clip_data/myvideo_yolo_detection_val2.csv', 'w')
    val_file3 = open('./clip_data/myvideo_yolo_detection_val3.csv', 'w')
    test_file1 = open('./clip_data/myvideo_yolo_detection_test1.csv', 'w')
    test_file2 = open('./clip_data/myvideo_yolo_detection_test2.csv', 'w')
    test_file3 = open('./clip_data/myvideo_yolo_detection_test3.csv', 'w')

    # 定义csv_writer
    csv_train1 = csv.writer(train_file1)
    csv_train1.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])
    csv_train2 = csv.writer(train_file2)
    csv_train2.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])
    csv_train3 = csv.writer(train_file3)
    csv_train3.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])

    csv_val1 = csv.writer(val_file1)
    csv_val1.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])
    csv_val2 = csv.writer(val_file2)
    csv_val2.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])
    csv_val3 = csv.writer(val_file3)
    csv_val3.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])

    csv_test1 = csv.writer(test_file1)
    csv_test1.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])
    csv_test2 = csv.writer(test_file2)
    csv_test2.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])
    csv_test3 = csv.writer(test_file3)
    csv_test3.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track","detection_length", "Height"])

    for i in range(len(myvideo_result)):
        tmp_path = myvideo_result[i][0]
        city = tmp_path.split('/')[0]
        # train
        if city in Train_cities1:
            csv_train1.writerow(myvideo_result[i])
        if city in Train_cities2:
            csv_train2.writerow(myvideo_result[i])
        if city in Train_cities3:
            csv_train3.writerow(myvideo_result[i])

        # val
        if city in Validation_cities1:
            csv_val1.writerow(myvideo_result[i])
        if city in Validation_cities2:
            csv_val2.writerow(myvideo_result[i])
        if city in Validation_cities3:
            csv_val3.writerow(myvideo_result[i])

        # test
        if city in Test_cities1:
            csv_test1.writerow(myvideo_result[i])
        if city in Test_cities2:
            csv_test2.writerow(myvideo_result[i])
        if city in Test_cities3:
            csv_test3.writerow(myvideo_result[i])

    f.close()
    train_file1.close()
    train_file2.close()
    train_file3.close()
    val_file1.close()
    val_file2.close()
    val_file3.close()
    test_file1.close()
    test_file2.close()
    test_file3.close()

    '''
        写入STED模型所需要的.npy文件
        size = (n, 8, 25)
    '''
    # box numpy数据保存
    train_val_test_box(result)
    # output label数据保存
    train_val_test_label(result)

    """
        my video detection write
        保存成DTP需要的跟踪格式
    """
    f = open('./clip_data/myvideo_yolo_detection.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["filename", "frame_num", "bb1", "bb2",
                         "bb3", "bb4", "track", "detection_length", "Height"])
    # 按照跟踪序号排序
    myvideo_result.sort(key=lambda x: x[6])

    # 处理track条目
    for i in range(len(myvideo_result)):
        if i == 0:
            pass
        else:
            # 如何filename相同，track序号相同，帧数连续则表示跟踪
            if myvideo_result[i][6] == myvideo_result[i - 1][6] and eval(myvideo_result[i][1]) == eval(myvideo_result[i - 1][1]) + 1 and myvideo_result[i][0] == myvideo_result[i - 1][0]:
                myvideo_result[i][7] = str(eval(myvideo_result[i - 1][7]) + 1)
            else:
                myvideo_result[i][7] = '0'

    for i in range(len(myvideo_result)):
        csv_writer.writerow(myvideo_result[i])

    f.close()

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
