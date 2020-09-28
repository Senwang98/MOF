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

warnings.filterwarnings('ignore')


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = '/media/wangsen/新加卷/MOF_data/citywalks/clips/PADUA/clip_000001.mp4'

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
    result = []
    ped_cnt = 0
    ped_index_store = []
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
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
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # for det in detections:
        #     bbox = det.to_tlbr()
        #     if show_detections and len(classes) > 0:
        #         det_cls = det.cls
        #         score = "%.2f" % (det.confidence * 100) + "%"
        #         cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
        #                     1e-3 * frame.shape[0], (0, 255, 0), 1)
        #         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            if int(bbox[0]) < 0 or int(bbox[1]) < 0 or int(bbox[2]) < 0 or int(bbox[3]) < 0:
                continue
            # 行人图像的裁剪
            crop_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # 裁剪
            crop_image = cv2.resize(crop_image, (128, 256))
            cv2.imwrite('./crop_image/frame_' + str(frame_num).zfill(4) + '_ped_' + str(track.track_id) + '.png', crop_image)

            adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 1)

            if not show_detections:
                track_cls = track.cls
                cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (0, 255, 0),
                            1)
                cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

            w = int(int(bbox[2]) - int(bbox[0]))
            h = int(int(bbox[3]) - int(bbox[1]))
            tmp_result = []
            # filename
            tmp_result.append("clip_000001.mp4")
            # frame_num
            tmp_result.append(str(frame_num))
            # bb1~bb4
            tmp_result.append(int(bbox[0]))
            tmp_result.append(int(bbox[1]))
            tmp_result.append(int(bbox[2]))
            tmp_result.append(int(bbox[3]))
            # track
            tmp_result.append(str(track.track_id))
            # detection length
            tmp_result.append(str(0))
            # Height
            tmp_result.append(str(h))
            result.append(tmp_result)
        frame_num += 1

        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 正式写入
    f = open('myvideo_yolo_detection.csv', 'w')
    csv_writer = csv.writer(f)
    # 写入头
    csv_writer.writerow(["filename", "frame_num", "bb1", "bb2", "bb3", "bb4", "track", "detection_length", "Height"])
    # 按照跟踪序号排序
    result.sort(key=lambda x: x[6])

    # Item handle: track
    for i in range(len(result)):
        if i == 0:
            pass
        else:
            # 如何filename相同，track序号相同，帧数连续则表示跟踪
            if result[i][6] == result[i - 1][6] and eval(result[i][1]) == eval(result[i - 1][1]) + 1 and result[i][0] == \
                    result[i - 1][0]:
                result[i][7] = str(eval(result[i - 1][7]) + 1)
            else:
                result[i][7] = '0'

    for i in range(len(result)):
        csv_writer.writerow(result[i])

    f.close()

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
