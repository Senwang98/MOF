import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import copy
import argparse
import pandas as pd
import numpy as np
import math
from PIL import Image
import scipy.ndimage
import sys
import time
from torchsummary import summary
import os

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import model_utils as utils
import video_transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_modified_resnet(NUM_FLOW_FRAMES):
    model = models.resnet50(pretrained=True)

    # Reshape resnet
    model = model.apply(utils.freeze_bn)
    model.bn1.train(True)

    pretrained_weights = model.conv1.weight
    avg_weights = torch.mean(pretrained_weights, 1)
    avg_weights = avg_weights.expand(NUM_FLOW_FRAMES * 2, -1, -1, -1)
    avg_weights = avg_weights.permute(1, 0, 2, 3)
    model.conv1 = nn.Conv2d(NUM_FLOW_FRAMES * 2,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3)
    model.conv1.weight.data = avg_weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)

    return model


class DynamicTrajectoryPredictor(nn.Module):
    def __init__(self, NUM_FLOW_FRAMES):
        super(DynamicTrajectoryPredictor, self).__init__()
        self.resnet50 = get_modified_resnet(NUM_FLOW_FRAMES)

    def forward(self, flow):
        out = self.resnet50(flow)
        return out


class LocationDatasetBDD(Dataset):
    def __init__(self,
                 filename,
                 root_dir,
                 img_root,
                 transform,
                 NUM_FLOW_FRAMES,
                 proportion=100):
        """
        Args:
            filename (string): Pkl file name with data. This must contain the
            optical flow image filename and the label.
            root_dir (string): Path to directory with the pkl file.
            proportion(int): Proportion of dataset to use for training
                            (up to 100, which is 100 percent of the dataset)
        """
        np.random.seed(seed=26)  # Set seed for reproducability
        self.df = pd.read_pickle(root_dir + filename)
        print('Loaded data from ', root_dir + filename)
        unique_filenames = self.df['filename'].unique()
        np.random.shuffle(unique_filenames)
        unique_filenames = unique_filenames[
            0:int(len(unique_filenames) * proportion / 100)]
        self.df = self.df[self.df['filename'].isin(unique_filenames)]
        self.df = self.df.reset_index()
        self.transform = transform
        self.img_root = img_root
        self.NUM_FLOW_FRAMES = NUM_FLOW_FRAMES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        NUM_FLOW_FRAMES = self.NUM_FLOW_FRAMES

        dir_name = self.df.loc[idx, 'filename']
        track = self.df.loc[idx, 'track']

        # Frame number is part of the filename
        frame_num = int(self.df.loc[idx, 'frame_num'])
        print('Output: ', dir_name, '/', frame_num, '/', track)
        flow_stack = np.zeros((256, 256, NUM_FLOW_FRAMES * 2)).astype('uint8')

        # Read in the optical flow images
        for frame in range(frame_num - (NUM_FLOW_FRAMES - 1) * 3,
                           frame_num + 1, 3):
            frame_name = dir_name + '/frame_' + \
                str(frame).zfill(4) + '_ped_' + str(int(track)) + '.png'
            print('Flow dir = ', frame_name)
            img_name_hor = str(self.img_root + 'horizontal/' + frame_name)
            img_name_ver = str(self.img_root + 'vertical/' + frame_name)

            try:
                hor_flow = Image.open(img_name_hor).resize((256, 256))
                ver_flow = Image.open(img_name_ver).resize((256, 256))
            except:
                print('Error: file not loaded. Could not find image file:')
                print(img_name_hor)
                print('Line = ', idx)
                hor_flow = np.zeros((256, 256))
                ver_flow = np.zeros((256, 256))

            flow_stack[:, :,
                       int((frame - (frame_num - (NUM_FLOW_FRAMES - 1) * 3)) //
                           3 * 2)] = hor_flow
            flow_stack[:, :,
                       int((frame - (frame_num -
                                     (NUM_FLOW_FRAMES - 1) * 3)) // 3 * 2 +
                           1)] = ver_flow

        flow_stack = self.transform(flow_stack)

        sample = {'flow_stack': flow_stack}
        return sample


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DynamicTrajectoryPredictor(9).to(device)
    model = model.float()
    model = nn.DataParallel(model)
    # 输出网络结构
    # summary(model, input_size=(18, 224, 224))
    model.load_state_dict(torch.load('./model.weights'), False)

    load_path = './data_inference/'
    img_root = '../../../flow_result/'

    # Training settings
    epochs = 15
    batch_size = 1
    learning_rate = 1e-5
    num_workers = 8
    weight_decay = 1e-2
    NUM_FLOW_FRAMES = 9
    training_proportion = 100

    # Transformers
    transform_val = video_transforms.Compose([
        video_transforms.Scale((224)),
        video_transforms.ToTensor(),
    ])

    valset = LocationDatasetBDD(filename='myvideo_val_yolo_0.pkl',
                                root_dir=load_path,
                                transform=transform_val,
                                img_root=img_root,
                                NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    # for param in model.parameters():
    #     param.requires_grad = False

    model.eval()
    tmp_result = []
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            if batch_idx % 100 == 0:
                end_time = time.time()
                print(' Batch ', batch_idx, ' of ', len(val_loader),
                      ' Cost time: ', end_time - start_time)
                start_time = end_time

            flow = data['flow_stack'].to(device)
            flow = flow.float()
            output = model(flow).detach().cpu().numpy()
            tmp_result.append(output)
            # print(output.shape)
            # if batch_idx == 1:
            #     print(batch_idx, ' : \n', output)
            # print(batch_idx, ' : \n', output)
            if batch_idx == 0:
                break

    ans = np.array(tmp_result).reshape(-1, 120)
    print(ans)
    print(ans.shape)

    # with open('record_extract.txt', 'w') as f:
    #     f.write(fold_type + ' ' + str(fold_num) + ' ' + str(ans.shape))

    # np.save('./data_inference/model_val_inference_result.npy', ans)
    np.save('./data_inference/val_prediction.npy', ans)


if __name__ == '__main__':
    main()
