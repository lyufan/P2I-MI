import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torchvision import transforms, utils

class MyDataSet(data.Dataset):
    def __init__(self, prediction_dir=None, label_dir=None, output_size=(256, 256), noise_in=None, training_set=True, video_data=False, train_split=0.9):
        self.prediction_dir = prediction_dir
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # crop_size = 704
        # offset_height = (1024 - crop_size) // 2
        # offset_width = (1024 - crop_size) // 2
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        self.resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])
        self.noise_in = noise_in
        self.video_data = video_data
        self.random_rotation = transforms.Compose([
            transforms.Resize(output_size),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.ToTensor()
        ])

        # load image file
        train_len = None
        self.length = 0
        self.prediction_dir = prediction_dir
        if prediction_dir is not None:
            # img_list = [glob.glob1(self.prediction_dir, ext) for ext in ['*jpg','*png']]
            prediction_list = [glob.glob1(self.prediction_dir, ext) for ext in ['*npy']]
            prediction_list = [item for sublist in prediction_list for item in sublist]
            prediction_list.sort()
            train_len = int(train_split*len(prediction_list))
            self.prediction_list = prediction_list
            # if training_set:
            #     self.image_list = image_list[:train_len]
            # else:
            #     self.image_list = image_list[train_len:]
            self.length = len(self.prediction_list)

        # load label file
        self.label_dir = label_dir
        if label_dir is not None:
            self.seeds = np.load(label_dir)
            if train_len is None:
                train_len = int(train_split*len(self.seeds))
            # if training_set:
            #     self.seeds = self.seeds[:train_len]
            # else:
            #     self.seeds = self.seeds[train_len:]
            if self.length == 0:
                self.length = len(self.seeds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        prediction = None
        if self.prediction_dir is not None:
            prediction_name = os.path.join(self.prediction_dir, self.prediction_list[idx])
            imgname = os.path.join("./data/celeba_select-VGG16", self.prediction_list[idx].strip().split('.n')[0]+".png")
            image = Image.open(imgname)
            prediction = np.load(prediction_name)
            # print(image.shape)
            prediction = torch.from_numpy(prediction)
            # img = torch.tensor(image)
            img = self.resize(image)
            if img.size(0) == 1:
                # print(img.size(0))
                # print(img.shape)
                img = torch.cat((img, img, img), dim=0)
                # print(img.shape)
            img = self.normalize(img)

        # generate image
        if self.label_dir is not None:
            torch.manual_seed(self.seeds[idx])
            z = torch.randn(1, 512)[0]
            if self.noise_in is None:
                n = [torch.randn(1, 1)]
            else:
                n = [torch.randn(noise.size())[0] for noise in self.noise_in]
            if prediction is None:
                return z, n
            else:
                return z, prediction, n
        else:
            return prediction, img


class MyDataSet2(data.Dataset):
    def __init__(self, prediction_dir=None, label_dir=None, output_size=(256, 256), noise_in=None, training_set=True, video_data=False, train_split=0.9):
        self.prediction_dir = prediction_dir
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        crop_size1 = 704  # 568
        crop_size2 = 704  # 644
        offset_height = (1024 - crop_size1) // 2
        offset_width = (1024 - crop_size2) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size1, offset_width:offset_width + crop_size2]
        # crop_size = 108
        # offset_height = (218 - crop_size) // 2
        # offset_width = (178 - crop_size) // 2
        # crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        self.resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])
        self.noise_in = noise_in
        self.video_data = video_data
        self.random_rotation = transforms.Compose([
            transforms.Resize(output_size),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.ToTensor()
        ])

        # load image file
        train_len = None
        self.length = 0
        self.prediction_dir = prediction_dir
        if prediction_dir is not None:
            # img_list = [glob.glob1(self.prediction_dir, ext) for ext in ['*jpg','*png']]
            prediction_list = [glob.glob1(self.prediction_dir, ext) for ext in ['*npy']]
            prediction_list = [item for sublist in prediction_list for item in sublist]
            prediction_list.sort()
            train_len = int(train_split*len(prediction_list))
            self.prediction_list = prediction_list
            self.length = len(self.prediction_list)

        # load label file
        self.label_dir = label_dir
        if label_dir is not None:
            self.seeds = np.load(label_dir)
            if train_len is None:
                train_len = int(train_split*len(self.seeds))
            if self.length == 0:
                self.length = len(self.seeds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        prediction = None
        if self.prediction_dir is not None:
            prediction = os.path.join(self.prediction_dir, self.prediction_list[idx])
            imgname = os.path.join("./data/stylegan2-generate-images/ims10000-celeba-vgg", self.prediction_list[idx].strip().split('.n')[0] + ".jpg")
            image = Image.open(imgname)
            prediction = np.load(prediction)
            # print(image.shape)
            prediction = torch.from_numpy(prediction)
            # print(img.shape)
            img = self.resize(image)
            if img.size(0) == 1:
                # print(img.size(0))
                # print(img.shape)
                img = torch.cat((img, img, img), dim=0)
                # print(img.shape)
            img = self.normalize(img)

        # generate image
        if self.label_dir is not None:
            torch.manual_seed(self.seeds[idx])
            z = torch.randn(1, 512)[0]
            if self.noise_in is None:
                n = [torch.randn(1, 1)]
            else:
                n = [torch.randn(noise.size())[0] for noise in self.noise_in]
            if prediction is None:
                return z, n
            else:
                return z, prediction, img, n
        else:
            return prediction

