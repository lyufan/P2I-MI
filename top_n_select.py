import os
import sys
import queue
import shutil
import torch
import torch.nn.functional as F
import torch.utils.data as data
from copy import deepcopy
import torchvision.transforms as transforms
from torchvision import utils as vutils
from PIL import Image
from argparse import ArgumentParser
from torch import nn
from tqdm import tqdm
import numpy as np
sys.path.append('inversion/')
from inversion.baselines.classify import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = ArgumentParser(description='Reclassify the public dataset with the target model')
parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
parser.add_argument('--data_name', type=str, default='celeba', help='celeba | ffhq | facescrub')
parser.add_argument('--top_n', type=int, default=10, help='the n of top-n selection strategy.')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--save_img_root', type=str, default='celeba_select-VGG16')
parser.add_argument('--save_npy_root', type=str, default='celeba_selectnpy-VGG16')

args = parser.parse_args()


class PublicFFHQ(torch.utils.data.Dataset):
    def __init__(self, root='./data/ffhq1024', transform=None):
        super(PublicFFHQ, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.path = self.root

        # num_classes = len([lists for lists in os.listdir(
        #     self.path) if os.path.isdir(os.path.join(self.path, lists))])

        # for idx in range(num_classes):
        #     class_path = os.path.join(self.path, str(idx * 1000).zfill(5))
        for _, _, files in os.walk(self.path):
            for img_name in files:
                self.images.append(os.path.join(self.path, img_name))

    def __getitem__(self, index):

        img_path = self.images[index]
        # print(img_path)
        img = Image.open(img_path)
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)


class PublicCeleba(torch.utils.data.Dataset):
    def __init__(self, file_path='./data/celeba_ganset.txt',
                 img_root='./data/img_align_celeba_png', transform=None):
        super(PublicCeleba, self).__init__()
        self.file_path = file_path
        self.img_root = img_root
        self.transform = transform
        self.images = []

        self.name_list, self.label_list = [], []

        f = open(self.file_path, "r")
        for line in f.readlines():
            img_name = line.strip()
            # img_name, iden = line.strip().split(' ')
            # self.label_list.append(int(iden))
            self.name_list.append(img_name)
            self.images.append(os.path.join(self.img_root, img_name))

    def __getitem__(self, index):
        # img_name = self.name_list[index]
        # img_path = os.path.join(self.img_root, img_name)
        # id = self.label_list[index]
        img_path = self.images[index]
        img = Image.open(img_path)
        if self.transform != None:
            img = self.transform(img)

        return img, img_path
        # return img, img_path, id

    def __len__(self):
        return len(self.images)
        # return len(self.name_list)


class PublicFaceScrub(torch.utils.data.Dataset):
    def __init__(self, img_root='./data/facescrub_public', transform=None):
        super(PublicFaceScrub, self).__init__()
        # self.file_path = file_path
        self.img_root = img_root
        self.transform = transform
        self.images = []

        name_list, label_list = [], []
        for foldername in os.listdir(self.img_root):
            FolderName = os.path.join(self.img_root, foldername)
            for filename in os.listdir(FolderName):
                self.images.append(os.path.join(FolderName, filename))


    def __getitem__(self, index):

        img_path = self.images[index]
        img = Image.open(img_path)
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)

def cutout(imgs, area):
    mask = get_cutout_mask(area)
    out = imgs - imgs * mask
    return out
def get_cutout_mask(area):
    mask = torch.zeros(64, 64).cuda().float()
    xmin, ymin = area[0]
    w, h = area[1]
    mask[xmin:xmin+w, ymin:ymin+h] = 1
    mask = mask.repeat(3, 1, 1)
    return mask


class FixedCutOut(nn.Module):
    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, img):
        out = cutout(img, self.area)
        return out


def define_trans():
    trans = []
    for i in range(8):
        for j in range(8):
            area = ((8 * i, 8 * j), (8, 8))
            trans.append(FixedCutOut(area))
    return trans



def top_n_selection(args, T, data_loader):
    """
    liuyufansb
    Top-n selection strategy.
    :param args: top-n, save_path
    :param T: target model
    :param data_loader: dataloader of
    :return:
    """
    print("=> start inference ...")
    save_img_root = args.save_img_root
    os.makedirs(args.save_img_root, exist_ok=True)
    os.makedirs(args.save_npy_root, exist_ok=True)
    all_images_prob = None
    all_images_path = None
    # get the predict confidence of each image in the public data
    with torch.no_grad():
        for i, (images, img_path) in enumerate(tqdm(data_loader)):
            bs = images.shape[0]
            images = images.cuda()
            logits = T(images)[-1]
            prob = F.softmax(logits, dim=1)  # (bs, 1000)
            prob = prob.cpu()
            if i == 0:
                all_images_prob = prob
                all_images_path = img_path
                # print(type(all_images_path))
            else:
                all_images_prob = torch.cat([all_images_prob, prob], dim=0)
                all_images_path = all_images_path + img_path
                # print(all_images_path)
            # snpy_path = os.path.join(args.save_root, str(os.path.basename(img_path[0]).strip().split('.')[0]) + '.npy')
            # np.save(snpy_path, prob.cpu().numpy())
    print("=> start reclassify ...")
    save_img_path = os.path.join("./data/public_select", args.save_img_root)
    save_npy_path = os.path.join("./data/public_select", args.save_npy_root)
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_npy_path, exist_ok=True)
    print(" top_n: ", args.top_n)
    print(" save_img_path: ", save_img_path)
    print(" save_npy_path: ", save_npy_path)

    # top-n selection
    for class_idx in range(args.num_classes):
        bs = all_images_prob.shape[0]
        ccc = 0
        # maintain a priority queue
        q = queue.PriorityQueue()
        class_idx_prob = all_images_prob[:, class_idx]

        for j in range(bs):
            current_value = float(class_idx_prob[j])

            image_path = all_images_path[j]
            # Maintain a priority queue with confidence as the priority
            if q.qsize() < args.top_n:
                q.put([current_value, image_path, all_images_prob[j]])
            else:
                current_min = q.get()
                if current_value < current_min[0]:
                    q.put(current_min)
                else:
                    q.put([current_value, image_path, all_images_prob[j]])
        # reclassify and move the images
        for m in range(q.qsize()):
            q_value = q.get()
            q_prob = round(q_value[0], 6)
            q_image_path = q_value[1]
            q_score = q_value[2].unsqueeze(0)
            # print(q_score.shape)
            # ori_save_path = os.path.join(save_path, str(class_idx))
            # if not os.path.exists(ori_save_path):
            #     os.makedirs(ori_save_path)
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)

            new_image_path = os.path.join(save_img_path, str(class_idx) + '_' + str(q_prob) + '.png')
            shutil.copy(q_image_path, new_image_path)

            snpy_path = os.path.join(save_npy_path, str(class_idx) + '_' + str(q_prob) + '.npy')
            np.save(snpy_path, torch.log(torch.tensor(q_score)))
            ccc += 1


print(args)
print("=> load target model ...")

model_name_T = args.model
if model_name_T.startswith("VGG16"):
    T = VGG16(1000)
    path_T = './inversion/checkpoints/target_model/VGG16_88.26.tar'
elif model_name_T.startswith('IR152'):
    T = IR152(1000)
    path_T = './inversion/checkpoints/target_model/IR152_91.16.tar'
elif model_name_T == "FaceNet64":
    T = FaceNet64(50)
    # T = FaceNet64(1000)
    # T = FaceNet64(200)
    path_T = './inversion/checkpoints/target_model/FaceNet64_96.00.tar'


T = torch.nn.DataParallel(T).cuda()
ckp_T = torch.load(path_T)
T.load_state_dict(ckp_T['state_dict'], strict=False)
T.eval()

print("=> load public dataset ...")
if args.data_name == 'celeba':
    re_size = 64
    crop_size = 108
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    celeba_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    data_set = PublicCeleba(file_path='./data/celeba_ganset.txt',
                            img_root='./data/img_align_celeba_png',
                            transform=celeba_transform)
    data_loader = data.DataLoader(data_set, batch_size=1)
elif args.data_name == 'ffhq':
    re_size = 64
    crop_size = 704
    offset_height = (1024 - crop_size) // 2
    offset_width = (1024 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    # crop = lambda x: x[:, 10:80, 15:85]
    ffhq_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    data_set = PublicFFHQ(root='./data/ffhq1024', transform=ffhq_transform)
    data_loader = data.DataLoader(data_set, batch_size=1, shuffle=False)
elif args.data_name == 'facescrub':
    re_size = 64

    faceScrub_transform = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    data_set = PublicFaceScrub(img_root='./data/facescrub_public', transform=faceScrub_transform)
    data_loader = data.DataLoader(data_set, batch_size=1, shuffle=False)

top_n_selection(args, T, data_loader)