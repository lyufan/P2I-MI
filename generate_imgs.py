import argparse
import os
import sys
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
sys.path.append('utils/')
from utils.functions import *
sys.path.append('inversion/')
from inversion.baselines.classify import *
from PIL import Image
from torchvision import transforms, utils
from tensorboard_logger import Logger
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('pixel2style2pixel/')
from pixel2style2pixel.models.stylegan2.model import Generator, get_keys

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./data/stylegan2-generate-images/', help='dataset path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan model')
parser.add_argument('--target_model_path', type=str, default='./inversion/checkpoints/target_model/VGG16_88.26.tar', help='pretrained stylegan model')
opts = parser.parse_args()


StyleGAN = Generator(1024, 512, 8)
state_dict = torch.load(opts.stylegan_model_path, map_location='cpu')
StyleGAN.load_state_dict(get_keys(state_dict, 'decoder'), strict=True)
StyleGAN.to(device)


targetmodel = VGG16(1000)
targetmodel = nn.DataParallel(targetmodel).to(device)
targetmodel.load_state_dict(torch.load(opts.target_model_path)['state_dict'], strict=False)
targetmodel.eval()


crop_size = 704
offset_height = (1024 - crop_size) // 2
offset_width = (1024 - crop_size) // 2
crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
img_transform = transforms.Compose([
    transforms.Lambda(crop),
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

seeds = np.load(opts.dataset_path + 'seeds_pytorch_1.8.1.npy')

seeds2 = []
count = 0
labels = [0 for i in range(1000)]
all_images_prob = None
all_images_path = []
all_seeds = []
if not os.path.exists(os.path.join(opts.dataset_path, 'all_images_prob-vgg.npy')):
    with torch.no_grad():
        for i, seed in enumerate(tqdm(seeds)):
            torch.manual_seed(seed)
            z = torch.randn(1, 512).to(device)
            n = StyleGAN.make_noise()
            w = StyleGAN.get_latent(z)
            x, _ = StyleGAN([w], input_is_latent=True, noise=n)
            x = clip_img(x)
            x_1 = img_transform(x.squeeze(0)).unsqueeze(0)
            _, output_t = targetmodel(x_1)
            prob = F.softmax(output_t, dim=1)
            prob = prob.cpu()
            if i == 0:
                all_images_prob = prob
                all_seeds.append(seed)

            else:
                all_images_prob = torch.cat([all_images_prob, prob], dim=0)
                all_seeds.append(seed)

    np.save(os.path.join(opts.dataset_path, 'all_images_prob-vgg.npy'), np.array(all_images_prob))
    np.save(os.path.join(opts.dataset_path, 'all_seeds-vgg.npy'), np.array(all_seeds))
    print(all_images_prob.shape)
print("=> start reclassify ...")
save_path = os.path.join(opts.dataset_path, 'ims10000-celeba-vgg')
os.makedirs(save_path, exist_ok=True)
npy_path = os.path.join(opts.dataset_path, 'sythnpy-celeba-vgg')
os.makedirs(npy_path, exist_ok=True)
print(" save_path: ", save_path)
all_images_prob = torch.tensor(np.load(os.path.join(opts.dataset_path, 'all_images_prob-vgg.npy')))
all_seeds = np.load(os.path.join(opts.dataset_path, 'all_seeds-vgg.npy')).tolist()
# top-n selection
for class_idx in tqdm(range(1000)):
    bs = all_images_prob.shape[0]
    ccc = 0
    # maintain a priority queue
    q = queue.PriorityQueue()
    class_idx_prob = all_images_prob[:, class_idx]

    for j in range(bs):
        current_value = float(class_idx_prob[j])

        current_seed =  all_seeds[j]
        if q.qsize() < 10:
            q.put([current_value, all_images_prob[j], current_seed])
        else:
            current_min = q.get()
            if current_value < current_min[0]:
                q.put(current_min)
            else:
                q.put([current_value, all_images_prob[j], current_seed])

    for m in range(q.qsize()):
        q_value = q.get()
        q_prob = round(q_value[0], 6)
        q_score = q_value[1].unsqueeze(0)
        q_seed = q_value[2]
        with torch.no_grad():
            torch.manual_seed(q_seed)
            z = torch.randn(1, 512).to(device)
            n = StyleGAN.make_noise()
            w = StyleGAN.get_latent(z)
            x, _ = StyleGAN([w], input_is_latent=True, noise=n)
            x = clip_img(x)
            new_image_path = os.path.join(save_path, str(class_idx) + '_' + str(q_prob) + '.jpg')
            utils.save_image(x, new_image_path)


        snpy_path = os.path.join(npy_path, str(class_idx) + '_' + str(q_prob) + '.npy')
        np.save(snpy_path, torch.log(torch.tensor(q_score)))

        seeds2.append(q_seed)

        ccc += 1



np.save(opts.dataset_path + 'seeds_10000-celeba-vgg.npy', np.array(seeds2))