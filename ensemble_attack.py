import argparse
import os
import sys
import queue
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
from utils.functions import *
sys.path.append('inversion/baselines/')
from inversion.baselines.classify import IR152, VGG16
from PIL import Image
from torchvision import transforms, utils
from tensorboard_logger import Logger
from tqdm import tqdm
from trainer0 import *
import sys
sys.path.append('pixel2style2pixel/')
from pixel2style2pixel.models.stylegan2.model import Generator, get_keys
from torch.autograd import Variable
from scipy.optimize import fsolve
# from trainer_old import Trainer as oldTrainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--pretrained_model_path', type=int, default='./logs_celeba-vgg/001/enc30.pth.tar', help='pretrained stylegan2 model')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan2 model')
parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained arcface model')
parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
parser.add_argument('--input', type=str, default='./data/celeba_selectnpy-VGG16', help='input path')
parser.add_argument('--seed', type=int, default=38, help='seed')
opts = parser.parse_args()

config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)

# Initialize trainer
trainer = Trainer(config, opts)
trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)
trainer.to(device)

trainer.enc.load_state_dict(torch.load(opts.pretrained_model_path),strict=False)
trainer.enc.eval()

StyleGAN = trainer.StyleGAN
StyleGAN_state_dict = torch.load(opts.stylegan_model_path, map_location='cpu')
StyleGAN.load_state_dict(get_keys(StyleGAN_state_dict, 'decoder'), strict=True)
StyleGAN.to(device)

dlatent_avg = trainer.dlatent_avg

all_path = [[] for i in range(1000)]
for (root, dirs, files) in sorted(os.walk(opts.input)):
    for file in files:
        all_path[int(file.strip().split('_')[0])].append(os.path.join(root,file))
        
w = [0 for i in range(1000)]
img = 0
with torch.no_grad():
    for i in tqdm(range(1000)):
        bs = len(all_path[i])
        div = 0
        all_path[i].sort(reverse=True)
        for j in range(bs):
            latent = np.load(all_path[i][j])
            latent = torch.from_numpy(latent).to(device)
            # -------------------------------------------------------------------------
            if latent[0][i]<0.965:
                max = latent[0][i] + 0.035
                latent = latent - 0.035*(latent/(1-latent[0][i]))
                latent[0][i] = max
                latent=torch.log(latent)
            else:
                max = latent[0][i]
                latent = torch.log(latent)
            # -------------------------------------------------------------------------
            w_recon, fea, outimg = trainer.enc(latent)
            w_recon = w_recon + dlatent_avg

            w[i] += float(max) * w_recon
            div += float(max)

        w[i] = w[i]/div


torch.manual_seed(opts.seed)
with torch.no_grad():
    for i in tqdm(range(1000)):
        features = None
        x_1_recon, fea_recon = StyleGAN([w[i]], input_is_latent=True, return_features=True, features_in=features, feature_scale=min(1.0, 0.0001 * 1e5))
        save_dir = './output/'
        os.makedirs(save_dir, exist_ok=True)
        utils.save_image(clip_img(x_1_recon), save_dir + "{}.png".format(i))

