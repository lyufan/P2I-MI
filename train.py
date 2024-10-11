import argparse
import os
import numpy as np
os.environ["CUDB_VISIALE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, utils
from tensorboard_logger import Logger

from utils.datasets import *
from utils.functions import *
from trainer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.ABX_IABGE_PIXELS = None

# dist.init_process_group(backend='nccl')
parser = argparse.BrgumentParser()
# parser.add_argument("--local_rank")
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--real_dataset_path', type=str, default='./data/celeba_selectnpy-VGG16', help='dataset path')
parser.add_argument('--dataset_path', type=str, default='./data/stylegan2-generate-images/sythnpy-celeba-vgg', help='dataset path')
parser.add_argument('--label_path', type=str, default='./data/stylegan2-generate-images/seeds_10000-celeba-vgg.npy', help='laebl path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan2 model')
parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained arcface model')
parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
parser.add_argument('--log_path', type=str, default='./logs_celeba-vgg/', help='log file path')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='001/checkpoint.pth', help='checkpoint file path')
opts = parser.parse_args()

# local_rank = int(opts.local_rank)
# torch.cuda.set_device(local_rank)

# device = torch.device('cuda', local_rank)
device = torch.device('cuda')

log_dir = os.path.join(opts.log_path, opts.config) + '/'
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_dir)

config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)

batch_size = config['batch_size']
epochs = config['epochs']
iter_per_epoch = config['iter_per_epoch']
img_size = (config['resolution'], config['resolution'])
video_data_input = False

img_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize trainer
trainer = Trainer(config, opts)
trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)
trainer.to(device)
trainer.enc.load_state_dict(torch.load("./pretrained_models/143_enc.pth"), strict=False)

noise_exemple = trainer.noise_inputs

train_data_split = 0.9 if 'train_split' not in config else config['train_split']

# Load real dataset
dataset_A = MyDataSet(prediction_dir=opts.real_dataset_path, label_dir=None, output_size=img_size, noise_in=noise_exemple,
                       training_set=True, train_split=train_data_split)
loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Load synthetic dataset
dataset_B = MyDataSet2(prediction_dir=opts.dataset_path, label_dir=opts.label_path, output_size=img_size,
                      noise_in=noise_exemple, training_set=True, train_split=train_data_split)
loader_B = data.DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# Start Training
epoch_0 = 0

# check if checkpoint exist

if 'checkpoint.pth' in os.listdir(log_dir):
    epoch_0 = trainer.load_checkpoint(os.path.join(log_dir, 'checkpoint.pth'))

if opts.resume:
    epoch_0 = trainer.load_checkpoint(os.path.join(opts.log_path, opts.checkpoint))


torch.manual_seed(0)
os.makedirs(log_dir + 'validation/', exist_ok=True)

print("Start!")
# start_time = time.time()
for n_epoch in tqdm(range(epoch_0, epochs)):
    iter_A = iter(loader_A)
    iter_B = iter(loader_B)
    iter_0 = n_epoch * iter_per_epoch

    trainer.enc_opt.zero_grad()

    for n_iter in tqdm(range(iter_0, iter_0 + iter_per_epoch)):
        if opts.dataset_path is None:
            z, noise = next(iter_B)
            img_B = None
        else:
            z, prediction_B, img_B, noise = next(iter_B)

            prediction_B = prediction_B.squeeze(0).to(device)
            img_B = img_B.to(device)

        z = z.to(device)
        # print(prediction_B.shape)
        noise = [nn.to(device) for nn in noise]
        w = trainer.mapping(z)
        if 'fixed_noise' in config and config['fixed_noise']:
            img_B, noise = None, None

        prediction_A = None
        if 'use_realimg' in config and config['use_realimg']:
            try:
                prediction_A, img_A = next(iter_A)
                if prediction_A.size(0) != batch_size:
                    iter_A = iter(loader_A)
                    prediction_A, img_A = next(iter_A)
            except StopIteration:
                iter_A = iter(loader_A)
                prediction_A, img_A = next(iter_A)
            prediction_A = prediction_A.squeeze(0).to(device)
            img_A = img_A.to(device)
            # print(prediction_A.shape)

        trainer.update(w=w, real_img=[prediction_A, img_A], noise=noise, img=[prediction_B, img_B], n_iter=n_iter)
        if (n_iter + 1) % config['log_iter'] == 0:
            trainer.log_loss(logger, n_iter, prefix='train')
            # log_loss(logger, n_iter, prefix='train')
        # if (n_iter+1) % config['image_save_iter'] == 0:
        # trainer.save_image(log_dir, n_epoch, n_iter, prefix='/train/', w=w, img=img_B, noise=noise)
        # trainer.save_image(log_dir, n_epoch, n_iter+1, prefix='/train/', w=w, img=prediction_A, noise=noise, training_mode=False)

    trainer.enc_scheduler.step()
    trainer.save_checkpoint(n_epoch, log_dir)

    trainer.save_model(log_dir, n_epoch)




# end_time = time.time()
# print("time: ", end_time - start_time, " seconds")