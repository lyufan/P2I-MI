import numpy as np
import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import statistics
from sklearn.model_selection import train_test_split
from copy import deepcopy
sys.path.append('~/P2I_MI/inversion/baselines')
import classify
import dataloader as dataloader
import utils as utils
import shutil


device = "cuda"

def test(T, A, I, mse_loss, dataloader=None, device='cuda'):
    T.eval()
    A.eval()
    I.eval()
    loss, cnt, MSE, correct_top5 = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (img, iden, _) in enumerate(dataloader):
            img = img.to(device)
            iden = iden.to(device)
            out_te = T(img)
            out_te = out_te.view(out_te.size(0), -1)
            out_t = A(img)[-1]
            out = I(out_t)
            mseloss = mse_loss(out_te, out)
            print("mseloss: {}".format(mseloss.item()))
            MSE += mseloss.item()
    return MSE / (i+1)

def main(testloader):

    E = classify.FaceNet(50)
    # E = classify.FaceNet(200)
    # E = classify.FaceNet(1000)
    # E = classify.IR152(1000)
    # E = classify.VGG16(1000)
    E = torch.nn.DataParallel(E).cuda()
    path_E = '~/P2I_MI/inversion/checkpoints/evaluate_model/FaceNet_95.88.tar'

    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)
    E.eval()

    mse_loss = nn.MSELoss().cuda()
    res = []
    res5 = []
    res_mse = []
    success = []
    record = []
    all = []

    print("Start Attack!")
    with torch.no_grad():
        for batch_idx, (data, iden, name) in enumerate(testloader):
            data = data.to(device)
            batch_size = len(data)
            eval_prob = E(data)[-1]
            # eval_prob = E(utils.high2low(fake))[-1]
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            max, _ = torch.max(F.softmax(eval_prob, dim=1), dim=1)

            cnt, cnt5 = 0, 0
            for i in range(batch_size):
                gt = iden[i].item()
                if eval_iden[i].item() == gt:
                    success.append([name[i],max[i].item()])
                    cnt += 1
                    shutil.copy(os.path.join(args["dataset"]["img_path"],"{}.png".format(gt)), os.path.join("./inversion_success-celeba-vgg","{}.png".format(gt)))

                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1
            res.append(cnt * 1.0 / batch_size)
            res5.append(cnt5 * 1.0 / batch_size)
        success.sort()
        # all.sort()
        print(success)
        # print(all)
        acc, acc_5 = statistics.mean(res), statistics.mean(res5)
        acc_var = statistics.variance(res)
        acc_var5 = statistics.variance(res5)
        print("Acc:{:.3f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var,acc_var5))
        return acc, acc_5, acc_var, acc_var5





if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("---------------------attack---------------------")
    # os.makedirs("./inversion/data_files/test5", exist_ok=True)
    os.makedirs("./inversion_success-celeba-vgg", exist_ok=True)
    # test_file = "./inversion/data_files/pubfig.txt"
    # test_file = "./inversion/data_files/facescrub.txt"
    test_file = "./inversion/data_files/celeba.txt"
    args = {"dataset": {"name": "celeba", "img_path": "./output", "model_name": "FaceNet"}}
    # args = {"dataset": {"name": "ffhq", "img_path": "./output", "model_name": "FaceNet"}}

    _, testloader = utils.init_dataloader(args, test_file, 2, mode="style", iterator=False)

    main(args, testloader)
