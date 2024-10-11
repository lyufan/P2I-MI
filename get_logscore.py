import numpy as np
import os
import torch
import torch.nn.functional as F
# np.set_printoptions(suppress=True)
# torch.set_printoptions(precision=4,sci_mode=False)
savedir = "./data/public_select/celeba_selectnpy-log-VGG16"
os.makedirs(savedir, exist_ok=True)
count = 0
for (root, dirs, files) in sorted(os.walk("./data/public_select/celeba_selectnpy-VGG16")):

    for file in files:
        data = np.load(os.path.join(root,file))
        np.save(os.path.join(savedir, file), torch.log(torch.tensor(data)))