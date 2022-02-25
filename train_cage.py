import torch
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import numpy as np
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
from IPython.display import clear_output

from cycle_gan.model.models import CycleGan
from cycle_gan.data.datasets import ImgDataset
from cycle_gan.utils import show_image, show_images, show_epoch_res


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


train_dataset = ImgDataset('datasets/selfie_dataset/', mode='train', image_size=(256, 256))
test_dataset = ImgDataset('datasets/selfie_dataset/', mode='test', image_size=(256, 256))

train_dataLoader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3)
test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=3)

model = CycleGan(n_residual=9, 
                 device=device
                )

for epoch in tqdm(range(10)):
    print('start epoch: {epoch}')
    
    for data in tqdm(train_dataLoader):
        model.train_step(data)    
        
    for data in test_dataLoader:
        pred =  model.test_step(data)
        break
    
    # if epoch % 10 == 0:
        # torch.save(model.state_dict(), f'checkpoints/CycleGan_epoch_{epoch:03d}.pth')
    torch.save(model.state_dict(), f'checkpoints/last_CycleGan_cage.pth') 
    clear_output(wait=True)
    show_epoch_res(pred)  
    plt.savefig(f'val_img/cage/{epoch:03d}.jpg', pad_inches=0, transparent=True)
