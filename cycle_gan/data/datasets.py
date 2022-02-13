from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as tt
import numpy as np
np.random.seed(seed=12)

import os
from PIL import Image


class ImgDataset(Dataset):
    def __init__(self, base_path, image_size=(100, 100), mode='train', name_A='A', name_B='B'):
        super().__init__()
        self.mode= mode
        self.a_files = glob(f'{base_path}/{mode}{name_A}/*.jpg')
        self.b_files = glob(f'{base_path}/{mode}{name_B}/*.jpg')
        
        self.a_size = len(self.a_files)
        self.b_size = len(self.b_files)
        self.max_size = max(self.a_size, self.b_size)
        
        if self.a_size < self.max_size:
            a_idx = np.arange(self.a_size)
            self.base_idx = np.hstack((a_idx, np.random.choice(a_idx, self.max_size - self.a_size)))
            self.second_idx = np.arange(self.max_size)
        else:
            b_idx = np.arange(self.b_size)
            self.base_idx = np.arange(self.a_size)
            self.second_idx = np.hstack((b_idx, np.random.choice(b_idx, self.max_size - self.b_size)))
            
        self.transforms = tt.Compose([
                              tt.Resize(image_size),
                              tt.ToTensor(),
#                               tt.Normalize(*stats),
        ])
        
    def __getitem__(self, idx):
        A_img = Image.open(self.a_files[self.base_idx[idx]]).convert(mode='RGB')
        A_img = self.transforms(A_img)
        
        if self.mode == 'train':
            B_img = Image.open(np.random.choice(self.b_files)).convert(mode='RGB')
        else:
            B_img = Image.open(self.b_files[self.second_idx[idx]]).convert(mode='RGB')

        B_img = self.transforms(B_img)

        
        return {'x': A_img, 'y': B_img}
       
    def __len__(self):
        return self.max_size