import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class Proteins(data.Dataset):
    'Characterize a dataset for Pytorch'
    def __init__(self, root):
        'Initialization'
        imgs = os.listdir(root)
        #这里不实际加载图片，只是指定路径
        self.imgs = [os.path.join(root,img) for img in imgs]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)

    def __getitem__(self, index):
        'Generate one sample of data'
        # select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X,y

