import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class Proteins(data.Dataset):
    'Characterize a dataset for Pytorch'
    def __init__(self, root, transforms=None):
        'Initialization'
        imgs = os.listdir(root)
        #这里不实际加载图片，只是指定路径
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms
        labels = []
        for i in range(len(self.imgs)):
            labels.append(1 if "AB-40" in imgs[i].split('/')[-1] else 0)
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)

    def __getitem__(self, index):
        'Generate one sample of data'
        img_path = self.imgs[index]
        # AB-40 -> "AB-40"; Alpha-syn -> "Alpha-syn"
        label = 1 if "AB-40" in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label




