# This project is to classify AB-40 and Alpha-syn

# Start from 2019/11/20
# Yunyi Qiu (York) y63qiu@edu.uwaterloo.ca

# %% import the packages
import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.io import loadmat
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# %% Functions
def return_labels_idx(labels):
    return [i for i in range(labels.shape[0]) if labels[i] == 'AB-40' or labels[i] == 'Alpha-syn']


# %% load the data
dbtpath = "E:\\Polarimetry Database\\Pure Proteins\\dbt_633nm.csv"
dbt_633nm = pd.read_csv(dbtpath)
# %% prepare the label
label_633nm = dbt_633nm['Subject']
AB40_SYN_idx = return_labels_idx(label_633nm)
label_AB40_SYN = [label_633nm[i] for i in AB40_SYN_idx]

# %% load the data


# %% Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models import resnet101
model_AB40_SYN = resnet101(pretrained=False)

# %% Training