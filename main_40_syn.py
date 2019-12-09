# This project is to classify AB-40 and Alpha-syn

# Start from 2019/11/20
# Yunyi Qiu (York) y63qiu@edu.uwaterloo.ca

# %% import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms as T
from torch.utils.data import SubsetRandomSampler as SRS
from torch.utils.data import DataLoader
from data.Dataset import Proteins
from models import resnet101
from sklearn.model_selection import train_test_split
from config import DefaultConfig
import time
import copy


# %% Functions
def return_labels_idx(labels):
    return [i for i in range(labels.shape[0]) if labels[i] == 'AB-40' or labels[i] == 'Alpha-syn']

def imshow(img):
    img = img/2 + 0.5   # unnormlize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# %% load the data
dbtpath = "E:\\Polarimetry Database\\Pure Proteins\\dbt_633nm.csv"
dbt_633nm = pd.read_csv(dbtpath)
# %% prepare the label
label_633nm = dbt_633nm['Subject']
AB40_SYN_idx = return_labels_idx(label_633nm)
label_AB40_SYN = [label_633nm[i] for i in AB40_SYN_idx]

# %% load the data
opt = DefaultConfig()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

all_dataset = Proteins(root=opt.root,transforms=transform)

# split training and test set 7:3
train_idx, test_idx = train_test_split(
    np.arange(len(all_dataset.labels)),
    test_size=0.3,
    stratify=all_dataset.labels
)

train_sampler = SRS(train_idx)
test_sampler = SRS(test_idx)
train_loader = DataLoader(all_dataset, batch_size=opt.batch_size, sampler=train_sampler)
test_loader = DataLoader(all_dataset, batch_size=opt.batch_size, sampler=test_sampler)



# %% Training
def train_model(model, criterion, optimizer, scheduler, num_epochs=25,train_loader=train_loader):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()       # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) # 1 is the dimension
                loss = criterion(outputs, labels)

                # backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

        epoch_loss = running_loss/train_loader.sampler.indices.shape[0]
        epoch_acc = running_corrects.double()/train_loader.sampler.indices.shape[0]

        print('{} Loss: {: .4f} Acc: {: .4f}'.format(
            'Train', epoch_loss, epoch_acc
        ))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))

    # load best model weights
    return model

#%% Visualizing the model predictions
#   Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %% Finetuning the network
# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_AB40_SYN = resnet101(pretrained=True)
# The size of each output sample is set to 2
num_in_ft = model_AB40_SYN.fc.in_features
model_AB40_SYN.fc = nn.Linear(num_in_ft, 2)
model_AB40_SYN.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_AB40_SYN = torch.optim.SGD(model_AB40_SYN.parameters(),
                                     lr=0.01, momentum=0.9)

scheduler = lr_scheduler.StepLR(optimizer_AB40_SYN,
                                step_size=1, gamma=0.8)

resnet101_AB40_SYN = train_model(model_AB40_SYN,criterion,
                                       optimizer_AB40_SYN,scheduler,
                                       num_epochs=30)

#%% Test the network
test_iter = iter(test_loader)
imagesaa, labelsaa = test_iter.next()

# print images
imshow(torchvision.utils.make_grid(imagesaa))

def test(test_loader=test_loader, model=resnet101_AB40_SYN):

    test_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # 1 is the dimension

        test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double()/test_loader.sampler.indices.shape[0]

    return test_corrects, test_acc

