import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from glob import glob
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setting the path for dataset
train_path = '/home/asus/arun_folder/inaturalist_12K/train'
test_path = '/home/asus/arun_folder/inaturalist_12K/val'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Image transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder(train_path, transform=train_transforms)
train_size = int(0.8*len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
test_dataset = ImageFolder(test_path, transform=test_transforms)

# Importing Resnet50 model
resnet = models.resnet50(pretrained = True)

# Setting preference to update parameters
for param in resnet.parameters():
    param.requires_grad = False

# Changing last layer
num_fltrs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_fltrs, 512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, 10),
)
# Setting loss and optimizer
resnet = resnet.to(device)
loss_function = nn.CrossEntropyLoss()
la = optim.AdamW(resnet.parameters(), lr=0.001, weight_decay = 0.2)
scheduler = StepLR(la, step_size = 5, gamma = 0.5)

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers = 20)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers = 20)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers = 20)

def accuracy(dataloader):
  total, correct_predictions = 0, 0
  for data in dataloader:
    X, Y = data
    X, Y = X.to(device), Y.to(device)
    Y_pred = resnet(X)
    _, pred = torch.max(Y_pred.data, 1)
    total += Y.size(0)
    correct_predictions += (pred == Y).sum().item()
    accuracy = (correct_predictions/total)*100
  return accuracy

loss_batch = []
loss_epoch = []
epochs = 15
for epoch in tqdm(range(epochs)):
  for i, data in enumerate(train_loader, 0):
      images, labels = data
      images, labels = images.to(device), labels.to(device)

      la.zero_grad()
      y_pred = resnet(images)

      loss = loss_function(y_pred, labels)
      loss.backward()
      la.step()
      loss_batch.append(loss.item())

  scheduler.step()
  loss_epoch.append(loss.item())
  accuracy_val = accuracy(val_loader)
  accuracy_train = accuracy(train_loader)
  accuracy_test = accuracy(test_loader)
  print('Status:')
  print('Train acc: %0.2f, Validation acc: %0.2f, Test acc: %0.2f' % (accuracy_train, accuracy_val, accuracy_test ))
