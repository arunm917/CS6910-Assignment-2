import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import copy

wandb.login()

train_path = '/home/asus/arun_folder/inaturalist_12K/train'
test_path = '/home/asus/arun_folder/inaturalist_12K/val'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#### Preprocessing #####

# Image transforms
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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
torch.manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
test_dataset = ImageFolder(test_path, transform=test_transforms)

dataset = DataLoader(train_dataset, shuffle=True, num_workers = 20)
dataiter = iter(dataset)
images, labels = next(dataiter)
image_dim = images.shape[2]

class CNN(nn.Module):
  def __init__(self, architecture, batch_norm, num_layers, num_filters, conv_filter_size, dropout, activation, dense_neurons):
    super(CNN, self).__init__()
    
    self.layers = []
    self.architecture = architecture
    self.batch_norm = batch_norm
    self.num_conv_layers = num_layers
    self.num_filters = num_filters
    self.conv_filter_size = conv_filter_size
    self.dropout = dropout
    self.dense_neurons = dense_neurons
    self.input_filters = 3
    self.maxpool_filter_size = 2

    if activation == 'ReLU':
      self.activation = nn.ReLU
    if activation == 'GELU':
      self.activation = nn.GELU
    if activation == "SiLU":
      self.activation = nn.SiLU
    if activation == "Mish":
      self.activation = nn.Mish
    if activation == 'LeakyReLU':
      self.activation = nn.LeakyReLU
    if activation == 'ELU':
      self.activation = nn.ELU

    if self.batch_norm == 'YES':
      if self.architecture == 'DOUBLE':
        self.num_filters = 16
        for i in range(self.num_conv_layers):
          self.layers.append(nn.Conv2d(self.input_filters,self.num_filters, self.conv_filter_size))
          self.layers.append(nn.BatchNorm2d(self.num_filters))
          self.layers.append(self.activation())
          self.layers.append(nn.MaxPool2d(self.maxpool_filter_size, stride = 2))
          self.layers.append(nn.Dropout(self.dropout))
          self.input_filters = self.num_filters
          self.num_filters = int(self.num_filters*2)
        
      if self.architecture == 'HALF':
        self.num_filters = 128
        for i in range(self.num_conv_layers):
          self.layers.append(nn.Conv2d(self.input_filters,self.num_filters, self.conv_filter_size))
          self.layers.append(nn.BatchNorm2d(self.num_filters))
          self.layers.append(self.activation())
          self.layers.append(nn.MaxPool2d(self.maxpool_filter_size, stride = 2))
          self.layers.append(nn.Dropout(self.dropout))
          self.input_filters = self.num_filters
          self.num_filters = int(self.num_filters/2)
      
      if self.architecture == 'EQUAL':
        for i in range(self.num_conv_layers):
          self.layers.append(nn.Conv2d(self.input_filters,self.num_filters, self.conv_filter_size))
          self.layers.append(nn.BatchNorm2d(self.num_filters))
          self.layers.append(self.activation())
          self.layers.append(nn.MaxPool2d(self.maxpool_filter_size, stride = 2))
          self.layers.append(nn.Dropout(self.dropout))
          self.input_filters = self.num_filters
          
    if self.batch_norm == 'NO':
      if self.architecture == 'DOUBLE':
        self.num_filters = 16
        for i in range(self.num_conv_layers):
          self.layers.append(nn.Conv2d(self.input_filters,self.num_filters, self.conv_filter_size))
          self.layers.append(self.activation())
          self.layers.append(nn.MaxPool2d(self.maxpool_filter_size, stride = 2))
          self.layers.append(nn.Dropout(self.dropout))
          self.input_filters = self.num_filters
          self.num_filters = int(self.num_filters*2)
        
      if self.architecture == 'HALF':
        self.num_filters = 128
        for i in range(self.num_conv_layers):
          self.layers.append(nn.Conv2d(self.input_filters,self.num_filters, self.conv_filter_size))
          self.layers.append(self.activation())
          self.layers.append(nn.MaxPool2d(self.maxpool_filter_size, stride = 2))
          self.layers.append(nn.Dropout(self.dropout))
          self.input_filters = self.num_filters
          self.num_filters = int(self.num_filters/2)
      
      if self.architecture == 'EQUAL':
        for i in range(self.num_conv_layers):
          self.layers.append(nn.Conv2d(self.input_filters,self.num_filters, self.conv_filter_size))
          self.layers.append(self.activation())
          self.layers.append(nn.MaxPool2d(self.maxpool_filter_size, stride = 2))
          self.layers.append(nn.Dropout(self.dropout))
          self.input_filters = self.num_filters

   # Construct sequential module
    self.cnn_model = nn.Sequential()
    for i, layer in enumerate(self.layers):
      self.cnn_model.add_module(str(i), layer)
  
    output_dim = self.compute_output_dim(self.num_conv_layers, self.conv_filter_size, self.maxpool_filter_size)
    input_dense = (output_dim**2)*self.input_filters

    self.fc_model = nn.Sequential(
        nn.Linear(input_dense, self.dense_neurons),
        self.activation(),
        nn.Linear(self.dense_neurons,10)
    )
  
  def compute_output_dim(self, num_conv_layers, conv_filter_size, maxpool_filter_size):
    input_size = image_dim
    for i in range (num_conv_layers):
      conv_output_dim = (input_size - conv_filter_size) + 1
      maxpool_output_dim = np.floor((conv_output_dim - maxpool_filter_size)/2) + 1
      input_size = maxpool_output_dim
    return int(maxpool_output_dim)

  def forward(self, x):
    x = self.cnn_model(x)
    x = x.view(x.size(0), -1)
    x = self.fc_model(x)
    return(x)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers = 20)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers = 20)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers = 20)

sweep_configuration = {
  'method': 'grid',
  'name': 'sweep',
  'metric': {'goal': 'maximize', 'name': 'accuracy_val'},
  'parameters': {
      'epochs':{'values':[15]},
      'architecture':{'values':['DOUBLE']},
      'batch_norm':{'values':['YES']},
      'num_layers': {'values': [5]},
      'num_filters': {'values': [32]},
      'conv_filter_size': {'values': [3]},
      'dropout': {'values': [0.1]},
      'activation':{'values':['GELU']},
      'dense_neurons': {'values': [256]},
      'learning_rate': {'values': [1e-3]},
      'weight_decay': {'values': [0.5]},
      'optimizer': {'values': ['AdamW']},
    } }

def wandbsweeps():
  wandb.init(project = 'CS6910_Assignment_2')
  wandb.run.name = (
        "bn"
        + str(wandb.config.batch_norm)
        + "nf"
        + str(wandb.config.num_filters)
        + "fs"
        + str(wandb.config.conv_filter_size)
        + "do"
        + str(wandb.config.dropout)
        + "lr"
        + str(wandb.config.learning_rate)
        + "opt"
        + wandb.config.optimizer
        + "af"
        + str(wandb.config.activation)
    )
  model = CNN(
    architecture= wandb.config.architecture,
    batch_norm = wandb.config.batch_norm,
    num_layers = wandb.config.num_layers,
    num_filters = wandb.config.num_filters, 
    conv_filter_size = wandb.config.conv_filter_size,
    dropout = wandb.config.dropout, 
    activation = wandb.config.activation,
    dense_neurons = wandb.config.dense_neurons).to(device)

  ## Loss and optimizer ##  
  loss_cr = nn.CrossEntropyLoss()
  if wandb.config.optimizer == 'Adam':
    la = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = CosineAnnealingLR(la, T_max=wandb.config.epochs/2, eta_min = 0.0001)
    
  elif wandb.config.optimizer == 'NAdam':
    la = optim.NAdam(model.parameters(), lr=wandb.config.learning_rate, weight_decay= wandb.config.weight_decay)
    scheduler = CosineAnnealingLR(la, T_max=wandb.config.epochs/2, eta_min = 0.0001)
    
  elif wandb.config.optimizer == 'RAdam':
    la = optim.RAdam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = CosineAnnealingLR(la, T_max=wandb.config.epochs/2, eta_min = 0.0001)
    
  elif wandb.config.optimizer == 'AdamW':
    la = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = CosineAnnealingLR(la, T_max=wandb.config.epochs/2, eta_min = 0.0001)
    
  elif wandb.config.optimizer == 'SGD':
    la = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = CosineAnnealingLR(la, T_max=wandb.config.epochs/2, eta_min = 0.0001)
    
  else:
      raise ValueError("Invalid optimizer type.")
  def accuracy(dataloader):
      total, correct_predictions = 0, 0
      for data in dataloader:
        X, Y = data
        X, Y = X.to(device), Y.to(device)
        Y_pred = model(X)
        _, pred = torch.max(Y_pred.data, 1)
        total += Y.size(0)
        correct_predictions += (pred == Y).sum().item()
        accuracy = (correct_predictions/total)*100
      return accuracy
    
  loss_batch = []
  loss_epoch = []
  epochs = wandb.config.epochs
  for i in tqdm(range(epochs)):
    epoch = i+1
    for j, data in enumerate(train_loader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        la.zero_grad()
        y_pred = model(images)
        loss = loss_cr(y_pred, labels)
        loss.backward()
        la.step()
        loss_batch.append(loss.item())
    
    scheduler.step()   
    loss_epoch.append(loss.item())
    accuracy_val = accuracy(val_loader)
    accuracy_train = accuracy(train_loader)
    print('training loop')     
    print('Validation acc: %0.2f, Train acc: %0.2f, Test acc: %0.2f' % (accuracy_val, accuracy_train))
    wandb.log({'loss_epoch': loss_epoch, 'accuracy_val':accuracy_val, 'accuracy_training':accuracy_train, 'epoch':epoch})
  wandb.finish()

sweep_id = wandb.sweep(sweep= sweep_configuration, project = 'CS6910_Assignment_2')
wandb.agent(sweep_id, function = wandbsweeps)
