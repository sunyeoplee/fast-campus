# title: image analysis
# author: Sun Yeop Lee

import os
from glob import glob

import numpy as np 

import matplotlib.pyplot as plt  
from PIL import Image

data_paths = glob('dataset/mnist_png/training/*/*.png')
len(data_paths)

path = data_paths[0]

label_nums = os.listdir('dataset/mnist_png/training')
len(label_nums)

nums_dataset = []

for lbl_n in label_nums:
    data_per_class = os.listdir('dataset/mnist_png/training/' + lbl_n)
    nums_dataset.append(len(data_per_class))

nums_dataset
plt.bar(label_nums, nums_dataset)
plt.show()

image_pil = Image.open(path)
image = np.array(image_pil)
image.shape
plt.imshow(image, 'gray')
plt.show()

path

def get_label(path):
    class_name = path.split('\\')[-2]
    label = int(class_name)
    return label

path, get_label(path)

heights = []
widths = []
for path in data_paths[:10]: # it takes too long so we just select the first 10
    img_pil = Image.open(path)
    image = np.array(img_pil)
    h, w = image.shape
    heights.append(h)
    widths.append(w)

np.unique(heights)
np.unique(widths)

len(heights)

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.hist(heights)
plt.title('Heights')
plt.show()



# pytorch: dataset loader - torchvision.ImageFolder
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

seed = 1 
lr = 0.001
momentum = 0.5 
batch_size = 64
test_batch_size = 64
epochs = 1
no_cuda = False
log_interval = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500,10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim =1)

os.listdir('dataset/mnist_png/training/')
train_dir = 'dataset/mnist_png/training'
test_dir = 'dataset/mnist_png/testing'

torch.manual_seed(seed)
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

train_dataset = datasets.ImageFolder(root = train_dir, # this does not work for grayscale
                                     transform=transforms.Compose([
                                         transforms.ToTensor(), 
                                         transforms.Normalize((0.1307,),(0.3081,))
                                     ]))

test_dataset = datasets.ImageFolder(root = test_dir, # this does not work for grayscale
                                     transform=transforms.Compose([
                                         transforms.ToTensor(), 
                                         transforms.Normalize((0.1307,),(0.3081,))
                                     ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

for epoch in range(1, epochs+1):
    model.train() 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clear gradients
        output = model(data) # enter the data into the model
        loss = F.nll_loss(output, target) # negative log likelihood loss
        loss.backward() # calculate gradients
        optimizer.step() # update parameters

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), # len(data) = batch_size, 
                100 * batch_idx / len(train_loader), loss.item()
            ))

    model.eval() # deactivate layers like batch normalization and dropout

    test_loss = 0
    correct = 0

    with torch.no_grad(): # turn off backpropagation, gradient calculation to increase speed and reduce memory usage
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # a class with the highest probability
            correct = pred.eq(target.view_as(pred)).sum().item() # reshape target as pred and check equality. 1 if correct, 0 if wrong. 

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))

# pytorch: dataset loader - custom dataset
import os 
from glob import glob 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

seed = 1 
lr = 0.001
momentum = 0.5
batch_size = 64
test_batch_size = 64
epochs = 1
no_cuda = False
log_interval = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500,10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2) # insert print(x.shape) after this line to check the shape of the output (64, 50, 5, 5). Then you can reshape to 5*5*50 in the next line.
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim =1)

os.chdir(r'C:\Users\USER\Desktop\Backup\Gap year\데이터 청년 캠퍼스\패스트 캠퍼스\올인원 패키지 - 딥러닝인공지능\[전체강의자료]-딥러닝-인공지능\Part3) 이미지 분석으로 배우는 tensorflow 2.0과 Pytorch')

train_paths = glob('dataset/cifar/train/*.png')
path = train_paths[0]
test_paths = glob('dataset/cifar/test/*.png')

def get_label1(path):
    return os.path.basename(path).split('_')[-1].replace('.png','')
label_names = [get_label1(path) for path in train_paths]

classes = np.unique(label_names)
label = np.argmax(classes == get_label1(path))

def get_label2(path):
    lbl_name = os.path.basename(path).split('_')[-1].replace('.png','')
    label = np.argmax(classes == lbl_name)
    return label

class Dataset(Dataset):
    def __init__(self, data_paths, transform=None):

        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        image = Image.open(path)
        label = get_label2(path)
        
        if self.transform:
            image = self.transform(image)

        return image, label

torch.manual_seed(seed)

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    Dataset(train_paths, 
            transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize(
                    mean=[0.406], 
                    std=[0.225])])
           ),
    batch_size=batch_size, 
    shuffle=True, 
    **kwargs
)

test_loader = torch.utils.data.DataLoader(
    Dataset(test_paths,
           transforms.Compose([
               transforms.ToTensor(), 
               transforms.Normalize(
                   mean=[0.406], 
                   std=[0.225])])
           ),
    batch_size=test_batch_size, 
    shuffle=False, 
    **kwargs
)
    
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)



for epoch in range(1, epochs + 1):
    # Train Mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # https://pytorch.org/docs/stable/nn.html#nll-loss
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # Test mode
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
# save model weights
save_path = 'model_weight.pt'
torch.save(model.state_dict(), save_path)

model = Net().to(device)
weight_dict = torch.load(save_path)

weight_dict.keys()

weight_dict['conv1.weight'].shape

model.load_state_dict(weight_dict)
model.eval()

save_path = 'model.pt'

torch.save(model, save_path)

model = torch.load(save_path)
model.eval()

checkpoint_path = 'checkpoint.pt'

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, checkpoint_path)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

checkpoint = torch.load(checkpoint_path)

checkpoint.keys()

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.train()
