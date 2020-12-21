# title: Pytorch basics


# pytorch basic
import numpy as np
import torch

nums = torch.arange(9)
nums.shape

type(nums)

nums.numpy()

nums.reshape(3,3)

zeros = torch.zeros(3,3)
torch.zeros_like(zeros)

# operations
nums*3
nums = nums.reshape(3,3)

nums+nums

result = torch.add(nums, 3)
result.numpy()

# view = reshape
range_nums = torch.arange(9).reshape(3,3)
range_nums
range_nums.view(-1)

# slice and index
nums[1]
nums[1,1]

# compile
arr = np.array([1,1,1])

arr_torch = torch.from_numpy(arr)
arr_torch.float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

arr_torch.to(device)

# autograd
x = torch.ones(2,2, requires_grad=True)
x
y = x + 2
y
print(y.grad_fn)
z = y * y * 3
out = z.mean()

print(z, out)

out.backward()

x.grad

print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)


# pytorch data process
import torch
from torchvision import datasets, transforms
import os
from glob import glob

seed = 1 

torch.manual_seed(seed)
batch_size = 32
test_batch_size = 32

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train = True, download = True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])),
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=False, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=test_batch_size,
    shuffle=True
)

images, labels = next(iter(train_loader)) # just bring the first one after one iteration
images.shape # batch size, channel, height, width
labels.shape

import numpy as np
import matplotlib.pyplot as plt

images[0].shape
torch_image = torch.squeeze(images[0])

image = torch_image.numpy()
image.shape

label = labels[0].numpy()
label.shape
label
plt.title(label)
plt.imshow(image, 'gray')
plt.show()

# pytorch layer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 


layer = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1).to(device)
layer

weight = layer.weight
weight.shape

weight = weight.detach().numpy() # trainable weights cannot be converted into numpy so we detach it first
weight

plt.imshow(weight[0,0,:,:], 'jet')
plt.colorbar()
plt.show()

image = images[0]
output_data = layer(images)
output_data = output_data.data
output = output_data.cpu().numpy() # compile in cpu and then convert into numpy
output.shape

image_arr = image.numpy()
image_arr.shape

plt.figure(figsize=(15,30))
plt.subplot(131)
plt.title('Input')
plt.imshow(np.squeeze(image_arr), 'gray')
plt.subplot(132)
plt.title('Weight')
plt.imshow(weight[0,0,:,:], 'jet')
plt.subplot(133)
plt.title('Output')
plt.imshow(output[0,0,:,:], 'gray')
plt.show()

image.shape
pool = F.max_pool2d(image, 2,2)
pool.shape
pool_arr = pool.numpy()
pool_arr.shape

plt.figure(figsize=(10,15))
plt.subplot(121)
plt.title('Input')
plt.imshow(np.squeeze(image_arr), 'gray')
plt.subplot(122)
plt.title('Output')
plt.imshow(np.squeeze(pool_arr), 'gray')
plt.show()

images.shape
flatten = images.reshape(1, -1)
flatten.shape

lin = nn.Linear(flatten.shape[1], 10)(flatten)
lin 

plt.imshow(lin.detach().numpy(),'jet')
plt.show()

with torch.no_grad(): # remove weights with no_grad. Also can use detach like above.
    flatten = image.reshape(1, -1)
    lin = nn.Linear(flatten.shape[1], 10)(flatten)
    softmax = F.softmax(lin, dim=1)

softmax
np.sum(softmax.numpy())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
result = model.forward(images)
result

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
params = list(model.parameters())
for i in range(8): # show weights and bias in order. No summary function like tensorflow
    print(params[i].size())


data, target = next(iter(train_loader))
data.shape, target.shape


epochs = 1
log_interval = 100
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

    # model.eval() # deactivate layers like batch normalization and dropout

    # test_loss = 0
    # correct = 0

    # with torch.no_grad(): # turn off backpropagation, gradient calculation to increase speed and reduce memory usage
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += F.nll_loss(output, target, reduction='sum').item()
    #         pred = output.argmax(dim=1, keepdim=True) # a class with the highest probability
    #         correct = pred.eq(target.view_as(pred)).sum().item() # reshape target as pred and check equality. 1 if correct, 0 if wrong. 

    # test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    # ))



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










