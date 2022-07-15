#!/usr/bin/env python3pytho
import torchvision
import network
import sklearn.metrics as metrics
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


from torch.utils.data import Dataset, random_split


dataset = "./data"
train_val_split = 0.7
batch_size = 150
epochs = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
data = torchvision.datasets.ImageFolder(root=dataset)
data.len=len(data)
train_len = int((train_val_split)*data.len)
test_len = data.len - train_len
train_subset, test_subset = random_split(data, [train_len, test_len])
trainset = DatasetFromSubset(train_subset, transform=network.transform('train'))
testset = DatasetFromSubset(test_subset, transform=network.transform('test'))

trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset,
                batch_size=batch_size, shuffle=False)





net = network.net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def test_network(net,testloader,print_confusion=False):
    net.eval()
    total_images = 0
    total_correct = 0
    conf_matrix = np.zeros((8,8))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                labels.cpu(),predicted.cpu(),labels=[0,1,2,3,4,5,6,7])

    model_accuracy = total_correct / total_images * 100
    print(', {0} test {1:.2f}%'.format(total_images,model_accuracy))
    net.train()
print("Using device: {}"
          "\n".format(str(device)))
print("Start training...")
for epoch in range(1,epochs+1):
    total_loss = 0
    total_images = 0
    total_correct = 0

    for batch in trainloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        preds = net(images)             # Process batch

        loss = criterion(preds, labels) # Calculate loss

        optimizer.zero_grad()
        loss.backward()                 # Calculate gradients
        optimizer.step()                # Update weights

        output = preds.argmax(dim=1)

        total_loss += loss.item()
        total_images += labels.size(0)
        total_correct += output.eq(labels).sum().item()

    model_accuracy = total_correct / total_images * 100
    print('ep {0}, loss: {1:.2f}, {2} train {3:.2f}%'.format(
           epoch, total_loss, total_images, model_accuracy), end='')

    test_network(net,testloader,
                 print_confusion=(epoch % 10 == 0))
    sys.stdout.flush()