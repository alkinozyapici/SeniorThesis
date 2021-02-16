import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
# import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter

import vgg

def get_datasets(*args, **kwargs):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    trainset = torchvision.datasets.CIFAR10(train=True, transform=transform, *args, **kwargs)
    testset = torchvision.datasets.CIFAR10(train=False, transform=transform, *args, **kwargs)
    return trainset, testset

def get_dataloaders(trainset, testset, batch_size=100, num_worker=4):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return trainloader, testloader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VGG_11 = vgg.__dict__['vgg11_bn']()
    VGG_11.to(device)

    criterion = nn.CrossEntropyLoss()
    lr_ = 0.02
    optimizer = optim.SGD(VGG_11.parameters(), lr=lr_)

    trainset, testset = get_datasets(root='./data', download=True)
    trainloader, testloader = get_dataloaders(trainset, testset, batch_size=100, num_worker=16)

    for epoch in range(150):  # loop over the dataset multiple times
        running_loss = 0.0
        print("Training the network. Epoch =", epoch)
        if(epoch % 30 == 0):
            lr_ = lr_ / 2
            optimizer = optim.SGD(VGG_11.parameters(), lr=lr_)

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = VGG_11(inputs)

            
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

    print('Finished Training')

    PATH = './VGG_11_mlp.pth'
    torch.save(VGG_11.state_dict(), PATH)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = mlp(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


if __name__ == '__main__':
    main()
