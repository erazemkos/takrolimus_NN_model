import os
from turtle import forward
import torch
from torch import float32, nn, relu
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from main import NUM_EPOCH

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,28*28),
            nn.ReLU(),
            nn.Linear(28*28,28*28),
            nn.ReLU(),
            nn.Linear(28*28,1),
            nn.ReLU()
            )
    def forward(self, input):
        logits = self.conv_layer(input)
        return logits

mnist_trainset  = torchvision.datasets.MNIST(root='MNIST/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_trainset ,
                        batch_size=4,
                        shuffle=True,
                        num_workers=4)

mnist_testset  = torchvision.datasets.MNIST(root='MNIST/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(mnist_testset ,
                        batch_size=4,
                        shuffle=True,
                        num_workers=4)

model = NeuralNetwork().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
NUM_EPOCH = 50
TRAIN_ARG = False
TEST_ARG = True
if __name__ == '__main__':
    if TRAIN_ARG:
        for epoch in tqdm(range(NUM_EPOCH)):
            full_loss = 0.0
            for x, (imgs, lbls) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(imgs)
                output = output.squeeze().to(dtype=torch.float)
                lbls = lbls.to(dtype=torch.float)
                loss = loss_func(output, lbls)
                full_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(full_loss/(len(train_loader)))

        torch.save(model.state_dict(), 'testni_run.pth')

    if TEST_ARG:
        model.load_state_dict(torch.load('testni_run.pth'))
        model.eval()
        full_loss = 0.0
        for x, (imgs, lbls) in enumerate(test_loader):
            output = model(imgs)
            output = output.squeeze().to(dtype=torch.float)
            lbls = lbls.to(dtype=torch.float)
            loss = loss_func(output, lbls)
            full_loss += loss.item()
        print(full_loss/len(test_loader))

        
