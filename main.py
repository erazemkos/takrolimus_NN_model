import os
import torch
from torch import float32, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/test_run_3/')



df = pd.read_excel('C:/Users/Erazem/Desktop/JS_projects/takrolimus/test_neuralnet_julij_brez_imen.xlsx', sheet_name='Sheet1', engine="openpyxl")

#normalized_df=(df-df.min())/(df.max()-df.min())
arr = df.to_numpy()
ARR_SCALE_MAX = np.max(arr[:,0])
ARR_SCALE_MIN = np.min(arr[:,0])

arr_edit = np.delete(arr, 2, 1)

min_max_scaler = preprocessing.MinMaxScaler()
arr_scaled = min_max_scaler.fit_transform(arr_edit)


device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 20
NUM_EPOCH = 100

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(70, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()


x_test = torch.tensor(arr_scaled[::5,1:]).float()
y_test = torch.unsqueeze(torch.tensor(arr_scaled[::5,0]),1).float()

remove_test = np.delete(arr_scaled, list(range(0, arr_scaled.shape[0], 5)), axis=0)

x_train = torch.tensor(remove_test[:,1:]).float()
y_train = torch.unsqueeze(torch.tensor(remove_test[:,0]),1).float()

train_dataset = TensorDataset(x_train,y_train)
train_loader = DataLoader(
    dataset=train_dataset,      
    batch_size=BATCH_SIZE,      
    shuffle=True,               
    num_workers=2,              
)

test_dataset = TensorDataset(x_test,y_test)
test_loader = DataLoader(
    dataset=test_dataset,      
    batch_size=BATCH_SIZE,      
    shuffle=True,               
    num_workers=2,              
)

def denormalize(value, ARR_SCALE_MAX=ARR_SCALE_MAX, ARR_SCALE_MIN=ARR_SCALE_MIN):
    return value*(ARR_SCALE_MAX-ARR_SCALE_MIN)+ARR_SCALE_MIN

def results():
    out_arr = np.zeros((x_test.shape[0],2))
    for idx, case in enumerate(x_test):

        output = model(case)
        out_arr[idx, 0] = denormalize(output)
        out_arr[idx, 1] = denormalize(y_test[idx])
    print(out_arr)


if __name__ == '__main__':
    for epoch in tqdm(range(NUM_EPOCH)):   
        full_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):  

            optimizer.zero_grad()   # clear gradients for next train
            
            output = model(batch_x)     # input x and predict based on x

            loss = loss_func(output, batch_y)     # must be (1. nn output, 2. target)
            full_loss += loss.item()

            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
        if epoch % 5 == 0:
            full_val_loss = 0.0
            for step, (batch_x, batch_y) in enumerate(test_loader):      
                output = model(batch_x)     # input x and predict based on x

                loss = loss_func(output, batch_y)     # must be (1. nn output, 2. target)
                full_val_loss += loss.item()
            writer.add_scalar('Loss/test', full_val_loss/(len(test_dataset)*1.0), epoch)
            print('##################################')
            print('MSE VALIDATION LOSS: ', full_val_loss/(len(test_dataset)*1.0))
            print('##################################')
        writer.add_scalar('Loss/train', full_loss/(len(train_dataset)*1.0), epoch)
        print('MSE TRAIN LOSS: ', full_loss/(len(train_dataset)*1.0))

    
    torch.save(model.state_dict(), 'test_run_100e.pth')
    model.eval()
    results()

