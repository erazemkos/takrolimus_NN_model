from cv2 import CV_32SC2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CatsDogsDataset(Dataset):
    def __init__(self, csv_file :str, root_dir :str, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = cv2.imread(img_path, mode='RGB')
        label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return (image, label)

