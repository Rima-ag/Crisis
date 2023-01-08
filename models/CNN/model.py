import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet152

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.resnet = resnet152(pretrained= True)
        self.resnet.fc = nn.Sequential(nn.Linear(8192, 64), nn.Linear(64, 64),nn.Linear(64, 2))


    def forward(self, batch_data):
        return self.resnet(batch_data)