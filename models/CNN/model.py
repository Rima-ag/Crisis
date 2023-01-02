import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet152

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.resnet = resnet152(pretrained= True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)


    def forward(self, batch_data):
        return self.resnet(batch_data)