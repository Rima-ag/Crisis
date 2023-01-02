from CNN.model import CNN
from robertaGRU.model import RobertaGRU
import torch
import torch.nn as nn

class MultiModal(torch.nn.Module):
    def __init__(self):
        super(MultiModal, self).__init__()
        self.rnn = RobertaGRU()
        self.cnn = CNN()

    #batch_data = image, text, label
    def forward(self, batch_data, device):
        c_output = self.cnn(batch_data[0].float())
        r_output = self.rnn(list(batch_data[1]), device)
        output = torch.div(c_output.add(r_output), 2)
        return output