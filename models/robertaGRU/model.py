import torch
import torch.nn as nn
import torchtext

class RobertaGRU(torch.nn.Module):
    def __init__(self):
        super(RobertaGRU, self).__init__()
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        self.pre = xlmr_base.transform()
        self.emb = xlmr_base.get_model()
        self.hidden_units = 64
        
        self.gru = nn.GRU(input_size = 768, hidden_size = self.hidden_units, batch_first = True, num_layers = 6)
        self.head = nn.Sequential(nn.Linear(self.hidden_units, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, 2))
    
    def initialize_hidden_state(self, device):
        return torch.zeros((1, 124, self.hidden_units)).to(device)

    def forward(self, batch_data, device):
        _input = torchtext.functional.to_tensor(self.pre(batch_data), padding_value = 1)
        embs = self.emb(_input)
        
        self.hidden = self.initialize_hidden_state(device)
        output_3, _ = self.gru(embs)
        output_3 = output_3[:, -1, :]

        output = self.head(output_3)
        return output