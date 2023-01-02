import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import torchtext

class Model(nn.Module):
  
    def __init__(self, vocab, num_classes):
        super().__init__()
        self.num_layers = 6
        self.hidden_dim = 64
        vocab_size, emb_size = vocab.vectors.shape
        self.emb = nn.Embedding(vocab_size, emb_size, _weight=vocab.vectors)
        
        self.rnn = nn.RNN(emb_size, self.hidden_dim, self.num_layers, batch_first=True)  
        # self.lstm = nn.LSTM(input_size = emb_size, hidden_size = 64, batch_first = True, num_layers = self.num_layers)
        
        self.head = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))
        
    def forward(self, batch_data):

        token_embs = self.emb(batch_data)
        
        hidden = torch.zeros(self.num_layers, 4096, self.hidden_dim)
        out, hidden = self.rnn(token_embs, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)

        logits = self.head(out)
        
        return logits

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 1)
        self.lstm = nn.LSTM(input_size = 768, hidden_size = 64, batch_first = True, num_layers = 2)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    
    def forward(self, ids, mask, token_type_ids):
        output_1, _= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        embs = self.l2(output_1)
        output_3, (h_n, c_n) = self.lstm(embs)
        last_hidden_state = h_n
        last_hidden_state = last_hidden_state.permute(1,0,2)
        output_3 = last_hidden_state.flatten(start_dim=1)

        output = self.head(output_3)
        return output

class Roberta(torch.nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        self.pre = xlmr_base.transform()
        self.emb = xlmr_base.get_model()
        
        self.lstm = nn.LSTM(input_size = 768, hidden_size = 64, batch_first = True, num_layers = 6)
        self.head = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, batch_data):
        _input = torchtext.functional.to_tensor(self.pre(batch_data), padding_value = 1)
        embs = self.emb(_input)

        output_3, (h_n, c_n) = self.lstm(embs)
        last_hidden_state = h_n
        last_hidden_state = last_hidden_state.permute(1,0,2)
        output_3 = last_hidden_state.flatten(start_dim=1)

        output = self.head(output_3)
        return output

class RobertaGRU(torch.nn.Module):
    def __init__(self):
        super(RobertaGRU, self).__init__()
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        self.pre = xlmr_base.transform()
        self.emb = xlmr_base.get_model()
        self.hidden_units = 64
        
        self.gru = nn.GRU(input_size = 768, hidden_size = self.hidden_units, batch_first = True, num_layers = 6)
        self.head = nn.Sequential(nn.Linear(self.hidden_units, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, 1))
    
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