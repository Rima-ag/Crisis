import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from sklearn import metrics

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

import models.MultiModal.data as data
import models.MultiModal.model as model
# Data initialization and loading
# from data import train_dataset, valid_dataset, my_vocab, test_dataset
from data import LEARNING_RATE, EPOCHS
training_loader, testing_loader = data.get_dataset('./CrisisMMD_v2.0/')

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

device = 'cuda' if use_cuda else 'cpu'
model = model.MultiModal()
model = model.float()
print(model)
model.to(device)

# loss_fn = nn.CrossEntropyLoss()
def loss_fn(outputs, targets):
    fn = torch.nn.CrossEntropyLoss(reduction='mean')
    return fn(outputs, targets)
    # targets = targets.type(torch.float32)
    # return torch.nn.BCEWithLogitsLoss()(outputs, targets.view(targets.size()[0], 1))

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for batch_idx, data in enumerate(training_loader):
        (features, targets) = data
        # features = list(features)
        optimizer.zero_grad()
        outputs = model(features, device)
        
        loss = loss_fn(outputs, targets)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(targets), len(training_loader.dataset),
                100. * batch_idx / len(training_loader), loss.data.item()))
        
        loss.backward()
        optimizer.step()

def validation(epoch ,loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data in enumerate(loader):
            (features, targets) = data
            # features = list(features)
            outputs = model(features, device)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            validation_loss += loss_fn(outputs, targets).data.item()
            # get the index of the max log-probability
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

        validation_loss /= len(loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))

    return fin_outputs, fin_targets

for epoch in range(EPOCHS):
    train(epoch)
    for loader in [training_loader, testing_loader]:
        outputs, targets = validation(1, loader)
        # outputs = np.array(outputs) >= 0.5
        # accuracy = metrics.accuracy_score(targets, outputs)
        # f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        # f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        # print(f"Accuracy Score = {accuracy}")
        # print(f"F1 Score (Micro) = {f1_score_micro}")
        # print(f"F1 Score (Macro) = {f1_score_macro}")


