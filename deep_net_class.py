import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn import metrics

def loss_fn(outputs, targets):
    fn = torch.nn.CrossEntropyLoss(reduction='mean')
    return fn(outputs, targets)

# Training settings
parser = argparse.ArgumentParser(description='Crisis Training script')
parser.add_argument('--model', type=str, default='MM', metavar='M',
                    help='Model to train (default: MM)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epoch', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--f1', default=False, help='show f1 micro and macro score')

args = parser.parse_args()
assert args.model in ['MM', 'CNN', 'RNN']

if args.model == 'MM':
    import MultiModal.data as data
    import MultiModal.model as model
    from model import MultiModal as TheModel
elif args.model == 'CNN':
    import CNN.data as data
    import CNN.model as model
    from model import CNN as TheModel
else:
    import robertaGRU.data as data
    import robertaGRU.model as model
    from model import RobertaGRU as TheModel

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
model = TheModel()
model = model.float()
print(model)
model.to(device)

optimizer = torch.optim.Adam(params = model.parameters(), lr=args.lr)
training_loader, testing_loader = data.get_dataset('./CrisisMMD_v2.0/')

def train(epoch):
    model.train()
    for batch_idx, data in enumerate(training_loader):
        (features, targets) = data
        optimizer.zero_grad()
        outputs = model(features, device)
        
        loss = loss_fn(outputs, targets)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(targets), len(training_loader.dataset),
                100. * batch_idx / len(training_loader), loss.data.item()))
        
        loss.backward()
        optimizer.step()

def validation(loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data in enumerate(loader):
            (features, targets) = data
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

for epoch in range(args.epoch):
    train(epoch)
    for loader in [training_loader, testing_loader]:
        outputs, targets = validation(loader)
        if args.f1:
            outputs = np.array(outputs) >= 0.5
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")


