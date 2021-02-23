import torch
from torch import nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms

BS = 256
lr = 0.1

trans = transforms.Compose([transforms.ToTensor()])

train_set = dataset.MNIST(root="./data", train=True, transform=trans, download=True)
test_set = dataset.MNIST(root="./data", train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=BS,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=BS,
                                         shuffle=True)

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)

net = nn.Sequential(Reshape(),nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.1)

def train_batch(X, y, opt, net, criterion):
  opt.zero_grad()
  y_hat = net(X)
  loss = criterion(y_hat, y)
  loss.backward()
  opt.step()
  return loss.data

n_epochs = 20
for epoch in range(n_epochs):
  av_loss = 0
  net.train()
  for batch_idx,(X, y) in enumerate(train_loader):
    av_loss += train_batch(X, y, opt, net, criterion)
  print("epoch {}/{}, average loss : {:.5f}".format(epoch, n_epochs, av_loss))

acc = 0
for _, (X,y) in enumerate(test_loader):
  corr = torch.sum(torch.argmax(net(X),dim=1) == y)
  acc += corr/len(X)
print("Précision sur le jeu de test : ", acc/len(test_loader))
