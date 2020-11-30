import torch
from torch import nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms

batch_size=256
trans = transforms.Compose([transforms.ToTensor()])

train_set = dataset.MNIST(root="./data", train=True, transform=trans, download=True)
test_set = dataset.MNIST(root="./data", train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=batch_size,
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
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

n_epochs = 40
for epoch in range(n_epochs):
    av_loss = 0
    net.train()
    for batch_idx,(X, y) in enumerate(train_loader):
        trainer.zero_grad()
        y_hat = net(X)
        loss = criterion(y_hat, y)
        loss.backward()
        trainer.step()
        av_loss += loss.data
        if (batch_idx+1) == len(train_loader):
            print("epoch {}, average loss : {:.5f}".format(epoch,av_loss))


acc = 0
for _, (X,y) in enumerate(test_loader):
    corr = 0
    for idx in range(len(X)):
        if torch.argmax(net(X[idx])) == y[idx]:
            corr += 1
    acc_b = corr/len(X)
    print("acc : ", acc_b)
    acc += acc_b
print("Précision sur le jeu de test : ", acc/len(test_loader))
