import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

print("Python Version:", torch.__version__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_dataloader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = F.nll_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, idx, loss.item()))


def test(model, device, test_dataloader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(test_dataloader.dataset)
    acc = correct / len(test_dataloader.dataset) * 100
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))


mnist_data = datasets.MNIST("./mnist_data", train=True, download=True,
                            transform = transforms.Compose([
                                transforms.ToTensor(),
                            ]))
# print(mnist_data)
# print(mnist_data[233][0].shape)

data = [d[0].data.cpu().numpy() for d in mnist_data]
np.mean(data)
np.std(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("./fashion_mnist_data", train=False, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("./fashion_mnist_data", train=False, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
)

lr = 0.01
momentum = 0.5

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

num_epochs = 2
for epoch in range(num_epochs):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)

torch.save(model.state_dict(), "fashion_mnist_cnn.pt")


