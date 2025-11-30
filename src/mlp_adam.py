import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from read_data import build_earnings_dataloader, EarningsWindowDataset
from parquet_helpers import data_Dir
from torchvision.transforms import v2

class NeuralNet(nn.Module):
    def __init__(self, windowSize):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6 * windowSize, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    i = 0
    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        inputs = (inputs - mean.to(device)) / std.to(device)
        labels = batch["labels"].to(device).long()

        # compute prediction error
        pred = model(inputs)
        loss = loss_fn(pred, labels)

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(inputs)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        i +=1

def test(dataloader, model, loss_fn, accuracies):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            #X, y = X.to(device), y.to(device)
            inputs = batch["inputs"].to(device)
            inputs = (inputs - mean.to(device)) / std.to(device)
            labels = batch["labels"].to(device).long()
            tickers = batch["ticker"]

            pred = model(inputs)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
        accuracies.append(100 * correct)

if __name__ == "__main__":
    windowSize = 100
    companies = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META",
    "GOOGL", "JPM", "WMT", "HD", "MCD",
    "TGT", "NKE", "BAC", "KO", "DIS",
    ]

    batch_size = 64

    full_data = EarningsWindowDataset(dataDir=data_Dir,window=windowSize, tickers=companies)
    N = len(full_data)
    trainLen = int(0.6 * N)
    valLen = int(0.2 * N)
    testLen = N - trainLen - valLen

    trainIndices = list(range(0, trainLen))
    valIndices = list(range(trainLen, trainLen + valLen))
    testIndices = list(range(trainLen + valLen, N))

    train_set = Subset(full_data, trainIndices)
    val_set = Subset(full_data, valIndices)
    test_set = Subset(full_data, testIndices)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set,   batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set,  batch_size=32, shuffle=False)

    tr_data = next(iter(train_loader))
    te_data = next(iter(test_loader))
    mean = tr_data["inputs"].mean(dim=[0, 1])
    std = tr_data["inputs"].std(dim=[0, 1])

    transforms = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean.tolist(), std.tolist())
    ])

    train_loader.transform = transforms
    test_loader.transform = transforms

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using device: {device}")
    model = NeuralNet(windowSize).to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1000
    all_accuracies = []
    for i in range(5):
        print(f"Run {i+1}")
        model = NeuralNet(windowSize).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        accuracies = []
        for epoch in range(epochs):
            print(f"epoch: {epoch}\n-----------------------------")
            train(train_loader, model, loss_fn, optimizer)
            test(test_loader, model, loss_fn, accuracies)
        all_accuracies.append(accuracies)

    all_accuracies = np.array(all_accuracies).T
    acc_mean = all_accuracies.mean(axis=1)
    acc_min = all_accuracies.min(axis=1)
    acc_max = all_accuracies.max(axis=1)

    e = range(1, epochs+1)
    plt.plot(e, acc_mean, label="Mean Accuracy", color="blue")
    plt.plot(e, acc_min, label="Min Accuracy", color="red")
    plt.plot(e, acc_max, label="Max Accuracy", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"MLP Test Accuracy over {epochs} epochs (5 runs)")
    plt.legend()
    plt.grid(True)
    plt.show()