import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer 
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import sys
from read_data import build_earnings_dataloader, EarningsWindowDataset
from parquet_helpers import data_Dir
import matplotlib.pyplot as plt

d_model = 128

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

t = Transformer(d_model).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(t.parameters(), lr=1e-4)
epochs = 100

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        labels = batch["labels"].to(device).long()

        attention_mask = pad_mask[:, None, None, :]

        if torch.isnan(attention_mask).any():
                print("NaNs in inputs")
                sys.exit(1)
        optimizer.zero_grad()

        logits = model(inputs, attention_mask)

        if torch.isnan(logits).any():
            print("NaNs in logit")
            sys.exit(1)

        loss = loss_fn(logits, labels)

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        bs = inputs.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

    if torch.isnan(torch.tensor(total_loss)):
        print("NaNs in inputs")
        sys.exit(1)

    return total_loss / max(1, total_examples)

def test(model, dataloader, loss_fn, device, accuracies):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, total_examples, total_loss, correct = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            labels = batch["labels"].to(device).long()

            attention_mask = pad_mask[:, None, None, :]
            pred = model(inputs, attention_mask)
            test_loss = loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

            bs = inputs.size(0)
            total_loss += test_loss * bs
            total_examples += bs
    correct /= size
    accuracies.append(correct * 100)

    return total_loss / max(1, total_examples)

if __name__ == "__main__":    
    windowSize = 60
    companies = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META",
    "GOOGL", "JPM", "WMT", "HD", "MCD",
    "TGT", "NKE", "BAC", "KO", "DIS",
    ]

    full_data = EarningsWindowDataset(dataDir=data_Dir, window=windowSize, tickers=companies)
    N = len(full_data)
    # 80 / 20 train test split
    trainLen = int(0.8 * N)
    testLen = N - trainLen
    trainIndices = list(range(0, trainLen))
    testIndices = list(range(trainLen, N))

    train_set = Subset(full_data, trainIndices)
    test_set = Subset(full_data, testIndices)  

    train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)

    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)

    accuracies = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n--------------------")
        train_loss = train(t, train_loader, optimizer, loss_fn, device)
        test_loss = test(t, test_loader, loss_fn, device, accuracies)
        print(f"train loss: {train_loss} | test loss: {test_loss}")

    plt.plot(range(0, epochs), accuracies, label="Accuracies", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over {epochs} epochs")
    plt.grid(True)
    plt.show()


    
    
