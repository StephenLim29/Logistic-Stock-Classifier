import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer 
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import sys
from read_data import build_earnings_dataloader, EarningsWindowDataset
from parquet_helpers import data_Dir
from collections import Counter

d_model = 512

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

t = Transformer(d_model).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(t.parameters(), lr=1e-4)
epochs = 100

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    correct = 0

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

        predicted_labels = torch.argmax(logits, dim=-1)
        correct += (predicted_labels == labels).sum().item()


    if torch.isnan(torch.tensor(total_loss)):
        print("NaNs in inputs")
        sys.exit(1)

    average_loss = total_loss / max(1, total_examples)
    accuracy = correct / max(1, total_examples)

    return average_loss, accuracy

def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, total_examples, total_loss = 0, 0, 0
    correct = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            labels = batch["labels"].to(device).long()

            attention_mask = pad_mask[:, None, None, :]
            pred = model(inputs, attention_mask)
            test_loss = loss_fn(pred, labels).item()

            bs = inputs.size(0)
            total_loss += test_loss * bs
            total_examples += bs

            predicted_labels = torch.argmax(pred, dim=-1)
            correct += (predicted_labels == labels).sum().item()

        average_loss = total_loss / max(1, total_examples)
        accuracy = correct / max(1, total_examples)
    return average_loss, accuracy


def evaluation(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []
    all_tickers = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            labels = batch["labels"].to(device).long()
            tickers = batch["ticker"]

            attention_mask = pad_mask[:, None, None, :]
            logits = model(inputs, attention_mask)
            predictions = logits.argmax(dim=-1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_tickers.extend(tickers)

            # Build Confusion Matrix
            tn = sum(1 for p, y in zip(all_predictions, all_labels) if p == 0 and y == 0)
            fp = sum(1 for p, y in zip(all_predictions, all_labels) if p == 1 and y == 0)
            fn = sum(1 for p, y in zip(all_predictions, all_labels) if p == 0 and y == 1)
            tp = sum(1 for p, y in zip(all_predictions, all_labels) if p == 1 and y == 1)

            print(f"CONFUSION MATRIX tn={tn}, fp={fp}, fn={fn}, tp={tp}")


            # Per Ticket Accuracy
            per_ticker = {}

            for p, y, t in zip(all_predictions, all_labels, all_tickers):
                if t not in per_ticker:
                    per_ticker[t] = {"Correct": 0, "Total": 0}
                per_ticker[t]["Total"] += 1
                if p == y:
                    per_ticker[t]["Correct"] += 1
            print("Per Ticker Accuracy:")
            for t, stats in per_ticker.items():
                acc = stats["Correct"] / max(1, stats["Total"])
                print(f"{t}: Right: {stats["Correct"]}, Total: {stats["Total"]}, Accuracy: {acc:.2f}")



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

    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n--------------------")
        train_loss, train_acc = train(t, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = test(t, test_loader, loss_fn, device)
        print(f"train loss: {train_loss:.2f} | train accuracy: {train_acc:.2f}\n"
              f"test loss: {test_loss:.2f} | test accuracy: {test_acc:.2f}")


        print("\nPer Ticker and Confusion Matrix evaluation on each test:")
        evaluation(t, test_loader, device)
        print("\n")


    

