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
# loss_fn = nn.CrossEntropyLoss()
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
            print(f"{t}: Right: {stats['Correct']}, Total: {stats['Total']}, Accuracy: {acc:.2f}")



if __name__ == "__main__":
    windowSize = 60
    companies = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META",
    "GOOGL", "JPM", "WMT", "HD", "MCD",
    "TGT", "NKE", "BAC", "KO", "DIS",
    ]

    full_data = EarningsWindowDataset(dataDir=data_Dir, window=windowSize, tickers=companies)
    N = len(full_data)

    # Compute class weights from the dataset (There is an Imbalance in the data set)
    all_labels = [sample["labels"].item() for sample in full_data]
    num_neg = sum(1 for y in all_labels if y == 0)
    num_pos = sum(1 for y in all_labels if y == 1)

    print(f"Class Stats: num_neg={num_neg}, num_pos={num_pos}")


    # weight class 1 higher due to seen imbalance in data (1s less likely)
    w0 = 1.0
    w1 = num_neg / num_pos
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    print(f"Class weights: w0={w0:.3f}, w1={w1:.3f}")

    # Move weights to the right device and define loss_fn
    class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)


    N = len(full_data)

    # 80 / 20 train test split
    """
    trainLen = int(0.8 * N)
    testLen = N - trainLen
    trainIndices = list(range(0, trainLen))
    testIndices = list(range(trainLen, N))

    train_set = Subset(full_data, trainIndices)
    test_set = Subset(full_data, testIndices)  

    train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)

    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)
    """


    # 60 / 20 / 20
    # train / val / test split
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

    best_val_acc = 0.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n--------------------")
        train_loss, train_acc = train(t, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = test(t, val_loader, loss_fn, device)

        print(
            f"train loss: {train_loss:.2f} | train accuracy: {train_acc:.2f}\n"
            f"val   loss: {val_loss:.2f} | val   accuracy: {val_acc:.2f}"
        )

        # Track best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = t.state_dict()  # save a copy of parameters




    print(f"\nBest epoch by val accuracy: {best_epoch} (val_acc={best_val_acc:.3f})")

    # Load best model before evaluating on test
    if best_state_dict is not None:
        t.load_state_dict(best_state_dict)

    print("\nFinal evaluation on TEST set:")
    test_loss, test_acc = test(t, test_loader, loss_fn, device)
    print(f"test loss: {test_loss:.2f} | test accuracy: {test_acc:.2f}")


    evaluation(t, test_loader, device)
    print()

    

