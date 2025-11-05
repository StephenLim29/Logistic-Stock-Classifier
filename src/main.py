import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer 

d_model = 512

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

t = Transformer(d_model).to()(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(t.parameters(), lr=1e-3)
epochs = 10

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        labels = batch["labels"].to(device)

        attention_mask = pad_mask[:, None, None, :]
        optimizer.zero_grad()

        logits = model(inputs, attention_mask, is_causal=False)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

    return total_loss / max(1, total_examples)

def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, total_examples = 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            labels = batch["labels"].to(device)

            attention_mask = pad_mask[:, None, None, :]
            pred = model(inputs, attention_mask)
            test_loss = loss_fn(pred, labels).item()

            bs = inputs.size(0)
            total_loss += test_loss * bs
            total_examples += bs

    return total_loss / max(1, total_examples)
        
train_loader = 1 # change
test_loader = 1 # change
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n--------------------")
    train_loss = train(t, train_loader, optimizer, loss_fn, device)
    test_loss = test(t, test_loader, loss_fn, device)
    