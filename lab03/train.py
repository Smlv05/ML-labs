import json
import time
from pathlib import Path

import torch
from torch import nn

from data import get_loaders
from models import MODELS

RESULTS = Path(__file__).resolve().parent / "results"
EPOCHS, LR = 5, 1e-3


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train_one(name, model, loaders, device):
    train_dl, val_dl, test_dl = loaders
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running, n = time.perf_counter(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        val_acc = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": running / n,
                        "val_acc": val_acc, "epoch_s": time.perf_counter() - t0})
        print(f"[{name}] epoch {epoch}: loss={running / n:.4f} val_acc={val_acc:.4f}")
    test_acc = evaluate(model, test_dl, device)
    print(f"[{name}] TEST acc={test_acc:.4f}")
    torch.save(model.state_dict(), RESULTS / f"{name}.pt")
    return {"test_acc": test_acc, "history": history}


def main():
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = get_loaders()
    results = {}
    for name, cls in MODELS.items():
        torch.manual_seed(0)
        results[name] = train_one(name, cls(), loaders, device)
    (RESULTS / "train.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()