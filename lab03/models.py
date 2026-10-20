import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear(49, 64) -> 2 encoder layers -> mean pool -> Linear(64, 10)."""

    def __init__(self, patch=7, d_model=64, nhead=4, dim_ff=128, num_layers=2, use_pos=True):
        super().__init__()
        self.patch = patch
        self.seq_len = (28 // patch) ** 2
        self.embed = nn.Linear(patch * patch, d_model)
        # Deviation from manual: learnable positional embedding (documented in report.md)
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model)) if use_pos else None
        if self.pos is not None:
            nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, 10)

    def patchify(self, x):
        # (B, 1, 28, 28) -> (B, 1, 4, 4, 7, 7) -> (B, 16, 49)
        b, p = x.shape[0], self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)
        return x.reshape(b, self.seq_len, p * p)

    def forward(self, x):
        tokens = self.embed(self.patchify(x))
        if self.pos is not None:
            tokens = tokens + self.pos
        return self.head(self.encoder(tokens).mean(dim=1))


MODELS = {"mlp": MLP, "cnn": CNN, "attention": PatchAttention}