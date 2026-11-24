"""Lab 7 - magnitude pruning (global / layer-wise), fixed masks, structured filter pruning."""
import torch
import torch.nn as nn

from common import BigCNN

PRUNABLE = (nn.Conv2d, nn.Linear)


def prunable_layers(model):
    """[(name, module)] for every conv/linear layer. Only weights are pruned (not biases / BN)."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, PRUNABLE)]


def _keep_mask(scores: torch.Tensor, n_prune: int) -> torch.Tensor:
    """Rank by |w| and mark the n_prune smallest as pruned (False)."""
    flat = scores.flatten()
    keep = torch.ones_like(flat, dtype=torch.bool)
    if n_prune > 0:
        keep[torch.argsort(flat)[:n_prune]] = False
    return keep.view_as(scores)


@torch.no_grad()
def magnitude_masks(model, sparsity: float, scope: str = "global"):
    """scope='global': one ranking over ALL prunable weights (layers end up with different sparsity).
    scope='layer'   : every layer is pruned to exactly `sparsity`."""
    layers = prunable_layers(model)
    if scope == "global":
        scores = [m.weight.abs().flatten() for _, m in layers]
        all_scores = torch.cat(scores)
        keep_all = _keep_mask(all_scores, int(sparsity * all_scores.numel()))
        masks, start = {}, 0
        for (name, m), s in zip(layers, scores):
            masks[name] = keep_all[start : start + s.numel()].view_as(m.weight)
            start += s.numel()
        return masks
    if scope == "layer":
        return {name: _keep_mask(m.weight.abs(), int(sparsity * m.weight.numel())) for name, m in layers}
    raise ValueError(scope)


@torch.no_grad()
def apply_masks(model, masks):
    modules = dict(model.named_modules())
    for name, mask in masks.items():
        modules[name].weight.mul_(mask)


def mask_keeper(masks):
    """Callback for train_epochs(after_step=...): re-applies the fixed mask after every
    optimizer step, so pruned weights stay exactly zero during fine-tuning."""
    def _after_step(model):
        apply_masks(model, masks)
    return _after_step


def layer_sparsity(model):
    return {name: float((m.weight == 0).float().mean()) for name, m in prunable_layers(model)}


@torch.no_grad()
def structured_filter_prune(model: BigCNN, ratio: float) -> BigCNN:
    """Removes `ratio` of the filters in EVERY conv layer (lowest L2 norm first) and builds
    a physically smaller dense BigCNN: next conv's input channels, BatchNorm and the first
    Linear layer's input columns are sliced accordingly."""
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    bns = [m for m in model.features if isinstance(m, nn.BatchNorm2d)]

    keep_idx = []
    for conv in convs:
        n_keep = max(1, int(round(conv.out_channels * (1 - ratio))))
        l2 = conv.weight.flatten(1).norm(p=2, dim=1)           # one L2 norm per filter
        keep_idx.append(torch.argsort(l2, descending=True)[:n_keep].sort().values)

    new = BigCNN(ch=tuple(len(k) for k in keep_idx), hidden=model.hidden)
    new_convs = [m for m in new.features if isinstance(m, nn.Conv2d)]
    new_bns = [m for m in new.features if isinstance(m, nn.BatchNorm2d)]

    prev = torch.arange(convs[0].in_channels)
    for conv, bn, nconv, nbn, k in zip(convs, bns, new_convs, new_bns, keep_idx):
        nconv.weight.copy_(conv.weight[k][:, prev])
        nconv.bias.copy_(conv.bias[k])
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(nbn, attr).copy_(getattr(bn, attr)[k])
        prev = k

    # Flatten order of (C, 3, 3) -> column index c*9 + spatial
    fc1_old, fc1_new = model.classifier[2], new.classifier[2]
    cols = (prev[:, None] * 9 + torch.arange(9)[None, :]).flatten()
    fc1_new.weight.copy_(fc1_old.weight[:, cols])
    fc1_new.bias.copy_(fc1_old.bias)
    new.classifier[5].load_state_dict(model.classifier[5].state_dict())
    return new