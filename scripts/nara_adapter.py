import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_ID = 126336


class NaRALinear(nn.Module):
    """Noise-aware low-rank adapter with a bucketed dynamic core matrix.

    This is a lightweight NaRA-style implementation for masked diffusion LMs:
    the adapter computes B C(lambda) A x, where the core C is selected from the
    current mask-ratio bucket. It is a mechanism-level baseline rather than a
    dependency on a specific external codebase.
    """

    def __init__(self, base, name, r=8, alpha=16, dropout=0.05, num_buckets=4):
        super().__init__()
        self.base = base
        self.name = name
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.num_buckets = int(num_buckets)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False)
        self.core = nn.Parameter(torch.zeros(self.num_buckets, self.r, self.r))
        self.register_buffer("current_bucket", torch.zeros(1, dtype=torch.long), persistent=False)
        self.reset_parameters()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        with torch.no_grad():
            eye = torch.eye(self.r)
            for i in range(self.num_buckets):
                self.core[i].copy_(eye)

    def set_bucket(self, bucket):
        if not torch.is_tensor(bucket):
            bucket = torch.tensor([int(bucket)], dtype=torch.long, device=self.current_bucket.device)
        bucket = bucket.detach().to(device=self.current_bucket.device, dtype=torch.long)
        bucket = torch.clamp(bucket, 0, self.num_buckets - 1)
        self.current_bucket = bucket

    def forward(self, x):
        base_out = self.base(x)
        dtype = self.lora_A.weight.dtype
        z = self.lora_A(self.dropout(x).to(dtype))
        bucket = self.current_bucket.to(z.device)
        if z.dim() == 3 and bucket.numel() == z.shape[0]:
            core = self.core.to(z.dtype)[bucket]
            z = torch.einsum("blr,brs->bls", z, core)
        else:
            core = self.core.to(z.dtype)[int(bucket.flatten()[0].item())]
            z = torch.matmul(z, core)
        update = self.lora_B(z).to(base_out.dtype) * self.scaling
        return base_out + update


def _find_parent(root, module_name):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _iter_target_linears(model, target_modules):
    for name, module in list(model.named_modules()):
        if not name:
            continue
        leaf = name.split(".")[-1]
        if leaf in target_modules and isinstance(module, nn.Linear):
            yield name, module


def _bucket_from_input(input_ids, num_buckets):
    if input_ids is None:
        return torch.zeros(1, dtype=torch.long)
    mask_ratio = (input_ids == MASK_ID).float().mean(dim=1)
    bucket = torch.floor(mask_ratio * num_buckets).long()
    return torch.clamp(bucket, 0, num_buckets - 1)


def _register_noise_hook(model, num_buckets):
    if getattr(model, "_nara_noise_hook_registered", False):
        return

    def hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids") if kwargs else None
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or not torch.is_tensor(input_ids):
            bucket = torch.zeros(1, dtype=torch.long, device=next(model.parameters()).device)
        else:
            bucket = _bucket_from_input(input_ids, num_buckets).to(input_ids.device)
        for layer in getattr(model, "_nara_layers", []):
            layer.set_bucket(bucket)

    try:
        model.register_forward_pre_hook(hook, with_kwargs=True)
    except TypeError:
        def old_hook(_module, args):
            input_ids = args[0] if args else None
            bucket = _bucket_from_input(input_ids, num_buckets).to(input_ids.device) if torch.is_tensor(input_ids) else torch.zeros(1, dtype=torch.long)
            for layer in getattr(model, "_nara_layers", []):
                layer.set_bucket(bucket)
        model.register_forward_pre_hook(old_hook)
    model._nara_noise_hook_registered = True


def install_nara_adapter(model, target_modules, r=8, alpha=16, dropout=0.05, num_buckets=4):
    for p in model.parameters():
        p.requires_grad_(False)
    layers = []
    names = []
    for name, module in _iter_target_linears(model, set(target_modules)):
        parent, child = _find_parent(model, name)
        wrapped = NaRALinear(module, name=name, r=r, alpha=alpha, dropout=dropout, num_buckets=num_buckets)
        setattr(parent, child, wrapped)
        layers.append(wrapped)
        names.append(name)
    if not layers:
        raise RuntimeError(f"No target Linear modules found for NaRA adapter: {target_modules}")
    model._nara_layers = layers
    model._nara_config = {
        "type": "nara_style",
        "target_modules": list(target_modules),
        "r": int(r),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "num_buckets": int(num_buckets),
        "module_names": names,
    }
    _register_noise_hook(model, int(num_buckets))
    return model


def nara_state_dict(model):
    state = {}
    for layer in getattr(model, "_nara_layers", []):
        prefix = layer.name
        state[f"{prefix}.lora_A.weight"] = layer.lora_A.weight.detach().cpu()
        state[f"{prefix}.lora_B.weight"] = layer.lora_B.weight.detach().cpu()
        state[f"{prefix}.core"] = layer.core.detach().cpu()
    return state


def save_nara_adapter(model, out_dir, tokenizer=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(nara_state_dict(model), out / "nara_adapter.pt")
    (out / "nara_config.json").write_text(json.dumps(model._nara_config, indent=2))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out))


def load_nara_adapter(model, adapter_dir):
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "nara_config.json").read_text())
    install_nara_adapter(
        model,
        config["target_modules"],
        r=config["r"],
        alpha=config["alpha"],
        dropout=config.get("dropout", 0.0),
        num_buckets=config.get("num_buckets", 4),
    )
    state = torch.load(adapter_dir / "nara_adapter.pt", map_location="cpu")
    by_name = {layer.name: layer for layer in model._nara_layers}
    missing = []
    for name, layer in by_name.items():
        for attr, target in [
            ("lora_A.weight", layer.lora_A.weight),
            ("lora_B.weight", layer.lora_B.weight),
            ("core", layer.core),
        ]:
            key = f"{name}.{attr}"
            if key not in state:
                missing.append(key)
                continue
            target.data.copy_(state[key].to(device=target.device, dtype=target.dtype))
    if missing:
        raise RuntimeError(f"NaRA adapter missing tensors: {missing[:5]}")
    return model


def is_nara_adapter(path):
    path = Path(path)
    return (path / "nara_config.json").exists() and (path / "nara_adapter.pt").exists()
