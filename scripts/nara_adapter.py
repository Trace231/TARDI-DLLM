import json
from pathlib import Path

import torch
import torch.nn as nn


MASK_ID = 126336


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim=64, scale=16.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.embed_dim = int(embed_dim)
        self.register_buffer("W", torch.randn(1, embed_dim // 2) * float(scale))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_proj = x.float() @ self.W.float()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).to(self.W.dtype)


class NARAMapper(nn.Module):
    def __init__(self, r, embedding_dim=64, hidden1=256, hidden2=512, input_dim=None):
        super().__init__()
        self.r = int(r)
        input_dim = int(input_dim or embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, self.r * self.r),
        )
        self.reset_parameters()

    def reset_parameters(self):
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, emb):
        return self.net(emb).view(emb.shape[0], self.r, self.r)


class NaRALinear(nn.Module):
    """Noise-aware low-rank adapter with a shared dynamic core matrix.

    The adapter computes B C(lambda) A x, where C(lambda)=I+eta F(e_lambda)
    is produced by a globally shared hypernetwork from a Gaussian Fourier
    embedding of the current mask ratio. This matches the mechanism described
    by NaRA while keeping the implementation local to this project.
    """

    def __init__(
        self,
        base,
        name,
        mapper,
        embedding,
        r=8,
        alpha=16,
        dropout=0.05,
        c_scale=0.1,
        task_embedding=None,
        task_mapper=None,
        task_gate=None,
        num_tasks=0,
        task_residual_scale=0.05,
        task_dropout=0.0,
    ):
        super().__init__()
        self.base = base
        self.name = name
        self.mapper = mapper
        self.embedding = embedding
        self.task_embedding = task_embedding
        self.task_mapper = task_mapper
        self.task_gate = task_gate
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.c_scale = float(c_scale)
        self.num_tasks = int(num_tasks)
        self.task_residual_scale = float(task_residual_scale)
        self.task_dropout = float(task_dropout)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False)
        self.register_buffer("current_noise_level", torch.ones(1, dtype=torch.float32), persistent=False)
        self.register_buffer("current_task_id", torch.zeros(1, dtype=torch.long), persistent=False)
        self.reset_parameters()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def set_noise_level(self, noise_level):
        if not torch.is_tensor(noise_level):
            noise_level = torch.tensor([float(noise_level)], dtype=torch.float32, device=self.current_noise_level.device)
        noise_level = noise_level.detach().to(device=self.current_noise_level.device, dtype=torch.float32)
        self.current_noise_level = torch.clamp(noise_level.view(-1), 0.0, 1.0)

    def set_task_id(self, task_id):
        if not torch.is_tensor(task_id):
            task_id = torch.tensor([int(task_id)], dtype=torch.long, device=self.current_task_id.device)
        task_id = task_id.detach().to(device=self.current_task_id.device, dtype=torch.long)
        if self.num_tasks > 0:
            task_id = torch.clamp(task_id.view(-1), 0, self.num_tasks - 1)
        else:
            task_id = task_id.view(-1)
        self.current_task_id = task_id

    def forward(self, x):
        base_out = self.base(x)
        dtype = self.lora_A.weight.dtype
        z = self.lora_A(self.dropout(x).to(dtype))
        noise = self.current_noise_level.to(device=z.device, dtype=torch.float32)
        if z.dim() == 3 and noise.numel() == z.shape[0]:
            pass
        elif noise.numel() == 1:
            noise = noise.expand(z.shape[0] if z.dim() == 3 else 1)
        else:
            noise = noise[:1].expand(z.shape[0] if z.dim() == 3 else 1)

        noise_emb = self.embedding(noise).to(dtype)
        emb = noise_emb
        task_emb = None
        task_id = None
        if self.task_embedding is not None:
            task_id = self.current_task_id.to(device=z.device, dtype=torch.long)
            batch = z.shape[0] if z.dim() == 3 else 1
            if task_id.numel() == batch:
                pass
            elif task_id.numel() == 1:
                task_id = task_id.expand(batch)
            else:
                task_id = task_id[:1].expand(batch)
            task_emb = self.task_embedding(task_id).to(dtype)
            if self.training and self.task_dropout > 0:
                keep = (torch.rand(task_emb.shape[0], 1, device=task_emb.device) >= self.task_dropout).to(dtype)
                task_emb = task_emb * keep
            if self.task_mapper is None:
                emb = torch.cat([emb, task_emb], dim=-1)
        core_delta = self.mapper(emb).to(dtype) * self.c_scale
        if self.task_mapper is not None and task_emb is not None:
            task_input = torch.cat([noise_emb, task_emb], dim=-1)
            task_delta = self.task_mapper(task_input).to(dtype)
            if self.task_gate is not None and task_id is not None:
                gate = torch.sigmoid(self.task_gate(task_id)).to(dtype).view(-1, 1, 1)
            else:
                gate = 1.0
            core_delta = core_delta + task_delta * gate * self.task_residual_scale
        eye = torch.eye(self.r, device=z.device, dtype=dtype).unsqueeze(0)
        core = eye + core_delta
        if z.dim() == 3:
            z = torch.einsum("blr,brs->bls", z, core)
        else:
            z = torch.matmul(z, core[0])
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


def _noise_level_from_input(input_ids):
    if input_ids is None:
        return torch.ones(1, dtype=torch.float32)
    return (input_ids == MASK_ID).float().mean(dim=1).clamp(0.0, 1.0)


def _register_noise_hook(model):
    if getattr(model, "_nara_noise_hook_registered", False):
        return

    def hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids") if kwargs else None
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or not torch.is_tensor(input_ids):
            noise_level = torch.ones(1, dtype=torch.float32, device=next(model.parameters()).device)
        else:
            noise_level = _noise_level_from_input(input_ids).to(input_ids.device)
        for layer in getattr(model, "_nara_layers", []):
            layer.set_noise_level(noise_level)

    try:
        model.register_forward_pre_hook(hook, with_kwargs=True)
    except TypeError:
        def old_hook(_module, args):
            input_ids = args[0] if args else None
            noise_level = _noise_level_from_input(input_ids).to(input_ids.device) if torch.is_tensor(input_ids) else torch.ones(1, dtype=torch.float32)
            for layer in getattr(model, "_nara_layers", []):
                layer.set_noise_level(noise_level)
        model.register_forward_pre_hook(old_hook)
    model._nara_noise_hook_registered = True


def set_nara_task(model, task):
    if not hasattr(model, "_nara_task_to_id"):
        return
    if isinstance(task, str):
        task_id = model._nara_task_to_id.get(task, 0)
    else:
        task_id = int(task)
    device = next(model.parameters()).device
    tensor = torch.tensor([task_id], dtype=torch.long, device=device)
    for layer in getattr(model, "_nara_layers", []):
        if hasattr(layer, "set_task_id"):
            layer.set_task_id(tensor)


def set_nara_task_batch(model, tasks):
    if not hasattr(model, "_nara_task_to_id"):
        return
    ids = [model._nara_task_to_id.get(str(t), 0) for t in tasks]
    device = next(model.parameters()).device
    tensor = torch.tensor(ids, dtype=torch.long, device=device)
    for layer in getattr(model, "_nara_layers", []):
        if hasattr(layer, "set_task_id"):
            layer.set_task_id(tensor)


def install_nara_adapter(
    model,
    target_modules,
    r=8,
    alpha=16,
    dropout=0.05,
    num_buckets=4,
    embedding_dim=64,
    mapper_hidden1=256,
    mapper_hidden2=512,
    c_scale=0.1,
    task_list=None,
    task_embedding_dim=0,
    task_conditioning="concat",
    task_residual_scale=0.05,
    task_dropout=0.0,
):
    for p in model.parameters():
        p.requires_grad_(False)
    task_list = [str(x) for x in (task_list or []) if str(x)]
    task_embedding_dim = int(task_embedding_dim or 0)
    task_conditioning = str(task_conditioning or "concat")
    if task_conditioning not in {"concat", "residual"}:
        raise ValueError(f"Unknown task_conditioning={task_conditioning}")
    mapper_input_dim = embedding_dim + (task_embedding_dim if task_list and task_conditioning == "concat" else 0)
    mapper = NARAMapper(r, embedding_dim=embedding_dim, hidden1=mapper_hidden1, hidden2=mapper_hidden2, input_dim=mapper_input_dim)
    embedding = GaussianFourierProjection(embedding_dim)
    task_embedding = nn.Embedding(len(task_list), task_embedding_dim) if task_list else None
    task_mapper = None
    task_gate = None
    if task_list and task_conditioning == "residual":
        task_mapper = NARAMapper(
            r,
            embedding_dim=embedding_dim,
            hidden1=mapper_hidden1,
            hidden2=mapper_hidden2,
            input_dim=embedding_dim + task_embedding_dim,
        )
        task_gate = nn.Embedding(len(task_list), 1)
        nn.init.constant_(task_gate.weight, -2.0)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mapper.to(device=device, dtype=dtype)
    embedding.to(device=device, dtype=dtype)
    if task_embedding is not None:
        task_embedding.to(device=device, dtype=dtype)
    if task_mapper is not None:
        task_mapper.to(device=device, dtype=dtype)
    if task_gate is not None:
        task_gate.to(device=device, dtype=dtype)
    layers = []
    names = []
    for name, module in _iter_target_linears(model, set(target_modules)):
        parent, child = _find_parent(model, name)
        wrapped = NaRALinear(
            module,
            name=name,
            mapper=mapper,
            embedding=embedding,
            r=r,
            alpha=alpha,
            dropout=dropout,
            c_scale=c_scale,
            task_embedding=task_embedding,
            task_mapper=task_mapper,
            task_gate=task_gate,
            num_tasks=len(task_list),
            task_residual_scale=task_residual_scale,
            task_dropout=task_dropout,
        )
        wrapped.to(device=device, dtype=dtype)
        setattr(parent, child, wrapped)
        layers.append(wrapped)
        names.append(name)
    if not layers:
        raise RuntimeError(f"No target Linear modules found for NaRA adapter: {target_modules}")
    model._nara_layers = layers
    model._nara_mapper = mapper
    model._nara_embedding = embedding
    model._nara_task_embedding = task_embedding
    model._nara_task_mapper = task_mapper
    model._nara_task_gate = task_gate
    model._nara_task_to_id = {task: i for i, task in enumerate(task_list)}
    model._nara_config = {
        "type": "task_noise_residual_nara_style_hypernetwork" if task_list and task_conditioning == "residual" else ("task_noise_nara_style_hypernetwork" if task_list else "nara_style_hypernetwork"),
        "target_modules": list(target_modules),
        "r": int(r),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "num_buckets": int(num_buckets),
        "embedding_dim": int(embedding_dim),
        "mapper_hidden1": int(mapper_hidden1),
        "mapper_hidden2": int(mapper_hidden2),
        "c_scale": float(c_scale),
        "task_list": task_list,
        "task_embedding_dim": int(task_embedding_dim),
        "task_conditioning": task_conditioning,
        "task_residual_scale": float(task_residual_scale),
        "task_dropout": float(task_dropout),
        "module_names": names,
        "official_mechanism": "C(lambda)=I+c_scale*MLP(GaussianFourier(lambda)); shared mapper across layers",
        "task_noise_mechanism": "concat: C(task,lambda)=I+c_scale*MLP([GaussianFourier(lambda); Emb(task)]); residual: C(task,lambda)=I+c_scale*F(lambda)+gate(task)*scale*G(lambda,task)",
    }
    _register_noise_hook(model)
    return model


def nara_state_dict(model):
    state = {}
    state["__mapper__"] = {k: v.detach().cpu() for k, v in model._nara_mapper.state_dict().items()}
    state["__embedding__"] = {k: v.detach().cpu() for k, v in model._nara_embedding.state_dict().items()}
    if getattr(model, "_nara_task_embedding", None) is not None:
        state["__task_embedding__"] = {k: v.detach().cpu() for k, v in model._nara_task_embedding.state_dict().items()}
    if getattr(model, "_nara_task_mapper", None) is not None:
        state["__task_mapper__"] = {k: v.detach().cpu() for k, v in model._nara_task_mapper.state_dict().items()}
    if getattr(model, "_nara_task_gate", None) is not None:
        state["__task_gate__"] = {k: v.detach().cpu() for k, v in model._nara_task_gate.state_dict().items()}
    for layer in getattr(model, "_nara_layers", []):
        prefix = layer.name
        state[f"{prefix}.lora_A.weight"] = layer.lora_A.weight.detach().cpu()
        state[f"{prefix}.lora_B.weight"] = layer.lora_B.weight.detach().cpu()
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
        embedding_dim=config.get("embedding_dim", 64),
        mapper_hidden1=config.get("mapper_hidden1", 256),
        mapper_hidden2=config.get("mapper_hidden2", 512),
        c_scale=config.get("c_scale", 0.1),
        task_list=config.get("task_list") or None,
        task_embedding_dim=config.get("task_embedding_dim", 0),
        task_conditioning=config.get("task_conditioning", "concat"),
        task_residual_scale=config.get("task_residual_scale", 0.05),
        task_dropout=config.get("task_dropout", 0.0),
    )
    state = torch.load(adapter_dir / "nara_adapter.pt", map_location="cpu")
    if "__mapper__" in state:
        model._nara_mapper.load_state_dict({k: v.to(next(model._nara_mapper.parameters()).device) for k, v in state["__mapper__"].items()})
    if "__embedding__" in state:
        model._nara_embedding.load_state_dict({k: v.to(model._nara_embedding.W.device) for k, v in state["__embedding__"].items()})
    if "__task_embedding__" in state and getattr(model, "_nara_task_embedding", None) is not None:
        model._nara_task_embedding.load_state_dict({k: v.to(next(model._nara_task_embedding.parameters()).device) for k, v in state["__task_embedding__"].items()})
    if "__task_mapper__" in state and getattr(model, "_nara_task_mapper", None) is not None:
        model._nara_task_mapper.load_state_dict({k: v.to(next(model._nara_task_mapper.parameters()).device) for k, v in state["__task_mapper__"].items()})
    if "__task_gate__" in state and getattr(model, "_nara_task_gate", None) is not None:
        model._nara_task_gate.load_state_dict({k: v.to(next(model._nara_task_gate.parameters()).device) for k, v in state["__task_gate__"].items()})
    by_name = {layer.name: layer for layer in model._nara_layers}
    missing = []
    for name, layer in by_name.items():
        for attr, target in [
            ("lora_A.weight", layer.lora_A.weight),
            ("lora_B.weight", layer.lora_B.weight),
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
