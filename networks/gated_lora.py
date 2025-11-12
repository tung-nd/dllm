# gated_peft_lora.py
import math, contextvars
import torch
import torch.nn as nn
import torch.nn.functional as F

# Context to pass a (broadcastable) binary gate mask without touching model.forward
_lora_gate_ctx = contextvars.ContextVar("lora_gate_ctx", default=None)

class LoraGateContext:
    """
    Usage:
        with LoraGateContext(mask):   # mask: broadcastable to layer outputs
            out = model(**batch)
    """
    def __init__(self, mask): self.mask = mask
    def __enter__(self):
        self._tok = _lora_gate_ctx.set(self.mask); return self
    def __exit__(self, exc_type, exc, tb):
        _lora_gate_ctx.reset(self._tok)


def _maybe_unsqueeze_last(mask, out_ndim):
    # If user passes (B,S) for (B,S,Hidden), add the trailing singleton
    if mask is None:
        return None
    if mask.dim() == out_ndim - 1:
        return mask.unsqueeze(-1)
    # Flattened token case: allow (N,) for (N,Hidden)
    if mask.dim() == 1 and out_ndim == 2:
        return mask.view(-1, 1)
    return mask


class GatedLoraLinear(nn.Linear):
    """
    Matches PEFT's LoRA for nn.Linear (non-quant/DoRA/RSLoRA), with an added per-token gate.

    Key parity points:
      - A: (r, in_features), B: (out_features, r)
      - scaling = lora_alpha / r
      - lora_dropout on the input
      - base weight/bias frozen
    Differences:
      - multiplies the LoRA delta by a binary gate mask (from LoraGateContext)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        fan_in_fan_out: bool = False,   # kept for interface parity; defaults False for Linear
    ):
        super().__init__(in_features, out_features, bias=bias)

        # ---- PEFT LoRA config ----
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if self.r > 0 else 0.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()
        self.fan_in_fan_out = bool(fan_in_fan_out)
        self.merged = False  # kept for parity; we don't auto-merge

        # Freeze base weights exactly like PEFT LoRA
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA params (PEFT shapes)
        if self.r > 0:
            # A: (r, in), B: (out, r)
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
            # PEFT/loralib-style init: A kaiming_uniform, B zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # -- PEFT uses a fan_in_fan_out toggle for some modules; for Linear keep False by default --
    @staticmethod
    def _transpose(weight, fan_in_fan_out: bool):
        return weight.T if fan_in_fan_out else weight

    def _base_linear(self, x):
        # Match PEFT: if fan_in_fan_out, use transposed weight in forward
        w = self._transpose(self.weight, self.fan_in_fan_out)
        return F.linear(x, w, self.bias)

    def forward(self, x):
        # Base path
        result = self._base_linear(x)

        # If no adapters or (optionally) merged, return base output
        if self.r == 0 or self.merged:
            return result

        # LoRA delta path (PEFT math)
        x_d = self.lora_dropout(x)                      # dropout on input
        delta = F.linear(x_d, self.lora_A)             # (..., r)   (weight: (r, in))
        delta = F.linear(delta, self.lora_B)           # (..., out) (weight: (out, r))
        delta = delta * self.scaling

        # --- GATING (the only functional difference) ---
        gate_mask = _lora_gate_ctx.get()
        if gate_mask is not None:
            gate_mask = _maybe_unsqueeze_last(gate_mask, delta.dim())
            # Rely on PyTorch broadcasting (must be 0/1 or [0.,1.])
            delta = delta * gate_mask

        return result + delta

    # Optional helpers for parity with PEFT utilities
    @torch.no_grad()
    def merge(self):
        """Permanently add LoRA weights into the base weight (not required for training)."""
        if self.r == 0 or self.merged:
            return
        # deltaW = B @ A  -> shape (out, in)
        delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        # write into stored weight orientation (out, in)
        self.weight.add_(self._transpose(delta_w, self.fan_in_fan_out).T if self.fan_in_fan_out else delta_w)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.r == 0 or not self.merged:
            return
        delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        self.weight.sub_(self._transpose(delta_w, self.fan_in_fan_out).T if self.fan_in_fan_out else delta_w)
        self.merged = False

def _get_parent_and_attr(model, dotted):
    parent = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def inject_gated_lora(
    model: nn.Module,
    target_keywords=("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"),
    r: int = 8,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    fan_in_fan_out: bool = False,
):
    """
    Replaces matching nn.Linear modules with GatedLoraLinear.
    Returns list of replaced module qualified names.
    """
    replaced = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(name.endswith(k) for k in target_keywords):
            parent, attr = _get_parent_and_attr(model, name)
            gl = GatedLoraLinear(
                in_features=mod.in_features,
                out_features=mod.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=(mod.bias is not None),
                fan_in_fan_out=fan_in_fan_out,
            )
            # cast gl to the same type as mod
            gl = gl.to(mod.weight.dtype)
            # copy over the pretrained base weights/bias (and freeze inside the class)
            gl.weight.data = mod.weight.data.clone()
            if gl.bias is not None and mod.bias is not None:
                gl.bias.data = mod.bias.data.clone()
            setattr(parent, attr, gl)
            replaced.append(name)
    return replaced