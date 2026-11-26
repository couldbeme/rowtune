"""The per-row scale module, 4-bit rounding, and folding the scales back into the weights."""
import torch
import torch.nn.utils.parametrize as P


def linears(model):
    """Linear layers of the decoder blocks (the output head and embeddings are left alone)."""
    return {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n}


class RowScale(torch.nn.Module):
    """One float32 factor per output row. As a parametrisation of a weight W it returns W * c[:, None]."""

    def __init__(self, rows, device):
        super().__init__()
        self.c = torch.nn.Parameter(torch.ones((rows,), device=device, dtype=torch.float32))

    def forward(self, W):
        return W * self.c.to(W.dtype)[:, None]


def quantize_rtn(W, bits=4, group_size=0):
    """Symmetric round-to-nearest, no zero point; one scale per row (group_size 0) or per group of columns.
    scale = max|w| / ((qmax - qmin) / 2). Returns (dequantised float32, integer codes int8, scale)."""
    W = W.float()
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    rows, cols = W.shape
    gs = cols if group_size == 0 else group_size
    if cols % gs:
        raise ValueError(f"columns {cols} not divisible by group size {gs}")
    amax = W.view(rows, cols // gs, gs).abs().amax(-1)
    scale = (amax / ((qmax - qmin) / 2)).clamp(min=1e-12)
    s = scale.repeat_interleave(gs, 1)
    codes = torch.clamp(torch.round(W / s), qmin, qmax)
    return codes * s, codes.to(torch.int8), scale


@torch.no_grad()
def rtn_student(model, bits=4, group_size=0):
    """Replace the weight of every decoder linear by its 4-bit round-to-nearest value, in place."""
    for m in linears(model).values():
        deq, _, _ = quantize_rtn(m.weight, bits, group_size)
        m.weight.copy_(deq.to(m.weight.dtype))
    return model


@torch.no_grad()
def copy_linears(src, dst):
    """Copy decoder linear weights of `src` into `dst` by name; returns the names copied."""
    s, done = linears(src), []
    for n, m in linears(dst).items():
        if n in s:
            m.weight.copy_(s[n].weight.to(m.weight.device, m.weight.dtype))
            done.append(n)
    return done


def attach_row_scales(model, only=None):
    """Freeze the model and put a RowScale parametrisation on every decoder linear weight (or only those named in
    `only`; the others stay plain and frozen). Returns {name: RowScale}."""
    for p in model.parameters():
        p.requires_grad_(False)
    mods = {}
    for n, m in linears(model).items():
        if only is not None and n not in only:
            continue
        rs = RowScale(m.weight.shape[0], m.weight.device)
        P.register_parametrization(m, "weight", rs)
        mods[n] = rs
    return mods


@torch.no_grad()
def fold_row_scales(model):
    """Replace each parametrised weight by its value W * c and drop the parametrisation."""
    for m in linears(model).values():
        if P.is_parametrized(m, "weight"):
            P.remove_parametrizations(m, "weight", leave_parametrized=True)


def rows_not_pure_rescale(W_after, W_before, tol=2e-2):
    """Rows in which W_after / W_before is not one constant (relative spread above tol) wherever W_before is non-zero;
    0 means every row differs from the start only by its factor, i.e. the integer grid is unchanged."""
    W, W0 = W_after.float(), W_before.float()
    nz = W0.abs() > 0
    ratio = torch.where(nz, W / torch.where(nz, W0, torch.ones_like(W0)), torch.zeros_like(W0))
    rmax = ratio.amax(dim=1, keepdim=True)
    rmin = torch.where(nz, ratio, rmax).amin(dim=1, keepdim=True)
    return int(((rmax - rmin).abs() > tol * rmax.abs().clamp(min=1e-6)).sum())
