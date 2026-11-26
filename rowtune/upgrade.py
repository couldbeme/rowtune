"""Upgrade a 4-bit model file in place of its format: read the codes and scales, tune one factor per row against the 16-bit
teacher, write the factors into the scales of a copy of the file."""
import numpy as np
import torch

from . import DEFAULT_LR
from .core import linears
from .formats import detect
from .tune import tune


def effective_factors(scales, dtype):
    """Row factors as the tuned model used them: rounded to the student's dtype (bfloat16 weights multiply by bf16(c))."""
    return {n: c.to(dtype).float().numpy() for n, c in scales.items()}


@torch.no_grad()
def student_from(q, factory, dtype):
    """A fresh model from `factory`, with the file's dequantised linears and norm gains; everything else (embeddings, head, any
    linear the file does not hold in a supported 4-bit form) stays as the factory model has it."""
    model = factory()
    mods = linears(model)
    for hf, lin in q.linears.items():
        m = mods[hf.removesuffix(".weight")]
        m.weight.copy_(torch.from_numpy(lin.dequant()).to(m.weight.device, m.weight.dtype))
    params = dict(model.named_parameters())
    for name, v in q.extras.items():
        p = params[name]
        if v is None:  # the exporter dropped a gain of exactly 1; the model must agree
            if not torch.allclose(p.float(), torch.ones_like(p, dtype=torch.float32), rtol=0, atol=1e-6):
                raise ValueError(f"the file holds no gain for {name} but the teacher's gain is not 1")
        else:
            p.copy_(torch.from_numpy(np.asarray(v)).to(p.device, p.dtype))
    return model


def upgrade(src, dst, teacher, factory, windows, lr=DEFAULT_LR, device="cuda:0", teacher_device=None, dtype=torch.bfloat16):
    """Returns the tuning statistics (with the learned `scales`), the format name, skipped tensors and the file check of dst against src."""
    fmt = detect(src)
    q = fmt.read(src)
    if not q.linears:
        raise ValueError(f"{src}: no supported 4-bit linear weights found")
    student = student_from(q, factory, dtype).to(device).eval()
    names = [n.removesuffix(".weight") for n in q.linears]
    info = tune(student, teacher, windows, lr=lr, device=device, teacher_device=teacher_device, only=names)
    factors = {n + ".weight": f for n, f in effective_factors(info["scales"], dtype).items()}
    fmt.write(src, dst, factors)
    info.update(format=fmt.NAME, skipped=q.skipped, file_check=fmt.verify(src, dst))
    if not info["file_check"]["PASS"]:
        raise RuntimeError(f"written file failed its check: {info['file_check']}")
    return info
