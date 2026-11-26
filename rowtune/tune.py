"""The tuning loop: match the 16-bit teacher's next-token distribution with one multiplier per output row."""
import torch
import torch.nn.functional as F

from . import DEFAULT_LR
from .core import attach_block_scales, attach_row_scales, fold_row_scales, groups_not_pure_rescale, linears, rows_not_pure_rescale


def tune(student, teacher, windows, lr=DEFAULT_LR, device="cuda:0", teacher_device=None, only=None, block=0):
    """Train the row multipliers of `student` (4-bit values in its linear weights) against `teacher` on token windows,
    one window per Adam step (only the linears named in `only`, if given; block > 0: one multiplier per `block` columns instead of per row), loss = mean KL(teacher || student) over positions; then fold them into the weights.
    The student's weights are frozen apart from the multipliers. Returns scalar statistics and the learned scales."""
    teacher_device = teacher_device or device
    for p in teacher.parameters():
        p.requires_grad_(False)
    start = {n: m.weight.detach().float().cpu().clone() for n, m in linears(student).items() if only is None or n in only}
    mods = attach_block_scales(student, block, only) if block else attach_row_scales(student, only)
    opt = torch.optim.Adam([{"params": [rs.c for rs in mods.values()], "lr": lr}])
    student.train(False)
    losses = []
    for x in windows:
        x = torch.tensor([x], device=device)
        with torch.no_grad():
            lt = F.log_softmax(teacher(input_ids=x.to(teacher_device)).logits.float(), -1).to(device)
        ls = F.log_softmax(student(input_ids=x).logits.float(), -1)
        loss = (lt.exp() * (lt - ls)).sum(-1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    scales = {n: rs.c.detach().cpu().clone() for n, rs in mods.items()}
    cs = torch.cat([c.flatten() for c in scales.values()])
    fold_row_scales(student)
    pure = (lambda a, b: groups_not_pure_rescale(a, b, block)) if block else rows_not_pure_rescale
    changed = sum(pure(m.weight.detach().float().cpu(), start[n]) for n, m in linears(student).items() if n in start)
    k = max(1, len(losses) // 10)
    return {"kl_first10pct": sum(losses[:k]) / k, "kl_last10pct": sum(losses[-k:]) / k, "c_min": cs.min().item(),
            "c_max": cs.max().item(), "c_mean": cs.mean().item(), "c_std": cs.std().item(), "n_scales": cs.numel(),
            "steps": len(losses), "rows_not_pure_rescale": changed, "scales": scales}
