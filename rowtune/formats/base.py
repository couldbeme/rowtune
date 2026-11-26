"""Shared pieces of the format readers and writers."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Linear4:
    """One 4-bit linear weight: integer codes in -8..7 [rows, cols] and float32 scales [rows, groups] (groups = 1 for one scale per row)."""
    codes: np.ndarray
    scale: np.ndarray

    def dequant(self):
        return self.codes.astype(np.float32) * np.repeat(self.scale, self.codes.shape[1] // self.scale.shape[1], axis=1)


@dataclass
class Quantised:
    """What a reader returns. linears: {HF weight name: Linear4}; extras: {HF parameter name: float32 vector, or None when the file
    does not hold it}; skipped: HF names of linear weights present in the file in a form that is not supported (left untouched)."""
    linears: dict
    extras: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)


def check_factors(factors, rows):
    """factors: {HF weight name: float vector}; rows: {HF weight name: row count} of what the file can patch."""
    for n, c in factors.items():
        if n not in rows:
            raise ValueError(f"no supported 4-bit linear named {n} in the file")
        if np.asarray(c).shape != (rows[n],):
            raise ValueError(f"{n}: {np.asarray(c).shape} factors for {rows[n]} rows")


def to_bf16_bits(x):
    """float32 -> bfloat16 bit patterns (uint16), round to nearest even, as torch .to(torch.bfloat16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def from_bf16_bits(b):
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)
