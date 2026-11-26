"""llama.cpp GGUF files with Q4_0 linear weights. Q4_0 block = float16 scale d + 16 bytes of 4-bit codes (32 weights); one row is
cols/32 blocks. A row factor multiplies the d of every block of the row; the code bytes are never written.
Layout and the GGUF-to-HF name and q/k row-order mapping follow my earlier write-back scripts (ggml-common.h block layouts)."""
import os
import re
import shutil
import struct

import numpy as np

from .base import Linear4, Quantised, check_factors

NAME = "gguf"
Q4_0 = 2
BLOCK = 18
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
LIN = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
       "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


def hf_linear_name(gname):
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gname)
    if m and m.group(2) in LIN:
        return f"model.layers.{m.group(1)}.{LIN[m.group(2)]}.weight", m.group(2)
    return None, None


def gguf_row_order(rows, n_head):
    """convert_hf_to_gguf.py LlamaModel.permute on an index vector: W_gguf = W_hf[perm] (rows of q or k)."""
    hd = rows // n_head
    if hd * n_head != rows or hd % 2:
        raise ValueError(f"cannot permute {rows} rows over {n_head} heads")
    return np.arange(rows).reshape(n_head, 2, hd // 2).swapaxes(1, 2).reshape(-1)


def _perm(arch, role, rows, n_head, n_head_kv):
    """Row permutation of a tensor, or None. Only the llama architecture is converted with the q/k permutation."""
    if arch != "llama":
        return None
    nh = n_head if role == "attn_q" else n_head_kv if role == "attn_k" else None
    return gguf_row_order(rows, nh) if nh else None


class _Info:
    pass


def parse(path):
    """Header of a GGUF file: metadata, and every tensor (name, ne dims, type, absolute data offset, byte length)."""
    with open(path, "rb") as f:
        buf = f.read(min(os.path.getsize(path), 1 << 26))  # header and tensor table sit in the first bytes
    if buf[:4] != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    ver, nt, nkv = struct.unpack_from("<IQQ", buf, 4)
    if ver < 2:
        raise ValueError(f"GGUF version {ver} not supported")
    pos = 24

    def rstr(p):
        n = struct.unpack_from("<Q", buf, p)[0]
        return buf[p + 8:p + 8 + n].decode("utf-8", "replace"), p + 8 + n

    def rval(t, p):
        if t == 8:
            return rstr(p)
        if t == 9:
            et, n = struct.unpack_from("<IQ", buf, p)
            p += 12
            out = []
            for _ in range(n):
                v, p = rval(et, p)
                out.append(v)
            return out, p
        fmt = _SCALAR[t]
        return struct.unpack_from(fmt, buf, p)[0], p + struct.calcsize(fmt)

    meta = {}
    for _ in range(nkv):
        k, pos = rstr(pos)
        t = struct.unpack_from("<I", buf, pos)[0]
        meta[k], pos = rval(t, pos + 4)
    raw = []
    for _ in range(nt):
        name, pos = rstr(pos)
        nd = struct.unpack_from("<I", buf, pos)[0]
        dims = struct.unpack_from(f"<{nd}Q", buf, pos + 4)
        typ, off = struct.unpack_from("<IQ", buf, pos + 4 + 8 * nd)
        pos += 4 + 8 * nd + 12
        raw.append((name, dims, typ, off))
    align = int(meta.get("general.alignment", 32))
    data0 = pos + (-pos) % align
    size = os.path.getsize(path)
    ends = sorted(data0 + o for _, _, _, o in raw) + [size]
    tensors = []
    for name, dims, typ, off in raw:
        a = data0 + off
        tensors.append((name, dims, typ, a, ends[ends.index(a) + 1] - a))
    info = _Info()
    arch = meta.get("general.architecture")
    info.meta, info.arch, info.data0, info.tensors = meta, arch, data0, tensors
    info.n_head, info.n_head_kv = meta.get(f"{arch}.attention.head_count"), meta.get(f"{arch}.attention.head_count_kv")
    return info


def _q4_0_linears(info):
    """{hf name: (gguf name, role, rows, cols, absolute offset, nbytes)} of Q4_0 linear tensors; and the other linears' hf names."""
    ok, skipped = {}, []
    for name, dims, typ, off, nb in info.tensors:
        hf, role = hf_linear_name(name)
        if hf is None or len(dims) != 2:
            continue
        if typ != Q4_0:
            skipped.append(hf)
            continue
        cols, rows = int(dims[0]), int(dims[1])
        ok[hf] = (name, role, rows, cols, off, rows * (cols // 32) * BLOCK)
    return ok, skipped


def read(path):
    info = parse(path)
    lin, skipped = _q4_0_linears(info)
    mm = np.memmap(path, mode="r")
    out = {}
    for hf, (name, role, rows, cols, off, n) in lin.items():
        blk = np.asarray(mm[off:off + n]).reshape(rows, cols // 32, BLOCK)
        d = np.ascontiguousarray(blk[:, :, :2]).view(np.float16)[:, :, 0].astype(np.float32)
        qs = blk[:, :, 2:]
        codes = np.concatenate([qs & 15, qs >> 4], axis=2).astype(np.int16) - 8  # element j low nibble, j + 16 high nibble
        codes, d = codes.reshape(rows, cols).astype(np.int8), d.reshape(rows, cols // 32)
        p = _perm(info.arch, role, rows, info.n_head, info.n_head_kv)
        if p is not None:
            inv = np.argsort(p)
            codes, d = codes[inv], d[inv]
        out[hf] = Linear4(codes, d)
    return Quantised(out, {}, skipped)


def write(src, dst, factors):
    """Copy src byte for byte, then multiply the block scales d of each Q4_0 linear by the factors (HF row order): one factor per
    row, or one per 32-weight block ([rows, cols/32])."""
    info = parse(src)
    lin, _ = _q4_0_linears(info)
    check_factors(factors, {hf: v[2] for hf, v in lin.items()}, {hf: v[3] // 32 for hf, v in lin.items()})
    if os.path.exists(dst):
        raise FileExistsError(dst)
    part = dst + ".part"
    shutil.copyfile(src, part)
    mm = np.memmap(part, mode="r+")
    for hf, c in factors.items():
        name, role, rows, cols, off, n = lin[hf]
        c = np.asarray(c, np.float32)
        p = _perm(info.arch, role, rows, info.n_head, info.n_head_kv)
        if p is not None:
            c = c[p]
        blk = mm[off:off + n].reshape(rows, cols // 32, BLOCK)
        d = np.ascontiguousarray(blk[:, :, :2]).view(np.float16).astype(np.float32)
        cb = c[:, None, None] if c.ndim == 1 else c[:, :, None]
        blk[:, :, :2] = (d * cb).astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
    mm.flush()
    del mm
    os.replace(part, dst)


def verify(src, dst):
    """Facts about dst against src: same size, same header/metadata/padding, 4-bit code bytes identical, other tensors identical."""
    a, b = parse(src), parse(dst)
    lin, _ = _q4_0_linears(a)
    ma, mb = np.memmap(src, mode="r"), np.memmap(dst, mode="r")
    same_size = os.path.getsize(src) == os.path.getsize(dst)
    same_structure = same_size and a.data0 == b.data0 and bytes(ma[:a.data0]) == bytes(mb[:b.data0]) and \
        [t[:4] for t in a.tensors] == [t[:4] for t in b.tensors]
    codes_identical, others_identical, changed_t, changed_rows, padding_ok = True, True, 0, 0, True
    if same_structure:
        patched = {v[4]: v[5] for v in lin.values()}
        for hf, (name, role, rows, cols, off, n) in lin.items():
            A = np.asarray(ma[off:off + n]).reshape(rows, cols // 32, BLOCK)
            B = np.asarray(mb[off:off + n]).reshape(rows, cols // 32, BLOCK)
            codes_identical &= bool(np.array_equal(A[:, :, 2:], B[:, :, 2:]))
            diff = (A[:, :, :2] != B[:, :, :2]).any(axis=(1, 2))
            changed_rows += int(diff.sum())
            changed_t += int(diff.any())
        for name, dims, typ, off, nb in a.tensors:
            start = off + patched.get(off, 0)  # past the patched blocks; whatever follows (padding) or the whole tensor must be identical
            same = bool(np.array_equal(ma[start:off + nb], mb[start:off + nb]))
            if off in patched:
                padding_ok &= same
            else:
                others_identical &= same
    rep = {"format": NAME, "same_size": same_size, "same_structure": same_structure, "codes_identical": codes_identical,
           "other_tensors_identical": others_identical, "padding_identical": padding_ok, "scale_tensors": len(lin),
           "scale_tensors_changed": changed_t, "scale_rows_changed": changed_rows, "bytes": os.path.getsize(dst)}
    rep["PASS"] = bool(same_size and same_structure and codes_identical and others_identical and padding_ok)
    return rep
