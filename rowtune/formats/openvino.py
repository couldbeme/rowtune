"""OpenVINO IR (openvino_model.xml + openvino_model.bin) with INT4 weights and one float16 scale per output channel.
Each decoder linear is stored by the exporter as: Const i4 codes [rows, cols] (two per byte, first element in the low nibble,
two's complement) -> Convert -> Multiply with Const f16 `<name>/scale` [rows, 1] -> Convert -> MatMul. A row factor is written
into that scale constant in place inside the .bin file: the xml, the codes and every other constant keep their bytes, so the
folder keeps its structure and size. Reading and writing need no OpenVINO installation (the xml is parsed directly)."""
import os
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .base import Linear4, Quantised, check_factors

NAME = "openvino"
WEIGHT = re.compile(r"self\.(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+_proj)\._openvino_orig_weight")
MATMUL = re.compile(r"__module\.model\.layers\.(\d+)\.(self_attn\.q_proj|mlp\.gate_proj)/ov_ext::linear/MatMul")
NORM_OF = {"self_attn.q_proj": "input_layernorm", "mlp.gate_proj": "post_attention_layernorm"}
NP = {"f16": np.float16, "f32": np.float32}


class _IR:
    pass


def _load(d):
    xml, binp = os.path.join(d, "openvino_model.xml"), os.path.join(d, "openvino_model.bin")
    if not (os.path.isfile(xml) and os.path.isfile(binp)):
        raise ValueError(f"{d} has no openvino_model.xml and openvino_model.bin")
    root = ET.parse(xml).getroot()
    ir = _IR()
    ir.layers, ir.consts, ir.src = {}, {}, {}
    for l in root.find("layers"):
        i = int(l.get("id"))
        ir.layers[i] = (l.get("name"), l.get("type"))
        dt = l.find("data")
        if l.get("type") == "Const" and dt is not None:
            shp = tuple(int(x) for x in dt.get("shape").replace(" ", "").split(",") if x) if dt.get("shape") else ()
            ir.consts[l.get("name")] = (i, dt.get("element_type"), shp, int(dt.get("offset")), int(dt.get("size")))
    for e in root.find("edges"):
        ir.src[(int(e.get("to-layer")), int(e.get("to-port")))] = int(e.get("from-layer"))
    ir.bin = binp
    return ir


def _linears(ir):
    """{hf name: (codes const, scale const)} with int4 codes and a [rows, 1] float scale; and names of unsupported ones."""
    ok, skipped = {}, []
    for name, c in ir.consts.items():
        m = WEIGHT.fullmatch(name)
        if not m:
            continue
        hf = m.group(1) + ".weight"
        s = ir.consts.get(name + "/scale")
        if c[1] == "i4" and s is not None and s[1] in NP and s[2] == (c[2][0], 1) and (name + "/zero_point") not in ir.consts:
            ok[hf] = (c, s)
        else:
            skipped.append(hf)
    return ok, skipped


def _unpack_i4(raw, shape):
    b = np.frombuffer(raw, np.uint8)
    v = np.stack([b & 15, b >> 4], 1).reshape(-1)[: int(np.prod(shape))].astype(np.int8)
    return np.where(v > 7, v - 16, v).reshape(shape).astype(np.int8)


def _norm_gain(ir, mm_id, hidden, mem):
    """The (1,1,hidden) gain constant of the Multiply that feeds MatMul input 0 (looking through Converts), or None."""
    n = ir.src.get((mm_id, 0))
    for _ in range(3):
        if n is None:
            return None
        name, typ = ir.layers[n]
        if typ == "Convert":
            n = ir.src.get((n, 0))
            continue
        if typ != "Multiply":
            return None
        for port in (0, 1):
            s = ir.src.get((n, port))
            if s is not None and ir.layers[s][1] == "Const":
                c = ir.consts[ir.layers[s][0]]
                if c[2] == (1, 1, hidden) and c[1] in NP:
                    return np.frombuffer(mem[c[3]:c[3] + c[4]], NP[c[1]]).astype(np.float32).copy()
        return None
    return None


def read(d):
    ir = _load(d)
    lin, skipped = _linears(ir)
    mem = np.memmap(ir.bin, mode="r")
    out = {}
    for hf, (c, s) in lin.items():
        codes = _unpack_i4(bytes(mem[c[3]:c[3] + c[4]]), c[2])
        scale = np.frombuffer(bytes(mem[s[3]:s[3] + s[4]]), NP[s[1]]).astype(np.float32).reshape(s[2])
        out[hf] = Linear4(codes, scale)
    extras = {}
    for i, (name, typ) in ir.layers.items():
        m = MATMUL.fullmatch(name) if typ == "MatMul" else None
        if m:
            hf = f"model.layers.{m.group(1)}.{m.group(2)}.weight"
            if hf in out:
                extras[f"model.layers.{m.group(1)}.{NORM_OF[m.group(2)]}.weight"] = _norm_gain(ir, i, out[hf].codes.shape[1], mem)
    return Quantised(out, extras, skipped)


def write(src, dst, factors):
    ir = _load(src)
    lin, _ = _linears(ir)
    check_factors(factors, {hf: c[2][0] for hf, (c, s) in lin.items()})
    if os.path.exists(dst):
        raise FileExistsError(dst)
    part = dst + ".part"
    shutil.copytree(src, part)
    mm = np.memmap(os.path.join(part, "openvino_model.bin"), mode="r+")
    for hf, c in factors.items():
        s = lin[hf][1]
        dt = NP[s[1]]
        old = np.frombuffer(bytes(mm[s[3]:s[3] + s[4]]), dt).astype(np.float32).reshape(s[2])
        new = (old * np.asarray(c, np.float32)[:, None]).astype(dt)
        mm[s[3]:s[3] + s[4]] = np.frombuffer(new.tobytes(), np.uint8)
    mm.flush()
    del mm
    os.replace(part, dst)


def _listing(d):
    return sorted((os.path.relpath(os.path.join(r, f), d), os.path.getsize(os.path.join(r, f))) for r, _, fs in os.walk(d) for f in fs)


def verify(src, dst):
    """dst against src: same file list and sizes, identical xml, identical int4 codes, only the scale constants differ."""
    a, b = _load(src), _load(dst)
    lin, _ = _linears(a)
    same_size = _listing(src) == _listing(dst)
    xml_same = open(os.path.join(src, "openvino_model.xml"), "rb").read() == open(os.path.join(dst, "openvino_model.xml"), "rb").read()
    ma, mb = np.memmap(a.bin, mode="r"), np.memmap(b.bin, mode="r")
    scale_spans = [(s[3], s[3] + s[4]) for c, s in lin.values()]
    code_spans = [(c[3], c[3] + c[4]) for c, s in lin.values()]
    codes_identical = all(np.array_equal(ma[x:y], mb[x:y]) for x, y in code_spans)
    changed = sum(int(not np.array_equal(ma[x:y], mb[x:y])) for x, y in scale_spans)
    mask = np.ones(len(ma), bool)
    for x, y in scale_spans:
        mask[x:y] = False
    rest_same = bool(np.array_equal(ma[mask], mb[mask])) if len(ma) == len(mb) else False
    rep = {"format": NAME, "same_size": bool(same_size), "same_structure": bool(same_size and xml_same), "codes_identical": bool(codes_identical),
           "all_bytes_outside_scales_identical": rest_same, "scale_tensors": len(lin), "scale_tensors_changed": changed, "bytes": int(len(mb))}
    rep["PASS"] = bool(same_size and xml_same and codes_identical and rest_same)
    return rep
