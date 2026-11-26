"""Independent writers of tiny 4-bit files in the three supported layouts (the oracle for the readers and writers in
rowtune.formats): a compressed-tensors style checkpoint, an OpenVINO IR with int4 constants, a llama.cpp GGUF with Q4_0."""
import json
import os
import struct

import numpy as np

from rowtune.core import linears, quantize_rtn

# ---- shared ---------------------------------------------------------------------------------------------------------


def quantised_linears(model, group_size=0):
    """{hf weight name: (codes int8 [rows, cols] in -8..7, scale float32 [rows, groups])} from the model's decoder linears."""
    out = {}
    for n, m in linears(model).items():
        _, codes, scale = quantize_rtn(m.weight.detach(), 4, group_size)
        out[n + ".weight"] = (codes.numpy().astype(np.int8), scale.numpy().astype(np.float32))
    return out


def deq(codes, scale):
    rows, cols = codes.shape
    return codes.astype(np.float32) * np.repeat(scale, cols // scale.shape[1], axis=1)


def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def from_bf16_bits(b):
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


# ---- GGUF -----------------------------------------------------------------------------------------------------------
F32, F16, Q4_0, Q8_0, Q6_K = 0, 1, 2, 8, 14


def q4_0_bytes(codes, d):
    """codes int8 -8..7 [rows, cols]; d float16 [rows, cols/32] -> raw Q4_0 blocks (2-byte d, 16 bytes: element j low nibble, j+16 high)."""
    rows, cols = codes.shape
    u = (codes.astype(np.int16) + 8).astype(np.uint8).reshape(rows, cols // 32, 32)
    qs = (u[..., :16] | (u[..., 16:] << 4)).astype(np.uint8)
    dd = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(rows, cols // 32, 2)
    return np.concatenate([dd, qs], axis=2).tobytes()


def _gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def build_gguf(path, tensors, arch="qwen3", n_head=1, n_head_kv=1, alignment=32):
    """tensors: list of (gguf name, ne dims [cols, rows], ggml type, raw bytes)."""
    kv = [("general.architecture", 8, _gstr(arch)), (f"{arch}.attention.head_count", 4, struct.pack("<I", n_head)),
          (f"{arch}.attention.head_count_kv", 4, struct.pack("<I", n_head_kv)), ("general.alignment", 4, struct.pack("<I", alignment))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        head += _gstr(k) + struct.pack("<I", t) + v
    off, infos, blobs = 0, b"", []
    for name, dims, typ, raw in tensors:
        infos += _gstr(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off)
        pad = (-len(raw)) % alignment
        blobs.append(raw + b"\0" * pad)
        off += len(raw) + pad
    pre = head + infos
    pre += b"\0" * ((-len(pre)) % alignment)
    with open(path, "wb") as f:
        f.write(pre + b"".join(blobs))


GGUF_ROLE = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
             "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def gguf_name(hf):
    _, _, L, rest = hf.split(".", 3)
    return f"blk.{L}.{GGUF_ROLE[rest.removesuffix('.weight')]}.weight"


def write_gguf_from(path, model, arch="qwen3", other_type_for=None, n_head=1, n_head_kv=1, group_size=32):
    """A Q4_0 GGUF of the model's decoder linears (one scale per 32 weights), a float32 norm and an F16 embedding.
    other_type_for: hf name written as Q8_0 bytes instead (to test that non-Q4_0 linears are left alone)."""
    q = quantised_linears(model, group_size)
    ts = []
    for hf, (codes, scale) in q.items():
        rows, cols = codes.shape
        if arch == "llama" and hf.endswith(("q_proj.weight", "k_proj.weight")):  # convert_hf_to_gguf.py permutes the rows of q and k
            hd = rows // n_head
            perm = np.arange(rows).reshape(n_head, 2, hd // 2).swapaxes(1, 2).reshape(-1)
            codes, scale = codes[perm], scale[perm]
        if hf == other_type_for:
            raw = bytes(range(256)) * ((rows * cols // 32 * 34) // 256 + 1)
            ts.append((gguf_name(hf), (cols, rows), Q8_0, raw[: rows * cols // 32 * 34]))
        else:
            ts.append((gguf_name(hf), (cols, rows), Q4_0, q4_0_bytes(codes, scale)))
    ts.append(("blk.0.attn_norm.weight", (32,), F32, np.linspace(0.5, 1.5, 32, dtype=np.float32).tobytes()))
    ts.append(("token_embd.weight", (32, 50), F16, np.arange(50 * 32, dtype=np.float16).tobytes()))
    build_gguf(path, ts, arch, n_head, n_head_kv)
    return q


# ---- OpenVINO IR ----------------------------------------------------------------------------------------------------
def _ov_pack_i4(codes):
    u = (codes.astype(np.int16) & 15).astype(np.uint8).reshape(-1)
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()


def write_ir_from(dirpath, model, norms=True):
    """openvino_model.xml/.bin: per linear i4 codes, f16 scale [rows,1], Convert, Multiply, Convert, MatMul as the exporter writes them;
    the RMSNorm gains as f32 (1,1,hidden) constants feeding the q/gate MatMuls; one int8 embedding constant that must stay untouched."""
    os.makedirs(dirpath, exist_ok=True)
    q = quantised_linears(model, 0)
    layers, edges, blob, nid = [], [], bytearray(), [0]

    def new(name, typ, data="", outs=""):
        i = nid[0]
        nid[0] += 1
        layers.append(f'<layer id="{i}" name="{name}" type="{typ}" version="opset1">{data}<output><port id="0" precision="FP16"/></output></layer>')
        return i

    def const(name, et, shape, raw):
        off = len(blob)
        blob.extend(raw)
        return new(name, "Const", f'<data element_type="{et}" shape="{", ".join(map(str, shape))}" offset="{off}" size="{len(raw)}"/>')

    def edge(a, b, port):
        edges.append(f'<edge from-layer="{a}" from-port="0" to-layer="{b}" to-port="{port}"/>')

    hidden = {}
    for l in range(len(model.model.layers)):
        for tag in ("input_layernorm", "post_attention_layernorm"):
            x = new("hidden_in", "Parameter", '<data shape="?,?,32" element_type="f32"/>')
            if norms:
                g = getattr(model.model.layers[l], tag).weight.detach().numpy().astype(np.float32)
                c = const(f"__module.model.layers.{l}.{tag}/aten::mul/Constant", "f32", (1, 1, g.size), g.tobytes())
                mulid = new(f"__module.model.layers.{l}.{tag}/aten::mul/Multiply_1", "Multiply", "")
                edge(x, mulid, 0)
                edge(c, mulid, 1)
                hidden[(l, tag)] = mulid
            else:
                hidden[(l, tag)] = x
    for hf, (codes, scale) in q.items():
        base = hf.removesuffix(".weight")
        l = int(base.split(".")[2])
        rows, cols = codes.shape
        cid = const(f"self.{base}._openvino_orig_weight", "i4", (rows, cols), _ov_pack_i4(codes))
        cv = new("Convert_a", "Convert", '<data destination_type="f16"/>')
        sid = const(f"self.{base}._openvino_orig_weight/scale", "f16", (rows, 1), scale.astype(np.float16).tobytes())
        mul = new(f"self.{base}._openvino_orig_weight/fq_weights_1", "Multiply", '<data auto_broadcast="numpy"/>')
        cv2 = new(f"__module.{base}/ov_ext::linear/Convert", "Convert", '<data destination_type="f32"/>')
        mm = new(f"__module.{base}/ov_ext::linear/MatMul", "MatMul", '<data transpose_a="false" transpose_b="true"/>')
        edge(cid, cv, 0)
        edge(cv, mul, 0)
        edge(sid, mul, 1)
        edge(mul, cv2, 0)
        edge(cv2, mm, 1)
        if base.endswith(("q_proj", "k_proj", "v_proj")):
            src = hidden[(l, "input_layernorm")]
        elif base.endswith(("gate_proj", "up_proj")):
            src = hidden[(l, "post_attention_layernorm")]
        else:
            src = new("act_in", "Parameter", '<data shape="?,?,32" element_type="f32"/>')
        edge(src, mm, 0)
    const("self.model.embed_tokens._openvino_orig_weight", "i8", (50, 32), bytes(range(50)) * 32)
    const("self.model.embed_tokens._openvino_orig_weight/scale", "f16", (50, 1), np.full(50, 0.01, np.float16).tobytes())
    xml = '<?xml version="1.0"?>\n<net name="Model0" version="11">\n<layers>\n' + "\n".join(layers) + "\n</layers>\n<edges>\n" + "\n".join(edges) + "\n</edges>\n</net>\n"
    with open(os.path.join(dirpath, "openvino_model.xml"), "w") as f:
        f.write(xml)
    with open(os.path.join(dirpath, "openvino_model.bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump({"_name_or_path": "tiny-teacher"}, f)
    return q


# ---- compressed-tensors style safetensors ----------------------------------------------------------------------------
def write_safetensors(path, tensors):
    """tensors: {name: (dtype string, shape, raw bytes)}"""
    hdr, off = {"__metadata__": {"format": "pt"}}, 0
    for n, (dt, shape, raw) in tensors.items():
        hdr[n] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for _, _, raw in tensors.values():
            f.write(raw)


def pack_int32(codes):
    """int8 codes -8..7 [rows, cols] -> int32 [rows, cols/8], offset binary (+8), element 0 in the lowest nibble."""
    rows, cols = codes.shape
    u = (codes.astype(np.int64) + 8).reshape(rows, cols // 8, 8)
    w = np.zeros((rows, cols // 8), np.uint64)
    for i in range(8):
        w |= u[..., i].astype(np.uint64) << np.uint64(4 * i)
    return w.astype(np.uint32).view(np.int32)


def write_ct_from(dirpath, model, group_size=0, scale_dtype="BF16", shards=1):
    os.makedirs(dirpath, exist_ok=True)
    q = quantised_linears(model, group_size)
    tens = {}
    for hf, (codes, scale) in q.items():
        base = hf.removesuffix(".weight")
        rows, cols = codes.shape
        tens[base + ".weight_packed"] = ("I32", (rows, cols // 8), pack_int32(codes).tobytes())
        if scale_dtype == "BF16":
            tens[base + ".weight_scale"] = ("BF16", scale.shape, bf16_bits(scale).tobytes())
        else:
            tens[base + ".weight_scale"] = ("F16", scale.shape, scale.astype(np.float16).tobytes())
        tens[base + ".weight_shape"] = ("I32", (2,), np.array([rows, cols], np.int32).tobytes())
    for n, p in model.named_parameters():
        if n.endswith(("layernorm.weight",)) or n in ("model.embed_tokens.weight", "lm_head.weight"):
            tens[n] = ("BF16", tuple(p.shape), bf16_bits(p.detach().numpy()).tobytes())
    names = list(tens)
    parts = [names[i::shards] for i in range(shards)]
    wm = {}
    for i, part in enumerate(parts):
        fn = "model.safetensors" if shards == 1 else f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        write_safetensors(os.path.join(dirpath, fn), {n: tens[n] for n in part})
        wm.update({n: fn for n in part})
    if shards > 1:
        with open(os.path.join(dirpath, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": wm}, f)
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump({"_name_or_path": "tiny-teacher", "quantization_config": {"format": "pack-quantized"}}, f)
    return q
