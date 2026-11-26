"""Readers and in-place writers for 4-bit model files. Each format module offers read(path), write(src, dst, factors) and
verify(src, dst); in every case only the per-row (or per-block-row) scale bytes change."""
import os

from . import ct, gguf, openvino

FORMATS = {"gguf": gguf, "openvino": openvino, "ct": ct}


def detect(path):
    """The format module of a file or folder: GGUF file, OpenVINO IR folder, or packed 4-bit safetensors checkpoint folder."""
    if os.path.isfile(path):
        with open(path, "rb") as f:
            if f.read(4) == b"GGUF":
                return gguf
    elif os.path.isdir(path):
        if os.path.exists(os.path.join(path, "openvino_model.xml")):
            return openvino
        if ct.is_checkpoint(path):
            return ct
    raise ValueError(f"{path}: not a GGUF file, an OpenVINO IR folder (openvino_model.xml) or a packed 4-bit safetensors folder")
