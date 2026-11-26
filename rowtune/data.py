"""Unlabelled text windows for tuning."""
import random

try:
    import pyarrow.parquet as pq
except ImportError:  # only needed for --parquet
    pq = None

MIN_CHARS = 200  # shorter documents are dropped
DOC_MARKER = "\n<<<DOC>>>\n"


def windows_from_texts(tokenizer, texts, n, length, seed):
    """Up to n token windows of `length`: documents longer than 200 characters, shuffled with `seed`, tokenised
    without special tokens and cut consecutively from one running token stream."""
    texts = [t for t in texts if len(t) > MIN_CHARS]
    random.Random(seed).shuffle(texts)
    out, cur = [], []
    for t in texts:
        cur += tokenizer(t, add_special_tokens=False).input_ids
        while len(cur) >= length and len(out) < n:
            out.append(cur[:length])
            cur = cur[length:]
        if len(out) >= n:
            break
    return out


def read_text_file(path):
    """Documents of a UTF-8 text file: split on a line holding only <<<DOC>>> if the file has one, else on lines."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    return raw.split(DOC_MARKER) if DOC_MARKER in raw else raw.split("\n")


def read_parquet_texts(path, column="text"):
    if pq is None:
        raise SystemExit("--parquet needs pyarrow (pip install pyarrow)")
    return pq.read_table(path, columns=[column]).column(column).to_pylist()
