"""The only module that touches Hugging Face transformers (imported if installed)."""
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = CompressedTensorsConfig = None


try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def _need():
    if AutoModelForCausalLM is None:
        raise SystemExit("this command needs transformers (pip install transformers)")


def load_causal_lm(name, dtype, device="cpu", compressed=False):
    """A causal LM from a model id or local folder, on `device`. compressed=True decompresses a compressed-tensors
    4-bit release to its dequantised weights."""
    _need()
    kw = {}
    if compressed:
        kw["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw).to(device).eval()


def load_tokenizer(name):
    _need()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def default_text_parquet():
    """The tuning text of the case study: WikiText-103 (raw), training shard 2 of 2, from the Hugging Face hub (downloaded)."""
    if hf_hub_download is None:
        raise SystemExit("downloading the default text needs huggingface_hub (pip install huggingface_hub), or pass --text-file / --parquet")
    return hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/train-00001-of-00002.parquet", repo_type="dataset")


def save_checkpoint(model, tokenizer, out_dir):
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
