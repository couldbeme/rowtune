<h1 align="center">rowtune</h1>

<p align="center"><b>Make a 4-bit model file more accurate in about two minutes.<br>Same format. Same size. Same integer codes.</b></p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-you-can-do-with-it">What you can do with it</a> ·
  <a href="#results">Results</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#limits">Limits</a>
</p>

<p align="center"><img src="docs/kept.svg" alt="Accuracy kept as a share of the 16-bit model. OpenVINO INT4 export 87.97 percent, 95.26 after rowtune. GGUF Q4_0 file 94.80 percent, 98.33 after rowtune." width="760"></p>

When a language model is rounded to 4 bits so that it fits on a laptop or a small chip, it loses accuracy. `rowtune` takes the
4-bit file you already have and gives back most of what the rounding cost. It does not retrain the model, it does not change a
single 4-bit code, and the file you get out has the same structure and the same number of bytes as the file you put in.

```
rowtune model-q4_0.gguf -o model-q4_0-rowtune.gguf --teacher path/to/16-bit-model
```

On a Qwen3-1.7B file this takes 145 seconds on one GPU and lifts accuracy from 94.80% to 98.33% of the 16-bit model.

## Quick start

```
pip install -e .                                      # torch, numpy
pip install transformers huggingface_hub pyarrow      # to load the teacher and the default tuning text
```

One command per supported input:

```
# a llama.cpp GGUF file with Q4_0 weights
rowtune models/Qwen3-1.7B-Q4_0.gguf -o out/Qwen3-1.7B-Q4_0-rowtune.gguf --teacher models/qwen3-1.7b

# an OpenVINO IR folder with INT4 weights, one scale per channel
rowtune models/qwen3-1.7b-int4-channel -o out/qwen3-1.7b-int4-rowtune --teacher models/qwen3-1.7b

# a 4-bit safetensors checkpoint in the compressed-tensors layout (see Limits: not yet verified on a real release)
rowtune models/Qwen3-1.7B-quantized.w4a16 -o out/Qwen3-1.7B-w4a16-rowtune
```

Next to the output you get two small files: `<output>.rowtune.json` with the statistics and the result of the file check, and
`<output>.row_scales.pt` with the learned factors.

**The file check.** After writing, the tool reads both files again and compares them. If one code byte, the structure or the size
differs, it stops with an error. You never get a file that only looks the same.

## What you can do with it

- **Upgrade the models you already run locally.** Point it at a Q4_0 GGUF file and load the result in the same runtime.
- **Get a better model onto an NPU.** OpenVINO's channel-wise INT4 format is the one its NPU path uses, and it is where the gain
  is largest: +4.13 points on the documented export.
- **Use it as the last step after another method.** It runs on top of GPTQ, AWQ or OmniQuant output and adds to them.
- **Aim it at your own domain.** Pass your own text with `--text-file`. Tuned on public Python code, a 4-bit release went from
  33.1% to 39.5% pass@1 on HumanEval and MBPP.
- **Add it to a release pipeline.** It needs two minutes and no labels, so it can run on every 4-bit build.
- **Study what rounding breaks.** `row_scales.pt` shows which rows of which layers had to move, and by how much.
- **Port it to another format.** The method fits any format that stores a real-valued scale next to the 4-bit codes. GGUF
  K-quants and MLX are the obvious next ones; see [Formats](#formats).

## Results

Everything below is on the same 10,277 questions from six zero-shot tasks (ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande,
LAMBADA). Differences are paired against the same file without the tuning, in points of mean accuracy, with 95% bootstrap
intervals over questions. "Kept" is the share of the 16-bit model's accuracy.

### Measured with this tool, end to end

One GPU each (RTX 5000 Ada), 26 November 2026, one run and one seed.

- **Qwen3-1.7B, GGUF Q4_0:** 145 seconds. +2.00 points [+1.43, +2.56]. Kept 94.80% → 98.33%. File size identical:
  1,129,782,944 bytes.
- **Qwen3-1.7B, OpenVINO INT4** (the documented AWQ + Scale Estimation export): 149 seconds. +4.13 points [+3.39, +4.85].
  Kept 87.97% → 95.26%. File size identical: 1,020,409,650 bytes.

The scores of the two input files come from my earlier runs of the same evaluation.

### Measured with the scripts this tool was built from

`REPRODUCE.md` lists the commands and says which of these have not been re-run with the tool.

- **Plain per-row rounding:** Qwen3-1.7B 82.4% → 94.7% kept; SmolLM2-1.7B 82.6% → 92.6%.
- **On top of GPTQ, per row:** Qwen3-1.7B 91.9% → 97.3% (+3.03 [+2.29, +3.79]; two repeat runs gave 95.0% and 94.9%);
  Qwen3-8B 94.9% → 97.9% (+2.00 [+1.36, +2.68]).
- **Published 4-bit releases:** Qwen3-1.7B +1.24 [+0.75, +1.71]; Qwen3-8B +1.13 [+0.72, +1.56], which leaves it at
  −0.20 [−0.72, +0.33] against 16-bit.
- **Three llama.cpp Q4_0 files:** +2.11, +1.17 and +1.25, all with intervals above zero. On Q4_K_M files the result is a tie.
- **Code** (HumanEval + MBPP, pass@1, tuned on public Python code): 33.1% → 39.5%, where 16-bit scores 42.0%. One run.
- **On a chip:** one run on Apple's Neural Engine, 500 LAMBADA items: 27.4% → 47.6%, against 50.6% for 16-bit on a GPU.

### Time

<p align="center"><img src="docs/minutes.svg" alt="GPU time to repair the same 4-bit model: OmniQuant 67.1 minutes for 92.7 percent, rowtune 2.0 minutes for 92.6 percent." width="760"></p>

About 2 minutes on an existing 4-bit file and about 4 minutes from the original weights, for a 1.7B model on one GPU. On the
same model OmniQuant's official code took 67.1 minutes to reach 92.7%; this method reached 92.6% in 2.0 minutes, a statistical
tie, and added 1.71 points [+1.21, +2.24] when run on top of OmniQuant.

## How it works

**In plain words.** Rounding snaps every weight to one of 16 values. Each snap is a small error, and the errors add up through
the layers. Every row of weights already shares one scale. I leave the snapped values alone and adjust only those scales, by
letting the 4-bit model and the original read the same text and moving the scales wherever their predictions differ.

**Precisely.** For every decoder linear layer I train one multiplier per output row so that the 4-bit model's next-token
distribution matches the 16-bit teacher's. The loss is the KL divergence to the teacher, the optimiser is Adam with learning
rate 1e-3, and the default run is 600 windows of 1,024 tokens of unlabelled text, one window per step (about 0.6 million tokens).
That is roughly 0.03% of the model's numbers. The learned factors are then multiplied into the scales the file already stores
(float16 for GGUF and OpenVINO, the stored type for safetensors), and nothing else is written.

Tuning text: `--text-file` or `--parquet`; by default a WikiText-103 training shard, downloaded.

`rowtune round --model M --out DIR` rounds a 16-bit model to 4 bits per row, tunes it, and writes a 16-bit checkpoint of the
result. Python entry points: `rowtune.core.attach_row_scales`, `rowtune.tune.tune`, `rowtune.core.fold_row_scales`,
`rowtune.upgrade.upgrade`, `rowtune.formats.{gguf,openvino,ct}`.

## Formats

- **Supported and tested with this tool:** llama.cpp GGUF Q4_0, and OpenVINO INT4 with one scale per channel.
- **Next, not yet tested:** GGUF K-quants such as Q4_K_M, whose scales are stored in 6 bits, so a tuned scale has to be rounded
  again and part of the gain may be lost; and MLX 4-bit.
- **Does not fit as it stands:** formats that store a codebook instead of a scale (palettised Core ML, the GGUF IQ types).

## What has been checked, and what has not

- **Tests** (67, no GPU, no download): on tiny synthetic files of each format, codes are bit-identical, scales changed,
  structure and byte size unchanged, the file loads again, and the upgraded file is closer to the teacher than the input.
  Run them with `python -I -m unittest`.
- **Real files, end to end:** the two results above. All 196 linear layers were handled in both files.
- **In llama.cpp:** the upgraded Qwen3-1.7B Q4_0 file (checksum identical to the scored file) loads in llama.cpp
  (`llama-cli`, build 11669) on an Apple M2 Max, is recognised as Q4_0 and answers three test prompts correctly at about
  190 tokens per second. That is a load-and-answer check, not a benchmark.
- **Not run yet:** the safetensors layout on a real release (it is written from library documentation); a second seed.

## Limits

- **Four bits is the floor.** At 2 bits SmolLM2 keeps 48.8% of 16-bit accuracy, where random answers give 48.1%. At 3 bits per
  row it keeps 64.2%.
- **GGUF:** only Q4_0 tensors are read and patched. A linear weight stored in another type is left byte-identical and, during
  tuning, stands in as the 16-bit teacher weight, so results on such files can differ from the numbers above.
- **OpenVINO:** symmetric INT4 with one float16 scale per output channel. Asymmetric constants are skipped. RMSNorm gains are
  read from the graph, so an AWQ export's folded gains are used.
- **Safetensors:** scales are stored in float16 or bfloat16, so the written factor is rounded (relative error up to 0.05% and
  0.4%).
- **Mostly GPU evaluation.** The accuracy numbers come from running the rounded weights on a GPU, not from integer kernels,
  apart from the one Neural Engine run.
- **Small sample of models.** Three models (SmolLM2-1.7B, Qwen3-1.7B, Qwen3-8B), one family of tuning text, and one run per
  result except where repeats are stated. Repeated runs vary by up to 2.4 points of kept share.
- **One variant is left out.** A further version of the method scores slightly higher but does not fit a stock 4-bit file
  format. It is not included and I make no claim for it.
- The 4-bit rounding in `rowtune round` is my own implementation and has not been checked bit for bit against other tools.

## Contributing

Ports to other formats are the most useful contribution. A format reader needs three things: find each linear layer's codes
and scales, write new scales back without touching anything else, and pass the file check. The GGUF and OpenVINO readers in
`rowtune/formats/` are short and can serve as a pattern.

## AI assistance

An AI assistant (Claude) helped write the code, tests and documentation in this repository. The method, the experiments and
the numbers come from my own work.

## Licence

Apache-2.0, see `LICENSE` and `NOTICE`. Copyright 2026 Alexandra Zhuravlyova.
