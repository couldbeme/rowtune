# Reproducing the case-study rows

Every number below is copied from my case study (23 November 2026) and its bootstrap table. Nothing in this file was
re-run for the release: **not run here** applies to every measured number, because the measurements need GPUs, model
downloads and the original evaluation driver, none of which were available when this repository was prepared. What was
run here is the test suite (see the end).

All gains are paired differences in mean accuracy (points) over six zero-shot tasks, with 95% intervals from a paired
bootstrap (2,000 resamples, seed 0, questions resampled within each task, the same indices for every version).
"Kept" is a version's mean divided by the 16-bit model's mean. Repeated runs of the method vary by up to 2.4 points
of kept share (different shuffles of the tuning text), so a fresh run will not match to the decimal.

## 1. Setup

    pip install -e .                 # torch
    pip install transformers         # model loading and saving
    pip install pyarrow              # only for --parquet
    pip install lm-eval              # only for the accuracy numbers

Models (the case study used these revisions; download to a local folder and pass the folder as `--model`):

    huggingface-cli download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir models/qwen3-1.7b
    huggingface-cli download HuggingFaceTB/SmolLM2-1.7B-Instruct --revision 31b70e2e869a7173562077fd711b654946d38674 --local-dir models/smollm2-1.7b
    huggingface-cli download Qwen/Qwen3-8B --local-dir models/qwen3-8b          # main branch as of 22 Nov 2026; no revision recorded

Tuning text: the second shard of the WikiText-103 (raw) training split, shuffled with seed 1, 600 windows of 1,024 tokens
(about 0.6 million tokens). The case study read this shard from a local copy; that it is byte-identical to the file
below is not verified here.

    huggingface-cli download Salesforce/wikitext wikitext-103-raw-v1/train-00001-of-00002.parquet --repo-type dataset --local-dir data
    TEXT=data/wikitext-103-raw-v1/train-00001-of-00002.parquet

## 2. Accuracy (the same for every row)

The case study scored the first 2,000 documents of each task (all of them where a task is smaller), zero-shot,
log-likelihood scoring without a chat template, batch size 32, metric `acc` (not `acc_norm`). 16-bit reference: the
original bfloat16 weights.

    lm_eval --model hf --model_args pretrained=OUT,dtype=bfloat16 \
      --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,lambada_openai \
      --num_fewshot 0 --batch_size 32 --limit 2000 --output_path OUT/eval

Mean accuracy is the unweighted mean of the six `acc,none` values. The case study ran lm-eval through its own driver
(the harness version 0.4.13 is recorded for some runs only), and the paired bootstrap script reads that driver's
per-question files, so the intervals below are not reproducible from `lm_eval` output without writing that step.
**Not run here.**

## 3. Rows

In each row `--device cuda:0` is one 32 GB GPU unless stated; add `--dtype bf16` (the default).

### 3.1 Per-row 4-bit from plain rounding, Qwen3-1.7B

    rowtune round --model models/qwen3-1.7b --parquet $TEXT --out out/q17-rtn-row

Expected, from the case study: 94.7% kept [93.3, 96.2] (mean accuracy 53.6) against 82.4% [80.8, 83.9] (46.6) for plain
per-row rounding without tuning. The run recorded in the result file: first-10% KL 0.472, last-10% KL 0.288,
573,440 scales, scale range 0.657 to 1.369, 136.5 s on one GPU (this includes rounding and loading). `tuning.json`
written by the command has the same fields. Not run here.

### 3.2 Per-row 4-bit from plain rounding, SmolLM2-1.7B

    rowtune round --model models/smollm2-1.7b --parquet $TEXT --out out/smol-rtn-row

Expected: 92.6% kept [91.3, 93.8] (mean 56.1) against 82.6% [81.2, 84.0] (50.1) plain; first-10% KL 0.446, last-10% KL
0.276, 638,976 scales, 122.6 s. Not run here.

### 3.3 Per-row 4-bit from GPTQ, Qwen3-1.7B and Qwen3-8B

The starting point is a GPTQ model rounded per row (llm-compressor, symmetric 4-bit, one scale per channel, 128
calibration sequences of 2,048 tokens of WikiText-103 train, seed 0, output head kept in 16 bit), stored as dequantised
16-bit weights in an ordinary checkpoint folder. This repository does not contain a GPTQ step; build that folder with
your own GPTQ run and pass it as `--student`.

    rowtune round --model models/qwen3-1.7b --student gptq-row/qwen3-1.7b --parquet $TEXT --out out/q17-gptq-row
    rowtune round --model models/qwen3-8b --student gptq-row/qwen3-8b --parquet $TEXT --out out/q8-gptq-row \
      --device cuda:0 --teacher-device cuda:1

Expected: Qwen3-1.7B 97.3% kept [96.0, 98.7] against 91.9% [90.6, 93.4] untuned, difference +3.03 points [+2.29, +3.79];
two repeat runs with other shuffles gave 95.0% and 94.9%. Qwen3-8B 97.9% [97.0, 98.9] against 94.9% [93.9, 95.9],
difference +2.00 [+1.36, +2.68]; 734 s on two GPUs. Not run here, and the GPTQ starting points were not rebuilt here.

### 3.4 Published 4-bit release, upgraded in place

    rowtune round --model models/qwen3-1.7b --student RedHatAI/Qwen3-1.7B-quantized.w4a16 --student-compressed \
      --parquet $TEXT --out out/q17-redhat
    rowtune round --model models/qwen3-8b --student RedHatAI/Qwen3-8B-quantized.w4a16 --student-compressed \
      --parquet $TEXT --out out/q8-redhat --device cuda:0 --teacher-device cuda:1

Expected: Qwen3-1.7B +1.24 points [+0.75, +1.71] (96.7% to 98.9% kept; 128 s); Qwen3-8B +1.13 [+0.72, +1.56]
(98.0% to 99.7% kept), and against 16-bit -0.20 [-0.72, +0.33], within noise (353 s on two GPUs). The same release
rebuilt through the case study's pipeline scored +0.09 and +0.07 against the published files, so the gain is not an
artefact of the pipeline. `--student-compressed` (decompression of the compressed-tensors format) is a one-line
call in `rowtune/hf.py` and was not exercised here. `rowtune round` writes a 16-bit checkpoint of the tuned weights; to get the same release back as a 4-bit file use the in-place command of 3.6. Not run here.

### 3.5 GPTQ per 128 weights, plus my method

Same as 3.3 with a group-128 GPTQ checkpoint as `--student`. Expected gains: Qwen3-1.7B +2.10 [+1.42, +2.75];
SmolLM2-1.7B +3.55 [+2.83, +4.27]. Not run here.

### 3.6 Files already shipped (OpenVINO channel-wise INT4, llama.cpp GGUF Q4_0, packed 4-bit safetensors)

In the case study these rows tuned a 16-bit checkpoint of the dequantised weights of the shipped file, then wrote the factors
back into the file's scales with my own scripts (same size, codes untouched). `rowtune INPUT -o OUTPUT` now does the whole
chain in one command and writes the file in the same format:

    rowtune gguf/Qwen3-1.7B-Q4_0.gguf -o out/Qwen3-1.7B-Q4_0-rowtune.gguf --teacher models/qwen3-1.7b --parquet $TEXT
    rowtune openvino/qwen3-1.7b-int4-sym-channel -o out/qwen3-1.7b-int4-rowtune --teacher models/qwen3-1.7b --parquet $TEXT
    rowtune redhat/Qwen3-1.7B-quantized.w4a16 -o out/qwen3-1.7b-w4a16-rowtune --teacher models/qwen3-1.7b --parquet $TEXT

Expected, Qwen3-1.7B (case study, measured with the earlier scripts): OpenVINO export with AWQ and Scale Estimation 88.0% to
95.3% kept, +4.16 points [+3.44, +4.85]; GGUF Q4_0 +2.11 [+1.54, +2.66] (plain file), +1.17 [+0.67, +1.67] (importance-matrix
file), +1.25 [+0.73, +1.74] (a third file); Q4_K_M files and one mixed-precision file tie (+0.31, +0.32, +0.10, all intervals
crossing zero); RedHatAI release: row 3.4. **Not run here: the tuning step on any real file.**

What was run here on real files (read-only on the originals, outputs in a separate folder, random row factors, no tuning):
- GGUF: a community Q4_0 file of Qwen3-1.7B (193 of its 196 decoder linears are Q4_0, three are stored in another type and left
  untouched): read, patched, byte size identical, all 4-bit code bytes identical, 193 of 193 tensors' scales changed, checked by
  `rowtune.formats.gguf.verify`. The Q4_0 dequantisation was checked on a plain Q4_0 file made from the Q8_0 file: relative
  Frobenius error 0.088 to 0.090 against the Q8_0 weights (three layers); a wrong nibble order would give an error above 1.
- OpenVINO: the INT4 channel-wise IR of Qwen3-1.7B: 196 linears and the 56 RMSNorm gains found, patched, byte size identical,
  xml identical, codes identical, 196 of 196 scale constants changed; the output loaded in the OpenVINO 2026.4.1 Python API and its
  scale constants equal the old scales times the factors (first 20 linears compared).
- Packed 4-bit safetensors (compressed-tensors layout): **no real file on this machine; not run.** Only the synthetic files of the tests.

### 3.6b Tuning the original GGUF and OpenVINO rows with the earlier scripts

Not part of this repository: the earlier dequantise and write-back scripts and the lm-eval driver stay private. Their steps (read codes
and scales, dequantise, tune, multiply scales by the factors) are what `rowtune INPUT -o OUTPUT` performs; the differences that could
move a number are listed in README "Limits".

### 3.7 Code, domain text

Same command as 3.4 for Qwen3-1.7B with `--text-file` pointing at a file of public Python code (documents separated by
a line holding only `<<<DOC>>>`). The case study used `codeparrot/codeparrot-clean-valid`; the script that built the
file is not in this repository, so the exact text is not reproducible from here. Expected, pass@1 on HumanEval + MBPP
(664 problems, greedy; MBPP 3-shot as lm-eval sets it): 16-bit 42.0%, the release 33.1%, the release with my method
on code text 39.5%, +6.33 points [+3.46, +8.89] (67 problems gained, 25 lost). Scoring executed generated code in a
sandbox that is not part of this repository. One run. Not run here.

## 4. Tests (run here)

    python -I -m unittest

67 tests, all passing at the time of writing; they use a tiny random torch model and tiny synthetic files of each format and need neither a GPU nor a download.
