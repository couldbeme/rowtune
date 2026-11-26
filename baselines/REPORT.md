# Baselines for rowtune: closed-form multiplier and per-block training (results, 26 Nov 2026)

Plan: baselines/PREREG-FLOOR.md (frozen, with arm D). Nothing in the plan was changed. One run per arm, no second seed, nothing tuned after seeing results.

## Headline

- Primary file (GGUF Q4_0): the published closed-form multiplier does not reproduce the tuned gain. C1 is not better than the input (accuracy -0.26 points, held-out KL up), C2 is clearly worse. R is negative for both.
- Per-block training (arm D, 44.0 M values, 76.8 times more than per-row) ties arm B: accuracy B - D = -0.14 [-0.60, +0.30]; held-out KL equal to four decimals.
- Secondary file (OpenVINO AWQ + Scale Estimation): the basis check failed on 8 of 196 linears, so by the plan's rule C1 and C2 were not computed for it (not run, not void). Arms A and B were re-scored.

## Checks before any arm

| Check | Result |
| --- | --- |
| Held-out windows disjoint from tuning windows | PASS. Held-out windows come only from documents the tuning stream never read: 0 shared documents, 0 identical windows (3,726 documents read by the 600 tuning windows, 1,231 by the 200 held-out windows, 350,446 documents of 200+ characters). Done at document level, which is stronger than the token-offset check the plan names (no shared document means no shared offset). |
| Tuning windows equal the tool's own windows | PASS (`windows_from_texts`, 600 x 1,024, seed 1, same parquet; bit-equal). |
| Basis check GGUF (dequantised scoring checkpoint) | PASS: max ||W - W^|| / ||W|| = 0.095, median 0.089, 196 of 196 below 0.5. Checkpoint equals the tool reader's dequantisation (max abs diff 0.0). |
| Basis check OpenVINO | **FAIL**: 8 linears at or above 0.5: layers.0.up 0.53, layers.2.up 7.91, layers.2.down 0.98, layers.24.down 0.51, layers.25.down 0.57, layers.26.down 0.90, layers.27.up 0.62, layers.27.down 4.58 (median over all 0.139). The plan says stop and report. I stopped the C1 and C2 arms for this file and did nothing to repair it. |
| Code-byte check (tool's own writer) | PASS for C1, C2, D on GGUF (codes_identical, same size 1,129,782,944 bytes, other tensors and padding identical). No void arm. |

## Scoring drivers and hashes

The released run's driver hashes were never recorded (no hash exists in the released result files or in the repo notes), so a match by hash cannot be shown. What I could show instead: the frozen driver (bench_levers.py, mtime later than the released GGUF run) re-scores arm A on both files, and arm B on both files, with **every one of the 10,277 per-question outcomes and answers identical** to the released result files (compare_released_*.json, compare_prev_*.json). The driver edit therefore changed nothing for these inputs.

sha256 (full list incl. inputs, mtimes and sizes in results/hashes.txt):
- bench_levers.py d11c8a98ac12a254424bcf8385a18bd087eeb7ba85f42f291b19a38bae8fedb2 (28,254 bytes)
- bench_levers_ov.py ad84a0b15add347b04362fca1feed38570a7c6f71aebb298692cbde8484aa5bf
- paired_ci.py d49dc66c827b0cbb04a76f279c444cdb9e47591df693d6f62d7b84bf05b99af8 (same file as server-test/paired_ci.py)
- dequant_to_hf.py (copy, only the output folder changed, see results/dequant_to_hf.diff) 73cfebc955df42ca9d476190b403d7bfd8e93c81008c879534d48ed43a3be3e6; original 22e3ba09...
- ir_dequant.py d4937f5e..., ir_norm.py 69686c37..., gguf_hf_map.py 0a12c99c...
- bf16 reference: results/methods_qwen3_1.7b.json:bf16 f4150bf6a6f70a3224ee0ecd6f625f44ad9a0fd481d364c28423e5637bea5719 (mean 56.60)
- input GGUF c1ede04f067441d1bb5d185949073c0a9694e1d29ed47151ff5215bd8ebb2899; held/tune windows.json 616bc09042b032e81ec573b448e4369f11d9857b1fe5ccee9cb56ae9fcbd1286; tuning parquet ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720

## Results, GGUF Q4_0 (primary file)

16-bit reference: mean 56.60 [55.68, 57.54] over the same 10,277 questions. Paired bootstrap: 2,000 resamples, seed 0, released procedure (paired_ci.py logic; unit-tested to give identical numbers). KL: forward KL(teacher || arm), mean per token over 200 held-out windows x 1,024 tokens, windows resampled 2,000 times, seed 0.

| Arm | Trained values | Time | Mean accuracy | Kept share (of 16-bit) | Held-out KL |
| --- | --- | --- | --- | --- | --- |
| A input | 0 | - | 53.66 [52.72, 54.59] | 94.80% [93.58, 96.00] | 0.2031 [0.1988, 0.2075] |
| B tuned per row (existing files) | 573,440 | 140.7 s (earlier run, tool timer) | 55.66 [54.76, 56.60] | 98.33% [97.20, 99.48] | 0.1097 [0.1075, 0.1120] |
| C1 closed form, 16-bit inputs | 573,440 (computed, not trained) | 180.9 s compute, one streaming pass, no gradients | 53.41 [52.51, 54.35] | 94.35% [93.05, 95.64] | 0.2256 [0.2210, 0.2302] |
| C2 closed form, quantised-model inputs | 573,440 | 730.0 s compute | 52.18 [51.30, 53.11] | 92.19% [90.93, 93.55] | 0.3022 [0.2978, 0.3067] |
| D per 32-weight block | 44,040,192 (76.8 x B) | 199.3 s tool timer (205 s wall) | 55.80 [54.86, 56.71] | 98.58% [97.44, 99.75] | 0.1098 [0.1074, 0.1123] |

Paired differences, accuracy in points, 95% interval:

| Pair | Difference | KL difference |
| --- | --- | --- |
| B - A | +2.00 [+1.43, +2.56] | -0.0934 [-0.0964, -0.0905] |
| C1 - A | -0.26 [-0.87, +0.38] | +0.0225 [+0.0202, +0.0249] |
| C2 - A | -1.48 [-2.20, -0.76] | +0.0991 [+0.0958, +0.1022] |
| B - C1 | +2.25 [+1.61, +2.90] | -0.1160 [-0.1190, -0.1128] |
| B - C2 | +3.48 [+2.78, +4.18] | -0.1925 [-0.1954, -0.1894] |
| D - A | +2.14 [+1.57, +2.71] | -0.0933 [-0.0965, -0.0902] |
| B - D | -0.14 [-0.60, +0.30] | -0.0001 [-0.0011, +0.0008] |

Share of the tuned gain reached, R = (C - A) / (B - A), question- or window-resampled 95% interval:

| Arm | R, accuracy | R, KL reduction |
| --- | --- | --- |
| C1 | -0.128 [-0.522, +0.168] | -0.241 [-0.270, -0.212] |
| C2 | -0.741 [-1.378, -0.335] | -1.060 [-1.118, -0.999] |
| D | +1.072 [+0.861, +1.349] | +0.998 [+0.988, +1.009] |

B reproduces the earlier server test exactly (same files, same checkpoint): +2.00 [+1.43, +2.56] over A.

### Layer-local quantities (196 linears, held-out text, 16-bit inputs, bf16-rounded multipliers as written)

- Quantity 1, floor fraction F_l = E(m*) / E(1) with m* from C1: median 0.933, so median recovered fraction 1 - F_l = 0.067 (range 0.007 to 0.621; quartiles 0.034 / 0.067 / 0.125).
- Quantity 2, tuned fraction T_l = E(m_B) / E(1): median 1.083 (range 1.010 to 3.004); T_l is above 1 in 196 of 196 linears. The tuned multipliers raise the layer-local error in every linear. T_l > F_l in 196 of 196.
- C2 at the layer-local objective with 16-bit inputs: median 1 - F_l = -0.719; C2 is below 1 in 8 of 196 linears only. (C2 optimises a different objective, so this is expected to look bad here; it is reported, not interpreted.)
- Quantity 3, where the tuned multipliers sit (all 573,440 rows): Spearman(m_B, m*) = 0.027 (identical to Spearman of logs where defined); Pearson of logs 0.029 over the 573,421 rows with both positive; same sign of (m - 1) in 42.9% of rows; median |log m*| 0.0118 (positive rows), median |log m_B| 0.0155, median |log m_B - log m*| 0.0228. Ranges: m* from -2.47 to 10.69, m_B from 0.471 to 1.500. m* quantiles 1/5/25/50/75/95/99%: 0.922 / 0.953 / 0.980 / 0.992 / 1.000 / 1.031 / 1.055.
- Per projection (Spearman / same-sign share / median |log m*| / median |log m_B|): q 0.071 / 45.7% / 0.0157 / 0.0155; k 0.062 / 44.0% / 0.0155 / 0.0155; v 0.116 / 47.9% / 0.0155 / 0.0157; o 0.065 / 44.6% / 0.0118 / 0.0157; gate -0.052 / 39.9% / 0.0155 / 0.0155; up 0.049 / 43.3% / 0.0118 / 0.0155; down 0.060 / 43.6% / 0.0078 / 0.0155. Full table in gguf_layer_quantities.json.
- Caveat on resolution: bf16 has a step of 0.0078 near 1, so most closed-form corrections (median |m* - 1| about 1%) land on one to two grid points. All arms use bf16 multipliers as the plan states; I did not test an unrounded C1.

## Predictions against outcomes (the plan's wording, primary file)

- P1 "m* stays close to 1: median |log m*| below 0.02, ... (median |log m_B| larger than median |log m*|)". HOLDS: 0.0118 below 0.02; 0.0155 above 0.0118; tuned span 0.471 to 1.500 confirmed. (Reported on the 573,421 rows where log is defined; see amendment.)
- P2 "median over the 196 linears of 1 - F_l below 0.10". HOLDS: 0.067.
- P3 "T_l > F_l in at least 90% of linears while arm B has lower held-out KL than C1". HOLDS: 196 of 196 (100%); B KL 0.1097 vs C1 0.2256.
- P4 "overall Spearman below 0.3; same-sign share between 0.4 and 0.7". HOLDS: 0.027; 0.429 (inside the band by 0.029).
- P5 "C1 reaches less than 30% of B's accuracy gain and less than 30% of B's KL reduction; C1 - A between -0.5 and +1.0". HOLDS: R = -0.128 and -0.241; C1 - A = -0.26.
- P6 "C2 does at least as well as C1 on held-out KL, and its R stays below 0.5". FALSE on the first part: C2 KL 0.3022 is worse than C1 0.2256 (difference +0.0766). The second part holds trivially (R = -0.741 accuracy, -1.060 KL).
- P7 (secondary file, median |log m*| below 0.01, median 1 - F_l below 0.05): NOT TESTED, arm not run (basis check failed).
- P8 (secondary file, C1 - A within +-0.5; B - A at least 8 times C1 - A): NOT TESTED. Only the B - A part is known: +4.13 [+3.39, +4.85], equal to the released figure.
- Arm D, "reaches a lower held-out KL than B": FALSE as a point estimate. D 0.1098 vs B 0.1097; B - D = -0.0001 [-0.0011, +0.0008]; the two are tied to four decimals.
- Arm D, "on accuracy B - D lies within +-0.6 points, i.e. a statistical tie": HOLDS: -0.14 [-0.60, +0.30] (the lower end of the interval touches -0.60).

What each outcome means, in the plan's words: R below 0.3 for C1 and C2 ("the end-to-end objective carries the gain; report as an ablation next to ThinkBit's result"). "C below A (closed form hurts end to end)": C1 is below A on accuracy by a margin whose interval contains zero, and above A on held-out KL by +0.0225 with an interval excluding zero; C2 is below A on both with intervals excluding zero. The plan makes no statement about a limit of scale-only repair, and neither does this report. D is within +-0.6 of B in accuracy: one-per-row restriction showed no accuracy cost in this run (one run, one seed).

## Secondary file (OpenVINO AWQ + Scale Estimation)

- C1, C2: not run, because the basis check failed (above). The cause is in the data, not the code: ckpt-vs-file difference is 0.0, so the dequantised checkpoint is exactly the file's weights; for 8 linears (all up_proj or down_proj) the dequantised weights are far from the teacher's, which fits a scale folded between up and down projections that the plan assumed was undone; I did not verify that explanation.
- A re-scored: 49.79 [48.85, 50.78], kept 87.97% [86.56, 89.55], held-out KL 0.4799. B re-scored: 53.92 [52.98, 54.91], kept 95.26% [93.95, 96.69], KL 0.2527. B - A = +4.13 [+3.39, +4.85], KL difference -0.2272 [-0.2323, -0.2222]. Item-identical to the released result files.

## Amendments and deviations (listed here because the plan asks for them before results are read; the first one was decided after the C1 factors existed, see its note)

1. Non-positive closed-form multipliers. The closed form is unconstrained and gives 19 non-positive rows out of 573,440 (all in down_proj; minimum -2.47, maximum 10.69), so "log m*" is undefined for them. Decided after seeing the C1 range, before any layer-local or alignment number was computed: arms are unchanged (the multipliers were written as computed, negative scale included; the file check passes); alignment statistics use Spearman and same-sign of (m - 1) over all rows and log statistics over rows where the logs exist (counts reported). The make_arm JSON field `median_abs_log` is NaN for that reason; ignore it, use gguf_layer_quantities.json.
2. C2 granularity. The plan says layers are processed in forward order. Implemented per decoder layer: for layer L the student has the C2 multipliers of layers below L applied and none yet in L, so within a layer the o_proj, gate, up and down linears see inputs that do not yet carry the same layer's q, k, v multipliers. y_i is the teacher row on the teacher input, y~_i the dequantised row on the student input.
3. Arm B scoring and KL use the existing dequantised checkpoints of the earlier test (rowtune-test/ckpt/q4_0_input and q4_0_rowtune, ov_rowtune), read only; the OpenVINO input checkpoint was regenerated into this folder with the earlier ir_dequant.py (the older copy had been removed from the server).
4. Disjointness was checked at document level (see above), not by raw token offsets.
5. Tool change for arm D: `--per-block 32` (GGUF only) in the tool; the tool's JSON now also carries a `per_block` field. Tool tuning settings for D otherwise identical to the earlier run (Adam, lr 1e-3, 600 x 1,024, seed 1, bf16, KL loss).
6. Process slips: commit 85d5709 contains one failing test (a wrongly hand-computed rank in a new test); fixed in f613eca, before any result depended on it. I deleted the local copy of windows.json (4.6 MB) from floor/results/ by mistake while tidying; it is unchanged on the server (sha256 above) and reproducible; nothing else was deleted anywhere.
7. Not done: unrounded-C1 diagnostic; any second seed; the OpenVINO C arms.

## Tests

Tool tests: 67 before; now 104 (14 new for per-block scales, 23 new for the closed form, held-out split, checks, bootstraps; one of those 23 added after the C1 range showed non-positive rows). All 104 pass locally (Python 3.11, torch 2.14) and on the server (Python 3.10, torch 2.11, CPU). Test-first throughout: each group failed at import for the right reason before the code existed.

## Commands (all GPU jobs under nohup, CUDA_VISIBLE_DEVICES=0, HF_HUB_OFFLINE=1; scripts in results/server-scripts/ and repo/floor/)

    nohup bash floor-2026-11-26/scripts/stage1.sh  > logs/stage1.log    # windows, disjointness, OV input checkpoint, basis checks
    nohup bash floor-2026-11-26/scripts/stage2.sh  > logs/stage2.log    # C1, C2, D (OpenVINO C1/C2 skipped by its own test)
    nohup bash floor-2026-11-26/scripts/stage3.sh  > logs/stage3.log    # dequantise, score A/B/C1/C2/D (+ OV A/B), KL, layer quantities
    # arm D inside stage2: python -m rowtune.cli gguf/src/Qwen3-1.7B-Q4_0-from-Q8_0-ggufpy.gguf -o floor-2026-11-26/files/gguf_d.gguf --teacher <snapshot 70d244cc...> --parquet <wikitext-103 shard 2> --per-block 32
    # scoring, as the earlier run: python bench_levers.py --repo Qwen/Qwen3-1.7B --revision 70d244cc... --out <json> --methods ext_rebuilt --label <label> --pub_repo <ckpt> --limit 2000   (bench_levers_ov.py for OpenVINO)
    # statistics: floor/final_stats.py (CPU), floor/compare_released.py (CPU)


## Left on the server


The scripts that ran the arms are in `floor/` (set `ROWTUNE_BENCH_DIR` and `ROWTUNE_TUNING_PARQUET`; see `floor/common.py`).
