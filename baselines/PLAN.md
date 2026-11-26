# Pre-registration: closed-form per-row multiplier versus end-to-end tuning (draft to freeze, 26 Nov 2026)

Status: FROZEN 26 Nov 2026 on the author's go ("all lanes"), with the addition at the end (arm D). Nothing had been run when this was frozen.
Prior art and the reason for this plan: baselines/FLOOR-PRIOR-ART.md.

## What this plan is, and what it is not

The "per-layer floor" from the closed-form least-squares multiplier is published (COMQ eq. 10, Beacon Prop. 2.1 and
sec. 3, PiSO eq. 7) and is not a limit on end-to-end scale-only repair (ThinkBit appendix H). This plan makes no limit
claim. It measures one thing no paper reports: on the same stock 4-bit files, with codes frozen, how much of the
end-to-end KL-tuned per-row multiplier's gain the published one-pass closed form reaches.

Added value, stated in advance: a backprop-free, one-pass baseline at matched constraints (same file, same codes, same
one-factor-per-row parameterisation, same tuning text). If it matches the tuned method, the tuned method is unnecessary
for that file and the paper must say so. If it does not, the paper has the ablation that shows why the end-to-end
objective is needed.

## Model, files, data

- Teacher: Qwen3-1.7B, the same snapshot as the released results (revision 70d244cc...).
- Primary file: the plain GGUF Q4_0 file used in the released result (made from Q8_0 with gguf-py, no importance
  matrix; 196 quantised linears, 573,440 rows).
- Secondary file: the OpenVINO INT4 symmetric per-channel file with AWQ + Scale Estimation (nncf 3.4.0) used in the
  released result.
- Tuned multipliers (arm B): the existing tool outputs, local copies
  <released>/qwen3-1.7b-q4_0-rowtune.gguf.row_scales.pt and
  <released>/q17_awq_se_rowtune.row_scales.pt. No new tuning run.
- Calibration text for the closed form: the same tuning text and windows as the tool (WikiText-103 train shard 2,
  600 windows x 1,024 tokens, seed 1), so the closed form sees exactly the data the tuned method saw.
- Held-out text for KL and layer errors: 200 windows x 1,024 tokens from the same shard, disjoint from the 600 tuning
  windows, chosen with seed 12345 after removing every tuning window. Disjointness is checked by token offsets before
  any number is computed; if the check fails, stop.
- Accuracy: the released suite and drivers unchanged: ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, LAMBADA,
  first 2,000 items per task (10,277 questions), "kept" share against the 16-bit reference, paired bootstrap with
  2,000 resamples, seed 0 (paired_ci.py). The exact driver file hashes are recorded before the first run; the scoring
  driver was edited after the GGUF runs, so input file (arm A) and arm B are re-scored with the frozen driver too.

## Arms (one run each, all reported)

- A: input file, multipliers all 1.
- B: tuned multipliers from the released tool run.
- C1 (closed form, 16-bit inputs): for every linear and every output row i, with x the 16-bit model's input to that
  linear on the calibration text, y_i = w_i . x (teacher) and y^_i = w^_i . x (dequantised row of the file, codes times
  its scales): m_i = sum(y_i y^_i) / sum(y^_i^2), summed over all calibration tokens. Computed in one streaming pass;
  no gradients.
- C2 (closed form, quantised-model inputs): as C1, but x~ is the input the linear receives in the quantised model
  whose earlier linears already carry their C2 multipliers (layers processed in forward order):
  m_i = sum(y_i y~_i) / sum(y~_i^2) with y~_i = w^_i . x~. This is the PiSO / Qronos cross-activation objective.
- All multipliers are rounded to bf16 and written by the release tool's own writer into the same file format; the tool's
  check that every code byte is unchanged must pass, or the arm is void and reported as such.

Basis check (before any arm is computed): all quantities use the dequantised scoring checkpoints the released scores
used. For each linear, ||W_teacher - W^||_F / ||W_teacher||_F < 0.5 must hold (the dequantised weight acts in the
teacher's basis; AWQ factors folded). If any linear fails, stop and report.

## Quantities

Per linear l (on held-out text, 16-bit inputs), with E(m) = sum_i ||y_i - m_i y^_i||^2:
1. Floor fraction F_l = E(m*) / E(1) and recovered fraction 1 - F_l, with m* from arm C1. Reported as the layer-local
   optimum for this one-factor-per-row family, not as a limit of any method.
2. Tuned fraction T_l = E(m_B) / E(1) for the tuned multipliers.
3. Where the tuned multipliers sit: over all rows and per projection type (q, k, v, o, gate, up, down): Spearman and
   Pearson correlation of log m_B with log m*; share of rows where log m_B and log m* have the same sign; median
   |log m*|, median |log m_B|, median |log m_B - log m*|.
End to end, per arm:
4. Held-out forward KL(teacher || arm), mean per token over the 200 windows.
5. Mean accuracy and kept share; paired differences C1 - A, C2 - A, B - C1, B - C2 with 95% bootstrap intervals.
6. Share of the tuned gain reached: R = (C - A) / (B - A), for accuracy and for KL reduction, with bootstrap
   intervals (resampling questions for accuracy, windows for KL).

## Predictions (written before any run)

Primary file (GGUF plain Q4_0):
- P1. m* stays close to 1: median |log m*| below 0.02, while the tuned factors are known to span 0.47 to 1.50
  (median |log m_B| larger than median |log m*|).
- P2. Layer-local recovery is small: median over the 196 linears of 1 - F_l below 0.10.
- P3. T_l > F_l in at least 90% of linears (tuned multipliers are worse than the closed form at the layer-local
  objective) while arm B has lower held-out KL than C1.
- P4. Weak alignment: overall Spearman(log m_B, log m*) below 0.3; same-sign share between 0.4 and 0.7.
- P5. End to end, C1 reaches less than 30% of B's accuracy gain (point estimate of R below 0.3) and less than 30% of
  B's KL reduction. C1 - A lies between -0.5 and +1.0 points.
- P6. C2 does at least as well as C1 on held-out KL, and its R stays below 0.5.
Secondary file (OpenVINO AWQ + Scale Estimation):
- P7. m* even closer to 1 (median |log m*| below 0.01) and median 1 - F_l below 0.05, because Scale Estimation already
  searched this objective at export.
- P8. C1 - A between -0.5 and +0.5 points; B - A (+4.13 released) at least 8 times C1 - A in point estimate.

## What each outcome means

- R at or above 0.8 with B - C interval containing 0, on a file: the published closed form does the job for that file.
  The paper then credits COMQ / Beacon / PiSO for the gain and rests its own claim only on the in-place, format-
  preserving write-back; the tool should offer the closed form as its default for that format.
- R between 0.3 and 0.8: the closed form carries part of the gain; report both, and test (as a separate, new plan) the
  closed form as the starting point for tuning.
- R below 0.3 (predicted): the end-to-end objective carries the gain; report as an ablation next to ThinkBit's result.
- C below A (closed form hurts end to end): the layer-local optimum is harmful in this network, consistent with
  ThinkBit table 21; report it plainly.
- P3 false (tuned multipliers also near the layer-local optimum): the two methods move the same direction and the
  end-to-end gain must come from somewhere else; this contradicts the ThinkBit pattern at 4 bits and is reported as such.
- In no outcome does this plan support a statement about the limit of scale-only repair. An end-to-end limit is not
  identified by these runs; the tuned result remains only an upper bound on the best reachable KL.

## Cost and boundaries

- One streaming pass over 0.6M tokens per arm and file for C1 and C2 (teacher and student forward, no backward), the
  write-back, and the released scoring. Expected under one GPU-hour in total; not verified.
- Nothing runs on the Mac. Nothing is installed or deleted. Results go to a new folder next to the released test
  results, and copies to baselines/results/ before any report.
- Deviations from this plan after freezing are listed in an amendment with the reason, before results are read.


## Addition before freezing: arm D, per-block scale training (26 Nov 2026)

Reason: the published paper (version 1) states that no head-to-head with per-group scale training has been run. Arm D
is that comparison on the primary file.
- D (GGUF Q4_0 only): one trainable multiplier for every 32-weight block of every quantised linear (the granularity
  at which the Q4_0 format stores its scales), initialised at 1, everything else frozen; the same loss (KL to the
  16-bit model), optimiser (Adam, learning rate 1e-3), tuning text, windows, order and seed as the released tool run.
  One run. Written back with the tool's writer; the code-byte check must pass or the arm is void.
- Reported for D: number of trained values; tuning time; held-out KL; mean accuracy and kept share; paired
  differences D - A and B - D with 95% bootstrap intervals; the same quantities 4 to 6 as the other arms.
- Prediction, written now: D reaches a lower held-out KL than B (it has about 30 to 100 times more trained values);
  on accuracy B - D lies within +-0.6 points, i.e. a statistical tie. If D beats B by more than 0.6 points with an
  interval excluding zero, the one-per-row restriction costs accuracy and the paper's next version says so. If B
  beats D by that margin, the restriction acts as a useful regulariser at this data size, and that is reported with
  the caveat of one run and one seed.
- Nothing in arms A, B, C1, C2 changes.
