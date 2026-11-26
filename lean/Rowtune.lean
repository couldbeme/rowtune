/-
Rowtune.lean — four elementary facts used by the row-scale tuning paper.

All quantities are real-valued and finite (rows indexed by a Fintype, vocabularies by a Fintype).
No `sorry`, no `axiom`, no `native_decide`. `#print axioms` is run on every theorem at the end.
-/
import Mathlib

open Real Finset

namespace Rowtune

/-! ## 1. Folding a per-row multiplier into the stored scale -/

/-- Dequantised weight from a per-row scale `s` and integer codes `q`: `W i j = s i * q i j`. -/
def dequant {r c : Type*} (s : r → ℝ) (q : r → c → ℤ) : r → c → ℝ :=
  fun i j => s i * (q i j : ℝ)

/-- Folding, entrywise: `m_i * (s_i * q_ij) = (m_i * s_i) * q_ij`. -/
theorem fold_entry (m s : ℝ) (q : ℤ) : m * (s * (q : ℝ)) = (m * s) * (q : ℝ) := by
  ring

/-- Folding, whole layer: multiplying row `i` of the dequantised weight by `m i` gives exactly the
dequantised weight of the same integer codes `q` with the new scales `m i * s i`. -/
theorem fold_layer {r c : Type*} (m s : r → ℝ) (q : r → c → ℤ) :
    (fun i j => m i * dequant s q i j) = dequant (fun i => m i * s i) q := by
  funext i j
  simp only [dequant]
  ring

/-! ## 2. Row multipliers act on the layer output coordinate by coordinate -/

/-- `(diag(m) * W) x = m ∘ (W x)` componentwise. -/
theorem diag_mul_mulVec {r c : Type*} [Fintype r] [Fintype c] [DecidableEq r]
    (m : r → ℝ) (W : Matrix r c ℝ) (x : c → ℝ) (i : r) :
    ((Matrix.diagonal m * W).mulVec x) i = m i * (W.mulVec x) i := by
  rw [← Matrix.mulVec_mulVec, Matrix.mulVec_diagonal]

/-- The same fact stated for the dequantised matrix: `diag(m) * dequant s q = dequant (m*s) q`,
so the row-rescaled layer is again a stored layer with unchanged codes. -/
theorem diag_mul_dequant {r c : Type*} [Fintype r] [DecidableEq r]
    (m s : r → ℝ) (q : r → c → ℤ) :
    Matrix.diagonal m * (Matrix.of (dequant s q)) = Matrix.of (dequant (fun i => m i * s i) q) := by
  ext i j
  rw [Matrix.diagonal_mul]
  simp only [Matrix.of_apply, dequant]
  ring

/-! ## 3. The objective: KL divergence, cross-entropy and entropy on a finite set -/

section KL

variable {ι : Type*} [Fintype ι]

/-- `KL(p‖q) = Σ p_i log(p_i / q_i)` (Mathlib's `Real.log` with `log 0 = 0`, so `0 log 0 = 0`). -/
noncomputable def kl (p q : ι → ℝ) : ℝ := ∑ i, p i * Real.log (p i / q i)

/-- Cross-entropy `H(p, q) = - Σ p_i log q_i`. -/
noncomputable def crossEntropy (p q : ι → ℝ) : ℝ := - ∑ i, p i * Real.log (q i)

/-- Entropy `H(p) = - Σ p_i log p_i`. -/
noncomputable def entropy (p : ι → ℝ) : ℝ := - ∑ i, p i * Real.log (p i)

/-- Hypotheses on `(p, q)`: both are probability vectors and `q` is positive wherever `p` is. -/
structure Admissible (p q : ι → ℝ) : Prop where
  p_nonneg : ∀ i, 0 ≤ p i
  q_nonneg : ∀ i, 0 ≤ q i
  p_sum : ∑ i, p i = 1
  q_sum : ∑ i, q i = 1
  q_pos_of_p_pos : ∀ i, 0 < p i → 0 < q i

/-- Pointwise Gibbs term: for `a > 0`, `b > 0`, `a log(a/b) + b - a ≥ 0`. -/
theorem gibbs_term_nonneg {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    0 ≤ a * Real.log (a / b) + b - a := by
  have h1 : Real.log (b / a) ≤ b / a - 1 := Real.log_le_sub_one_of_pos (div_pos hb ha)
  have h2 : Real.log (a / b) = - Real.log (b / a) := by
    rw [← Real.log_inv, inv_div]
  have h3 : a * (b / a - 1) = b - a := by field_simp
  have h4 : a * Real.log (b / a) ≤ b - a := by
    calc a * Real.log (b / a) ≤ a * (b / a - 1) := mul_le_mul_of_nonneg_left h1 ha.le
      _ = b - a := h3
  rw [h2]
  linarith

/-- Equality case of the pointwise Gibbs term. -/
theorem gibbs_term_eq_zero {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
    (h : a * Real.log (a / b) + b - a = 0) : a = b := by
  by_contra hne
  have hx : b / a ≠ 1 := by
    intro h'
    apply hne
    have := (div_eq_one_iff_eq ha.ne').mp h'
    exact this.symm
  have h1 : Real.log (b / a) < b / a - 1 := Real.log_lt_sub_one_of_pos (div_pos hb ha) hx
  have h2 : Real.log (a / b) = - Real.log (b / a) := by
    rw [← Real.log_inv, inv_div]
  have h3 : a * (b / a - 1) = b - a := by field_simp
  have h4 : a * Real.log (b / a) < b - a := by
    calc a * Real.log (b / a) < a * (b / a - 1) := mul_lt_mul_of_pos_left h1 ha
      _ = b - a := h3
  rw [h2] at h
  linarith

/-- The per-coordinate term `t_i = p_i log(p_i/q_i) + q_i - p_i`, which is `≥ 0` and sums to `KL`. -/
theorem term_nonneg {p q : ι → ℝ} (h : Admissible p q) (i : ι) :
    0 ≤ p i * Real.log (p i / q i) + q i - p i := by
  rcases (h.p_nonneg i).lt_or_eq with hp | hp
  · exact gibbs_term_nonneg hp (h.q_pos_of_p_pos i hp)
  · rw [← hp]; simp [h.q_nonneg i]

theorem kl_eq_sum_terms {p q : ι → ℝ} (h : Admissible p q) :
    kl p q = ∑ i, (p i * Real.log (p i / q i) + q i - p i) := by
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, h.p_sum, h.q_sum, kl]
  ring

/-- Gibbs' inequality: `KL(p‖q) ≥ 0`. -/
theorem kl_nonneg {p q : ι → ℝ} (h : Admissible p q) : 0 ≤ kl p q := by
  rw [kl_eq_sum_terms h]
  exact Finset.sum_nonneg (fun i _ => term_nonneg h i)

/-- `KL(p‖q) = 0` iff `p = q`. -/
theorem kl_eq_zero_iff {p q : ι → ℝ} (h : Admissible p q) : kl p q = 0 ↔ p = q := by
  constructor
  · intro h0
    rw [kl_eq_sum_terms h] at h0
    have hall := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => term_nonneg h i)).mp h0
    funext i
    have hi := hall i (Finset.mem_univ i)
    rcases (h.p_nonneg i).lt_or_eq with hp | hp
    · exact gibbs_term_eq_zero hp (h.q_pos_of_p_pos i hp) hi
    · rw [← hp] at hi ⊢
      simp at hi
      linarith
  · rintro rfl
    unfold kl
    apply Finset.sum_eq_zero
    intro i _
    rcases (h.p_nonneg i).lt_or_eq with hp | hp
    · rw [div_self hp.ne', Real.log_one, mul_zero]
    · rw [← hp, zero_mul]

/-- `KL(p‖q) = H(p, q) − H(p)`. -/
theorem kl_eq_crossEntropy_sub_entropy {p q : ι → ℝ} (h : Admissible p q) :
    kl p q = crossEntropy p q - entropy p := by
  unfold kl crossEntropy entropy
  rw [sub_neg_eq_add, neg_add_eq_sub, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  rcases (h.p_nonneg i).lt_or_eq with hp | hp
  · rw [Real.log_div hp.ne' (h.q_pos_of_p_pos i hp).ne']
    ring
  · rw [← hp]; ring

/-- Consequence used in the paper: for a fixed teacher `p`, `KL(p‖q)` and the cross-entropy
`H(p, q)` differ by a constant (`H(p)`) that does not depend on the student `q`. -/
theorem kl_sub_kl_eq_ce_sub_ce {p q q' : ι → ℝ} (h : Admissible p q) (h' : Admissible p q') :
    kl p q - kl p q' = crossEntropy p q - crossEntropy p q' := by
  rw [kl_eq_crossEntropy_sub_entropy h, kl_eq_crossEntropy_sub_entropy h']
  ring

end KL

/-! ## 4. Re-rounding a tuned scale to a grid of spacing `d` -/

/-- Round `t` to the grid `d ℤ` with Mathlib's `round` (nearest integer, ties rounded up:
`round x = ⌊x + 1/2⌋`). -/
noncomputable def roundTo (d t : ℝ) : ℝ := d * (round (t / d) : ℝ)

/-- `|round_d(t) − t| ≤ d/2`. -/
theorem abs_roundTo_sub_le {d : ℝ} (hd : 0 < d) (t : ℝ) : |roundTo d t - t| ≤ d / 2 := by
  have key : roundTo d t - t = - d * (t / d - (round (t / d) : ℝ)) := by
    unfold roundTo
    field_simp
    ring
  rw [key, abs_mul, abs_neg, abs_of_pos hd]
  have hr := abs_sub_round (t / d)
  calc d * |t / d - (round (t / d) : ℝ)| ≤ d * (1 / 2) := mul_le_mul_of_nonneg_left hr hd.le
    _ = d / 2 := by ring

/-- Characterisation of the rounding cell: `round_d(t) = k d` iff `(k − 1/2) d ≤ t < (k + 1/2) d`. -/
theorem roundTo_eq_iff {d : ℝ} (hd : 0 < d) (t : ℝ) (k : ℤ) :
    roundTo d t = d * k ↔ ((k : ℝ) - 1 / 2) * d ≤ t ∧ t < ((k : ℝ) + 1 / 2) * d := by
  unfold roundTo
  rw [mul_right_inj' hd.ne', Int.cast_inj, round_eq, Int.floor_eq_iff]
  constructor
  · rintro ⟨h1, h2⟩
    constructor
    · have : (k : ℝ) - 1 / 2 ≤ t / d := by linarith
      calc ((k : ℝ) - 1 / 2) * d ≤ t / d * d := mul_le_mul_of_nonneg_right this hd.le
        _ = t := div_mul_cancel₀ t hd.ne'
    · have : t / d < (k : ℝ) + 1 / 2 := by linarith
      calc t = t / d * d := (div_mul_cancel₀ t hd.ne').symm
        _ < ((k : ℝ) + 1 / 2) * d := mul_lt_mul_of_pos_right this hd
  · rintro ⟨h1, h2⟩
    have e1 : (k : ℝ) - 1 / 2 ≤ t / d := by rw [le_div_iff₀ hd]; exact h1
    have e2 : t / d < (k : ℝ) + 1 / 2 := by rw [div_lt_iff₀ hd]; exact h2
    constructor <;> linarith

/-- Corollary: two values in the same cell `[(k − 1/2) d, (k + 1/2) d)` round to the same stored value. -/
theorem roundTo_eq_of_same_cell {d : ℝ} (hd : 0 < d) (k : ℤ) {t t' : ℝ}
    (ht : ((k : ℝ) - 1 / 2) * d ≤ t ∧ t < ((k : ℝ) + 1 / 2) * d)
    (ht' : ((k : ℝ) - 1 / 2) * d ≤ t' ∧ t' < ((k : ℝ) + 1 / 2) * d) :
    roundTo d t = roundTo d t' := by
  rw [(roundTo_eq_iff hd t k).mpr ht, (roundTo_eq_iff hd t' k).mpr ht']

/-- Erasure: if the stored scale `s = k d` is on the grid and the tuned value `t` moved it by
strictly less than `d/2`, re-rounding returns exactly `s`, so the tuning is erased. -/
theorem roundTo_erases {d : ℝ} (hd : 0 < d) (k : ℤ) {t : ℝ} (h : |t - d * k| < d / 2) :
    roundTo d t = d * k := by
  rw [roundTo_eq_iff hd t k]
  rw [abs_lt] at h
  constructor <;> nlinarith [h.1, h.2]

end Rowtune

#print axioms Rowtune.fold_entry
#print axioms Rowtune.fold_layer
#print axioms Rowtune.diag_mul_mulVec
#print axioms Rowtune.diag_mul_dequant
#print axioms Rowtune.gibbs_term_nonneg
#print axioms Rowtune.gibbs_term_eq_zero
#print axioms Rowtune.term_nonneg
#print axioms Rowtune.kl_eq_sum_terms
#print axioms Rowtune.kl_nonneg
#print axioms Rowtune.kl_eq_zero_iff
#print axioms Rowtune.kl_eq_crossEntropy_sub_entropy
#print axioms Rowtune.kl_sub_kl_eq_ce_sub_ce
#print axioms Rowtune.abs_roundTo_sub_le
#print axioms Rowtune.roundTo_eq_iff
#print axioms Rowtune.roundTo_eq_of_same_cell
#print axioms Rowtune.roundTo_erases
