# Machine-checked statements

`Rowtune.lean` proves four elementary facts the method relies on, in Lean 4 (4.35.0) with Mathlib, with no `sorry`
and only the standard axioms (`propext`, `Classical.choice`, `Quot.sound`):

1. Folding a per-row multiplier into the stored scale gives the same integer codes.
2. Multiplying a layer's weight matrix by a diagonal matrix rescales that layer's output coordinate by coordinate.
3. For finite distributions, the KL divergence is non-negative, zero exactly at equality, and equals cross-entropy
   minus entropy.
4. Rounding to a grid of spacing d moves a value by at most d/2, and a change smaller than d/2 from a grid point is
   erased.

These are statements in exact real arithmetic about one linear layer. They say nothing about float16 bit-exactness,
about the whole network, or about the measured results.

Check it inside any Lean project that has Mathlib built: `lake env lean Rowtune.lean`.
