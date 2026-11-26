"""rowtune: train one multiplier per output row of every linear layer of a 4-bit model, with the integer codes fixed."""

DEFAULT_STEPS = 600  # tuning windows (one optimiser step each)
DEFAULT_LENGTH = 1024  # tokens per window
DEFAULT_LR = 1e-3  # Adam learning rate of the row multipliers
