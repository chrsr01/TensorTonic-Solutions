import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(0)

    keep = 1.0 - p
    mask = (rng.random(x.shape) < keep).astype(float) / keep
    return x*mask, mask
    
    