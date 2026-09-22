import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape

    w = np.zeros(D)
    b = 0.0

    for step in range(0, steps):
        z = X @ w + b
        preds = _sigmoid(z)
        grad_w = 1/len(y) * X.T@(preds-y)
        grad_b = np.mean(preds - y)
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w,b