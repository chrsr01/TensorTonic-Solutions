def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    def derivative(a,b,x):
        return 2*a*x+b
    x = x0
    for step in range(0, steps):
        deriv = derivative(a,b,x)
        x_new = x - lr * deriv
        x = x_new
    return x
    