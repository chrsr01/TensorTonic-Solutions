import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    L = math.sqrt(6/fan_in)

    for i in range(0, len(W)):
        for j in range(0,len(W[0])):
            W[i][j] = W[i][j] * 2*L -L
    return W