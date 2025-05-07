import numpy as np

def attention(Q, K, V):
    """Standard scaled dot-product attention using NumPy."""
    dk = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(dk)

    # Softmax with numerical stability
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    scores /= np.sum(scores, axis=-1, keepdims=True)

    return np.matmul(scores, V)
