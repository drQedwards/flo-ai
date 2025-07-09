import numpy as np

# ---------- Helper functions ----------

def randn(shape, scale=0.5, rng=None):
    rng = rng or np.random
    return (rng.rand(*shape) * 2 - 1) * scale


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax (in-place safe)."""
    x_shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x_shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Simple row-wise layer norm (no learned γ/β)."""
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ---------- Multi-Head Attention ----------

class Head:
    def __init__(self, d_model: int, d_head: int, rng=None):
        rng = rng or np.random
        self.Wq = randn((d_model, d_head), rng=rng)
        self.Wk = randn((d_model, d_head), rng=rng)
        self.Wv = randn((d_model, d_head), rng=rng)
        self.Wo = randn((d_head, d_model), rng=rng)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d_model) -> returns (N, d_head)"""
        Q = X @ self.Wq  # (N, d_head)
        K = X @ self.Wk  # (N, d_head)
        V = X @ self.Wv  # (N, d_head)

        scores = (Q @ K.T) / np.sqrt(Q.shape[1])  # (N, N)
        probs = softmax_rows(scores)
        ctx = probs @ V  # (N, d_head)
        return ctx @ self.Wo  # (N, d_model)


class MHA:
    def __init__(self, n_heads: int, d_model: int, rng=None):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.heads = [Head(d_model, self.d_head, rng=rng) for _ in range(n_heads)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        outs = [h.forward(X) for h in self.heads]  # each (N, d_model)
        concat = np.concatenate(outs, axis=1)  # (N, d_model * n_heads)
        # Project back to d_model using simple truncation / average.
        # For demo simplicity we just take first d_model cols.
        return concat[:, : self.d_model]


# ---------- Feed-Forward ----------

class FFN:
    def __init__(self, d_model: int, d_ff: int, rng=None):
        rng = rng or np.random
        self.W1 = randn((d_model, d_ff), rng=rng)
        self.W2 = randn((d_ff, d_model), rng=rng)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1)  # ReLU
        return h @ self.W2


# ---------- Transformer Block ----------

class Block:
    def __init__(self, n_heads: int, d_model: int, d_ff: int, rng=None):
        self.mha = MHA(n_heads, d_model, rng=rng)
        self.ffn = FFN(d_model, d_ff, rng=rng)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Multi-head attention + residual + norm
        mha_out = self.mha.forward(X)
        X = layer_norm(X + mha_out)

        # Feed-forward + residual + norm
        ffn_out = self.ffn.forward(X)
        X = layer_norm(X + ffn_out)
        return X


# ---------- Positional Encoding ----------

def add_positional_encoding(X: np.ndarray) -> np.ndarray:
    N, d = X.shape
    P = np.zeros_like(X)
    for pos in range(N):
        for i in range(0, d, 2):
            angle = pos / (10000 ** (2 * (i // 2) / d))
            P[pos, i] = np.sin(angle)
            if i + 1 < d:
                P[pos, i + 1] = np.cos(angle)
    return X + P


# ---------- Lattice ----------

class Lattice:
    def __init__(self, n_layers: int, n_heads: int, d_model: int, d_ff: int, rng=None):
        self.layers = [Block(n_heads, d_model, d_ff, rng=rng) for _ in range(n_layers)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = add_positional_encoding(X)
        for block in self.layers:
            X = block.forward(X)
        return X


# ---------- Convenience API ----------

def run(input_emb, n_layers=2, n_heads=2, d_model=8, d_ff=16, seed=42):
    """Utility to forward an input embedding matrix through the lattice.

    Args:
        input_emb: list[list[float]] – shape (seq_len, d_model).
    Returns:
        list[list[float]] – transformed embeddings of same shape.
    """
    rng = np.random.default_rng(seed)
    X = np.array(input_emb, dtype=np.float32)
    lattice = Lattice(n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff, rng=rng)
    out = lattice.forward(X)
    return out.tolist()