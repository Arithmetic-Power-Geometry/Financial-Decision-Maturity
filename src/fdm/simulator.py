from __future__ import annotations
import numpy as np
from .core import sigmoid

def generate_episode(rng: np.random.Generator, steps: int = 6) -> dict:
    """Generate one finance-like sequential evidence episode."""
    base_rate = 0.08
    y = int(rng.random() < base_rate)
    latent = rng.normal(1.6 if y else -1.6, 0.8)

    ps = []
    observed_scores = []
    for t in range(steps):
        quality = 0.35 + 0.65 * (t / max(1, steps - 1))
        noise = rng.normal(0.0, 1.35 * (1.0 - 0.72 * quality))
        score = quality * latent + noise - 1.1
        p = sigmoid(score)
        observed_scores.append(score)
        ps.append(p)

    return {
        "y": y,
        "probabilities": np.asarray(ps, dtype=float),
        "scores": np.asarray(observed_scores, dtype=float),
    }

def sample_terminal_posteriors(current_p: float, t: int, total_steps: int,
                               rng: np.random.Generator, n: int = 300) -> np.ndarray:
    """Approximate possible terminal posterior states from the current state."""
    remaining = max(total_steps - 1 - t, 0)
    if remaining == 0:
        return np.repeat(float(current_p), n)

    p = np.clip(current_p, 1e-6, 1-1e-6)
    logit = np.log(p / (1-p))
    sigma = 1.25 * remaining / max(total_steps - 1, 1)
    future_logits = rng.normal(logit, sigma, size=n)
    return 1.0 / (1.0 + np.exp(-future_logits))
