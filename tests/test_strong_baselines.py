import numpy as np
from fdm.simulator import generate_episode
from fdm.baselines import confidence_threshold_policy, myopic_voi_policy, optimal_stopping_policy


def test_strong_baselines_terminate():
    ep = generate_episode(np.random.default_rng(123), steps=6)
    outs = [
        confidence_threshold_policy(ep, confidence=0.9),
        myopic_voi_policy(ep, np.random.default_rng(124), n_mc=80),
        optimal_stopping_policy(ep, np.random.default_rng(125), n_mc=80),
    ]
    for out in outs:
        assert 0 <= out["t"] < 6
        assert out["decision"] in (0, 1)
        assert out["cost"] >= 0
