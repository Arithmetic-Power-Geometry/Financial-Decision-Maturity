import numpy as np
from fdm.simulator import generate_episode
from fdm.policies import run_policy

def test_episode_shape():
    ep = generate_episode(np.random.default_rng(1), steps=6)
    assert len(ep["probabilities"]) == 6
    assert np.all((ep["probabilities"] >= 0) & (ep["probabilities"] <= 1))

def test_all_policies_terminate():
    ep = generate_episode(np.random.default_rng(2), steps=6)
    for i, policy in enumerate(["immediate", "fixed_delay", "always_escalate", "oracle", "fdm"]):
        out = run_policy(ep, policy, np.random.default_rng(100+i))
        assert 0 <= out["t"] < 6
        assert out["decision"] in (0, 1)
        assert out["cost"] >= 0
