import numpy as np
from fdm.strong_policies import _action_certainty, _binary_entropy, run_strong_policy
from fdm.simulator import generate_episode

def test_action_certainty_is_bounded():
    for p in np.linspace(0,1,21):
        c=_action_certainty(float(p),1.0,5.0)
        assert 0.0 <= c <= 1.0

def test_entropy_is_bounded():
    for p in np.linspace(0.001,0.999,21):
        h=_binary_entropy(float(p))
        assert 0.0 <= h <= 1.0 + 1e-12

def test_all_strong_policies_terminate():
    ep=generate_episode(np.random.default_rng(2))
    for i,p in enumerate(["immediate","fixed_delay","always_escalate","confidence_gate","entropy_gate","voi","fdm","terminal"]):
        r=run_strong_policy(ep,p,np.random.default_rng(100+i))
        assert 0 <= r["t"] < len(ep["probabilities"])
        assert r["decision"] in (0,1)
        assert np.isfinite(r["cost"])
