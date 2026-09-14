import numpy as np
from fdm.commitment_baseline import commitment_aware_optimal_stopping
from fdm.simulator import generate_episode

def test_commitment_optimal_stopping_terminates():
    ep=generate_episode(np.random.default_rng(22),steps=6)
    o=commitment_aware_optimal_stopping(ep,np.random.default_rng(23),n_mc=30)
    assert 0 <= o["t"] < 6
    assert o["decision"] in (0,1)
    assert o["cost"] >= 0
