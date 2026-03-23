import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from bposd.css import css_code
from tile_code_and_clifford_deformation.tile_code import build_periodic_tile_code
from infinite_bias_threshold.infinite_bias_threshold import simulate_single_trial

@pytest.fixture
def tiny_code_data():
    """Sets up a tiny periodic code representation to run simulations efficiently."""
    Hx, Hz, _ = build_periodic_tile_code(7, 7)
    qcode = css_code(Hx, Hz)
    
    L = qcode.l.toarray()
    H_in = qcode.h.toarray()
    
    return H_in, L, qcode.N

def test_seed_determinism(tiny_code_data):
    """
    Test 1: Seed Determinism. 
    Running identical single trials with identically seeded RNGs should yield precisely identical logical failure outcomes.
    """
    H_in, L, N = tiny_code_data
    p, q, error_rate = 0.25, 0.5, 0.1
    
    rng1 = np.random.default_rng(2024)
    logical_fail_1 = simulate_single_trial(H_in.copy(), L.copy(), p, q, error_rate, N, rng1)
    
    rng2 = np.random.default_rng(2024)
    logical_fail_2 = simulate_single_trial(H_in.copy(), L.copy(), p, q, error_rate, N, rng2)
    
    assert logical_fail_1 == logical_fail_2, "Simulations starting with identical RNG seeds mutated inconsistently."

def test_pipeline_output_constraints(tiny_code_data):
    """
    Test 2: Pipeline Constraints Test.
    Ensures that decoding the logic completes correctly and returns exactly an INT mapping to 0 or 1.
    """
    H_in, L, N = tiny_code_data
    p, q, error_rate = 0.25, 0.5, 0.2
    
    rng = np.random.default_rng(123)
    logical_fail = simulate_single_trial(H_in.copy(), L.copy(), p, q, error_rate, N, rng)
    
    assert isinstance(logical_fail, int), "Decoder loop must return an integer."
    assert logical_fail in [0, 1], "Decoder outcome must restrict to binary 0 or 1."

def test_e2e_low_noise(tiny_code_data):
    """
    Test 3: Smoke Pipeline for exceptionally low noise rate resulting in exactly 0 errors.
    """
    H_in, L, N = tiny_code_data
    p, q, error_rate = 0.0, 0.0, 0.00000001
    
    rng = np.random.default_rng(789)
    logical_fail = simulate_single_trial(H_in.copy(), L.copy(), p, q, error_rate, N, rng)
    
    # Highly probable to be 0 for this tiny error rate block
    assert logical_fail == 0, "At ~0 noise, pipeline fails to perfectly decode."
