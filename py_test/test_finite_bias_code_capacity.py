import pytest
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from finite_bias_code_capacity_model import code_capacity_css
from finite_bias_code_capacity_model.code_capacity_css import css_decode_sim, create_tile_code

@pytest.fixture
def dummy_code_matrices():
    """Generates an open tile code (l=4, m=4) for simulation setup tests."""
    Hx, Hz, lx, lz, n = create_tile_code(4, 4)
    return Hx, Hz, lx, lz, n

def test_bias_channel_initialization(dummy_code_matrices):
    """
    Test 1: Bias Channel Initialization
    Verifies the internal variables accurately map the [pX : pY : pZ] input distribution mathematically.
    """
    Hx, Hz, lx, lz, n = dummy_code_matrices
    
    # For a purely z-biased channel
    error_rate = 0.1
    eta = 100
    sim = css_decode_sim(
        hx=Hx,
        hz=Hz,
        lx=lx,
        lz=lz,
        xyz_error_bias=[1.0, 1.0, eta],
        error_rate=error_rate,
        run_sim=False  # Do not loop
    )
    
    expected_tot = 102.0
    expected_pz = (error_rate * eta) / expected_tot
    expected_px = (error_rate * 1.0) / expected_tot
    
    assert np.isclose(sim.pz, expected_pz), f"Expected Z probability {expected_pz}, got {sim.pz}"
    assert np.isclose(sim.px, expected_px), f"Expected X probability {expected_px}, got {sim.px}"
    
    # Assert they get propagated to the per-qubit arrays correctly
    assert np.allclose(sim.channel_probs_z, expected_pz)
    assert np.allclose(sim.channel_probs_x, expected_px)


def test_single_shot_decoding_correctness(dummy_code_matrices):
    """
    Test 2: Single Error Injection Decoding
    Verifies that BP+OSD inner loop can map a trivial correctable error back to a success without the multiprocessing loop.
    """
    Hx, Hz, lx, lz, n = dummy_code_matrices
    
    sim = css_decode_sim(
        hx=Hx,
        hz=Hz,
        lx=lx,
        lz=lz,
        error_rate=0.01,
        xyz_error_bias=[1.0, 1.0, 100.0],
        max_iter=100,
        osd_method="osd_e",
        osd_order=8,
        run_count=1,
        target_runs=1,
        run_sim=False # We trigger manually
    )
    
    # Forcefully inject a single Z error on the very first qubit
    sim.error_z = np.zeros(sim.N, dtype=np.uint8)
    sim.error_z[0] = 1
    sim.error_x = np.zeros(sim.N, dtype=np.uint8)
    
    # Mock the internal error generator to just return our deterministic trivial error
    sim._generate_error = MagicMock(return_value=(sim.error_x, sim.error_z))
    
    sim._single_run()
    
    # It's a weight 1 error on a generic open code, BP+OSD should trivially recover it.
    assert sim.osdw_success_count == 1, "The decoder failed to correct a weight-1 Z-error."

@patch("finite_bias_code_capacity_model.code_capacity_css.pd.DataFrame.to_csv")
@patch("finite_bias_code_capacity_model.code_capacity_css.os.makedirs")
def test_multiprocessing_dataframe_simulation_loop_smoke(mock_makedirs, mock_to_csv):
    """
    Test 3: Multiprocessing Data Aggregation Smoke Test.
    Ensures ProcessPoolExecutor doesn't deadlock and successfully flattens parallel outputs into pandas.
    """
    test_args = ["performance.py", "3", "3", "--bias", "100.0"]
    
    with patch("sys.argv", test_args), \
         patch("finite_bias_code_capacity_model.code_capacity_css.TARGET_RUNS", 2), \
         patch("finite_bias_code_capacity_model.code_capacity_css.np.linspace", return_value=[0.01]):
        
        # This will execute the `main()` function fully, which utilizes futures ProcessPoolExecutor.
        # It's patched heavily to only execute two total shots on one single error rate configuration.
        try:
            code_capacity_css.main()
        except Exception as e:
            pytest.fail(f"Simulation main loop crashed during parallel execution: {e}")
        
        # Verify the CSV saving routine was successfully reached at the very bottom
        mock_to_csv.assert_called_once()
        args, kwargs = mock_to_csv.call_args
        assert "csv" in args[0]
