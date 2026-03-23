import pytest
import numpy as np
import stim
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Inject a mock for stimbposd to prevent ModuleNotFoundError when pytest attempts collection!
sys.modules['stimbposd'] = MagicMock()

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from finite_bias_circuit_level_simulation.circuit_level_css import (
    xyz_from_bias,
    generate_circuit,
    CircuitGenParameters,
    wilson_interval,
)

# Custom np.load patch to correctly resolve the code_data relative to the module instead of the test execution root.
original_load = np.load

def mock_np_load(file_path, *args, **kwargs):
    path_str = str(file_path)
    if "code_data/tilecode" in path_str:
        # Resolve to the actual directory where circuit_level_css.py resides
        abs_path = project_root / "finite_bias_circuit_level_simulation" / path_str
        return original_load(abs_path, *args, **kwargs)
    return original_load(file_path, *args, **kwargs)


def test_xyz_from_bias():
    """Test 1: Bias Mathematics."""
    p_total = 0.01
    r_bias = 100.0
    
    px, py, pz = xyz_from_bias(p_total, r_bias)
    
    # Denominator is r_bias + 2 = 102
    assert math.isclose(pz, p_total * 100 / 102)
    assert math.isclose(px, p_total * 1 / 102)
    assert math.isclose(py, p_total * 1 / 102)

@patch("numpy.load", side_effect=mock_np_load)
def test_generate_circuit_smoke(mock_load):
    """Test 2: Native Circuit Generation Smoke Test."""
    # The smallest pre-computed code in finite_bias_circuit_level_simulation/code_data/ is l=6.
    circuit = generate_circuit(
        rounds=1,
        x_distance=6,
        z_distance=6,
        bias=100.0
    )
    
    assert isinstance(circuit, stim.Circuit)
    assert len(circuit) > 0, "Generated circuit is completely empty."


@patch("numpy.load", side_effect=mock_np_load)
def test_circuit_structural_constraints(mock_load):
    """Test 3: Circuit Topographic Constraints."""
    circuit = generate_circuit(
        rounds=1,
        x_distance=6,
        z_distance=6,
        bias=100.0
    )
    
    num_observables = circuit.num_observables
    assert num_observables >= 1, f"Expected at least 1 logical observable for an open tile code memory experiment, got {num_observables}."
    
    # Verify DETECTOR presence
    has_detector = any(instruction.name == "DETECTOR" for instruction in circuit)
    assert has_detector, "The circuit is completely missing DETECTOR annotations needed for Sinter."
    
    # Assert physical mapping boundaries were assigned natively
    assert circuit.num_qubits > 0


def test_circuit_noise_injection():
    """Test 4: Noise Application via CircuitGenParameters."""
    params = CircuitGenParameters(
        code_name="test_code",
        task="test_task",
        rounds=1,
        before_measure_flip_probability=1.0,  # Massive 100% noise injection
        bias=100.0
    )
    
    test_circuit = stim.Circuit()
    targets = [0, 1]
    
    # Appending a measure instruction should natively prepend a PAULI_CHANNEL_1 because of the noise param.
    params.append_measure(test_circuit, targets, "Z")
    
    instructions = list(test_circuit)
    assert len(instructions) == 2, "Expected noisy measure to expand into 2 instructions."
    
    assert instructions[0].name == "PAULI_CHANNEL_1", "Phenomenological noise failed to insert."
    assert instructions[1].name == "M", "Base measurement instruction missing."


def test_wilson_interval():
    """Test 5: Wilson Interval Math Statistic."""
    # Standard distribution centered exactly on 50%
    lo, hi = wilson_interval(k=50, n=100)
    assert lo < 0.5 < hi
    
    # Edge case 0 out of 100
    lo, hi = wilson_interval(k=0, n=100)
    assert lo < 0.05
    
    # Safe 0-division bypass mapping
    lo, hi = wilson_interval(k=0, n=0)
    assert math.isnan(lo) and math.isnan(hi), "wilson_interval(0,0) failed to safely catch division bounds mapping to NaNs."
