import numpy as np
import pytest
import galois
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from bposd.css import css_code
from tile_code_and_clifford_deformation.tile_code import build_periodic_tile_code
from tile_code_and_clifford_deformation.clifford_deformations import (
    apply_probabilistic_deformation,
    hadamard_on_quarters_with_logicals,
    Deformation_on_Translational_invariant,
    Deformation_XY_Translational_invariant
)

@pytest.fixture
def base_code_data():
    """Builds the periodic tile code (l=7, m=7), extracts canonical logicals, and creates initial full blocks."""
    l, m = 7, 7
    Hx, Hz, _ = build_periodic_tile_code(l, m)
    
    qcode = css_code(Hx, Hz)
    lx = qcode.lx.toarray()
    lz = qcode.lz.toarray()
    k, n = lx.shape
    
    # Canonicalize logicals so lx[i] anticommutes tightly with lz[i]
    comm_matrix = (lx @ lz.T) % 2
    GF = galois.GF(2)
    comm_inv = np.array(np.linalg.inv(GF(comm_matrix)), dtype=int)
    lx_canonical = (comm_inv @ lx) % 2
    
    # Symplectic matrix for H
    H_init = np.block([[Hx, np.zeros_like(Hx)], [np.zeros_like(Hz), Hz]])
    # Symplectic arrays for logicals
    LX_init = np.hstack([lx_canonical, np.zeros((k, n), dtype=int)])
    LZ_init = np.hstack([np.zeros((k, n), dtype=int), lz])
    
    return H_init, LX_init, LZ_init, k, n, l

def verify_commutations(H_def, LX_def, LZ_def, k):
    """Checks all symplectic commutation relationships mod 2."""
    def comm(A, B):
        n_qubits = A.shape[1] // 2
        Xa, Za = A[:, :n_qubits], A[:, n_qubits:]
        Xb, Zb = B[:, :n_qubits], B[:, n_qubits:]
        return (Xa @ Zb.T + Za @ Xb.T) % 2

    # Stabilizers globally commute with each other
    assert np.all(comm(H_def, H_def) == 0), "Some deformed stabilizers do not commute with each other."
    
    # Logicals commute with all stabilizers
    assert np.all(comm(H_def, LX_def) == 0), "Logical X operators do not commute with deformed stabilizers."
    assert np.all(comm(H_def, LZ_def) == 0), "Logical Z operators do not commute with deformed stabilizers."
    
    # LX commutes with LX, LZ commutes with LZ
    assert np.all(comm(LX_def, LX_def) == 0), "Deformed LX operators do not commute amongst themselves."
    assert np.all(comm(LZ_def, LZ_def) == 0), "Deformed LZ operators do not commute amongst themselves."
    
    # LX and LZ anticommute correctly (canonical pairs)
    np.testing.assert_array_equal(
        comm(LX_def, LZ_def), 
        np.eye(k, dtype=int), 
        err_msg="Deformed logicals lost canonical anticommutation structure."
    )

def test_probabilistic_deformation(base_code_data):
    H_init, LX_init, LZ_init, k, n, l = base_code_data
    rng = np.random.default_rng(42)
    
    # Needs a random deformation applied
    H_def, lx_def, lz_def = apply_probabilistic_deformation(H_init.copy(), LX_init.copy(), p=0.25, q=0.5, rng=rng)
    # The deformations must be exactly identical to process LX and LZ
    # But apply_probabilistic_deformation depends on RNG!
    # To check symplectic traits strictly under random Cliffords, we MUST use the same seed per column!
    # OR we apply the exact same unitary S to H, LX, LZ. 
    pass

def test_probabilistic_deformation_deterministic(base_code_data):
    """Forces RNG seed logic to apply identical single qubit cliffs to H, LX, LZ."""
    H_init, LX_init, LZ_init, k, n, l = base_code_data
    
    # Mocking a fixed seq of deformations
    H_full = np.vstack([H_init, LX_init, LZ_init])
    r = H_init.shape[0]
    
    # We pass the stacked matrix as "L" to apply exactly the same transformation columns
    _, full_def_x, full_def_z = apply_probabilistic_deformation(np.zeros_like(H_init), H_full.copy(), 0.25, 0.5, np.random.default_rng(123))
    
    H_full_def = np.hstack([full_def_x, full_def_z])
    H_def = H_full_def[:r, :]
    LX_def = H_full_def[r:r+k, :]
    LZ_def = H_full_def[r+k:, :]
    
    verify_commutations(H_def, LX_def, LZ_def, k)

def test_quarter_swap_deformation(base_code_data):
    H_init, LX_init, LZ_init, k, n, l = base_code_data
    
    H_def, lx_def, lz_def = hadamard_on_quarters_with_logicals(H_init.copy(), LX_init.copy())
    _, lx2, lz2 = hadamard_on_quarters_with_logicals(H_init.copy(), LZ_init.copy())
    
    LX_def = np.hstack([lx_def, lz_def])
    LZ_def = np.hstack([lx2, lz2])
    
    verify_commutations(H_def, LX_def, LZ_def, k)

def test_translational_invariant_deformation(base_code_data):
    H_init, LX_init, LZ_init, k, n, l = base_code_data
    
    H_def, lx_def, lz_def = Deformation_on_Translational_invariant(H_init.copy(), LX_init.copy(), l)
    _, lx2, lz2 = Deformation_on_Translational_invariant(H_init.copy(), LZ_init.copy(), l)
    
    LX_def = np.hstack([lx_def, lz_def])
    LZ_def = np.hstack([lx2, lz2])
    
    verify_commutations(H_def, LX_def, LZ_def, k)

def test_xy_translational_invariant_deformation(base_code_data):
    H_init, LX_init, LZ_init, k, n, l = base_code_data
    
    H_def, lx_def, lz_def = Deformation_XY_Translational_invariant(H_init.copy(), LX_init.copy())
    _, lx2, lz2 = Deformation_XY_Translational_invariant(H_init.copy(), LZ_init.copy())
    
    LX_def = np.hstack([lx_def, lz_def])
    LZ_def = np.hstack([lx2, lz2])
    
    verify_commutations(H_def, LX_def, LZ_def, k)
