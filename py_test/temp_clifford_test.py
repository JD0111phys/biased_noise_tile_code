import numpy as np
from bposd.css import css_code
import galois
from tile_code_and_clifford_deformation.tile_code import build_periodic_tile_code
from tile_code_and_clifford_deformation.clifford_deformations import (
    apply_probabilistic_deformation,
    hadamard_on_quarters_with_logicals,
    Deformation_on_Translational_invariant,
    Deformation_XY_Translational_invariant
)

# 1. Setup canonical CSS code
Hx, Hz, _ = build_periodic_tile_code(7, 7) # Need a multiple of 4 for quarter swaps, wait! 2n must be % 4 == 0.
# l=7, m=7 -> 98 edges. n=98. 2n=196. 196 % 4 == 0, yes!
qcode = css_code(Hx, Hz)
lx = qcode.lx.toarray()
lz = qcode.lz.toarray()
k, n = lx.shape
comm_matrix = (lx @ lz.T) % 2
GF = galois.GF(2)
comm_inv = np.array(np.linalg.inv(GF(comm_matrix)), dtype=int)
lx_canonical = (comm_inv @ lx) % 2

H_init = np.block([[Hx, np.zeros_like(Hx)], [np.zeros_like(Hz), Hz]])
LX_init = np.hstack([lx_canonical, np.zeros((k, n), dtype=int)])
LZ_init = np.hstack([np.zeros((k, n), dtype=int), lz])

def check_symplectic(H_def, LX_def, LZ_def):
    # Commutator function A M B^T mod 2
    def comm(A, B):
        n_qubits = A.shape[1] // 2
        Xa, Za = A[:, :n_qubits], A[:, n_qubits:]
        Xb, Zb = B[:, :n_qubits], B[:, n_qubits:]
        return (Xa @ Zb.T + Za @ Xb.T) % 2

    # H with H
    assert np.all(comm(H_def, H_def) == 0), "Stabilizers don't commute"
    # H with LX
    assert np.all(comm(H_def, LX_def) == 0), "Stabilizers don't commute with LX"
    # H with LZ
    assert np.all(comm(H_def, LZ_def) == 0), "Stabilizers don't commute with LZ"
    # LX with LX
    assert np.all(comm(LX_def, LX_def) == 0), "LX doesn't commute with LX"
    # LZ with LZ
    assert np.all(comm(LZ_def, LZ_def) == 0), "LZ doesn't commute with LZ"
    # LX with LZ
    assert np.array_equal(comm(LX_def, LZ_def), np.eye(k, dtype=int)), "LX and LZ don't anticommute to Identity"

# Test 1: Quarter Swap
H_q, lx_q, lz_q = hadamard_on_quarters_with_logicals(H_init.copy(), LX_init.copy())
_, lx_q2, lz_q2 = hadamard_on_quarters_with_logicals(H_init.copy(), LZ_init.copy())
LX_def = np.hstack([lx_q, lz_q])
LZ_def = np.hstack([lx_q2, lz_q2])
check_symplectic(H_q, LX_def, LZ_def)
print('Quarter Swap OK')

# Test 2: XY TI
H_xy, lx_xy, lz_xy = Deformation_XY_Translational_invariant(H_init.copy(), LX_init.copy())
_, lx_xy2, lz_xy2 = Deformation_XY_Translational_invariant(H_init.copy(), LZ_init.copy())
LX_def = np.hstack([lx_xy, lz_xy])
LZ_def = np.hstack([lx_xy2, lz_xy2])
check_symplectic(H_xy, LX_def, LZ_def)
print('XY TI OK')

print('All good')
