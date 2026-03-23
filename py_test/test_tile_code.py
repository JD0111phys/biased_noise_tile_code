import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tile_code_and_clifford_deformation.tile_code import (
    build_periodic_tile_code,
    build_open_tile_code,
    get_edge_indices,
    stabilizer_to_vector,
)
from bposd.css import css_code
import galois

def test_periodic_tile_code_css_property():
    Hx, Hz, H_parity = build_periodic_tile_code(7, 7)
    
    # Check CSS parity check commutativity condition: Hx @ Hz.T = 0 mod 2
    comm = (Hx @ Hz.T) % 2
    assert np.all(comm == 0), "Periodic tile code CSS property failed: Hx @ Hz.T != 0 mod 2"
    
    # Check shapes
    assert len(Hx.shape) == 2
    assert len(Hz.shape) == 2
    assert H_parity.shape[0] == Hx.shape[0] + Hz.shape[0]
    assert H_parity.shape[1] == Hx.shape[1] + Hz.shape[1]

    # Check logical anticommutation
    qcode = css_code(Hx, Hz)
    lx = qcode.lx.toarray()
    lz = qcode.lz.toarray()
    comm_matrix = (lx @ lz.T) % 2
    assert np.any(comm_matrix), "Logical operators lx and lz must not completely commute (matrix is all zeros)."

    # Canonicalize logicals so lx[i] anticommutes with lz[i]
    GF = galois.GF(2)
    comm_inv = np.array(np.linalg.inv(GF(comm_matrix)), dtype=int)
    lx_canonical = (comm_inv @ lx) % 2
    
    # Check that the canonical logicals form the identity matrix mod 2
    identity_comm = (lx_canonical @ lz.T) % 2
    k = lx.shape[0]
    np.testing.assert_array_equal(
        identity_comm, 
        np.eye(k, dtype=int), 
        err_msg="Canonical logicals do not anticommute to identity for periodic tile code."
    )

def test_open_tile_code_css_property():
    Hx, Hz, H_parity = build_open_tile_code(4, 4)
    
    # Check CSS parity check commutativity condition
    comm = (Hx @ Hz.T) % 2
    assert np.all(comm == 0), "Open tile code CSS property failed: Hx @ Hz.T != 0 mod 2"
    
    # Check shapes
    assert len(Hx.shape) == 2
    assert len(Hz.shape) == 2
    assert H_parity.shape[0] == Hx.shape[0] + Hz.shape[0]
    assert H_parity.shape[1] == Hx.shape[1] + Hz.shape[1]

    # Check logical anticommutation
    qcode = css_code(Hx, Hz)
    lx = qcode.lx.toarray()
    lz = qcode.lz.toarray()
    comm_matrix = (lx @ lz.T) % 2
    assert np.any(comm_matrix), "Logical operators lx and lz must not completely commute (matrix is all zeros)."

    # Canonicalize logicals so lx[i] anticommutes with lz[i]
    GF = galois.GF(2)
    comm_inv = np.array(np.linalg.inv(GF(comm_matrix)), dtype=int)
    lx_canonical = (comm_inv @ lx) % 2
    
    # Check that the canonical logicals form the identity matrix mod 2
    identity_comm = (lx_canonical @ lz.T) % 2
    k = lx.shape[0]
    np.testing.assert_array_equal(
        identity_comm, 
        np.eye(k, dtype=int), 
        err_msg="Canonical logicals do not anticommute to identity for open tile code."
    )

def test_periodic_tile_code_different_dimensions():
    # For periodic tile code, l and m should be multiples of 7
    for l, m in [(7, 7), (7, 14), (14, 7)]:
        Hx, Hz, H_parity = build_periodic_tile_code(l, m)
        comm = (Hx @ Hz.T) % 2
        assert np.all(comm == 0)

def test_open_tile_code_different_dimensions():
    # For open tile code, l must equal m
    for l, m in [(3, 3), (4, 4), (5, 5)]:
        Hx, Hz, H_parity = build_open_tile_code(l, m)
        comm = (Hx @ Hz.T) % 2
        assert np.all(comm == 0)

def test_parity_block_structure():
    Hx, Hz, H_parity = build_periodic_tile_code(7, 7)
    
    nx, n_qubits_x = Hx.shape
    nz, n_qubits_z = Hz.shape
    
    assert n_qubits_x == n_qubits_z
    
    # Verify the off-diagonal blocks are zero
    np.testing.assert_array_equal(H_parity[:nx, :n_qubits_x], Hx)
    np.testing.assert_array_equal(H_parity[:nx, n_qubits_x:], np.zeros((nx, n_qubits_x)))
    np.testing.assert_array_equal(H_parity[nx:, :n_qubits_x], np.zeros((nz, n_qubits_x)))
    np.testing.assert_array_equal(H_parity[nx:, n_qubits_x:], Hz)

def test_edge_indices():
    l, m = 7, 7
    edges = get_edge_indices(l, m)
    # Total edges = l * m (horizontal) + l * m (vertical)
    assert len(edges) == 2 * l * m
    
    # Verify the first half are horizontal
    for i in range(l * m):
        assert edges[i][1] == "h"
        
    # Verify the second half are vertical
    for i in range(l * m, 2 * l * m):
        assert edges[i][1] == "v"

def test_stabilizer_to_vector():
    stab = [0, 2, 4]
    length = 6
    vec = stabilizer_to_vector(stab, length)
    expected = np.array([1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(vec, expected)
