"""
Quantum State Representation in Hilbert Space

This module implements quantum states and their operations in Hilbert space,
including inner products, tensor products, and measurements.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la


class QuantumState:
    """Representation of quantum states in Hilbert space"""

    def __init__(
        self, coefficients: np.ndarray, basis_labels: Optional[List[str]] = None
    ):
        """
        Initialize quantum state |ψ⟩ = Σᵢ αᵢ|i⟩

        Args:
            coefficients: Complex amplitudes αᵢ
            basis_labels: Labels for basis states |i⟩
        """
        self.coefficients = np.array(coefficients, dtype=complex)
        self.normalize()

        n = len(coefficients)
        if basis_labels is None:
            # Default to computational basis
            self.basis_labels = [format(i, f"0{int(np.log2(n))}b") for i in range(n)]
        else:
            self.basis_labels = basis_labels

    def normalize(self):
        """Ensure ⟨ψ|ψ⟩ = 1"""
        norm = np.linalg.norm(self.coefficients)
        if norm > 0:
            self.coefficients /= norm

    def inner_product(self, other: "QuantumState") -> complex:
        """Compute ⟨ψ|φ⟩"""
        return np.vdot(self.coefficients, other.coefficients)

    def tensor_product(self, other: "QuantumState") -> "QuantumState":
        """Compute |ψ⟩ ⊗ |φ⟩"""
        new_coeffs = np.kron(self.coefficients, other.coefficients)
        new_labels = [
            f"{l1}{l2}" for l1 in self.basis_labels for l2 in other.basis_labels
        ]
        return QuantumState(new_coeffs, new_labels)

    def expectation_value(self, operator: np.ndarray) -> complex:
        """Compute ⟨ψ|A|ψ⟩"""
        return np.vdot(self.coefficients, operator @ self.coefficients)

    def measure(self, basis: Optional[np.ndarray] = None) -> Tuple[int, "QuantumState"]:
        """Perform quantum measurement"""
        if basis is None:
            # Computational basis measurement
            probabilities = np.abs(self.coefficients) ** 2
        else:
            # Change to measurement basis
            transformed = basis.conj().T @ self.coefficients
            probabilities = np.abs(transformed) ** 2

        # Sample outcome
        outcome = np.random.choice(len(probabilities), p=probabilities)

        # Collapse state
        new_state = np.zeros_like(self.coefficients)
        if basis is None:
            new_state[outcome] = 1
        else:
            new_state = basis[:, outcome]

        return outcome, QuantumState(new_state, self.basis_labels)


class DensityMatrix:
    """Density matrix representation for mixed states"""

    def __init__(self, matrix: np.ndarray):
        """
        Initialize density matrix ρ

        Properties:
        - Hermitian: ρ† = ρ
        - Positive semidefinite: ⟨ψ|ρ|ψ⟩ ≥ 0 for all |ψ⟩
        - Unit trace: Tr(ρ) = 1
        """
        self.matrix = np.array(matrix, dtype=complex)
        self.dimension = matrix.shape[0]

        # Verify properties
        assert np.allclose(
            self.matrix, self.matrix.conj().T
        ), "Density matrix must be Hermitian"
        assert np.allclose(
            np.trace(self.matrix), 1
        ), "Density matrix must have unit trace"

        eigenvalues = np.linalg.eigvalsh(self.matrix)
        assert np.all(
            eigenvalues >= -1e-10
        ), "Density matrix must be positive semidefinite"

    @classmethod
    def from_pure_state(cls, state: QuantumState) -> "DensityMatrix":
        """Create density matrix from pure state: ρ = |ψ⟩⟨ψ|"""
        coeffs = state.coefficients.reshape(-1, 1)
        matrix = coeffs @ coeffs.conj().T
        return cls(matrix)

    @classmethod
    def from_ensemble(
        cls, states: List[QuantumState], probabilities: List[float]
    ) -> "DensityMatrix":
        """Create mixed state: ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|"""
        assert abs(sum(probabilities) - 1) < 1e-10, "Probabilities must sum to 1"

        dim = len(states[0].coefficients)
        matrix = np.zeros((dim, dim), dtype=complex)

        for state, prob in zip(states, probabilities):
            coeffs = state.coefficients.reshape(-1, 1)
            matrix += prob * (coeffs @ coeffs.conj().T)

        return cls(matrix)

    def purity(self) -> float:
        """Compute purity: Tr(ρ²)"""
        return np.real(np.trace(self.matrix @ self.matrix))

    def von_neumann_entropy(self) -> float:
        """Compute von Neumann entropy: S = -Tr(ρ log ρ)"""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        # Remove zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    def partial_trace(
        self, subsystem_dims: List[int], trace_out: List[int]
    ) -> "DensityMatrix":
        """Compute partial trace over specified subsystems"""
        # Implementation of partial trace
        total_dims = subsystem_dims
        keep_dims = [d for i, d in enumerate(subsystem_dims) if i not in trace_out]

        # Reshape and trace
        reshaped = self.matrix.reshape(total_dims + total_dims)

        # Trace out specified indices
        for idx in sorted(trace_out, reverse=True):
            reshaped = np.trace(reshaped, axis1=idx, axis2=idx + len(subsystem_dims))
            total_dims.pop(idx)

        new_dim = np.prod(keep_dims)
        return DensityMatrix(reshaped.reshape(new_dim, new_dim))

    def entanglement_entropy(
        self, subsystem_dims: List[int], partition: List[int]
    ) -> float:
        """Compute entanglement entropy across partition"""
        reduced_density = self.partial_trace(subsystem_dims, partition)
        return reduced_density.von_neumann_entropy()
