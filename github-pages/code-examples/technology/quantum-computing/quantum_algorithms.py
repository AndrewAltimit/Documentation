"""
Advanced Quantum Algorithms

This module implements key quantum algorithms including:
- Quantum Phase Estimation (QPE)
- HHL Algorithm for Linear Systems
- Quantum Walks
"""

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.linalg as la
from quantum_gates import QuantumCircuit, QuantumGates


class QuantumPhaseEstimation:
    """Quantum Phase Estimation (QPE) algorithm"""

    @staticmethod
    def qpe_circuit(
        unitary: np.ndarray, eigenvector: np.ndarray, precision_qubits: int
    ) -> QuantumCircuit:
        """
        Estimate eigenvalue e^(2πiφ) of unitary U

        Given: U|ψ⟩ = e^(2πiφ)|ψ⟩
        Output: n-bit approximation of φ
        """
        n_qubits = int(np.log2(unitary.shape[0]))
        circuit = QuantumCircuit(precision_qubits + n_qubits)

        # Initialize eigenstate (simplified - would use state preparation)
        # circuit.initialize_state(eigenvector, list(range(precision_qubits, precision_qubits + n_qubits)))

        # Create superposition in precision register
        for i in range(precision_qubits):
            circuit.add_gate(QuantumGates.hadamard(), i)

        # Controlled unitary operations
        for i in range(precision_qubits):
            # Apply U^(2^i)
            power = 2 ** (precision_qubits - i - 1)
            controlled_U = QuantumGates.controlled_gate(
                la.fractional_matrix_power(unitary, power)
            )
            # Simplified - would properly handle multi-qubit controlled gates
            # circuit.add_controlled_unitary(controlled_U, i, list(range(precision_qubits, precision_qubits + n_qubits)))

        # Inverse QFT on precision register (simplified)
        # circuit.add_inverse_qft(list(range(precision_qubits)))

        return circuit

    @staticmethod
    def extract_phase(measurement: List[int], precision: int) -> float:
        """Extract phase from QPE measurement"""
        # Convert binary measurement to phase
        value = 0
        for i, bit in enumerate(measurement):
            value += bit * 2 ** (-(i + 1))
        return value


class HHLAlgorithm:
    """Harrow-Hassidim-Lloyd algorithm for solving Ax = b"""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        precision: int = 4,
        evolution_time: float = 2 * np.pi,
    ):
        """
        Solve linear system Ax = b on quantum computer

        Assumptions:
        - A is Hermitian (can be relaxed)
        - A is sparse
        - Condition number κ is small
        """
        self.A = A
        self.b = b / np.linalg.norm(b)  # Normalize
        self.precision = precision
        self.evolution_time = evolution_time

        # Compute eigendecomposition for analysis
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(A)

    def build_circuit(self) -> QuantumCircuit:
        """Build HHL quantum circuit"""
        n_qubits = int(np.log2(self.A.shape[0]))
        circuit = QuantumCircuit(n_qubits + self.precision + 1)  # +1 for ancilla

        # Step 1: Prepare |b⟩ (simplified - would use state preparation)
        # circuit.initialize_state(self.b, list(range(n_qubits)))

        # Step 2: Phase estimation on e^(iAt)
        hamiltonian_simulation = self._hamiltonian_evolution()
        # qpe_circuit = QuantumPhaseEstimation.qpe_circuit(
        #     hamiltonian_simulation,
        #     self.b,
        #     self.precision
        # )

        # Step 3: Controlled rotation on ancilla
        # circuit.add_eigenvalue_inversion(
        #     list(range(n_qubits, n_qubits + self.precision)),
        #     n_qubits + self.precision
        # )

        # Step 4: Uncompute phase estimation
        # circuit.add_inverse_qpe()

        # Step 5: Measure ancilla
        # Post-select on |1⟩

        return circuit

    def _hamiltonian_evolution(self) -> np.ndarray:
        """Implement e^(iAt) using Hamiltonian simulation"""
        return la.expm(1j * self.A * self.evolution_time)

    def classical_comparison(self) -> Dict[str, Any]:
        """Compare with classical solution"""
        x_classical = np.linalg.solve(self.A, self.b)

        # Quantum solution (simplified simulation)
        x_quantum = self._simulate_hhl()

        return {
            "classical_solution": x_classical,
            "quantum_solution": x_quantum,
            "speedup": self._estimate_speedup(),
            "condition_number": np.linalg.cond(self.A),
        }

    def _simulate_hhl(self) -> np.ndarray:
        """Simplified HHL simulation"""
        # This is a classical simulation of the quantum algorithm
        # In practice, would run on quantum hardware

        # Decompose b in eigenbasis of A
        coefficients = self.eigenvectors.T @ self.b

        # Apply inverse eigenvalues (with regularization)
        epsilon = 0.01  # Regularization parameter
        result_coeffs = np.zeros_like(coefficients)

        for i, (coeff, eigenval) in enumerate(zip(coefficients, self.eigenvalues)):
            if abs(eigenval) > epsilon:
                result_coeffs[i] = coeff / eigenval

        # Transform back
        x_quantum = self.eigenvectors @ result_coeffs

        # Normalize
        return x_quantum / np.linalg.norm(x_quantum)

    def _estimate_speedup(self) -> float:
        """Estimate quantum speedup"""
        n = self.A.shape[0]
        sparsity = np.count_nonzero(self.A) / n**2
        kappa = np.linalg.cond(self.A)

        # Classical: O(n√κ) for sparse systems
        # Quantum: O(log(n)κ²/ε)
        classical_complexity = n * np.sqrt(kappa)
        quantum_complexity = np.log2(n) * kappa**2

        return classical_complexity / quantum_complexity


class QuantumWalk:
    """Quantum walk algorithms for graph problems"""

    def __init__(self, adjacency_matrix: np.ndarray):
        self.adjacency = adjacency_matrix
        self.n_vertices = adjacency_matrix.shape[0]

        # Compute graph Laplacian
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        self.laplacian = degree_matrix - adjacency_matrix

    def continuous_time_walk(self, initial_vertex: int, time: float) -> np.ndarray:
        """
        Continuous-time quantum walk
        |ψ(t)⟩ = e^(-iHt)|ψ(0)⟩ where H is the adjacency matrix
        """
        initial_state = np.zeros(self.n_vertices)
        initial_state[initial_vertex] = 1

        # Evolution operator
        U = la.expm(-1j * self.adjacency * time)

        return U @ initial_state

    def discrete_time_walk(self, steps: int) -> np.ndarray:
        """
        Discrete-time quantum walk with coin operator
        """
        # Hilbert space: position ⊗ coin
        coin_dim = 2  # For regular graphs
        total_dim = self.n_vertices * coin_dim

        # Coin operator (Hadamard)
        C = np.kron(np.eye(self.n_vertices), QuantumGates.hadamard())

        # Shift operator
        S = self._build_shift_operator()

        # Walk operator W = S·C
        W = S @ C

        # Initial state: |vertex⟩ ⊗ |coin⟩
        initial_state = np.zeros(total_dim)
        initial_state[0] = 1  # Start at vertex 0, coin state |0⟩

        # Evolve
        state = initial_state
        for _ in range(steps):
            state = W @ state

        return state

    def _build_shift_operator(self) -> np.ndarray:
        """Build shift operator for discrete quantum walk"""
        coin_dim = 2
        total_dim = self.n_vertices * coin_dim
        S = np.zeros((total_dim, total_dim), dtype=complex)

        # Shift based on coin state and adjacency
        for v in range(self.n_vertices):
            neighbors = np.where(self.adjacency[v] > 0)[0]
            if len(neighbors) >= 2:
                # Map coin |0⟩ to first neighbor, |1⟩ to second
                n0, n1 = neighbors[0], neighbors[1]
                S[2 * n0, 2 * v] = 1  # |n0,0⟩ ← |v,0⟩
                S[2 * n1 + 1, 2 * v + 1] = 1  # |n1,1⟩ ← |v,1⟩

        return S

    def hitting_time(self, start: int, target: int, max_time: float = 100) -> float:
        """
        Estimate quantum hitting time from start to target vertex
        """
        times = np.linspace(0, max_time, 1000)
        probabilities = []

        for t in times:
            state = self.continuous_time_walk(start, t)
            prob = np.abs(state[target]) ** 2
            probabilities.append(prob)

        # Find first peak
        peaks = self._find_peaks(probabilities)
        if peaks:
            return times[peaks[0]]
        return max_time

    def _find_peaks(self, data: List[float], threshold: float = 0.1) -> List[int]:
        """Find peaks in probability distribution"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > threshold and data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks.append(i)
        return peaks

    def spatial_search(self, marked_vertices: List[int]) -> Dict[str, Any]:
        """
        Quantum spatial search for marked vertices
        Returns analysis of search performance
        """
        # Oracle Hamiltonian
        H_oracle = np.zeros((self.n_vertices, self.n_vertices))
        for v in marked_vertices:
            H_oracle[v, v] = 1

        # Search Hamiltonian: H = -γL + H_oracle
        gamma = 1 / np.sqrt(self.n_vertices)  # Optimal for complete graph
        H_search = -gamma * self.laplacian + H_oracle

        # Initial state: uniform superposition
        initial_state = np.ones(self.n_vertices) / np.sqrt(self.n_vertices)

        # Optimal search time
        spectral_gap = np.sort(np.linalg.eigvalsh(H_search))[1]
        optimal_time = np.pi / (2 * spectral_gap)

        # Evolution
        final_state = la.expm(-1j * H_search * optimal_time) @ initial_state

        # Success probability
        success_prob = sum(np.abs(final_state[v]) ** 2 for v in marked_vertices)

        return {
            "optimal_time": optimal_time,
            "success_probability": success_prob,
            "final_state": final_state,
            "speedup": np.sqrt(self.n_vertices),  # Grover-like speedup
        }
