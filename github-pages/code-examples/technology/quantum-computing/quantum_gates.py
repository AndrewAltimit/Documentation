"""
Quantum Gates and Circuit Implementation

This module implements quantum gates, circuit construction, and optimization techniques
for quantum computing applications.
"""

from typing import Dict, List, Union

import numpy as np
import scipy.linalg as la


class QuantumGates:
    """Implementation of quantum gates with matrix representations"""

    # Single-qubit gates
    @staticmethod
    def pauli_gates() -> Dict[str, np.ndarray]:
        """Pauli matrices: generators of SU(2)"""
        return {
            "I": np.array([[1, 0], [0, 1]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate: H = (X + Z)/√2"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def rotation_gates(theta: float) -> Dict[str, np.ndarray]:
        """Rotation gates: Rₐ(θ) = exp(-iθσₐ/2)"""
        return {
            "Rx": np.array(
                [
                    [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                    [-1j * np.sin(theta / 2), np.cos(theta / 2)],
                ],
                dtype=complex,
            ),
            "Ry": np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ],
                dtype=complex,
            ),
            "Rz": np.array(
                [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
                dtype=complex,
            ),
        }

    @staticmethod
    def phase_gates() -> Dict[str, np.ndarray]:
        """Common phase gates"""
        return {
            "S": np.array([[1, 0], [0, 1j]], dtype=complex),  # √Z
            "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),  # ⁴√Z
            "Sdg": np.array([[1, 0], [0, -1j]], dtype=complex),  # S†
            "Tdg": np.array(
                [[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex
            ),  # T†
        }

    # Two-qubit gates
    @staticmethod
    def cnot() -> np.ndarray:
        """Controlled-NOT gate"""
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )

    @staticmethod
    def controlled_gate(U: np.ndarray) -> np.ndarray:
        """Create controlled version of single-qubit gate U"""
        n = U.shape[0]
        CU = np.eye(2 * n, dtype=complex)
        CU[n:, n:] = U
        return CU

    @staticmethod
    def swap() -> np.ndarray:
        """SWAP gate"""
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    @staticmethod
    def iswap() -> np.ndarray:
        """iSWAP gate"""
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    # Three-qubit gates
    @staticmethod
    def toffoli() -> np.ndarray:
        """Toffoli (CCNOT) gate"""
        toffoli = np.eye(8, dtype=complex)
        toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
        return toffoli

    @staticmethod
    def fredkin() -> np.ndarray:
        """Fredkin (CSWAP) gate"""
        fredkin = np.eye(8, dtype=complex)
        fredkin[5, 5], fredkin[6, 6] = 0, 0
        fredkin[5, 6], fredkin[6, 5] = 1, 1
        return fredkin


class QuantumCircuit:
    """Quantum circuit implementation with composition rules"""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        self.gates = []
        self.unitary = np.eye(self.dim, dtype=complex)

    def add_gate(self, gate: np.ndarray, qubits: Union[int, List[int]]):
        """Add gate to circuit"""
        if isinstance(qubits, int):
            qubits = [qubits]

        # Build full unitary
        full_gate = self._expand_gate(gate, qubits)
        self.unitary = full_gate @ self.unitary
        self.gates.append((gate, qubits))

    def _expand_gate(self, gate: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Expand gate to full Hilbert space"""
        n_gate_qubits = int(np.log2(gate.shape[0]))

        if n_gate_qubits == 1:
            # Single-qubit gate
            ops = [np.eye(2, dtype=complex) for _ in range(self.num_qubits)]
            ops[qubits[0]] = gate

            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return result

        else:
            # Multi-qubit gate - more complex tensor product structure
            # Implementation depends on qubit ordering convention
            return self._multi_qubit_expansion(gate, qubits)

    def _multi_qubit_expansion(self, gate: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Expand multi-qubit gate to full space"""
        # This is a simplified implementation
        # Full implementation would handle arbitrary qubit permutations
        n = self.num_qubits
        full_dim = 2**n
        expanded = np.zeros((full_dim, full_dim), dtype=complex)

        # Map gate indices to full space indices
        n_gate_qubits = len(qubits)
        gate_dim = 2**n_gate_qubits

        for i in range(full_dim):
            for j in range(full_dim):
                # Extract relevant qubit values
                i_bits = format(i, f"0{n}b")
                j_bits = format(j, f"0{n}b")

                # Check if non-gate qubits match
                match = True
                for k in range(n):
                    if k not in qubits and i_bits[k] != j_bits[k]:
                        match = False
                        break

                if match:
                    # Map to gate indices
                    gate_i = int("".join(i_bits[q] for q in qubits), 2)
                    gate_j = int("".join(j_bits[q] for q in qubits), 2)
                    expanded[i, j] = gate[gate_i, gate_j]

        return expanded

    def depth(self) -> int:
        """Circuit depth (assuming parallel execution)"""
        layers = []

        for gate, qubits in self.gates:
            # Find first layer where qubits are free
            placed = False
            for i, layer in enumerate(layers):
                if not any(q in layer for q in qubits):
                    layer.update(qubits)
                    placed = True
                    break

            if not placed:
                layers.append(set(qubits))

        return len(layers)

    def gate_count(self) -> Dict[str, int]:
        """Count gates by type"""
        counts = {}
        for gate, _ in self.gates:
            gate_type = self._identify_gate(gate)
            counts[gate_type] = counts.get(gate_type, 0) + 1
        return counts

    def _identify_gate(self, gate: np.ndarray) -> str:
        """Identify gate type from matrix"""
        # Compare with known gates
        gates = {
            "H": QuantumGates.hadamard(),
            "X": QuantumGates.pauli_gates()["X"],
            "Y": QuantumGates.pauli_gates()["Y"],
            "Z": QuantumGates.pauli_gates()["Z"],
            "CNOT": QuantumGates.cnot(),
        }

        for name, known_gate in gates.items():
            if gate.shape == known_gate.shape and np.allclose(gate, known_gate):
                return name

        return f"Unknown_{gate.shape[0]}x{gate.shape[0]}"


class CircuitOptimizer:
    """Quantum circuit optimization techniques"""

    @staticmethod
    def commutation_dag(circuit: QuantumCircuit) -> np.ndarray:
        """Build commutation DAG for circuit optimization"""
        n_gates = len(circuit.gates)
        dag = np.zeros((n_gates, n_gates), dtype=bool)

        for i in range(n_gates):
            for j in range(i + 1, n_gates):
                gate1, qubits1 = circuit.gates[i]
                gate2, qubits2 = circuit.gates[j]

                # Check if gates act on overlapping qubits
                if set(qubits1) & set(qubits2):
                    # Check if they commute
                    if not CircuitOptimizer._gates_commute(
                        gate1, qubits1, gate2, qubits2
                    ):
                        dag[i, j] = True

        return dag

    @staticmethod
    def _gates_commute(
        gate1: np.ndarray, qubits1: List[int], gate2: np.ndarray, qubits2: List[int]
    ) -> bool:
        """Check if two gates commute"""
        # Simplified check - in practice would expand gates and check [A,B] = 0
        if set(qubits1).isdisjoint(set(qubits2)):
            return True

        # If gates share qubits, check matrix commutation
        # This is a simplified implementation
        return False

    @staticmethod
    def optimize_two_qubit_gates(circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize sequences of two-qubit gates using KAK decomposition"""
        # Implementation of Cartan decomposition for SU(4)
        # This is a placeholder for the actual optimization algorithm
        optimized = QuantumCircuit(circuit.num_qubits)

        # Copy gates with optimization
        for gate, qubits in circuit.gates:
            optimized.add_gate(gate, qubits)

        return optimized
