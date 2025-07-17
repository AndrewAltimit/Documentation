"""
Physical Implementations of Quantum Computing

Implementation details for various quantum computing hardware platforms including
superconducting qubits, trapped ions, topological qubits, and photonic systems.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la


class SuperconductingQubit:
    """Transmon qubit implementation details"""

    def __init__(self, E_J: float, E_C: float, n_g: float = 0):
        """
        Args:
            E_J: Josephson energy
            E_C: Charging energy
            n_g: Offset charge
        """
        self.E_J = E_J
        self.E_C = E_C
        self.n_g = n_g

        # Transmon regime: E_J/E_C >> 1
        self.regime = "transmon" if E_J / E_C > 50 else "charge"

    def hamiltonian(self, n_levels: int = 4) -> np.ndarray:
        """Transmon Hamiltonian in charge basis"""
        H = np.zeros((n_levels, n_levels), dtype=complex)

        # Charging energy term
        for n in range(n_levels):
            H[n, n] = 4 * self.E_C * (n - self.n_g) ** 2

        # Josephson coupling term
        for n in range(n_levels - 1):
            H[n, n + 1] = -self.E_J / 2
            H[n + 1, n] = -self.E_J / 2

        return H

    def energy_levels(self, n_levels: int = 4) -> np.ndarray:
        """Compute energy levels"""
        H = self.hamiltonian(n_levels)
        eigenvalues = np.linalg.eigvalsh(H)
        return eigenvalues - eigenvalues[0]  # Set ground state to zero

    def anharmonicity(self) -> float:
        """Compute anharmonicity α = E12 - E01"""
        levels = self.energy_levels(4)
        return (levels[2] - levels[1]) - (levels[1] - levels[0])

    def charge_dispersion(self, n_g_values: np.ndarray) -> np.ndarray:
        """Calculate charge dispersion ε_01(n_g)"""
        dispersions = []

        for n_g in n_g_values:
            self.n_g = n_g
            levels = self.energy_levels()
            dispersions.append(levels[1] - levels[0])

        return np.array(dispersions)

    def t1_estimate(self, temperature: float, omega_01: float) -> float:
        """Estimate T1 relaxation time"""
        # Simplified model
        k_B = 1.380649e-23  # Boltzmann constant
        hbar = 1.054571817e-34

        # Thermal occupation
        n_thermal = 1 / (np.exp(hbar * omega_01 / (k_B * temperature)) - 1)

        # Purcell limit (assuming Q ~ 10^6)
        Q = 1e6
        kappa = omega_01 / Q

        T1 = 1 / (kappa * (1 + n_thermal))
        return T1

    def gate_fidelity(self, gate_time: float, T1: float, T2: float) -> float:
        """Estimate single-qubit gate fidelity"""
        # Simple exponential decay model
        decay_T1 = np.exp(-gate_time / T1)
        decay_T2 = np.exp(-gate_time / T2)

        # Average fidelity including both relaxation and dephasing
        fidelity = (2 + decay_T1 + decay_T2) / 4
        return fidelity


class TrappedIonQubit:
    """Trapped ion qubit implementation"""

    def __init__(self, mass: float, charge: float, trap_frequency: float):
        """
        Args:
            mass: Ion mass (kg)
            charge: Ion charge (C)
            trap_frequency: Trap frequency (Hz)
        """
        self.mass = mass
        self.charge = charge
        self.omega_trap = 2 * np.pi * trap_frequency

        # Lamb-Dicke parameter
        self.lamb_dicke = self._calculate_lamb_dicke()

    def _calculate_lamb_dicke(self) -> float:
        """Calculate Lamb-Dicke parameter η"""
        hbar = 1.054571817e-34
        k = 2 * np.pi / 369e-9  # Typical laser wavelength

        eta = k * np.sqrt(hbar / (2 * self.mass * self.omega_trap))
        return eta

    def motional_states(self, n_max: int = 10) -> Dict[str, np.ndarray]:
        """Compute coupled spin-motional states"""
        states = {}

        # |g,n⟩ and |e,n⟩ basis states
        for n in range(n_max):
            states[f"g,{n}"] = np.zeros(2 * n_max)
            states[f"g,{n}"][n] = 1

            states[f"e,{n}"] = np.zeros(2 * n_max)
            states[f"e,{n}"][n_max + n] = 1

        return states

    def molmer_sorensen_gate(self, ions: List[int], phase: float = 0) -> np.ndarray:
        """
        Mølmer-Sørensen gate for entangling ions

        Implements XX-type interaction: exp(-i θ/2 σₓ⊗σₓ)
        """
        n_ions = len(ions)
        dim = 2**n_ions

        # Build XX operator
        XX = np.zeros((dim, dim), dtype=complex)

        # Pauli X matrix
        sigma_x = np.array([[0, 1], [1, 0]])

        # Construct full XX operator
        for i in range(n_ions):
            for j in range(i + 1, n_ions):
                if i in ions and j in ions:
                    # Build σₓ^(i) ⊗ σₓ^(j)
                    op = 1
                    for k in range(n_ions):
                        if k == i or k == j:
                            op = np.kron(op, sigma_x)
                        else:
                            op = np.kron(op, np.eye(2))
                    XX += op

        # Evolution under XX interaction
        U = la.expm(-1j * phase / 2 * XX)
        return U

    def addressing_errors(
        self, target_ion: int, beam_width: float, ion_spacing: float
    ) -> Dict[int, float]:
        """Calculate addressing errors from beam spillover"""
        errors = {}

        for ion in range(5):  # Assume 5-ion chain
            distance = abs(ion - target_ion) * ion_spacing
            # Gaussian beam profile
            spillover = np.exp(-2 * (distance / beam_width) ** 2)
            errors[ion] = spillover

        return errors


@dataclass
class TopologicalQubit:
    """Topological qubit based on Majorana zero modes"""

    gap: float  # Topological gap
    coherence_length: float  # Superconducting coherence length
    wire_length: float  # Nanowire length

    def braiding_operator(self, exchange_type: str = "clockwise") -> np.ndarray:
        """
        Braiding operator for Majorana zero modes

        Implements non-Abelian statistics: σ₁σ₂ = e^(iπ/4)
        """
        if exchange_type == "clockwise":
            # Clockwise exchange
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0, 0, 0, 1],
                ],
                dtype=complex,
            ) * np.exp(1j * np.pi / 4)
        else:
            # Counter-clockwise (inverse)
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0, -1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0, 0, 0, 1],
                ],
                dtype=complex,
            ) * np.exp(-1j * np.pi / 4)

    def fusion_rules(self) -> Dict[str, str]:
        """Majorana fermion fusion rules"""
        return {
            "1 × 1": "1",  # Vacuum
            "1 × ψ": "ψ",  # Majorana
            "ψ × 1": "ψ",
            "ψ × ψ": "1 + ψ",  # Non-Abelian
        }

    def topological_protection(self) -> float:
        """Estimate topological protection factor"""
        # Protection exponential in L/ξ
        return np.exp(-self.wire_length / self.coherence_length)

    def quasiparticle_poisoning_rate(self, temperature: float) -> float:
        """Estimate quasiparticle poisoning rate"""
        k_B = 1.380649e-23

        # Arrhenius-like activation
        rate = 1e6 * np.exp(-self.gap / (k_B * temperature))  # Hz
        return rate


class PhotonicQubit:
    """Photonic qubit implementations"""

    @staticmethod
    def dual_rail_encoding(photon_state: str) -> np.ndarray:
        """
        Dual-rail encoding: |0⟩ = |1,0⟩, |1⟩ = |0,1⟩
        """
        if photon_state == "0":
            return np.array([1, 0], dtype=complex)  # Photon in mode 0
        elif photon_state == "1":
            return np.array([0, 1], dtype=complex)  # Photon in mode 1
        else:
            raise ValueError("Invalid photon state")

    @staticmethod
    def polarization_encoding(polarization: str) -> np.ndarray:
        """
        Polarization encoding: |0⟩ = |H⟩, |1⟩ = |V⟩
        """
        if polarization == "H":
            return np.array([1, 0], dtype=complex)
        elif polarization == "V":
            return np.array([0, 1], dtype=complex)
        elif polarization == "D":  # Diagonal
            return np.array([1, 1], dtype=complex) / np.sqrt(2)
        elif polarization == "A":  # Anti-diagonal
            return np.array([1, -1], dtype=complex) / np.sqrt(2)
        elif polarization == "R":  # Right circular
            return np.array([1, 1j], dtype=complex) / np.sqrt(2)
        elif polarization == "L":  # Left circular
            return np.array([1, -1j], dtype=complex) / np.sqrt(2)
        else:
            raise ValueError("Invalid polarization")

    @staticmethod
    def kly_gate(alpha: float, phi: float) -> np.ndarray:
        """
        KLM (Knill-Laflamme-Milburn) probabilistic gate

        Success probability depends on ancilla measurement
        """
        # Simplified version of KLM CNOT
        success_prob = np.sin(alpha) ** 2

        gate = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(phi), -np.sin(phi)],
                [0, 0, np.sin(phi), np.cos(phi)],
            ],
            dtype=complex,
        )

        return gate, success_prob

    @staticmethod
    def cluster_state_generation(n_photons: int) -> np.ndarray:
        """Generate linear cluster state for measurement-based QC"""
        # Create GHZ-like entangled state
        state = np.zeros(2**n_photons, dtype=complex)

        # |+⟩^⊗n followed by CZ gates
        plus_state = np.array([1, 1]) / np.sqrt(2)

        # Initial product state
        state[0] = 1
        for i in range(n_photons):
            state = np.kron(state, plus_state)

        # Apply CZ between neighbors
        # (simplified - actual implementation would use proper CZ gates)

        return state / np.linalg.norm(state)


class NeutralAtomQubit:
    """Neutral atom qubit in optical lattices or tweezers"""

    def __init__(self, atom_type: str, wavelength: float):
        self.atom_type = atom_type
        self.wavelength = wavelength

        # Rydberg blockade radius
        self.blockade_radius = self._calculate_blockade_radius()

    def _calculate_blockade_radius(self) -> float:
        """Calculate Rydberg blockade radius"""
        # Simplified calculation
        C6 = 1e-60  # Van der Waals coefficient (Jm^6)
        Omega = 1e6  # Rabi frequency (Hz)

        R_b = (C6 / Omega) ** (1 / 6)
        return R_b

    def rydberg_gate(self, control: int, target: int, distance: float) -> np.ndarray:
        """
        Rydberg blockade controlled-Z gate
        """
        if distance < self.blockade_radius:
            # Strong blockade regime
            return np.diag([1, 1, 1, -1])  # CZ gate
        else:
            # Weak/no blockade
            theta = self.blockade_radius**6 / distance**6
            return np.diag([1, 1, 1, np.exp(1j * theta)])

    def optical_lattice_potential(self, position: np.ndarray) -> float:
        """Calculate optical lattice potential"""
        k = 2 * np.pi / self.wavelength

        # 3D lattice potential
        V = np.sum(np.sin(k * position) ** 2)
        return V
