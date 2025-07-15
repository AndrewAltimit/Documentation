"""
Implementation of the Five Postulates of Quantum Mechanics

This module demonstrates the fundamental postulates that form the
mathematical foundation of quantum mechanics.
"""

import numpy as np
import scipy.linalg as la
from typing import List, Tuple


class QuantumMechanics:
    """Implementation of quantum mechanics postulates"""
    
    @staticmethod
    def postulate1_state_space():
        """
        Postulate 1: The state of a quantum system is described by 
        a unit vector |ψ⟩ in a Hilbert space ℋ
        """
        # Example: Two-level system (qubit)
        H_qubit = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩
        H_qubit_1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩
        H_qubit_superposition = (H_qubit + H_qubit_1) / np.sqrt(2)  # (|0⟩ + |1⟩)/√2
        return H_qubit_superposition
    
    @staticmethod
    def postulate2_evolution(H: np.ndarray, psi: np.ndarray, t: float) -> np.ndarray:
        """
        Postulate 2: Time evolution is governed by the Schrödinger equation
        iℏ ∂|ψ⟩/∂t = H|ψ⟩
        
        Solution: |ψ(t)⟩ = U(t)|ψ(0)⟩ where U(t) = exp(-iHt/ℏ)
        """
        # Set ℏ = 1 in natural units
        U = la.expm(-1j * H * t)
        return U @ psi
    
    @staticmethod
    def postulate3_measurement(psi: np.ndarray, 
                             observables: List[np.ndarray]) -> Tuple[int, np.ndarray]:
        """
        Postulate 3: Measurement is described by a set of measurement 
        operators {Mₘ} satisfying Σₘ Mₘ†Mₘ = I
        
        Probability of outcome m: p(m) = ⟨ψ|Mₘ†Mₘ|ψ⟩
        Post-measurement state: |ψ'⟩ = Mₘ|ψ⟩/√p(m)
        """
        probabilities = []
        
        for M in observables:
            prob = np.real(psi.conj().T @ M.conj().T @ M @ psi)
            probabilities.append(prob)
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        
        # Sample outcome
        outcome = np.random.choice(len(observables), p=probabilities)
        
        # Apply measurement operator
        M_outcome = observables[outcome]
        new_state = M_outcome @ psi
        new_state /= np.linalg.norm(new_state)
        
        return outcome, new_state
    
    @staticmethod
    def postulate4_composite_systems(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
        """
        Postulate 4: The state space of a composite system is the 
        tensor product of component state spaces
        
        |ψ⟩₁₂ = |ψ⟩₁ ⊗ |ψ⟩₂
        """
        return np.kron(psi1, psi2)
    
    @staticmethod
    def postulate5_projective_measurement(psi: np.ndarray, 
                                        observable: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Postulate 5: Projective measurements correspond to Hermitian observables
        
        Observable A = Σᵢ aᵢPᵢ where Pᵢ are projectors
        """
        # Diagonalize observable
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        # Compute probabilities for each eigenvalue
        probabilities = []
        for i in range(len(eigenvalues)):
            proj = np.outer(eigenvectors[:, i], eigenvectors[:, i].conj())
            prob = np.real(psi.conj().T @ proj @ psi)
            probabilities.append(prob)
        
        # Sample outcome
        outcome_idx = np.random.choice(len(eigenvalues), p=probabilities)
        measured_value = eigenvalues[outcome_idx]
        
        # Collapse to eigenstate
        new_state = eigenvectors[:, outcome_idx]
        
        return measured_value, new_state