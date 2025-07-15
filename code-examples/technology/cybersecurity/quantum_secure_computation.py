"""
Quantum Cryptanalysis and Secure Computation

Implementation of quantum attacks, post-quantum defenses,
and secure multi-party computation protocols.
"""

import numpy as np
import os
import random
import hashlib
from typing import List, Tuple, Dict, Callable


class QuantumAttacks:
    """Quantum algorithms for cryptanalysis"""
    
    @staticmethod
    def grovers_search(oracle, n_bits):
        """
        Grover's algorithm for searching unsorted database
        Breaks symmetric crypto in O(sqrt(N)) time
        
        Args:
            oracle: Function that returns True for target item
            n_bits: Number of bits in search space
        
        Returns:
            Quantum circuit for Grover's algorithm
        """
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        
        # Number of iterations
        iterations = int(np.pi / 4 * np.sqrt(2**n_bits))
        
        # Create quantum circuit
        qr = QuantumRegister(n_bits)
        cr = ClassicalRegister(n_bits)
        qc = QuantumCircuit(qr, cr)
        
        # Initialize superposition
        qc.h(qr)
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle
            oracle(qc, qr)
            
            # Diffusion operator
            qc.h(qr)
            qc.x(qr)
            qc.h(qr[-1])
            qc.mcx(qr[:-1], qr[-1])
            qc.h(qr[-1])
            qc.x(qr)
            qc.h(qr)
            
        # Measure
        qc.measure(qr, cr)
        
        return qc
    
    @staticmethod
    def shors_factoring(n):
        """
        Shor's algorithm outline for factoring
        Breaks RSA in polynomial time
        
        Args:
            n: Number to factor
        
        Returns:
            Factors of n (if found)
        """
        # Classical preprocessing
        if n % 2 == 0:
            return 2, n // 2
            
        # Check if n = a^b for some a, b > 1
        for b in range(2, int(np.log2(n)) + 1):
            a = int(n**(1/b))
            if a**b == n:
                return a, n // a
                
        # Main algorithm loop
        while True:
            # 1. Choose random a < n
            a = random.randrange(2, n)
            
            # 2. Check gcd(a, n)
            from math import gcd
            g = gcd(a, n)
            if g > 1:
                return g, n // g
                
            # 3. Find period r using quantum period finding
            # (Simulated here - actual implementation requires quantum computer)
            r = QuantumAttacks._simulated_period_finding(a, n)
            
            if r and r % 2 == 0:
                # 4. Compute factors
                x = pow(a, r // 2, n)
                if x != n - 1:
                    p = gcd(x - 1, n)
                    q = gcd(x + 1, n)
                    if p > 1 and q > 1:
                        return p, n // p
    
    @staticmethod
    def _simulated_period_finding(a, n):
        """Simulate quantum period finding (classical)"""
        # In reality, this would use quantum Fourier transform
        # This is a classical simulation for demonstration
        seen = {}
        x = 1
        for r in range(1, n):
            x = (x * a) % n
            if x in seen:
                return r - seen[x]
            seen[x] = r
        return None
    
    @staticmethod
    def quantum_walk_search(graph, marked_vertices):
        """
        Quantum walk algorithm for graph search
        
        Args:
            graph: Adjacency matrix of graph
            marked_vertices: Set of target vertices
        
        Returns:
            Quantum walk operator
        """
        n = len(graph)
        
        # Coin operator (Grover diffusion)
        def coin_operator():
            C = (2/n) * np.ones((n, n)) - np.eye(n)
            return C
            
        # Shift operator based on graph structure
        def shift_operator():
            S = np.zeros((n*n, n*n))
            for i in range(n):
                for j in range(n):
                    if graph[i][j] > 0:
                        # Connect vertex i coin state j to vertex j coin state i
                        S[j*n + i, i*n + j] = 1
            return S
            
        # Oracle for marked vertices
        def oracle():
            O = np.eye(n*n)
            for v in marked_vertices:
                for c in range(n):
                    O[v*n + c, v*n + c] = -1
            return O
            
        # Quantum walk operator
        C = coin_operator()
        S = shift_operator()
        O = oracle()
        
        # Walk operator: S(C⊗I)O
        C_tensor_I = np.kron(C, np.eye(n))
        W = S @ C_tensor_I @ O
        
        return W
    
    @staticmethod
    def hhl_algorithm(A, b):
        """
        HHL algorithm for solving linear systems Ax = b
        Exponential speedup for certain matrices
        
        Args:
            A: Matrix (must be Hermitian)
            b: Vector
        
        Returns:
            Solution vector x
        """
        # Simplified classical simulation
        # Real HHL requires quantum phase estimation
        
        # Ensure A is Hermitian
        if not np.allclose(A, A.conj().T):
            raise ValueError("Matrix must be Hermitian")
            
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        # Quantum phase estimation would extract these
        # Controlled rotation based on eigenvalues
        # Simplified: directly compute solution
        
        # Project b onto eigenbasis
        b_eigen = eigenvectors.conj().T @ b
        
        # Apply inverse eigenvalues (with regularization)
        epsilon = 1e-10
        x_eigen = b_eigen / (eigenvalues + epsilon)
        
        # Transform back
        x = eigenvectors @ x_eigen
        
        return x / np.linalg.norm(x)


class GarbledCircuit:
    """Yao's Garbled Circuit Protocol"""
    
    def __init__(self):
        self.wire_labels = {}
        self.garbled_tables = {}
        
    def garble_gate(self, gate_type, input_wires, output_wire):
        """Garble a single gate"""
        # Generate wire labels
        for wire in input_wires + [output_wire]:
            if wire not in self.wire_labels:
                self.wire_labels[wire] = {
                    0: os.urandom(16),
                    1: os.urandom(16)
                }
                
        # Create garbled truth table
        garbled_table = []
        
        for a in [0, 1]:
            for b in [0, 1]:
                # Compute gate output
                if gate_type == 'AND':
                    output = a & b
                elif gate_type == 'XOR':
                    output = a ^ b
                elif gate_type == 'OR':
                    output = a | b
                elif gate_type == 'NOT':
                    output = 1 - a
                    
                # Encrypt output label
                key_a = self.wire_labels[input_wires[0]][a]
                if len(input_wires) > 1:
                    key_b = self.wire_labels[input_wires[1]][b]
                else:
                    key_b = b'\x00' * 16
                    
                output_label = self.wire_labels[output_wire][output]
                
                # Double encryption
                encrypted = self.encrypt_label(key_a, key_b, output_label)
                garbled_table.append(encrypted)
                
        # Randomly permute table
        random.shuffle(garbled_table)
        self.garbled_tables[output_wire] = garbled_table
        
        return garbled_table
    
    def encrypt_label(self, key1, key2, label):
        """Encrypt label with two keys"""
        # Simple encryption scheme (should use proper authenticated encryption)
        combined_key = hashlib.sha256(key1 + key2).digest()[:16]
        
        # XOR encryption (simplified)
        encrypted = bytes(a ^ b for a, b in zip(label, combined_key))
        
        return encrypted
    
    def decrypt_label(self, key1, key2, encrypted):
        """Decrypt label with two keys"""
        return self.encrypt_label(key1, key2, encrypted)  # XOR is self-inverse
    
    def evaluate_gate(self, gate_table, input_labels):
        """Evaluate garbled gate"""
        # Try all entries in garbled table
        for entry in gate_table:
            if len(input_labels) == 2:
                decrypted = self.decrypt_label(input_labels[0], input_labels[1], entry)
            else:
                decrypted = self.decrypt_label(input_labels[0], b'\x00' * 16, entry)
                
            # Check if decryption is valid (would use authentication in practice)
            if len(decrypted) == 16:  # Valid label length
                return decrypted
                
        raise ValueError("Invalid gate evaluation")
    
    def garble_circuit(self, circuit):
        """Garble entire circuit"""
        garbled_circuit = {
            'gates': {},
            'output_map': {},
            'input_labels': {}
        }
        
        # Topological sort of gates
        for gate in circuit['gates']:
            gate_type = gate['type']
            inputs = gate['inputs']
            output = gate['output']
            
            # Garble gate
            table = self.garble_gate(gate_type, inputs, output)
            garbled_circuit['gates'][output] = {
                'type': gate_type,
                'inputs': inputs,
                'table': table
            }
            
        # Create output decoding table
        for output_wire in circuit['outputs']:
            garbled_circuit['output_map'][output_wire] = {
                self.wire_labels[output_wire][0]: 0,
                self.wire_labels[output_wire][1]: 1
            }
            
        # Input labels for garbler
        for input_wire in circuit['inputs']:
            garbled_circuit['input_labels'][input_wire] = self.wire_labels[input_wire]
            
        return garbled_circuit


class SecureMultipartyComputation:
    """Advanced MPC protocols"""
    
    @staticmethod
    def bgw_protocol(parties, inputs, circuit, threshold):
        """
        BGW protocol for secure computation
        
        Args:
            parties: List of party IDs
            inputs: Dict mapping party to their input
            circuit: Arithmetic circuit to evaluate
            threshold: Corruption threshold
        
        Returns:
            Output shares for each party
        """
        n = len(parties)
        
        # Input sharing phase
        shared_inputs = {}
        for party, value in inputs.items():
            # Each party shares their input
            shares = SecureMultipartyComputation._shamir_share(
                value, threshold, n
            )
            shared_inputs[party] = shares
            
        # Computation phase
        wire_shares = {}
        
        # Initialize input wires
        for wire, party in circuit['input_mapping'].items():
            wire_shares[wire] = shared_inputs[party]
            
        # Evaluate gates
        for gate in circuit['gates']:
            if gate['type'] == 'ADD':
                # Addition is local
                a_shares = wire_shares[gate['inputs'][0]]
                b_shares = wire_shares[gate['inputs'][1]]
                output_shares = [
                    (a[0] + b[0]) % gate['modulus']
                    for a, b in zip(a_shares, b_shares)
                ]
                
            elif gate['type'] == 'MUL':
                # Multiplication requires communication
                a_shares = wire_shares[gate['inputs'][0]]
                b_shares = wire_shares[gate['inputs'][1]]
                output_shares = SecureMultipartyComputation._bgw_multiply(
                    a_shares, b_shares, threshold, gate['modulus']
                )
                
            wire_shares[gate['output']] = output_shares
            
        # Output reconstruction
        outputs = {}
        for output_wire in circuit['outputs']:
            shares = wire_shares[output_wire]
            value = SecureMultipartyComputation._shamir_reconstruct(
                shares[:threshold+1], gate['modulus']
            )
            outputs[output_wire] = value
            
        return outputs
    
    @staticmethod
    def _shamir_share(secret, threshold, num_shares):
        """Generate Shamir secret shares"""
        # Random polynomial of degree threshold
        coefficients = [secret]
        coefficients.extend([
            random.randrange(1, 1000000)
            for _ in range(threshold)
        ])
        
        shares = []
        for x in range(1, num_shares + 1):
            y = sum(coef * (x ** i) for i, coef in enumerate(coefficients))
            shares.append(y)
            
        return shares
    
    @staticmethod
    def _shamir_reconstruct(shares, modulus):
        """Reconstruct secret from shares"""
        result = 0
        
        for i, (xi, yi) in enumerate(shares):
            # Lagrange coefficient
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator *= -xj
                    denominator *= (xi - xj)
                    
            coefficient = (numerator * pow(denominator, -1, modulus)) % modulus
            result = (result + yi * coefficient) % modulus
            
        return result
    
    @staticmethod
    def _bgw_multiply(a_shares, b_shares, threshold, modulus):
        """BGW multiplication protocol"""
        n = len(a_shares)
        
        # Local multiplication
        c_shares = [(a * b) % modulus for a, b in zip(a_shares, b_shares)]
        
        # Degree reduction
        # Each party shares their c_i value
        all_shares = []
        for i, c_i in enumerate(c_shares):
            shares = SecureMultipartyComputation._shamir_share(c_i, threshold, n)
            all_shares.append(shares)
            
        # Compute linear combination
        result_shares = []
        for i in range(n):
            # Lagrange coefficients for degree reduction
            value = 0
            for j in range(n):
                # Simplified - should use proper Lagrange interpolation
                value = (value + all_shares[j][i]) % modulus
                
            result_shares.append(value)
            
        return result_shares
    
    @staticmethod
    def gmc_protocol(circuit, inputs, num_parties):
        """
        GMW (Goldreich-Micali-Wigderson) protocol
        
        Args:
            circuit: Boolean circuit
            inputs: Binary inputs for each party
            num_parties: Number of parties
        
        Returns:
            Circuit output
        """
        # Initialize wire values (XOR shares)
        wire_values = {}
        
        # Input sharing
        for wire, (party, bit_index) in circuit['input_mapping'].items():
            # Each party creates XOR shares of their input
            shares = [0] * num_parties
            shares[party] = inputs[party][bit_index]
            
            # Random shares for others
            for i in range(num_parties):
                if i != party:
                    shares[i] = random.randint(0, 1)
                    shares[party] ^= shares[i]
                    
            wire_values[wire] = shares
            
        # Evaluate gates
        for gate in circuit['gates']:
            if gate['type'] == 'XOR':
                # XOR is local
                a_shares = wire_values[gate['inputs'][0]]
                b_shares = wire_values[gate['inputs'][1]]
                output_shares = [a ^ b for a, b in zip(a_shares, b_shares)]
                
            elif gate['type'] == 'AND':
                # AND requires OT
                a_shares = wire_values[gate['inputs'][0]]
                b_shares = wire_values[gate['inputs'][1]]
                output_shares = SecureMultipartyComputation._gmw_and_gate(
                    a_shares, b_shares, num_parties
                )
                
            wire_values[gate['output']] = output_shares
            
        # Output reconstruction
        output = 0
        for share in wire_values[circuit['output']]:
            output ^= share
            
        return output
    
    @staticmethod
    def _gmw_and_gate(a_shares, b_shares, num_parties):
        """GMW AND gate using OT"""
        output_shares = [0] * num_parties
        
        # Each pair of parties runs OT
        for i in range(num_parties):
            for j in range(i + 1, num_parties):
                # Party i is sender, party j is receiver
                # Simplified OT simulation
                
                # Party i's inputs to OT
                m00 = random.randint(0, 1)
                m01 = m00 ^ a_shares[i]
                m10 = m00 ^ b_shares[i]
                m11 = m00 ^ a_shares[i] ^ b_shares[i] ^ 1
                
                # Party j chooses based on their shares
                choice = (a_shares[j], b_shares[j])
                
                # OT output
                if choice == (0, 0):
                    ot_output = m00
                elif choice == (0, 1):
                    ot_output = m01
                elif choice == (1, 0):
                    ot_output = m10
                else:
                    ot_output = m11
                    
                # Update shares
                output_shares[i] ^= m00
                output_shares[j] ^= ot_output
                
        return output_shares