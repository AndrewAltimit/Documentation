"""
Advanced Cryptographic Foundations

Implementation of various cryptographic algorithms including number theory,
elliptic curves, post-quantum cryptography, and zero-knowledge proofs.
"""

import random
import hashlib
import os
import numpy as np
from math import gcd


# Number Theory for Cryptography

def miller_rabin(n, k=5):
    """Miller-Rabin primality test"""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witness loop
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bits):
    """Generate a prime number with specified bit length"""
    while True:
        p = random.getrandbits(bits)
        p |= (1 << bits - 1) | 1  # Ensure MSB is 1 and number is odd
        if miller_rabin(p):
            return p


# Elliptic Curve Cryptography Theory

class EllipticCurve:
    """Elliptic curve over finite field: y² = x³ + ax + b (mod p)"""
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
        
    def is_on_curve(self, point):
        if point is None:
            return True  # Point at infinity
        x, y = point
        return (y * y - x * x * x - self.a * x - self.b) % self.p == 0
    
    def point_addition(self, P, Q):
        """Add two points on the elliptic curve"""
        if P is None: return Q
        if Q is None: return P
        
        x1, y1 = P
        x2, y2 = Q
        
        if x1 == x2:
            if y1 == y2:
                # Point doubling
                s = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, self.p) % self.p
            else:
                return None  # Point at infinity
        else:
            # Point addition
            s = (y2 - y1) * pow(x2 - x1, -1, self.p) % self.p
        
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_multiplication(self, k, P):
        """Multiply point P by scalar k using double-and-add"""
        if k == 0: return None
        if k == 1: return P
        
        result = None
        addend = P
        
        while k:
            if k & 1:
                result = self.point_addition(result, addend)
            addend = self.point_addition(addend, addend)
            k >>= 1
            
        return result

# Example: secp256k1 parameters (used in Bitcoin)
secp256k1 = EllipticCurve(
    a=0,
    b=7,
    p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
)


# Post-Quantum Cryptography - Lattice-Based Cryptography

class LearningWithErrors:
    """Learning with Errors (LWE) cryptosystem"""
    def __init__(self, n, q, sigma):
        self.n = n  # Dimension
        self.q = q  # Modulus
        self.sigma = sigma  # Error distribution parameter
        
    def generate_keypair(self):
        """Generate public and private keys"""
        # Private key: random vector s
        s = np.random.randint(0, self.q, size=self.n)
        
        # Public key: (A, b = As + e)
        A = np.random.randint(0, self.q, size=(self.n, self.n))
        e = np.random.normal(0, self.sigma, size=self.n).astype(int) % self.q
        b = (A @ s + e) % self.q
        
        return (A, b), s
    
    def encrypt(self, public_key, message_bit):
        """Encrypt a single bit"""
        A, b = public_key
        
        # Random vector r
        r = np.random.randint(0, 2, size=self.n)
        
        # Ciphertext: (u = A^T r, v = b^T r + m * q/2)
        u = (A.T @ r) % self.q
        v = (np.dot(b, r) + message_bit * (self.q // 2)) % self.q
        
        return u, v
    
    def decrypt(self, private_key, ciphertext):
        """Decrypt a single bit"""
        s = private_key
        u, v = ciphertext
        
        # Compute v - s^T u
        result = (v - np.dot(s, u)) % self.q
        
        # Decode: check if closer to 0 or q/2
        if result > self.q // 4 and result < 3 * self.q // 4:
            return 1
        else:
            return 0


# Hash-Based Signatures (Merkle Trees)

class MerkleSignature:
    """Merkle tree signature scheme"""
    def __init__(self, height, hash_func=hashlib.sha256):
        self.height = height
        self.hash = hash_func
        self.num_leaves = 2 ** height
        
    def generate_keypair(self):
        """Generate Merkle tree keypair"""
        # Generate one-time signing keys
        leaves = []
        private_keys = []
        
        for i in range(self.num_leaves):
            priv = os.urandom(32)
            pub = self.hash(priv).digest()
            private_keys.append(priv)
            leaves.append(pub)
        
        # Build Merkle tree
        tree = self._build_tree(leaves)
        root = tree[0][0]
        
        return root, (private_keys, tree)
    
    def _build_tree(self, leaves):
        """Build Merkle tree from leaves"""
        tree = [leaves]
        
        while len(tree[-1]) > 1:
            level = []
            for i in range(0, len(tree[-1]), 2):
                left = tree[-1][i]
                right = tree[-1][i + 1] if i + 1 < len(tree[-1]) else left
                parent = self.hash(left + right).digest()
                level.append(parent)
            tree.append(level)
            
        return tree
    
    def sign(self, private_key, message, index):
        """Sign message using leaf at given index"""
        private_keys, tree = private_key
        
        # One-time signature
        signature = private_keys[index]
        
        # Authentication path
        auth_path = []
        idx = index
        for level in range(self.height):
            sibling_idx = idx ^ 1
            if sibling_idx < len(tree[level]):
                auth_path.append((sibling_idx % 2, tree[level][sibling_idx]))
            idx //= 2
            
        return signature, auth_path


# Zero-Knowledge Proofs - Schnorr Protocol

class SchnorrProtocol:
    """Non-interactive Schnorr zero-knowledge proof"""
    def __init__(self, p, q, g):
        self.p = p  # Prime modulus
        self.q = q  # Prime order of g
        self.g = g  # Generator
        
    def prove_knowledge(self, x, message):
        """Prove knowledge of x such that y = g^x mod p"""
        # Public value
        y = pow(self.g, x, self.p)
        
        # Commitment
        r = random.randrange(1, self.q)
        t = pow(self.g, r, self.p)
        
        # Challenge (Fiat-Shamir)
        c = int(hashlib.sha256(
            f"{self.g}{y}{t}{message}".encode()
        ).hexdigest(), 16) % self.q
        
        # Response
        s = (r + c * x) % self.q
        
        return (y, t, s), c
    
    def verify_proof(self, proof, challenge, message):
        """Verify zero-knowledge proof"""
        y, t, s = proof
        
        # Recompute challenge
        c_verify = int(hashlib.sha256(
            f"{self.g}{y}{t}{message}".encode()
        ).hexdigest(), 16) % self.q
        
        if c_verify != challenge:
            return False
            
        # Verify: g^s = t * y^c mod p
        left = pow(self.g, s, self.p)
        right = (t * pow(y, challenge, self.p)) % self.p
        
        return left == right


# Homomorphic Encryption - Paillier Cryptosystem

class PaillierCrypto:
    """Additively homomorphic Paillier cryptosystem"""
    def __init__(self, bits=1024):
        self.bits = bits
        
    def generate_keypair(self):
        """Generate Paillier keypair"""
        # Generate two large primes
        p = generate_prime(self.bits // 2)
        q = generate_prime(self.bits // 2)
        
        n = p * q
        g = n + 1  # Simple choice for g
        
        # Private key components
        lambda_n = (p - 1) * (q - 1)
        mu = pow(self._L(pow(g, lambda_n, n * n), n), -1, n)
        
        public_key = (n, g)
        private_key = (lambda_n, mu)
        
        return public_key, private_key
    
    def _L(self, x, n):
        """L function for Paillier"""
        return (x - 1) // n
    
    def encrypt(self, public_key, plaintext):
        """Encrypt plaintext"""
        n, g = public_key
        
        # Random r where gcd(r, n) = 1
        r = random.randrange(1, n)
        while gcd(r, n) != 1:
            r = random.randrange(1, n)
        
        # Ciphertext: c = g^m * r^n mod n^2
        ciphertext = (pow(g, plaintext, n * n) * pow(r, n, n * n)) % (n * n)
        return ciphertext
    
    def decrypt(self, public_key, private_key, ciphertext):
        """Decrypt ciphertext"""
        n, g = public_key
        lambda_n, mu = private_key
        
        # Plaintext: m = L(c^lambda mod n^2) * mu mod n
        plaintext = (self._L(pow(ciphertext, lambda_n, n * n), n) * mu) % n
        return plaintext
    
    def add_encrypted(self, public_key, c1, c2):
        """Add two encrypted values (homomorphic property)"""
        n, _ = public_key
        return (c1 * c2) % (n * n)
    
    def multiply_encrypted(self, public_key, ciphertext, scalar):
        """Multiply encrypted value by scalar"""
        n, _ = public_key
        return pow(ciphertext, scalar, n * n)