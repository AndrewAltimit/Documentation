"""
Formal Security Models and Frameworks

Implementation of provable security frameworks, universal composability,
and information-theoretic security schemes.
"""

import random


# Provable Security Framework

class SecurityReduction:
    """Framework for security reductions and proofs"""
    
    @staticmethod
    def discrete_log_to_cdh_reduction(dlog_solver, g, h, p):
        """
        Reduce Computational Diffie-Hellman to Discrete Log
        If we can solve DLog, we can solve CDH
        """
        # Given g^a and g^b, compute g^(ab)
        # Step 1: Use DLog solver to find a from g^a
        a = dlog_solver(g, h, p)
        
        # Step 2: Compute g^(ab) = (g^b)^a
        return lambda gb: pow(gb, a, p)
    
    @staticmethod
    def ind_cpa_game(encryption_scheme, adversary, security_parameter):
        """
        IND-CPA (Indistinguishability under Chosen Plaintext Attack) game
        """
        # Key generation
        pk, sk = encryption_scheme.keygen(security_parameter)
        
        # Adversary chooses two messages
        m0, m1, state = adversary.choose_messages(pk)
        
        # Challenger encrypts random message
        b = random.randint(0, 1)
        c = encryption_scheme.encrypt(pk, m0 if b == 0 else m1)
        
        # Adversary guesses
        b_prime = adversary.guess(c, state)
        
        # Advantage: |Pr[b' = b] - 1/2|
        return b == b_prime
    
    @staticmethod
    def ind_cca_game(encryption_scheme, adversary, security_parameter):
        """
        IND-CCA (Indistinguishability under Chosen Ciphertext Attack) game
        """
        # Key generation
        pk, sk = encryption_scheme.keygen(security_parameter)
        
        # Phase 1: Adversary queries decryption oracle
        oracle_queries = []
        
        def decryption_oracle(ciphertext):
            if ciphertext not in oracle_queries:
                oracle_queries.append(ciphertext)
                return encryption_scheme.decrypt(sk, ciphertext)
            return None
        
        # Adversary chooses messages with oracle access
        m0, m1, state = adversary.choose_messages(pk, decryption_oracle)
        
        # Challenge
        b = random.randint(0, 1)
        c_star = encryption_scheme.encrypt(pk, m0 if b == 0 else m1)
        oracle_queries.append(c_star)  # Cannot decrypt challenge
        
        # Phase 2: More queries (except c*)
        b_prime = adversary.guess(c_star, state, decryption_oracle)
        
        return b == b_prime


# Universal Composability Framework

class UCFramework:
    """Universal Composability security framework"""
    
    class IdealFunctionality:
        """Base class for ideal functionalities"""
        def __init__(self):
            self.parties = {}
            self.corrupted = set()
            self.sid = None  # Session ID
            
        def register_party(self, party_id):
            """Register a party in the functionality"""
            if party_id not in self.parties:
                self.parties[party_id] = {
                    'registered': True,
                    'input': None,
                    'output': None
                }
        
        def corrupt_party(self, party_id):
            """Mark a party as corrupted"""
            self.corrupted.add(party_id)
            
    class SecureComputation(IdealFunctionality):
        """Ideal functionality for secure computation"""
        def __init__(self, function):
            super().__init__()
            self.function = function
            self.inputs_received = {}
            
        def input(self, party_id, value):
            """Party provides input"""
            if party_id in self.parties:
                self.inputs_received[party_id] = value
                
                # Check if all parties provided input
                if len(self.inputs_received) == len(self.parties):
                    self._compute_and_deliver()
                    
        def _compute_and_deliver(self):
            """Compute function and deliver outputs"""
            # Collect non-corrupted inputs
            inputs = {}
            for pid, value in self.inputs_received.items():
                if pid not in self.corrupted:
                    inputs[pid] = value
                    
            # Compute function
            outputs = self.function(inputs)
            
            # Deliver to parties
            for pid in self.parties:
                if pid not in self.corrupted:
                    self.parties[pid]['output'] = outputs.get(pid)
                    
    class CommitmentFunctionality(IdealFunctionality):
        """Ideal commitment functionality"""
        def __init__(self):
            super().__init__()
            self.commitments = {}
            
        def commit(self, party_id, value):
            """Party commits to a value"""
            if party_id in self.parties:
                commitment_id = f"com_{party_id}_{len(self.commitments)}"
                self.commitments[commitment_id] = {
                    'committer': party_id,
                    'value': value,
                    'opened': False
                }
                return commitment_id
                
        def open(self, party_id, commitment_id):
            """Open a commitment"""
            if commitment_id in self.commitments:
                com = self.commitments[commitment_id]
                if com['committer'] == party_id and not com['opened']:
                    com['opened'] = True
                    return com['value']
            return None


# Information-Theoretic Security - Secret Sharing

class ShamirSecretSharing:
    """Shamir's (t,n) threshold secret sharing"""
    def __init__(self, threshold, num_shares, prime):
        self.t = threshold
        self.n = num_shares
        self.p = prime
        
    def share_secret(self, secret):
        """Split secret into n shares, need t to reconstruct"""
        # Generate random polynomial of degree t-1
        coefficients = [secret] + [random.randrange(0, self.p) 
                                  for _ in range(self.t - 1)]
        
        # Evaluate polynomial at n points
        shares = []
        for x in range(1, self.n + 1):
            y = sum(coef * pow(x, i, self.p) 
                   for i, coef in enumerate(coefficients)) % self.p
            shares.append((x, y))
            
        return shares
    
    def reconstruct_secret(self, shares):
        """Reconstruct secret from t shares using Lagrange interpolation"""
        if len(shares) < self.t:
            raise ValueError("Insufficient shares")
            
        # Use first t shares
        shares = shares[:self.t]
        
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            # Compute Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.p
                    denominator = (denominator * (xi - xj)) % self.p
                    
            # Compute Lagrange coefficient
            lagrange = (numerator * pow(denominator, -1, self.p)) % self.p
            secret = (secret + yi * lagrange) % self.p
            
        return secret
    
    def verify_share(self, share, commitment):
        """Verify share against public commitment (if using VSS)"""
        # Verifiable Secret Sharing extension
        # Would require commitment scheme implementation
        pass


class AdditiveSecretSharing:
    """Simple additive secret sharing over finite field"""
    def __init__(self, num_shares, modulus):
        self.n = num_shares
        self.modulus = modulus
        
    def share_secret(self, secret):
        """Split secret into n additive shares"""
        # Generate n-1 random shares
        shares = [random.randrange(0, self.modulus) for _ in range(self.n - 1)]
        
        # Last share ensures sum equals secret
        last_share = (secret - sum(shares)) % self.modulus
        shares.append(last_share)
        
        return shares
    
    def reconstruct_secret(self, shares):
        """Reconstruct by adding all shares"""
        if len(shares) != self.n:
            raise ValueError("Need all shares for additive scheme")
            
        return sum(shares) % self.modulus


class ReplicatedSecretSharing:
    """Replicated secret sharing for n=3, t=1"""
    def __init__(self, modulus):
        self.modulus = modulus
        
    def share_secret(self, secret):
        """Create replicated shares for 3 parties"""
        # Generate random values
        r1 = random.randrange(0, self.modulus)
        r2 = random.randrange(0, self.modulus)
        r3 = (secret - r1 - r2) % self.modulus
        
        # Each party gets two values
        share1 = (r1, r2)  # Party 1 gets r1, r2
        share2 = (r2, r3)  # Party 2 gets r2, r3
        share3 = (r3, r1)  # Party 3 gets r3, r1
        
        return [share1, share2, share3]
    
    def reconstruct_secret(self, shares):
        """Reconstruct from any 2 shares"""
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares")
            
        # Extract values and reconstruct
        if len(shares) == 3:
            # Have all shares
            return (shares[0][0] + shares[0][1] + shares[1][1]) % self.modulus
        else:
            # Need to determine which shares we have
            # Implementation depends on share identification
            pass


# Secure Multi-Party Computation Primitives

class ObliviousTransfer:
    """1-out-of-2 Oblivious Transfer"""
    def __init__(self, group_params):
        self.p = group_params['p']  # Prime
        self.g = group_params['g']  # Generator
        
    def sender_setup(self, m0, m1):
        """Sender has two messages m0, m1"""
        # Generate random values
        a = random.randrange(1, self.p - 1)
        A = pow(self.g, a, self.p)
        
        # Store for transfer phase
        self.sender_state = {
            'a': a,
            'A': A,
            'm0': m0,
            'm1': m1
        }
        
        return A
    
    def receiver_choose(self, b, A):
        """Receiver chooses bit b ∈ {0,1}"""
        # Generate random value
        x = random.randrange(1, self.p - 1)
        
        if b == 0:
            X = pow(self.g, x, self.p)
        else:
            X = (A * pow(self.g, x, self.p)) % self.p
            
        self.receiver_state = {
            'b': b,
            'x': x,
            'X': X
        }
        
        return X
    
    def sender_respond(self, X):
        """Sender computes encrypted messages"""
        a = self.sender_state['a']
        m0 = self.sender_state['m0']
        m1 = self.sender_state['m1']
        
        # Compute keys
        k0 = pow(X, a, self.p)
        k1 = pow(X / self.sender_state['A'], a, self.p)
        
        # Encrypt messages (simplified - should use proper encryption)
        e0 = (m0 * k0) % self.p
        e1 = (m1 * k1) % self.p
        
        return e0, e1
    
    def receiver_decrypt(self, e0, e1):
        """Receiver decrypts chosen message"""
        b = self.receiver_state['b']
        x = self.receiver_state['x']
        A = self.sender_state['A'] if hasattr(self, 'sender_state') else None
        
        if b == 0:
            k = pow(A, x, self.p) if A else None
            return (e0 * pow(k, -1, self.p)) % self.p if k else None
        else:
            k = pow(A, x, self.p) if A else None
            return (e1 * pow(k, -1, self.p)) % self.p if k else None