---
layout: docs
title: "Cybersecurity: Cryptography"
permalink: /docs/technology/cybersecurity/cryptography.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Cryptography</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Encryption, post-quantum algorithms, and the mathematics that protect your data</p>
</div>

<p class="breadcrumb"><a href="./">Cybersecurity</a> › Cryptography</p>

<div class="intro-card">
  <p class="lead-text">Cryptography is the foundation every other security control rests on. This page builds from symmetric and public-key encryption through modern techniques — elliptic curves, post-quantum algorithms, zero-knowledge proofs, and homomorphic encryption — and finishes with the mathematical foundations of RSA, elliptic curves, and secret sharing.</p>
</div>

## The Foundation: Encryption

Encryption is like a lock that protects your data. But unlike physical locks, digital encryption relies on mathematical problems that are easy to do in one direction but practically impossible to reverse without the key.

### Symmetric Encryption: One Key for Everything

The simplest form of encryption uses the same key to lock and unlock data. Imagine you and a friend have identical keys to a lockbox:

```python
from cryptography.fernet import Fernet

# Generate a key (both parties need this)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt a message
message = "Meet me at midnight"
encrypted = cipher.encrypt(message.encode())
print(f"Encrypted: {encrypted}")
# Output: b'gAAAAABh...long random-looking string...'

# Only someone with the key can decrypt
decrypted = cipher.decrypt(encrypted)
print(f"Decrypted: {decrypted.decode()}")
# Output: Meet me at midnight
```

This is how messaging apps like Signal protect your conversations. But there's a problem: how do you securely share that key with your friend? If you send it over the internet, an attacker might intercept it. This chicken-and-egg problem stumped cryptographers for centuries.

### The Public Key Revolution

In 1976, Whitfield Diffie and Martin Hellman proposed something radical: what if you could have two different keys—one to lock (encrypt) and another to unlock (decrypt)? This idea seemed impossible, but they found a way using the mathematics of prime numbers.

#### Why RSA Works: The Power of Prime Numbers

RSA encryption, named after Rivest, Shamir, and Adleman, relies on a simple fact: multiplying two large prime numbers is easy, but factoring the result back into those primes is extraordinarily difficult.

```python
# Easy direction: multiplication
p = 104729  # prime number
q = 103591  # prime number
n = p * q   # = 10,848,583,639

# Hard direction: factoring
# Given only n = 10,848,583,639, find p and q
# With small numbers, this is doable. With a 2048-bit modulus?
# It would take all the computers on Earth far longer than the
# age of the universe.
#
# Note: 1024-bit RSA is now considered weak and is deprecated;
# 2048-bit is today's practical minimum, with 3072/4096-bit
# recommended for long-lived keys.
```

This asymmetry—easy one way, hard the other—is the foundation of modern internet security. When you see the padlock icon in your browser, it's using this principle to protect your connection.

### How HTTPS Protects Your Banking

Now we can understand how HTTPS keeps your data safe:

1. **Your browser asks the bank's website for its public key**
2. **The website sends its public key (anyone can see this)**
3. **Your browser generates a random session key for fast symmetric encryption**
4. **Your browser encrypts the session key with the bank's public key**
5. **Only the bank can decrypt it with their private key**
6. **Now you both have the same session key for fast, secure communication**

This elegant dance happens in milliseconds every time you visit a secure website.

## Beyond Basic Encryption: Modern Cryptographic Techniques

As our digital world evolves, so do the threats. Modern cryptography has developed sophisticated techniques to address challenges that early internet pioneers never imagined.

### Elliptic Curve Cryptography: Doing More with Less

RSA requires large keys (2048-4096 bits) to be secure. But what about devices with limited power, like your smartphone or smart home devices? Enter Elliptic Curve Cryptography (ECC), which provides the same security with much smaller keys.

The math behind ECC involves points on special curves. Instead of factoring, the security relies on the difficulty of the "discrete logarithm problem" on elliptic curves:

```python
# Simplified elliptic curve example
# Curve: y² = x³ + ax + b (mod p)

# Point addition on curves follows special rules
# If you know point P and scalar k, computing k*P is easy
# But given P and Q = k*P, finding k is extremely hard

# This is why Bitcoin uses elliptic curves for digital signatures
# Your private key is k, your public key is k*G (where G is a known point)
```

### The Quantum Threat: Why We Need New Cryptography

Here's a sobering thought: quantum computers, once they're powerful enough, will break RSA and ECC. Shor's algorithm can factor large numbers and solve discrete logarithms efficiently on a quantum computer. This isn't science fiction—it's why organizations are already preparing.

A classical computer factoring a 2048-bit RSA modulus by trial division would need billions of years; a sufficiently large quantum computer running Shor's algorithm could do it in hours or days. The defense is not a faster classical algorithm but a switch to problems quantum computers do *not* solve efficiently — the post-quantum families below (lattice-based, hash-based, code-based, and multivariate schemes).

**2024 Update**: IBM's quantum computers have reached 1000+ qubits, and while error rates remain high, the timeline for "cryptographically relevant quantum computers" has shortened. NIST released standardized post-quantum algorithms in 2024, and organizations are beginning the migration.

#### Post-Quantum Cryptography: Preparing for Tomorrow

Cryptographers are developing new algorithms based on problems that even quantum computers find difficult:

**Lattice-Based Cryptography**: Imagine a multi-dimensional grid of points. Finding the shortest path between points when there's some random "error" added is surprisingly hard, even for quantum computers.

```python
# Simplified Learning with Errors (LWE) concept
# Secret: s = [2, 3, 1]
# Public: Random matrix A and b = A*s + small_error
# Even knowing A and b, finding s is hard due to the error

A = [[4, 2, 7],
     [1, 5, 3],
     [6, 8, 2]]
s = [2, 3, 1]
error = [0, 1, -1]  # Small random errors

# b = A*s + error (mod q)
# Given A and b, recover s? Extremely difficult!
```

**Hash-Based Signatures**: These rely only on the security of hash functions. Even if quantum computers arrive tomorrow, hash-based signatures would still be secure.

**Code-Based Cryptography**: Security rests on the hardness of decoding general error-correcting codes.

**Multivariate Cryptography**: Security rests on the difficulty of solving systems of multivariate polynomial equations.

The transition to post-quantum cryptography is accelerating. Major browsers including Chrome and Firefox now support post-quantum key exchange by default. NIST's 2024 standards include:
- **CRYSTALS-Kyber**: For key encapsulation
- **CRYSTALS-Dilithium**: For digital signatures
- **FALCON**: Alternative signature scheme
- **SPHINCS+**: Hash-based signatures for highest security

### Privacy-Preserving Technologies

As we share more data online, a crucial question emerges: can we use data without exposing it? This isn't just about hiding from hackers—it's about fundamental privacy rights.

#### Zero-Knowledge Proofs: Proving Without Revealing

Imagine you want to prove you're over 21 to enter a bar, but you don't want to show your driver's license (which reveals your exact age, address, and more). Zero-knowledge proofs make this possible.

**Real-world example**: You could prove you know your password without sending the password itself:

```python
# Simplified zero-knowledge proof concept
# Prover knows secret x, wants to prove they know it
# without revealing x

# 1. Commitment: Prover sends y = g^x (mod p)
# 2. Challenge: Verifier sends random challenge c
# 3. Response: Prover computes r = x + c*k (mod q)
# 4. Verify: Verifier checks that g^r = y * public_key^c

# The verifier learns nothing about x!
```

This technology is already being used in blockchain systems for private transactions and in identity verification systems that respect privacy.

#### Homomorphic Encryption: Computing on Encrypted Data

What if you could perform calculations on encrypted data without decrypting it? This sounds impossible, but homomorphic encryption makes it real.

**Why this matters**: Imagine using a cloud service to analyze your medical data. With homomorphic encryption, the cloud can process your encrypted data and return encrypted results—without ever seeing your actual medical information.

```python
# Homomorphic property of a Paillier-style ADDITIVE scheme.
# In Paillier, *multiplying* two ciphertexts decrypts to the
# *sum* of the plaintexts:
#   E(5) * E(3) = E(5 + 3) = E(8)
#
# (Not every scheme works this way: RSA/ElGamal are
# multiplicatively homomorphic, and fully homomorphic schemes
# like BGV/CKKS support both addition and multiplication at a
# much higher cost.)

# Real application: Private voting
# Each vote is encrypted; the tally is computed by multiplying
# the ciphertexts (which adds the plaintext votes).
# Only the final sum is decrypted—individual votes remain secret
```

## Advanced Cryptographic Foundations

Now that we've seen how cryptography protects us in practice, let's dive deeper into the mathematical foundations that make it all possible. Understanding these concepts helps you make informed decisions about security.

### The Mathematics Behind RSA

We touched on RSA earlier, but let's see exactly how the math works:

```python
import random
from math import gcd

def generate_rsa_keys(bits=2048):
    # Step 1: Generate two large primes
    # (2048-bit is the practical minimum today; 1024-bit is deprecated)
    p = generate_large_prime(bits // 2)
    q = generate_large_prime(bits // 2)

    # Step 2: Calculate n = p * q
    n = p * q

    # Step 3: Calculate Euler's totient
    phi = (p - 1) * (q - 1)

    # Step 4: Choose public exponent e
    e = 65537  # Common choice, must be coprime with phi

    # Step 5: Calculate private exponent d
    d = modular_inverse(e, phi)

    # Public key: (n, e)
    # Private key: (n, d)
    return (n, e), (n, d)

def encrypt_rsa(message, n, e):
    # Encryption: c = m^e mod n
    return pow(message, e, n)

def decrypt_rsa(ciphertext, n, d):
    # Decryption: m = c^d mod n
    return pow(ciphertext, d, n)

# The security relies on the fact that knowing n
# doesn't help you find p and q (factoring is hard)
```

### Elliptic Curves: The Elegant Alternative

Elliptic curves provide the same security as RSA with much smaller keys. The math is beautiful:

```python
# Elliptic curve: y² = x³ + ax + b (mod p)
# Example: Bitcoin uses secp256k1: y² = x³ + 7

class EllipticCurve:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p  # Prime modulus

    def point_addition(self, P, Q):
        """Add two points on the curve"""
        if P == Q:
            # Point doubling
            s = (3 * P[0]**2 + self.a) * modular_inverse(2 * P[1], self.p)
        else:
            # Point addition
            s = (Q[1] - P[1]) * modular_inverse(Q[0] - P[0], self.p)

        x3 = (s**2 - P[0] - Q[0]) % self.p
        y3 = (s * (P[0] - x3) - P[1]) % self.p
        return (x3, y3)

    def scalar_multiplication(self, k, P):
        """Multiply point P by scalar k"""
        # This is easy to compute
        # But given P and k*P, finding k is extremely hard
        # This is the elliptic curve discrete logarithm problem
```

### Secret Sharing: Distributing Trust

What if you need multiple people to authorize something, like launching a missile or accessing a bitcoin wallet? Shamir's Secret Sharing provides an elegant solution:

```python
def shamir_share_secret(secret, threshold, num_shares, prime):
    """
    Split secret into n shares, need k to reconstruct
    Uses polynomial: f(x) = secret + a1*x + a2*x² + ... + ak*x^(k-1)
    """
    # Generate random coefficients
    coefficients = [secret]
    for i in range(threshold - 1):
        coefficients.append(random.randint(0, prime - 1))

    # Generate shares: (x, f(x)) for x = 1, 2, ..., n
    shares = []
    for x in range(1, num_shares + 1):
        y = sum(coeff * pow(x, i, prime) for i, coeff in enumerate(coefficients)) % prime
        shares.append((x, y))

    return shares

def reconstruct_secret(shares, prime):
    """
    Reconstruct secret using Lagrange interpolation
    """
    secret = 0
    for i, (xi, yi) in enumerate(shares):
        numerator = 1
        denominator = 1
        for j, (xj, _) in enumerate(shares):
            if i != j:
                numerator = (numerator * -xj) % prime
                denominator = (denominator * (xi - xj)) % prime

        lagrange = (yi * numerator * modular_inverse(denominator, prime)) % prime
        secret = (secret + lagrange) % prime

    return secret

# Example: Nuclear launch codes requiring 3 of 5 generals
# Each general gets one share, any 3 can launch
```

---

<div class="page-nav">
  <span class="page-nav-prev"><a href="./">← Cybersecurity Hub</a></span>
  <span class="page-nav-next"><a href="application-and-cloud-security.html">Web, Cloud &amp; Container Security →</a></span>
</div>

## See Also

- [Web, Cloud & Container Security](application-and-cloud-security.html) — where encryption meets real applications
- [Operations, Response & Compliance](operations-and-response.html) — formal security proofs and secure multi-party computation
- [Quantum Computing](../quantumcomputing.html) — the hardware behind the post-quantum threat
