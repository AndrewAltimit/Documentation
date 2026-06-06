---
layout: docs
title: "Cybersecurity: Cryptography"
permalink: /docs/technology/cybersecurity/cryptography.html
toc: true
toc_sticky: true
hide_title: true
---

<p class="breadcrumb"><a href="./">Cybersecurity</a> › Cryptography</p>

# Cryptography

Cryptography is the foundation every other security control rests on. This page builds from symmetric and public-key encryption through modern techniques — elliptic curves, post-quantum algorithms, zero-knowledge proofs, and homomorphic encryption — and finishes with the mathematical foundations of RSA, elliptic curves, and secret sharing.

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

## TLS in Practice

The handshake sketched above is the *idea* behind HTTPS; **TLS (Transport Layer Security)** is the protocol that actually implements it on the wire. TLS is where the symmetric and public-key primitives from the previous sections come together into a single negotiated, authenticated, forward-secret channel. This section covers the modern handshake, how a cipher suite is chosen, how the server's identity is vouched for by certificates, and the operational machinery — issuance, renewal, and revocation — that keeps those certificates trustworthy.

### The TLS 1.3 Handshake

TLS 1.3 (RFC 8446, 2018) is a ground-up redesign of the protocol. The headline change is latency: where TLS 1.2 needed **two round trips** (2-RTT) before any application data could flow, TLS 1.3 completes the handshake in **one round trip** (1-RTT), and can resume an earlier session in **zero round trips** (0-RTT).

The key insight is that the client stops waiting to be told which key-exchange group to use. Instead it *guesses*: in its very first message it sends not just the list of groups it supports but an actual ephemeral Diffie–Hellman public key (a "key share") for its preferred group. If the server is happy with that group, it replies with its own key share and both sides can derive the shared secret immediately.

```
Client                                              Server

ClientHello
  + supported_versions (1.3)
  + key_share (X25519 ephemeral pubkey)   -------->
  + signature_algorithms
  + cipher suites
                                                ServerHello
                                          + key_share (server ephemeral pubkey)
                                          {EncryptedExtensions}
                                          {Certificate}
                                          {CertificateVerify}   (signs the transcript)
                                          {Finished}
                                  <--------  [Application Data possible]
{Finished}
[Application Data]              <------->  [Application Data]

   { } = encrypted with handshake keys
   [ ] = encrypted with application keys
```

After the single `ClientHello` / `ServerHello` exchange, *everything else is encrypted*. The server's certificate, which travelled in plaintext under TLS 1.2, is now protected under the handshake keys — a meaningful privacy improvement, since a passive observer can no longer trivially see which site you are connecting to from the certificate alone (the SNI hostname in the ClientHello is the remaining leak, addressed separately by Encrypted Client Hello).

**Improvements over TLS 1.2.** TLS 1.3 is as much about *removing* footguns as adding speed:

| Area | TLS 1.2 | TLS 1.3 |
|------|---------|---------|
| Handshake latency | 2-RTT | 1-RTT (0-RTT on resumption) |
| Forward secrecy | Optional (static-RSA key exchange allowed) | Mandatory (ephemeral (EC)DHE only) |
| Key exchange | RSA, DH, ECDH (static or ephemeral) | (EC)DHE only |
| Bulk ciphers | CBC, RC4, 3DES, AEAD | AEAD only (AES-GCM, ChaCha20-Poly1305, AES-CCM) |
| Renegotiation | Yes (source of attacks) | Removed (replaced by key update + post-handshake auth) |
| Compression | Allowed (enabled CRIME) | Removed |
| Cert / handshake privacy | Sent in cleartext | Encrypted |

The static-RSA key exchange is the most important casualty. Under TLS 1.2 a server could decrypt the premaster secret with its long-term private key, which meant anyone who later stole that key could decrypt *recorded* past traffic. TLS 1.3 mandates ephemeral (EC)DHE, so each session's keys are derived from one-time values and **forward secrecy** is guaranteed: compromising the server's private key tomorrow does not retroactively decrypt today's captured sessions. Likewise, the protocol drops every cipher with a track record of attacks (RC4, 3DES, CBC-mode padding oracles, TLS-level compression) and permits only AEAD ciphers, which authenticate and encrypt in one pass.

**0-RTT and its caveat.** On a resumed connection the client can send application data in its very first flight, encrypted with a key derived from a pre-shared key (PSK) established earlier. This is excellent for latency but **0-RTT data is replayable** — a network attacker can capture and re-send that first flight, and the server has no handshake context yet to reject the duplicate. For that reason 0-RTT must only carry idempotent requests (e.g. an HTTP `GET`), never a state-changing `POST`.

### Cipher-Suite Selection

A *cipher suite* names the combination of algorithms a connection will use. The naming convention itself changed in TLS 1.3, reflecting how much was simplified:

```
TLS 1.2:  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
          └┬─┘ └──┬──┘ └┬┘      └─────┬──────┘ └──┬──┘
        protocol  |   auth        bulk AEAD    HKDF/PRF
              key exchange         cipher        hash

TLS 1.3:  TLS_AES_128_GCM_SHA256
              └─────┬──────┘ └──┬──┘
                AEAD cipher   hash
```

In TLS 1.2 the suite bundled the **key exchange** (ECDHE), the **authentication** (RSA or ECDSA signature), the **bulk cipher** (AES-128-GCM), and the **hash** (SHA-256) into one identifier. TLS 1.3 splits these concerns: the suite now only names the AEAD cipher and hash, while the key-exchange group and signature algorithm are negotiated independently in their own extensions. This orthogonality is why TLS 1.3 has just **five** cipher suites instead of the hundreds that accumulated under TLS 1.2:

- `TLS_AES_128_GCM_SHA256`
- `TLS_AES_256_GCM_SHA384`
- `TLS_CHACHA20_POLY1305_SHA256`
- `TLS_AES_128_CCM_SHA256`
- `TLS_AES_128_CCM_8_SHA256`

**How the choice is made.** The client offers an ordered list; the server picks. By default the server honours its *own* preference order over the client's, so server configuration is what actually decides the outcome. The practical guidance:

- **Prefer AES-GCM where hardware acceleration exists.** Virtually all modern x86 and ARM CPUs have AES-NI instructions, making AES-GCM both fast and constant-time.
- **Prefer ChaCha20-Poly1305 on devices without AES hardware.** A pure-software AES implementation is slow *and* prone to cache-timing side channels; ChaCha20 is fast and constant-time in software, which is why mobile clients often list it first.
- **Choose the key-exchange group for both security and speed.** X25519 is the modern default — it is fast, has no known weak parameters, and avoids the invalid-curve and small-subgroup pitfalls of the NIST P-curves.

```bash
# Inspect what a server actually negotiated
openssl s_client -connect example.com:443 -tls1_3 </dev/null 2>/dev/null \
  | grep -E "Protocol|Cipher|Server Temp Key"
# Protocol  : TLSv1.3
# Cipher    : TLS_AES_256_GCM_SHA384
# Server Temp Key: X25519, 253 bits

# Enumerate every suite a server will accept (great for audits)
nmap --script ssl-enum-ciphers -p 443 example.com
```

### The Certificate Lifecycle

Encryption alone proves nothing about *who* you are talking to. A TLS certificate binds a public key to an identity (a domain name) and is signed by a **Certificate Authority (CA)** that your operating system or browser already trusts. The handshake's `CertificateVerify` message proves the server holds the matching private key, and the chain of signatures up to a trusted root proves the binding is legitimate.

**The chain of trust.** A leaf (server) certificate is signed by an intermediate CA, which is signed by a root CA whose public key ships preinstalled in the **trust store** of your OS or browser. Roots are kept offline; intermediates do the day-to-day signing so that a compromised intermediate can be revoked without distrusting the root.

```
Root CA  (self-signed, in OS/browser trust store, kept offline)
   │  signs
   ▼
Intermediate CA  (online, does day-to-day issuance)
   │  signs
   ▼
Leaf certificate  (your server: CN/SAN = example.com, your public key)
```

The server must send the leaf *and* the intermediate(s) during the handshake; a common misconfiguration is sending only the leaf, which breaks validation for clients that lack the intermediate cached.

**Validation levels** differ only in how much the CA checks before issuing — the cryptographic protection on the wire is identical:

- **Domain Validation (DV)** — the CA confirms you control the domain (e.g. by serving a token). Issued in seconds, free. The overwhelming majority of certificates.
- **Organization Validation (OV)** — the CA additionally verifies the organization exists.
- **Extended Validation (EV)** — the most rigorous vetting. Browsers no longer give EV a distinct UI treatment, so its practical value has diminished.

**ACME and Let's Encrypt.** The certificate lifecycle used to be a manual, annual chore: generate a CSR, paste it into a web form, pay, download, install, and remember to do it again next year. **Let's Encrypt** (launched 2015) and the **ACME protocol** (RFC 8555) automated the whole thing and made DV certificates free. The flow:

1. The ACME client generates a key pair and asks the CA for a certificate for `example.com`.
2. The CA issues a **challenge** to prove control of the domain — typically `http-01` (serve a token at `http://example.com/.well-known/acme-challenge/...`) or `dns-01` (publish a TXT record). `dns-01` is required for **wildcard** certificates and works behind firewalls.
3. The client satisfies the challenge; the CA verifies it and signs the certificate.
4. The client installs the certificate and schedules automatic renewal.

```bash
# Issue and auto-renew with certbot (the canonical ACME client)
certbot --nginx -d example.com -d www.example.com

# certbot installs a systemd timer / cron job that renews when a cert
# is within 30 days of expiry — renewal is a no-op if not yet due
certbot renew --dry-run
```

Because Let's Encrypt certificates are valid for only **90 days**, automation is not optional — and that short lifetime is itself a security feature: it limits the damage window of an undetected key compromise and makes revocation (which is unreliable, as we'll see) matter less. The industry is moving toward even shorter lifetimes; the CA/Browser Forum has voted to reduce the maximum certificate lifetime to **47 days** by 2029, which only deepens the dependence on ACME automation.

### Revocation: OCSP and CRLs

Certificates carry an expiry date, but sometimes a certificate must be killed *early* — a private key leaks, a domain changes hands, or a CA mis-issues. Revocation is the mechanism, and it has historically been the weakest link in the PKI.

- **CRL (Certificate Revocation List)** — the CA publishes a signed list of every revoked serial number. Clients download it and check membership. CRLs grow large and are fetched infrequently, so they can be badly stale.
- **OCSP (Online Certificate Status Protocol)** — the client asks the CA's responder about *one* specific certificate in real time. Fresher than a CRL, but it adds a network round trip to page load and, worse, **leaks the browsing history** to the CA (which sites you visit). Browsers responded by **soft-failing**: if the OCSP responder is unreachable, they assume the certificate is good — which lets a network attacker who can block the OCSP query simply suppress revocation entirely.
- **OCSP stapling** — the *server* periodically fetches a signed, time-stamped OCSP response from the CA and "staples" it into the TLS handshake. The client gets fresh revocation status with no extra round trip and no privacy leak. The `must-staple` certificate extension can be set to make a missing staple a hard failure, closing the soft-fail hole.

```nginx
# Enable OCSP stapling in nginx
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/ssl/chain.pem;
resolver 1.1.1.1 valid=300s;
```

Because online revocation is slow and leaky, browsers increasingly favour **pushed revocation**: Chrome's **CRLSets** and Firefox's **CRLite** aggregate revocations at the vendor and ship them to the browser out of band, sidestepping per-connection OCSP entirely. This, combined with ever-shorter certificate lifetimes, is the industry's pragmatic answer to revocation's chronic unreliability.

### DANE and TLSA

Who decides which CAs to trust? Today, your browser does — and it trusts *hundreds* of them, any one of which can issue a certificate for *any* domain. A single compromised or coerced CA (as in the 2011 DigiNotar breach) can mint a valid certificate for your bank. **DANE (DNS-Based Authentication of Named Entities, RFC 6698)** offers a different trust anchor: let the *domain owner* pin which certificate or CA is legitimate, published in DNS and protected by **DNSSEC**.

The pin lives in a **TLSA** record:

```
_443._tcp.example.com.  IN  TLSA  3 1 1  <SHA-256 of the server's public key>
                                  │ │ │
                                  │ │ └─ Matching type: 1 = SHA-256 of the selector
                                  │ └─── Selector: 1 = SubjectPublicKeyInfo (0 = full cert)
                                  └───── Usage: 3 = "this exact cert/key", bypassing the CA
                                          (2 = pin a private trust anchor; 0/1 = constrain a CA)
```

Usage `3` ("DANE-EE") is the most common: it says "ignore the public CA system, the only valid key for this service is *this* one." Because the record is signed with DNSSEC, an attacker cannot forge it without also breaking DNSSEC. The catch is deployment: **browsers do not support DANE for HTTPS** (the perceived gain over CA pinning and Certificate Transparency did not justify the DNSSEC dependency). Its real success story is **SMTP**: DANE is widely deployed for opportunistic-encryption email (MTA-to-MTA), where it upgrades the otherwise trivially-downgradeable STARTTLS into authenticated, mandatory encryption.

### Mutual TLS (mTLS)

In ordinary TLS only the *server* presents a certificate; the client stays anonymous to the cryptographic layer and authenticates later at the application level (a password, a token). **Mutual TLS** makes authentication symmetric: the server requests a certificate from the client too, and the client proves possession of its private key with its own `CertificateVerify`. Both ends are now cryptographically authenticated before a single byte of application data flows.

This is rarely used for public websites (you cannot hand a certificate to every visitor) but is the backbone of **machine-to-machine** trust:

- **Zero-trust networks and service meshes** — Istio, Linkerd, and Consul issue short-lived certificates to every workload and enforce mTLS on all internal traffic, so a service's *identity* is its certificate rather than its network location.
- **API and partner integrations** — a bank requiring a client certificate to call its payment API.
- **IoT device fleets** — each device provisioned with a unique certificate at manufacture.

```
Client                                              Server

ClientHello                              -------->
                                            ServerHello
                                          {Certificate}            (server's cert)
                                          {CertificateRequest}     <- asks client to authenticate
                                          {CertificateVerify}
                                  <--------  {Finished}
{Certificate}            (client's cert)
{CertificateVerify}      (client proves its key)
{Finished}                               -------->
[Application Data]               <------->  [Application Data]
```

The operational cost of mTLS is **certificate lifecycle at scale**: every client now needs a certificate, and every certificate needs issuing, rotating, and revoking. This is precisely why service meshes pair mTLS with an automated internal CA (e.g. SPIFFE/SPIRE) that mints certificates with lifetimes measured in hours — the same short-lifetime-plus-automation pattern that ACME brought to the public web, applied inside the datacenter.

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
