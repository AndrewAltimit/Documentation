# Cybersecurity

<html><header><link rel="stylesheet" href="https://andrewaltimit.github.io/Documentation/style.css"></header></html>

Cybersecurity is the practice of protecting systems, networks, and data from digital attacks, unauthorized access, and damage. It encompasses a wide range of technologies, processes, and practices designed to safeguard digital assets.

## Advanced Cryptographic Foundations

### Mathematical Foundations

**Core cryptographic primitives and their mathematical basis:**

#### Number Theory for Cryptography
- **Miller-Rabin Test**: Probabilistic primality testing with configurable accuracy
- **Prime Generation**: Secure random prime generation for RSA and DH
- **Modular Arithmetic**: Foundation for public-key cryptography
- **Discrete Logarithm Problem**: Security basis for many protocols

#### Elliptic Curve Cryptography Theory
- **Curve Equation**: y² = x³ + ax + b (mod p)
- **Point Addition**: Group operation on elliptic curves
- **Scalar Multiplication**: Efficient double-and-add algorithm
- **secp256k1**: Bitcoin's chosen curve parameters

### Post-Quantum Cryptography

**Cryptographic systems resistant to quantum attacks:**

#### Lattice-Based Cryptography
- **Learning with Errors (LWE)**: Hard problem based on lattice theory
- **Error Distribution**: Gaussian noise for security
- **Key Encapsulation**: Quantum-resistant key exchange
- **NIST Candidates**: Kyber, Dilithium, Falcon

#### Hash-Based Signatures
- **Merkle Trees**: Quantum-resistant digital signatures
- **One-Time Signatures**: Foundation for tree-based schemes
- **Authentication Paths**: Efficient proof generation
- **Stateful Security**: Managing signature indices

### Zero-Knowledge Proofs

**Proving knowledge without revealing information:**

#### Schnorr Protocol
- **Non-Interactive**: Fiat-Shamir transformation
- **Discrete Log Proof**: Prove knowledge of x where y = g^x
- **Commitment-Challenge-Response**: Three-phase protocol
- **Applications**: Authentication, digital signatures

### Homomorphic Encryption

**Computing on encrypted data:**

#### Paillier Cryptosystem
- **Additive Homomorphism**: E(a) × E(b) = E(a + b)
- **Semantic Security**: IND-CPA secure
- **Applications**: Electronic voting, secure auctions
- **Threshold Variants**: Distributed decryption

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/cybersecurity/cryptographic_foundations.py">cryptographic_foundations.py</a>
</div>

```python
# Example usage:
from cryptographic_foundations import EllipticCurve, LearningWithErrors, SchnorrProtocol

# Elliptic curve operations
curve = EllipticCurve(a=0, b=7, p=large_prime)
point = (x, y)
doubled = curve.point_addition(point, point)
multiplied = curve.scalar_multiplication(k, point)

# Post-quantum encryption
lwe = LearningWithErrors(n=512, q=12289, sigma=3.2)
public_key, private_key = lwe.generate_keypair()
ciphertext = lwe.encrypt(public_key, message_bit=1)

# Zero-knowledge proof
schnorr = SchnorrProtocol(p, q, g)
proof, challenge = schnorr.prove_knowledge(secret, "message")
valid = schnorr.verify_proof(proof, challenge, "message")

## Advanced Formal Security Models

### Computational Security Models

#### Provable Security Framework

**Security reductions and game-based proofs:**

- **Discrete Log to CDH Reduction**: Demonstrates how solving discrete logarithm implies solving Computational Diffie-Hellman
- **IND-CPA Security Game**: Formal game for proving semantic security of encryption schemes
- **IND-CCA Security Game**: Stronger game allowing chosen ciphertext queries
- **Advantage Computation**: Measuring adversary success probability vs random guessing

```python
# Example usage:
from formal_security_models import SecurityReduction

# Run IND-CPA game
advantage = SecurityReduction.ind_cpa_game(
    encryption_scheme=elgamal,
    adversary=cpa_adversary,
    security_parameter=128
)

# Security proof: advantage should be negligible
assert advantage < 1/2**128
```

#### Universal Composability Framework

**UC security model for protocol composition:**

- **Ideal Functionalities**: Abstract specification of protocol goals
- **Real/Ideal Paradigm**: Protocol secure if indistinguishable from ideal functionality
- **Composition Theorem**: UC-secure protocols remain secure under arbitrary composition
- **Common Functionalities**: Commitment, secure computation, oblivious transfer

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/cybersecurity/formal_security_models.py">formal_security_models.py</a>
</div>

```python
# Example: UC-secure commitment
from formal_security_models import UCFramework

commitment = UCFramework.CommitmentFunctionality()
commitment.register_party("Alice")
commitment.register_party("Bob")

# Alice commits to value
com_id = commitment.commit("Alice", secret_value)
# Later: Alice opens commitment
value = commitment.open("Alice", com_id)
```

### Information-Theoretic Security

#### Secret Sharing Schemes

**Unconditionally secure secret distribution:**

- **Shamir's (t,n) Threshold Scheme**: Based on polynomial interpolation
- **Additive Secret Sharing**: Simple XOR-based sharing
- **Replicated Secret Sharing**: Efficient for small number of parties
- **Verifiable Secret Sharing (VSS)**: Prevents dealer from cheating

**Key properties:**
- **Perfect Security**: No information from t-1 shares
- **Optimal Share Size**: Each share same size as secret
- **Lagrange Interpolation**: Efficient reconstruction algorithm
- **Applications**: Distributed key management, secure multi-party computation

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/cybersecurity/formal_security_models.py">formal_security_models.py</a>
</div>

```python
# Example: Split encryption key among 5 parties
from formal_security_models import ShamirSecretSharing

# Need 3 of 5 shares to reconstruct
sss = ShamirSecretSharing(threshold=3, num_shares=5, prime=large_prime)
shares = sss.share_secret(encryption_key)

# Later: reconstruct with any 3 shares
key = sss.reconstruct_secret(shares[0:3])
```

## Core Security Principles

### CIA Triad
The fundamental principles of information security:

- **Confidentiality**: Ensuring information is accessible only to authorized individuals
- **Integrity**: Maintaining the accuracy and completeness of data
- **Availability**: Ensuring authorized users have reliable access to resources

### Defense in Depth
Layered security approach with multiple defensive mechanisms:
- Perimeter defenses
- Network segmentation
- Host-based protections
- Application security
- Data encryption
- User training

### Principle of Least Privilege
Users and processes should have only the minimum access rights necessary to perform their functions.

## Cryptography

### Symmetric Encryption
Same key for encryption and decryption.

**AES (Advanced Encryption Standard)**:
- Key sizes: 128, 192, or 256 bits
- Block size: 128 bits
- Modes: ECB, CBC, CTR, GCM

Example with OpenSSL:
```bash
# Encrypt
openssl enc -aes-256-cbc -salt -in file.txt -out file.enc -k password

# Decrypt
openssl dec -aes-256-cbc -in file.enc -out file.txt -k password
```

### Asymmetric Encryption
Different keys for encryption (public) and decryption (private).

**RSA**:
- Based on factoring large prime numbers
- Key sizes: typically 2048 or 4096 bits

**Elliptic Curve Cryptography (ECC)**:
- Smaller key sizes for equivalent security
- Common curves: P-256, P-384, P-521

### Hashing
One-way functions producing fixed-size output.

- **SHA-256**: 256-bit output, widely used
- **SHA-3**: Latest standard, different algorithm
- **bcrypt/scrypt/Argon2**: For password hashing

### Digital Signatures
Provide authentication and non-repudiation.

Process:
1. Hash the message
2. Encrypt hash with private key
3. Recipient decrypts with public key and verifies

## Network Security

### Firewalls
Control network traffic based on security rules.

**Types**:
- Packet filtering
- Stateful inspection
- Application layer (Layer 7)
- Next-generation (NGFW)

**iptables example**:
```bash
# Allow SSH only from specific IP
iptables -A INPUT -p tcp --dport 22 -s 192.168.1.100 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j DROP

# Block outgoing connections to port 25
iptables -A OUTPUT -p tcp --dport 25 -j DROP
```

### VPN (Virtual Private Network)
Creates encrypted tunnel for secure communication.

**Types**:
- IPSec: Network layer encryption
- SSL/TLS VPN: Application layer
- WireGuard: Modern, efficient protocol

### IDS/IPS
**Intrusion Detection System (IDS)**: Monitors and alerts
**Intrusion Prevention System (IPS)**: Monitors and blocks

Common tools:
- Snort
- Suricata
- OSSEC

## Web Application Security

### OWASP Top 10
Most critical web application security risks:

1. **Injection**: SQL, NoSQL, Command injection
2. **Broken Authentication**: Session management flaws
3. **Sensitive Data Exposure**: Inadequate encryption
4. **XML External Entities (XXE)**: XML processor attacks
5. **Broken Access Control**: Authorization failures
6. **Security Misconfiguration**: Default settings, verbose errors
7. **Cross-Site Scripting (XSS)**: Client-side code injection
8. **Insecure Deserialization**: Object manipulation
9. **Using Components with Known Vulnerabilities**: Outdated libraries
10. **Insufficient Logging & Monitoring**: Delayed breach detection

### SQL Injection Prevention
**Vulnerable code**:
```python
query = f"SELECT * FROM users WHERE name = '{username}'"
```

**Secure code**:
```python
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (username,))
```

### Cross-Site Scripting (XSS) Prevention
- Input validation
- Output encoding
- Content Security Policy (CSP)
- HTTP-only cookies

### CSRF Protection
- Anti-CSRF tokens
- SameSite cookie attribute
- Verify referrer header

## Authentication and Authorization

### Multi-Factor Authentication (MFA)
Combines multiple verification methods:
- Something you know (password)
- Something you have (token, phone)
- Something you are (biometrics)

### OAuth 2.0
Authorization framework for third-party access.

**Grant types**:
- Authorization Code
- Implicit (deprecated)
- Client Credentials
- Resource Owner Password

### JWT (JSON Web Tokens)
Stateless authentication tokens.

Structure: `header.payload.signature`

Example:
```javascript
const jwt = require('jsonwebtoken');
const token = jwt.sign(
  { userId: 123, role: 'user' },
  process.env.JWT_SECRET,
  { expiresIn: '1h' }
);
```

## Cloud Security

### Shared Responsibility Model
- **Cloud Provider**: Physical security, infrastructure
- **Customer**: Data, applications, access management

### AWS Security Best Practices
- Enable MFA on root account
- Use IAM roles instead of access keys
- Enable CloudTrail logging
- Implement least privilege policies
- Encrypt data at rest and in transit

Example IAM policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject"],
    "Resource": "arn:aws:s3:::my-bucket/*",
    "Condition": {
      "IpAddress": {
        "aws:SourceIp": "192.168.1.0/24"
      }
    }
  }]
}
```

### Container Security
- Scan images for vulnerabilities
- Run containers as non-root
- Use minimal base images
- Implement network policies
- Enable security policies (AppArmor, SELinux)

## Incident Response

### Incident Response Phases
1. **Preparation**: Policies, tools, training
2. **Identification**: Detect and analyze incidents
3. **Containment**: Limit damage and prevent spread
4. **Eradication**: Remove threat from environment
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Post-incident analysis

### Digital Forensics
**Key principles**:
- Maintain chain of custody
- Preserve evidence integrity
- Document all actions

**Common tools**:
- Volatility (memory analysis)
- Autopsy/Sleuth Kit (disk forensics)
- Wireshark (network analysis)

## Security Operations

### SIEM (Security Information and Event Management)
Aggregates and analyzes security events.

Popular solutions:
- Splunk
- Elastic Security
- IBM QRadar

Example Splunk query:
```
index=security sourcetype=firewall action=blocked
| stats count by src_ip
| sort -count
| head 10
```

### Vulnerability Management
**Process**:
1. Asset inventory
2. Vulnerability scanning
3. Risk assessment
4. Remediation
5. Verification

**Common scanners**:
- Nessus
- OpenVAS
- Qualys

### Penetration Testing
Authorized simulated attacks to identify vulnerabilities.

**Methodology**:
1. Reconnaissance
2. Scanning
3. Enumeration
4. Exploitation
5. Post-exploitation
6. Reporting

**Tools**:
- Metasploit
- Burp Suite
- Nmap
- John the Ripper

## Compliance and Governance

### Common Frameworks
- **ISO 27001**: Information security management
- **NIST Cybersecurity Framework**: Risk-based approach
- **CIS Controls**: Prioritized security actions
- **PCI DSS**: Payment card security
- **GDPR**: EU data protection
- **HIPAA**: Healthcare data security

### Security Policies
Essential policies:
- Information Security Policy
- Acceptable Use Policy
- Incident Response Plan
- Business Continuity Plan
- Data Classification Policy

## Emerging Threats

### Zero-Day Exploits
Vulnerabilities unknown to vendors.

Mitigation:
- Behavior-based detection
- Application sandboxing
- Regular patching
- Threat intelligence

### Ransomware
Encrypts data and demands payment.

Prevention:
- Regular backups (3-2-1 rule)
- Email filtering
- User training
- Network segmentation
- Endpoint detection and response (EDR)

### Supply Chain Attacks
Target third-party vendors to reach ultimate victims.

Protection:
- Vendor risk assessment
- Software composition analysis
- Code signing verification
- Dependency scanning

### AI/ML in Security
**Defensive uses**:
- Anomaly detection
- Malware classification
- Automated threat hunting

**Offensive concerns**:
- Deepfakes
- Automated attacks
- Adversarial ML

## Advanced Attack Techniques and Defenses

### Side-Channel Attacks

#### Timing Attack Implementation

**Exploiting timing variations in code execution:**

- **Vulnerable Operations**: String comparison, password checking, cryptographic operations
- **Attack Method**: Measure execution time differences to infer secret data
- **Constant-Time Defense**: Implement operations that take same time regardless of input
- **Statistical Analysis**: Use multiple measurements to reduce noise

#### Power Analysis

**Differential Power Analysis (DPA) and Simple Power Analysis (SPA):**

- **Power Consumption Models**: Hamming weight, Hamming distance
- **Correlation Analysis**: Statistical correlation between power and hypothetical values
- **Key Extraction**: Recover cryptographic keys from power traces
- **Countermeasures**: Masking, shuffling, power line filtering

#### Electromagnetic Analysis

**EM emanation exploitation:**

- **Near-Field vs Far-Field**: Different measurement techniques
- **Signal Processing**: FFT analysis, pattern matching
- **Key Recovery**: Extract keys from EM radiation patterns
- **Shielding**: Faraday cages, EM-absorbing materials

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/cybersecurity/advanced_attacks.py">advanced_attacks.py</a>
</div>

```python
# Example: Timing attack on password check
from advanced_attacks import TimingAttack

attack = TimingAttack()
secret = attack.extract_secret(
    target_length=16,
    charset=string.ascii_letters + string.digits,
    oracle=password_checker
)
```

### Advanced Exploitation Techniques

#### Return-Oriented Programming (ROP)

**Code reuse attacks bypassing DEP:**

- **Gadget Discovery**: Finding useful instruction sequences ending in ret
- **Chain Building**: Constructing exploit payload from gadgets
- **Stack Pivoting**: Redirecting stack pointer to controlled data
- **Common Gadgets**: pop reg; ret, xchg reg; ret, syscall; ret
- **Defenses**: CFI, CET, stack canaries, ASLR

#### Heap Exploitation

**Modern heap exploitation techniques:**

- **House of Spirit**: Free fake chunks to control allocation
- **Tcache Poisoning**: Corrupt tcache to get arbitrary allocation
- **House of Force**: Overflow wilderness chunk size
- **Use-After-Free**: Exploit dangling pointers
- **Heap Feng Shui**: Precise heap layout manipulation

#### Format String Vulnerabilities

**Printf exploitation techniques:**

- **Memory Disclosure**: Read arbitrary memory with %s
- **Arbitrary Write**: Write memory using %n
- **GOT Overwrite**: Redirect function calls
- **Direct Parameter Access**: Use %n$x syntax
- **FORTIFY_SOURCE**: Compile-time protection

### Machine Learning Security

#### Adversarial Examples

**Attacks on ML models:**

- **FGSM**: Fast Gradient Sign Method - single-step attack
- **PGD**: Projected Gradient Descent - iterative attack
- **C&W**: Carlini & Wagner - optimization-based attack
- **DeepFool**: Minimal perturbation for misclassification
- **Universal Perturbations**: Single perturbation fools multiple inputs

#### Model Poisoning and Backdoors

**Training-time attacks:**

- **Data Poisoning**: Corrupt training data to degrade performance
- **Backdoor Triggers**: Hidden patterns that cause misclassification
- **Neuron Trojans**: Target specific neurons for activation
- **Clean-Label Attacks**: Poisoned samples with correct labels
- **Defenses**: Data sanitization, robust training, trigger detection

#### Privacy Attacks

**Information leakage from models:**

- **Membership Inference**: Determine if sample was in training set
- **Model Inversion**: Reconstruct training data from model
- **Property Inference**: Extract dataset properties
- **Model Extraction**: Steal model functionality through queries

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/cybersecurity/ml_security.py">ml_security.py</a>
</div>

```python
# Example: Generate adversarial example
from ml_security import AdversarialML

adv_ml = AdversarialML()
adv_image = adv_ml.pgd_attack(
    model=target_model,
    image=original_image,
    label=true_label,
    epsilon=0.03,
    num_iter=40
)
```

### Advanced Forensics

#### Memory Forensics

**Advanced memory analysis techniques:**

- **Process Enumeration**: Find EPROCESS/task_struct in memory
- **Encryption Key Extraction**: Locate keys using entropy analysis
- **Network Artifacts**: Extract IPs, URLs, domains from memory
- **Registry Analysis**: Parse registry hives from memory dumps
- **Code Injection Detection**: Find injected code and hooks
- **Timeline Reconstruction**: Build event timeline from artifacts

#### Artifact Analysis

**Digital evidence extraction:**

- **String Extraction**: ASCII and Unicode string analysis
- **Handle Tables**: Enumerate open handles and resources
- **MFT Parsing**: File system activity reconstruction
- **Volatility Plugins**: Automated memory forensics
- **Anti-Forensics Detection**: Identify evasion techniques

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/cybersecurity/memory_forensics.py">memory_forensics.py</a>
</div>

```python
# Example: Extract processes from memory dump
from memory_forensics import MemoryForensics

forensics = MemoryForensics(memory_dump)
processes = forensics.find_processes()
keys = forensics.extract_encryption_keys()
artifacts = forensics.find_network_artifacts()
```

## Security Best Practices

### Secure Development
- Security by design
- Threat modeling
- Code reviews
- Static and dynamic analysis
- Dependency management

### Password Security
- Minimum 12 characters
- Multi-factor authentication
- Password managers
- No password reuse
- Regular rotation for privileged accounts

### Network Segmentation
- DMZ for public-facing services
- VLAN separation
- Zero trust architecture
- Micro-segmentation

### Monitoring and Logging
- Centralized log management
- Real-time alerting
- Anomaly detection
- Regular log review
- Long-term retention

## Research Frontiers

### Quantum Cryptanalysis

**Quantum algorithms threatening classical cryptography:**

- **Grover's Algorithm**: O(√N) search - breaks symmetric crypto
- **Shor's Algorithm**: Polynomial-time factoring - breaks RSA/ECC
- **Quantum Period Finding**: Core of many quantum attacks
- **HHL Algorithm**: Solve linear systems exponentially faster
- **Quantum Walk**: Graph search with quadratic speedup

### Secure Multi-Party Computation

**Computing on private data:**

- **Garbled Circuits**: Yao's protocol for 2-party computation
- **BGW Protocol**: Information-theoretic MPC for honest majority
- **GMW Protocol**: Computational MPC using oblivious transfer
- **SPDZ**: Practical MPC with preprocessing
- **Applications**: Private set intersection, secure auctions, private ML

<div class="code-reference">
<i class="fas fa-code"></i> Full implementation: <a href="https://github.com/andrewaltimit/Documentation/blob/main/github-pages/code-examples/technology/cybersecurity/quantum_secure_computation.py">quantum_secure_computation.py</a>
</div>

```python
# Example: Grover's algorithm setup
from quantum_secure_computation import QuantumAttacks

# Create quantum circuit for database search
qc = QuantumAttacks.grovers_search(
    oracle=database_oracle,
    n_bits=20  # Search space of 2^20
)

# Breaks 160-bit key in 2^80 operations instead of 2^160
```

## References and Further Reading

### Graduate-Level Textbooks
1. **Katz & Lindell** - "Introduction to Modern Cryptography" (3rd Edition)
2. **Goldreich** - "Foundations of Cryptography" (Volumes 1 & 2)
3. **Boneh & Shoup** - "A Graduate Course in Applied Cryptography"
4. **Stinson** - "Cryptography: Theory and Practice" (4th Edition)

### Research Papers
1. **Post-Quantum Cryptography**
   - Bernstein et al. (2017) - "Post-quantum cryptography"
   - NIST PQC Standardization Process documents

2. **Zero-Knowledge Proofs**
   - Goldwasser, Micali, Rackoff (1989) - "The Knowledge Complexity of Interactive Proof Systems"
   - Ben-Sasson et al. (2014) - "Succinct Non-Interactive Zero Knowledge"

3. **Secure Multi-Party Computation**
   - Yao (1982) - "Protocols for Secure Computations"
   - Beaver, Micali, Rogaway (1990) - "The Round Complexity of Secure Protocols"

4. **Side-Channel Attacks**
   - Kocher (1996) - "Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS"
   - Brier, Clavier, Olivier (2004) - "Correlation Power Analysis"

### Advanced Topics for Research
1. **Fully Homomorphic Encryption** - Computing on encrypted data
2. **Oblivious RAM** - Hide memory access patterns
3. **Differential Privacy** - Statistical database privacy
4. **Verifiable Computation** - Prove computation correctness
5. **Blockchain Security** - Consensus mechanisms and smart contract security

## See Also
- [Networking](networking.html) - Network fundamentals
- [AWS](aws.html) - Cloud security specifics
- [Docker](docker.html) - Container security
- [Kubernetes](kubernetes.html) - Orchestration security
- [Quantum Computing](quantumcomputing.html) - Quantum algorithms and cryptography