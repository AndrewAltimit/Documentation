---
layout: docs
title: "Cybersecurity: Web, Cloud & Container Security"
permalink: /docs/technology/cybersecurity/application-and-cloud-security.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Web, Cloud &amp; Container Security</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Where most breaches actually happen — applications, cloud accounts, and containers</p>
</div>

<p class="breadcrumb"><a href="./">Cybersecurity</a> › Web, Cloud &amp; Container Security</p>

<div class="intro-card">
  <p class="lead-text">Cryptography protects data in transit, but most successful attacks target the application and infrastructure layers — injection flaws, misconfigured cloud permissions, and insecure containers. This page covers web application vulnerabilities, the cloud shared-responsibility model and IAM, and container hardening.</p>
</div>

## Web Application Security: Where Most Attacks Happen

While advanced cryptography protects data in transit, most successful attacks target vulnerabilities in web applications. Understanding these vulnerabilities is crucial because they're where real breaches occur.

### SQL Injection: The Database Killer

SQL injection remains one of the most dangerous vulnerabilities. Here's why it's so devastating:

```python
# Vulnerable code - NEVER do this!
username = request.form['username']
password = request.form['password']
query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"

# What an attacker enters:
# Username: admin' OR '1'='1' --
# Password: anything

# Resulting query:
# SELECT * FROM users WHERE username='admin' OR '1'='1' --' AND password='anything'
# This returns ALL users because '1'='1' is always true!

# Secure code - use parameterized queries
query = "SELECT * FROM users WHERE username=? AND password=?"
cursor.execute(query, (username, password))
# The database knows these are values, not SQL code
```

### Cross-Site Scripting (XSS): Hijacking Browsers

XSS attacks inject malicious scripts into websites that other users view. The impact can be devastating:

```html
<!-- Vulnerable code -->
<div>
  Welcome, <?php echo $_GET['name']; ?>!
</div>

<!-- Attacker sends link: site.com?name=<script>steal_cookies()</script> -->
<!-- When victims click it, the script runs in their browser! -->

<!-- Secure code - always escape output -->
<div>
  Welcome, <?php echo htmlspecialchars($_GET['name']); ?>!
</div>
<!-- Now the script is displayed as text, not executed -->
```

### The Authentication Challenge

Passwords alone are no longer enough. Modern authentication requires multiple factors:

1. **Something you know** (password)
2. **Something you have** (phone, hardware token)
3. **Something you are** (fingerprint, face recognition)

But implementing secure authentication is complex. This is where protocols like OAuth 2.0 come in, allowing you to "Login with Google" instead of creating yet another password.

### JSON Web Tokens: Stateless Security

JWTs solved a major problem in web applications: how to maintain user sessions without storing state on the server.

**Important Security Update**: Many JWT libraries had critical vulnerabilities. Always:
- Use strong, unique secrets (256+ bits)
- Validate the 'alg' header to prevent algorithm confusion attacks
- Set and validate expiration times
- Consider JWE (encrypted JWTs) for sensitive data
- Use refresh token rotation for long-lived sessions

```javascript
// JWT structure: header.payload.signature
const jwt = require('jsonwebtoken');

// Create a token
const token = jwt.sign(
  {
    userId: 123,
    role: 'user',
    // JWT `exp` is in SECONDS since the Unix epoch, not milliseconds.
    // Date.now() returns milliseconds, so divide by 1000:
    exp: Math.floor(Date.now() / 1000) + 3600  // Expires in 1 hour (3600 s)
  },
  process.env.JWT_SECRET
);

// Cleaner alternative: let the library set `exp` for you and avoid the
// unit mistake entirely:
// const token = jwt.sign({ userId: 123, role: 'user' },
//                        process.env.JWT_SECRET, { expiresIn: '1h' });

// Token contains:
// 1. Header: {"alg": "HS256", "typ": "JWT"}
// 2. Payload: {"userId": 123, "role": "user", "exp": ...}
// 3. Signature: HMAC-SHA256(header + payload, secret)

// Anyone can read the payload, but can't modify it
// without invalidating the signature
```

## Cloud Security: New Challenges, New Solutions

The cloud revolutionized how we build and deploy applications, but it also introduced new security challenges. You're no longer protecting a physical server in your data center—you're securing resources that exist "somewhere" in someone else's infrastructure.

### The Shared Responsibility Model

Understanding who secures what is crucial:

```python
# Cloud Provider Secures:
# - Physical data centers
# - Network infrastructure
# - Hypervisor layer
# - Physical storage

# You Secure:
# - Your data
# - Identity and access management
# - Application code
# - Operating system (in IaaS)
# - Network traffic controls
# - Encryption keys
```

### IAM: The Keys to Your Kingdom

In the cloud, Identity and Access Management (IAM) is your most critical security control. A misconfigured IAM policy can expose your entire infrastructure.

**Modern IAM Practices**:
- **Zero Trust Architecture**: Never trust, always verify (covered in full under [Attacks & Network Defense](attacks-and-defense.html#zero-trust-never-trust-always-verify))
- **Just-In-Time Access**: Temporary elevated privileges
- **Passwordless Authentication**: Passkeys, FIDO2 standards
- **Policy Intelligence**: AI-powered policy recommendations
- **CNAPP**: Cloud-Native Application Protection Platforms

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",      // DANGER: Anyone can access!
    "Action": "s3:*",       // DANGER: All permissions!
    "Resource": "arn:aws:s3:::my-bucket/*"
  }]
}

// Secure version:
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::123456789012:role/MyAppRole"},
    "Action": ["s3:GetObject"],  // Minimum necessary permission
    "Resource": "arn:aws:s3:::my-bucket/public/*",
    "Condition": {
      "IpAddress": {"aws:SourceIp": "203.0.113.0/24"}  // IP restriction
    }
  }]
}
```

### Container Security: Shipping Code Safely

Containers add another layer of complexity. You're not just securing an application—you're securing the entire environment it runs in.

**Container Security Best Practices**:
- **Software Bill of Materials (SBOM)**: Required by many regulations
- **Sigstore**: Sign and verify container images
- **Distroless images**: Reduce attack surface
- **Runtime security**: Falco, Sysdig for threat detection
- **Policy as Code**: OPA (Open Policy Agent) for enforcement

```dockerfile
# Insecure Dockerfile
FROM ubuntu:latest
USER root                    # Running as root!
RUN apt-get update && apt-get install -y curl
COPY app /app
CMD ["/app"]

# Secure Dockerfile
FROM ubuntu:22.04           # Specific version
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*  # Clean up
RUN useradd -m appuser      # Create non-root user
USER appuser                # Switch to non-root
COPY --chown=appuser:appuser app /app
CMD ["/app"]
```

---

<div class="page-nav">
  <span class="page-nav-prev"><a href="cryptography.html">← Cryptography</a></span>
  <span class="page-nav-next"><a href="attacks-and-defense.html">Attacks &amp; Network Defense →</a></span>
</div>

## See Also

- [Cryptography](cryptography.html) — the encryption protecting your application's data
- [Attacks & Network Defense](attacks-and-defense.html) — firewalls, VPNs, and Zero Trust in full
- [AWS](../aws/) — cloud IAM and the shared-responsibility model in practice
- [Docker](../docker/) — container fundamentals
- [Kubernetes](../kubernetes/) — orchestrating and securing containers at scale
