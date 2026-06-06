---
layout: docs
title: "Cybersecurity: Application Security"
permalink: /docs/technology/cybersecurity/application-and-cloud-security.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Application Security</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">The OWASP Top 10, injection, broken access control, and building security into the SDLC</p>
</div>

<p class="breadcrumb"><a href="./">Cybersecurity</a> › Application Security</p>

<div class="intro-card">
  <p class="lead-text">Cryptography protects data in transit, but most successful attacks target the application layer itself — injection flaws, broken access control, and weak authentication. This page covers the OWASP Top 10, SQL injection, cross-site scripting and CSRF, server-side request forgery, modern authentication and authorization (JWT/OAuth/OIDC), secure session management, API security, and building security into the software development lifecycle.</p>
</div>

<div class="notice--info" markdown="1">
**Looking for cloud and container security?** The shared-responsibility model, cloud IAM, posture management, image hardening, and Kubernetes runtime defense now live on their own page: [Cloud &amp; Container Security](cloud-and-container-security.html).
</div>

## Why the Application Layer Dominates Breaches

The network perimeter has hardened — TLS is ubiquitous, firewalls are standard, and patching has improved. As a result, attackers have moved up the stack. The application is the one component that *must* accept untrusted input from the internet and act on it, which makes it the richest target. Every form field, URL parameter, HTTP header, cookie, and uploaded file is an attacker-controlled channel into your logic and your data store.

The recurring root cause is the same across almost every class of vulnerability: **untrusted data is treated as trusted code or commands**. Injection mixes attacker data into a SQL query; XSS mixes it into HTML; SSRF mixes it into the URL of an outbound request. The defenses rhyme too — separate data from code (parameterization), and never trust input you did not generate yourself.

## The OWASP Top 10

The [OWASP Top 10](https://owasp.org/www-project-top-ten/) is the industry-standard awareness document for web application risk, refreshed every few years from real-world breach and bug-bounty data. It is not a checklist of *the* ten bugs to fix — it is ten *categories* of risk to design against. The 2021 edition (current at time of writing) reads:

| Rank | Category | Core problem |
|------|----------|--------------|
| A01 | **Broken Access Control** | Users can act outside their intended permissions |
| A02 | **Cryptographic Failures** | Sensitive data exposed through weak or missing crypto |
| A03 | **Injection** | Untrusted input interpreted as a command (SQLi, NoSQLi, OS, LDAP) |
| A04 | **Insecure Design** | Missing or flawed security controls at the design stage |
| A05 | **Security Misconfiguration** | Default creds, verbose errors, unnecessary features enabled |
| A06 | **Vulnerable & Outdated Components** | Known-vulnerable libraries and dependencies |
| A07 | **Identification & Authentication Failures** | Weak login, session, or credential handling |
| A08 | **Software & Data Integrity Failures** | Unverified updates, insecure deserialization, CI/CD tampering |
| A09 | **Security Logging & Monitoring Failures** | Breaches go undetected because nothing is logged |
| A10 | **Server-Side Request Forgery (SSRF)** | The server is tricked into making attacker-chosen requests |

Two structural shifts from earlier editions are worth noting. **Broken Access Control rose to #1** because authorization bugs are both pervasive and high-impact. And the new **Insecure Design** category recognizes that some flaws cannot be patched away — they stem from a missing control that should have been designed in from the start (e.g., no rate limit on a password-reset flow). The sections below drill into the categories that dominate real exploitation.

## A03 — Injection: SQL Injection, the Database Killer

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

The fix above is the single most important idea in application security: **parameterized queries (prepared statements) separate code from data.** The SQL structure is sent to the database first and compiled; the user-supplied values are then bound to placeholders and can never change the query's meaning. No amount of escaping, blacklisting, or "sanitizing" `'` characters is as reliable.

### Beyond `' OR '1'='1`: classes of SQLi

- **In-band (classic):** results returned directly, e.g. `UNION SELECT username, password FROM users`.
- **Blind boolean-based:** the page renders differently for true/false conditions, letting an attacker extract data one bit at a time (`AND SUBSTRING(password,1,1)='a'`).
- **Blind time-based:** no visible difference, so the attacker measures response time (`AND IF(condition, SLEEP(5), 0)`).
- **Out-of-band:** the database is coerced into making a DNS/HTTP request that exfiltrates data to the attacker.

### Defense in depth for injection

1. **Parameterized queries / prepared statements everywhere.** Use them even for "trusted" internal values — defense should not depend on classification being correct.
2. **Use a query builder or ORM** that parameterizes by default, but stay alert: raw-string escape hatches (e.g. `Model.objects.raw()`, `sequelize.literal()`) reopen the hole.
3. **Least-privilege database accounts.** The web app's DB user should not own `DROP TABLE` or read other schemas.
4. **Allowlist what cannot be parameterized.** Table and column names can't be bound as parameters; validate them against a fixed allowlist rather than interpolating user input.
5. The same pattern applies to **NoSQL injection** (e.g. MongoDB operator injection via `{"$gt": ""}`), **OS command injection** (use `subprocess` argument lists, never `shell=True` with interpolated strings), and **LDAP injection**.

## A03 — Cross-Site Scripting (XSS): Hijacking Browsers

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

XSS is injection aimed at the browser instead of the database: attacker-controlled data is interpreted as HTML or JavaScript in the victim's session. Once a script runs in the victim's context it can read cookies, make authenticated requests, keylog, or rewrite the page.

### The three flavors of XSS

- **Reflected XSS:** the payload travels in the request (a URL parameter) and is echoed straight back in the response — typically delivered via a crafted link.
- **Stored (persistent) XSS:** the payload is saved server-side (a comment, profile field) and served to every visitor. The most dangerous because it needs no per-victim delivery.
- **DOM-based XSS:** the vulnerability is entirely client-side — JavaScript reads from `location.hash` or `document.referrer` and writes it into the DOM via `innerHTML`, with no server round-trip.

### Defending against XSS

1. **Context-aware output encoding.** Escaping differs by sink: HTML body, HTML attribute, JavaScript string, URL, and CSS each need different encoding. Use the framework's auto-escaping templating (React, Angular, Django, Rails all escape by default) and treat raw-HTML escape hatches (`dangerouslySetInnerHTML`, `v-html`, `\| safe`) as red flags.
2. **Content Security Policy (CSP).** A strong CSP (`script-src 'self'` plus nonces/hashes, no `unsafe-inline`) means an injected `<script>` simply won't execute — a powerful second line of defense.
3. **Sanitize rich HTML** with a vetted library (DOMPurify) when you genuinely must render user-authored markup.
4. **Mark cookies `HttpOnly`** so that even a successful XSS cannot read the session cookie from JavaScript.
5. **Avoid dangerous sinks:** prefer `textContent` over `innerHTML`, and never pass user data to `eval`, `setTimeout(string, …)`, or `new Function`.

## Cross-Site Request Forgery (CSRF)

Where XSS abuses the *site's* trust in the browser, CSRF abuses the *browser's* trust in the site. The browser automatically attaches cookies to every request to a domain, so a malicious page can forge a state-changing request to a site where the victim is logged in:

```html
<!-- Hosted on evil.com; fires automatically when the victim visits -->
<form action="https://bank.example/transfer" method="POST" id="f">
  <input type="hidden" name="to" value="attacker">
  <input type="hidden" name="amount" value="10000">
</form>
<script>document.getElementById('f').submit();</script>
<!-- The victim's bank session cookie rides along — the transfer succeeds -->
```

### CSRF defenses

- **Synchronizer (anti-CSRF) tokens.** The server embeds an unpredictable per-session token in each form; the forged request from `evil.com` can't read or guess it. This is the canonical defense and is built into most frameworks.
- **`SameSite` cookies.** Setting `SameSite=Lax` (now the browser default) or `Strict` stops the cookie from being sent on cross-site requests, neutralizing most CSRF. Treat it as defense in depth, not a sole control, because `Lax` still permits top-level GET navigations.
- **Double-submit cookie** pattern for stateless APIs.
- **Verify `Origin`/`Referer`** for sensitive endpoints.
- Note that **token-based auth in an `Authorization` header is not automatically attached by the browser**, so pure bearer-token APIs are largely immune to classic CSRF — the risk returns the moment you put the token in a cookie.

## A10 — Server-Side Request Forgery (SSRF)

SSRF tricks the *server* into making an HTTP request to a destination the attacker chooses. It became a Top 10 category in its own right after high-profile cloud breaches (the 2019 Capital One incident pivoted through SSRF to the cloud metadata service).

```python
# Vulnerable: the server fetches whatever URL the user supplies
url = request.args['image_url']
resp = requests.get(url)          # attacker sets url=http://169.254.169.254/...
return resp.content

# The cloud instance metadata endpoint hands out temporary credentials:
#   http://169.254.169.254/latest/meta-data/iam/security-credentials/<role>
# SSRF -> credential theft -> full account compromise
```

Why it's dangerous: the server usually sits *inside* the trusted network. An attacker who steers its outbound requests can reach internal admin panels, databases, and — most damagingly — the **cloud instance metadata service** (`169.254.169.254`) to steal IAM credentials.

### SSRF defenses

1. **Allowlist destinations.** Permit only the specific hosts/schemes the feature needs; reject everything else. Blocklists of "internal" IP ranges are brittle (DNS rebinding, IPv6, redirects, decimal-encoded IPs).
2. **Resolve and validate the IP** after DNS resolution, and re-validate after every redirect, to defeat DNS rebinding and `Location:`-based bypasses.
3. **Use IMDSv2** (token-required, session-oriented metadata) on AWS so a naïve GET can't lift credentials; see [Cloud &amp; Container Security](cloud-and-container-security.html).
4. **Network-layer egress filtering** so the workload simply cannot reach the metadata endpoint or internal subnets it has no business contacting.

## A01 — Broken Access Control & Authentication

Access control is now the #1 OWASP category, and it splits into two questions the application must answer for every request: **who are you (authentication)** and **are you allowed to do this (authorization)?**

### The authentication challenge

Passwords alone are no longer enough. Modern authentication requires multiple factors:

1. **Something you know** (password)
2. **Something you have** (phone, hardware token)
3. **Something you are** (fingerprint, face recognition)

But implementing secure authentication is complex. This is where protocols like OAuth 2.0 come in, allowing you to "Login with Google" instead of creating yet another password.

Authentication hardening checklist (OWASP A07):

- **Store passwords with a memory-hard hash** — Argon2id (preferred), scrypt, or bcrypt — never plain SHA-256, and never plaintext.
- **Enforce MFA**, ideally phishing-resistant WebAuthn/passkeys (FIDO2) over SMS one-time codes.
- **Rate-limit and lock out** login and password-reset endpoints to blunt credential stuffing and brute force.
- **Return generic errors** ("invalid username or password") to avoid username enumeration.

### Authorization: the part that's usually broken

Authentication confirms identity; authorization is where most real bugs live. The classic failure is **IDOR (Insecure Direct Object Reference)** — also called Broken Object Level Authorization in the API world:

```python
# Vulnerable: trusts the client-supplied id with no ownership check
@app.get("/api/invoices/<invoice_id>")
def get_invoice(invoice_id):
    return db.invoices.find_one({"_id": invoice_id})   # any id works!

# Secure: scope every lookup to the authenticated user
@app.get("/api/invoices/<invoice_id>")
def get_invoice(invoice_id):
    inv = db.invoices.find_one({"_id": invoice_id, "owner": current_user.id})
    if inv is None:
        abort(404)        # don't leak existence to non-owners
    return inv
```

Principles for sound authorization:

- **Deny by default.** Every endpoint requires an explicit grant; new routes are locked until proven open.
- **Enforce on the server, every time.** Hiding a button in the UI is not access control — the API must re-check on each request.
- **Check ownership and function-level access together.** "Can this user touch *this* object?" (object level) *and* "is this user allowed to call *this* operation at all?" (function level).
- **Centralize the policy** (RBAC/ABAC) rather than scattering `if user.role == 'admin'` checks that drift out of sync.

## JSON Web Tokens: Stateless Security

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

### The `alg` confusion attack — and why you must verify with the right key

The signature is what makes a JWT trustworthy, so the classic attacks all target verification:

- **`alg: none`.** Early libraries accepted a token whose header declared `none` and skipped signature checking entirely. **Pin the expected algorithm on the verifier** and reject anything else.
- **RS256 → HS256 confusion.** A server that holds an RSA *public* key but lets the token pick the algorithm can be fooled: the attacker switches `alg` to `HS256` and signs with the (public, therefore known) key as the HMAC secret. Again the fix is to fix the algorithm at verification time, not read it from the attacker-controlled header.
- **Revocation is hard.** A signed JWT is valid until it expires; you can't easily invalidate one mid-life. Keep access-token lifetimes short (minutes), use a refresh-token rotation scheme for longevity, and keep a server-side denylist for "log out everywhere."

## Secure Session Management

Whether you use server-side sessions or token-based auth, the rules for keeping a session from being stolen or fixated are similar:

- **Generate session IDs with a CSPRNG** and enough entropy to be unguessable.
- **Set cookie flags:** `HttpOnly` (blocks JS theft), `Secure` (HTTPS only), and `SameSite=Lax`/`Strict` (CSRF defense).
- **Rotate the session ID on privilege change** — especially right after login — to prevent *session fixation*, where an attacker plants a known session ID before the victim authenticates.
- **Expire sessions** with both an idle timeout and an absolute timeout, and invalidate server-side on logout.
- **Scope cookies** with the narrowest `Domain`/`Path`, and avoid putting sensitive data in the cookie value itself.

## API Security

APIs are now the dominant attack surface, and OWASP maintains a separate [API Security Top 10](https://owasp.org/www-project-api-security/). The themes specialize the web list:

- **Broken Object Level Authorization (BOLA/IDOR)** is the #1 API risk — see the ownership-check example above. Every object access must be scoped to the caller.
- **Broken Object Property Level Authorization / mass assignment.** Binding a whole request body to a model lets a caller set fields they shouldn't (`{"role": "admin"}`). Bind to an explicit allowlist of writable fields.
- **Broken Function Level Authorization.** Don't rely on the client not calling `/admin/*` — enforce role checks server-side per route.
- **Unrestricted resource consumption.** Apply rate limiting, pagination with hard caps, payload-size limits, and query-depth/complexity limits (critical for GraphQL).
- **Strict input validation against a schema** (OpenAPI/JSON Schema), rejecting unknown fields rather than ignoring them.
- **Authenticate service-to-service** calls (mTLS or signed tokens) — internal does not mean trusted.

## A04/A08 — Building Security into the SDLC

The categories above are symptoms; a **Secure Software Development Lifecycle (SSDLC)** is how you stop reintroducing them. The principle is "shift left": catch flaws when they are cheap to fix (design, code review) rather than after deployment.

| SDLC phase | Security activity |
|------------|-------------------|
| **Requirements** | Define security/abuse requirements; data classification |
| **Design** | Threat modeling (STRIDE); address *Insecure Design* before code exists |
| **Implementation** | Secure-coding standards; **SAST** (static analysis) and **secret scanning** in the IDE/PR |
| **Build / CI-CD** | **SCA** (dependency scanning) for OWASP A06; sign artifacts; protect the pipeline (A08) |
| **Test** | **DAST** / **IAST**; fuzzing; targeted penetration testing |
| **Release** | Security gates in CD; verified, signed deployments |
| **Operate** | Logging & monitoring (A09); runtime protection; vulnerability disclosure & patching |

Key practices that pay off the most:

- **Threat model at design time.** STRIDE (Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, Elevation of privilege) gives a structured way to ask "how could this be abused?" before a line of code exists — the only effective defense against *Insecure Design (A04)*.
- **Automate the scanners in CI** so security feedback lands in the pull request, not a quarterly audit: SAST for code flaws, SCA for vulnerable dependencies (A06), and secret scanning to stop credentials reaching the repo.
- **Manage your software supply chain (A08).** Pin and verify dependencies, generate an SBOM, and sign build artifacts so a tampered component is detectable. (The container/image side of this is covered on [Cloud &amp; Container Security](cloud-and-container-security.html).)
- **Treat logging and monitoring as a feature (A09).** Log authentication, access-control, and input-validation failures in a tamper-resistant way so that the incident-response team has something to work with; see [Security Operations](security-operations.html) and [Incident Response](incident-response.html).

---

<div class="page-nav">
  <span class="page-nav-prev"><a href="cryptography.html">← Cryptography</a></span>
  <span class="page-nav-next"><a href="cloud-and-container-security.html">Cloud &amp; Container Security →</a></span>
</div>

## See Also

- [Cloud &amp; Container Security](cloud-and-container-security.html) — the shared-responsibility model, IAM, image hardening, and Kubernetes runtime defense that used to live here
- [Cryptography](cryptography.html) — the encryption protecting your application's data
- [Attacks &amp; Network Defense](attacks-and-defense.html) — firewalls, VPNs, and Zero Trust in full
- [Security Operations](security-operations.html) — monitoring, detection, and the SOC
- [Incident Response](incident-response.html) — what happens after an application is breached
