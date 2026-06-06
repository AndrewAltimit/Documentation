---
layout: docs
title: "Git: Authentication & Access Control"
permalink: /docs/technology/git/auth-and-access-control.html
toc: true
toc_sticky: true
hide_title: true
---

[Git Internals](./) ›

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Authentication &amp; Access Control</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">How Git proves who you are, who can write, and that a commit is genuinely yours — keys, tokens, signing, SSO, and leak defense</p>
</div>

Git itself stores no credentials and enforces no permissions: it is a content-addressable store that any clone can read or rewrite. **Authentication** (proving *who is connecting*) and **access control** (deciding *what they may do*) live in the **transport layer** and the **hosting platform** — SSH, HTTPS, the credential helper, and the server-side policy of GitHub/GitLab/Bitbucket/Gitea. A separate, independent layer — **commit and tag signing** — proves *who authored a change* and that it has not been altered, regardless of how it was transported.

This page covers all four: how you authenticate to a remote (SSH keys, deploy keys, personal-access tokens, credential helpers), how you cryptographically vouch for your own commits (GPG and SSH signing), how organizations centralize identity (SAML/OAuth SSO), and how to revoke, audit, and contain credentials when they leak.

<div class="tip-card">
  <h4>Two orthogonal questions</h4>
  <p><strong>Transport auth</strong> answers <em>"may this connection fetch or push?"</em> and is checked by the server at clone/fetch/push time. <strong>Signing</strong> answers <em>"who really wrote this commit?"</em> and is verified by anyone, any time, long after the push. A pushed commit can be authentic-by-transport but unsigned, or signed but pushed by a leaked token. You generally want both.</p>
</div>

## The Transport Landscape

A remote URL's scheme selects the transport and therefore the auth mechanism:

| URL form | Transport | Authenticates with |
|----------|-----------|--------------------|
| `git@github.com:org/repo.git` | SSH | SSH key pair (user or deploy key) |
| `ssh://git@host:22/org/repo.git` | SSH | SSH key pair |
| `https://github.com/org/repo.git` | HTTPS | Username + token/password via credential helper |
| `git://host/repo.git` | Git protocol | **None** — anonymous, read-only, unauthenticated |
| `/path/to/repo.git` or `file://` | Local filesystem | OS filesystem permissions |

The legacy `git://` protocol is unauthenticated and unencrypted; major hosts have disabled it. In practice you choose between **SSH** and **HTTPS**, and the rest of this page is about doing each securely.

```bash
git remote -v                         # inspect current remote URLs
git remote set-url origin git@github.com:org/repo.git   # switch to SSH
git remote set-url origin https://github.com/org/repo.git  # switch to HTTPS
```

## SSH Keys

SSH authenticates with **public-key cryptography**: you hold a private key, the server holds your public key, and a challenge-response handshake proves you possess the private key without ever transmitting it. No secret crosses the wire on every push the way a token does.

### Generating a Key

Prefer **Ed25519** (small, fast, modern). Use RSA only where you must interoperate with old servers, and then at 4096 bits.

```bash
# Modern default — Ed25519
ssh-keygen -t ed25519 -C "you@example.com" -f ~/.ssh/id_ed25519

# Hardware-backed (FIDO2 security key) — private key cannot be exfiltrated
ssh-keygen -t ed25519-sk -C "you@example.com" -f ~/.ssh/id_ed25519_sk

# Legacy interop only
ssh-keygen -t rsa -b 4096 -C "you@example.com" -f ~/.ssh/id_rsa
```

The `-sk` variants (`ed25519-sk`, `ecdsa-sk`) bind the key to a FIDO2 authenticator (YubiKey, Touch ID via a resident credential). The private key material lives in the hardware and a touch/PIN is required per use, so a compromised laptop cannot push on its own.

**Always set a passphrase** on a software key. The on-disk key is then encrypted; an attacker who copies `~/.ssh/id_ed25519` still needs the passphrase. Combine the passphrase with `ssh-agent` so you type it once per session.

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519          # add key; agent caches the decrypted key
ssh-add -t 8h ~/.ssh/id_ed25519    # auto-forget after 8 hours
ssh-add -l                         # list loaded identities
```

### Registering and Testing

Copy the **public** half (`~/.ssh/id_ed25519.pub`, never the file without `.pub`) into your account's SSH keys page on the host, then verify:

```bash
ssh -T git@github.com      # GitHub greets you by username on success
ssh -Tv git@github.com     # verbose — shows which key was offered and accepted
```

### Per-Host Configuration

`~/.ssh/config` pins which key, user, and options apply to each host. This is essential when you have multiple identities (work and personal accounts on the same host) — SSH cannot distinguish them by URL alone.

```sshconfig
# ~/.ssh/config
Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal
    IdentitiesOnly yes          # offer ONLY this key, do not spray all keys

Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
    IdentitiesOnly yes
```

Then clone with the alias: `git clone git@github-work:org/repo.git`. `IdentitiesOnly yes` prevents the agent from offering every loaded key in turn, which both leaks key fingerprints and can trip "too many authentication failures" lockouts.

### Verifying the Server's Host Key

SSH auth is mutual: the *server* also presents a host key, and trusting the wrong one enables a man-in-the-middle. The first connection records the host key in `~/.ssh/known_hosts`. Verify it against the host's **published** fingerprints before accepting, rather than blindly typing "yes". Hosts publish their key fingerprints (GitHub, for instance, documents its Ed25519/RSA/ECDSA fingerprints); compare with:

```bash
ssh-keygen -lf ~/.ssh/known_hosts -F github.com
```

A sudden host-key-changed warning on an established host is a red flag, not noise to suppress.

## Deploy Keys

A **deploy key** is an SSH public key attached to a **single repository** rather than to a user account. It is the right credential for CI servers, deployment hosts, and automation that needs exactly one repo.

| Property | User SSH key | Deploy key |
|----------|--------------|------------|
| Scope | All repos the user can access | One repository |
| Default access | Read/write per user role | **Read-only** unless write explicitly enabled |
| Tied to | A human account | A repo + a key pair |
| Reused across repos | Yes | A given key may be registered to only one repo |
| Blast radius if leaked | Everything the user can touch | That one repo |

```bash
# On the CI/deploy host, generate a key with NO passphrase (unattended use)
ssh-keygen -t ed25519 -C "ci-deploy: example-service" -f ~/.ssh/deploy_example -N ""
# Register deploy_example.pub on the repository's "Deploy keys" page,
# granting write access only if the pipeline must push (e.g. tags, gh-pages).
```

**Principle of least privilege:** add the deploy key read-only unless the pipeline genuinely pushes. A read-only deploy key that leaks cannot corrupt the repository. Because a deploy key with a passphrase cannot be used unattended, compensate with tight scope (one repo, read-only) and rotation rather than relying on the missing passphrase.

For automation that must span *several* repositories, prefer a **machine/bot account** with its own SSH key, or a host-native machine identity (GitHub Apps / installation tokens, GitLab project access tokens) over reusing one human's key — those issue short-lived, scoped, individually-revocable credentials and keep the audit trail attributable.

## HTTPS, Tokens, and Credential Helpers

Over HTTPS, Git authenticates with a username and a **secret** on every request. Plain account passwords for Git operations are deprecated or removed on the major hosts; you use a **token** instead.

### Personal Access Tokens (PATs)

A PAT is a long random string that stands in for your password for Git (and API) operations. The point of a token over a password is that it can be **scoped** (limited to specific permissions), **time-boxed** (expires), and **revoked individually** without touching your password or other tokens.

Two generations exist on GitHub, and the distinction matters:

- **Classic PATs** carry coarse, account-wide scopes (`repo`, `read:org`, `write:packages`, …). The `repo` scope grants access to *every* repository you can reach — broad blast radius.
- **Fine-grained PATs** are scoped to **specific repositories** (or a single org) with **per-resource permissions** (e.g. Contents: read-only, Pull requests: read/write) and a **mandatory expiry**. Prefer these.

Best practices for any token:

- **Scope to the minimum** permissions and repositories the task needs.
- **Set the shortest practical expiry** (e.g. 30–90 days, or hours for ephemeral CI).
- **One token per consumer** (per machine, per pipeline) so revocation is surgical and the audit log shows which consumer did what.
- **Never** embed a token in a remote URL (`https://user:TOKEN@host/...`) — it lands in `.git/config`, in `git remote -v` output, in shell history, and in CI logs. Use a credential helper instead.

```bash
git clone https://github.com/org/repo.git
# Username: your-username
# Password: paste the PAT (NOT your account password)
```

### GitHub App Installation Tokens

For server-to-server automation at scale, an **installation access token** minted by a GitHub App is superior to a PAT: it is **short-lived** (about an hour), **scoped** to the App's installed repositories and permissions, and **not** tied to any human's account. CI systems mint one per job and discard it. This is the modern replacement for "service account with a long-lived PAT."

### Credential Helpers

A **credential helper** stores and supplies your HTTPS secret so you are not prompted every push. Configure one — never the bare `store` helper, which writes the token to a **plaintext** file (`~/.git-credentials`, mode 600 but still cleartext).

| Helper | Storage | Notes |
|--------|---------|-------|
| `cache` | RAM, time-limited | `--timeout` seconds; nothing hits disk |
| `store` | **Plaintext file** | Avoid; cleartext on disk |
| `osxkeychain` | macOS Keychain | OS-encrypted |
| `manager` (GCM) | OS keystore (Keychain / libsecret / Windows Credential Manager) | Cross-platform; supports OAuth device flow |
| `libsecret` | Linux Secret Service (GNOME Keyring / KWallet) | OS-encrypted |

```bash
# In-memory only, forget after 1 hour
git config --global credential.helper 'cache --timeout=3600'

# OS-native secure storage (recommended)
git config --global credential.helper manager          # Git Credential Manager
git config --global credential.helper osxkeychain       # macOS
git config --global credential.helper libsecret         # Linux Secret Service

# Scope a helper to one host (useful for split work/personal identities)
git config --global credential.https://github.com.helper manager
```

The helper protocol is simple text on stdin/stdout (`protocol`, `host`, `username`, `password` key/value lines terminated by a blank line). Git calls `get` before a request and `store`/`erase` after, so any program implementing those three verbs can broker credentials — including ones that fetch a fresh short-lived token on demand. **Git Credential Manager** can run a full OAuth device-code flow, so you authenticate in a browser and GCM transparently caches and refreshes the OAuth token; you never handle a long-lived PAT at all, which is the most secure HTTPS option for interactive use.

<div class="tip-card">
  <h4>SSH or HTTPS — which to choose?</h4>
  <p><strong>SSH</strong> shines for interactive developer use (key + agent + passphrase, nothing to paste) and for fixed automation hosts (deploy keys). <strong>HTTPS</strong> is friendlier behind restrictive corporate proxies/firewalls (port 443) and pairs naturally with OAuth/SSO and short-lived App tokens in CI. Both are secure when configured per this page; pick per environment, and you can even mix them per host via <code>~/.ssh/config</code> and per-host credential helpers.</p>
</div>

## Commit and Tag Signing

Transport auth says nothing about *authorship*. Git records the author/committer fields as **plain, unverified strings** — anyone can `git -c user.email=ceo@example.com commit`. **Signing** attaches a cryptographic signature to a commit or tag so that anyone can later verify it was created by the holder of a specific key and has not been altered.

### What a signature actually covers

A commit object is a small text blob: tree hash, parent hash(es), author, committer, and message. Signing computes a digital signature over **the exact bytes of that object** and stores it in the commit's `gpgsig` header. Because the object includes the **parent hash**, and the parent's hash transitively commits to all ancestor content (the Merkle/DAG property of Git's object model), a valid signature on a commit certifies the **entire history reachable from it** is byte-for-byte unchanged. Tamper with any ancestor and every descendant hash changes, breaking the signature.

> See [Object Model &amp; Storage](object-model.html) for why every object is named by the hash of its content, which is exactly the property signing leans on.

### Signing with GPG

```bash
# Generate a signing key (Ed25519 or RSA-4096)
gpg --full-generate-key

# Find its long key id
gpg --list-secret-keys --keyid-format=long
#   sec   ed25519/3AA5C34371567BD2 ...

# Tell Git to use it and sign by default
git config --global user.signingkey 3AA5C34371567BD2
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# Export the PUBLIC key to register on the host (GitHub "GPG keys" page)
gpg --armor --export 3AA5C34371567BD2
```

The email in the GPG key's user ID must match `git config user.email` (and a verified email on the host) for the host to show the commit as **Verified**.

### Signing with SSH (simpler, key reuse)

Since Git 2.34 you can sign with your **SSH** key — no GPG keyring to manage, and you can reuse the same key family you already use for auth (use a *separate* key in practice; see below).

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

To **verify** SSH-signed commits locally, Git needs an `allowed_signers` file mapping identities to public keys:

```bash
# ~/.config/git/allowed_signers
you@example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... you@example.com
```
```bash
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
```

Hosts let you register the same SSH public key as a **signing key** (distinct from an authentication key, even if the bytes match).

<div class="tip-card">
  <h4>Separate auth keys from signing keys</h4>
  <p>An authentication key is exercised against many servers and may be forwarded by the agent; a signing key vouches for your identity in perpetuity. Keep them <strong>distinct</strong> so that compromise or rotation of one does not force the other, and so a forwarded auth key cannot be coaxed into signing. Hardware-backed keys (<code>-sk</code> / smartcard) are ideal for signing.</p>
</div>

### Verifying Signatures

```bash
git log --show-signature                 # show verification status per commit
git verify-commit <commit>               # verify a specific commit
git verify-tag v1.2.0                     # verify a signed tag
git log --pretty="%h %G? %an %s"          # %G? => G good, B bad, U unknown, N none
```

The `%G?` placeholder yields **G** (good signature), **B** (bad — content altered), **U** (good but untrusted key), **E** (cannot check), or **N** (no signature). CI can gate merges on this.

### Enforcing Signing in CI

Local config is advisory; enforcement happens on the server or in CI. Hosts offer **branch protection / rulesets** requiring signed commits, and you can additionally verify in a pipeline:

```bash
# Fail the build if any commit on the branch is unsigned or has a bad signature
set -e
for c in $(git rev-list origin/main..HEAD); do
  status=$(git log -1 --pretty="%G?" "$c")
  case "$status" in
    G|U) ;;                                   # good (U = good but untrusted key)
    *) echo "Unsigned/invalid commit: $c ($status)"; exit 1 ;;
  esac
done
```

> The platform's signed-commit branch rule is the real gate; the script above is a defense-in-depth check inside the same self-hosted runners that build this site (see [CI/CD](../ci-cd/)).

## Server-Side Access Control: SSO, SAML, and OAuth

Everything above proves *who you are at the transport*. **Authorization** — which repos you may read/write, who can force-push, who can change settings — is the host's responsibility and is layered on top.

### How the layers stack

```
                       ┌─────────────────────────────┐
   You ──auth──▶       │  Identity (who you are)      │  SSH key / PAT / OAuth / SAML
                       ├─────────────────────────────┤
                       │  Org / SSO policy gate       │  SAML SSO, IP allowlist, 2FA req.
                       ├─────────────────────────────┤
                       │  Authorization (what you may │  roles, teams, branch protection,
                       │  do on this repo/ref)        │  CODEOWNERS, ruleset required reviews
                       └─────────────────────────────┘
```

### SAML / OIDC Single Sign-On

Enterprises federate Git-host identity to a central **Identity Provider** (Okta, Entra ID/Azure AD, Google Workspace) via **SAML** or **OpenID Connect**. The IdP becomes the single source of truth for who exists and what groups they belong to, enabling central onboarding/offboarding, mandatory 2FA, and conditional-access policies.

A subtlety that trips everyone: **SAML applies to the web session, but Git operations use a key/token.** When SSO is enforced on an org, your existing **PAT or SSH key must be explicitly *authorized* for that org** before it works against the org's repos — even though the credential is valid in general. You authorize each credential once (a one-click "Configure SSO" / "Authorize" on the token or key), and a fresh SSO web login may be required periodically to keep it active.

```bash
# Symptom of an un-authorized credential under SAML SSO:
git push
# remote: ... SAML enforcement ... your credential is not authorized for org X
# Fix: open the token/SSH-key settings page and "Authorize" it for org X.
```

### OAuth Apps and Scopes

Third-party tools (CI, code-review bots, IDE integrations) connect via **OAuth**: you grant a named application a set of **scopes** through a browser consent screen, and it receives an access token (often with a refresh token) — your password is never shared. Audit and **revoke** these grants from your account's "Applications / Authorized OAuth Apps" page. Grant the narrowest scopes a tool asks for, and be suspicious of any integration requesting `repo`-wide or org-admin scope it does not obviously need.

### Repository-Level Authorization

Independent of how you signed in, the host enforces per-repo and per-ref policy:

- **Roles / teams**: read, triage, write, maintain, admin — assigned to users and teams.
- **Branch protection / rulesets**: require pull requests, required reviewers, required status checks, **required signed commits**, linear history, and restrictions on **force-push** and deletion of protected branches.
- **CODEOWNERS**: route required reviews to the owners of touched paths.
- **Server-side hooks**: `pre-receive`/`update` hooks reject pushes that violate policy (e.g. secret-scanning, commit-message format) — see [Algorithms &amp; Advanced Operations](algorithms-and-operations.html) for the hook model.

## Revocation, Rotation, and Audit

Credentials are liabilities; assume any of them will eventually leak. The defense is **fast revocation**, **routine rotation**, and **audit trails** that let you answer "what could this credential do, and what did it do?"

### Revocation

| Credential | How to revoke | Effect |
|------------|---------------|--------|
| SSH user key | Delete the public key from the account | All hosts using it lose access immediately |
| Deploy key | Remove from the repo's Deploy keys | That repo only |
| PAT | Delete/revoke the token | That token only; other tokens unaffected |
| OAuth grant | Revoke the app authorization | That application loses its token |
| App installation | Uninstall / rotate the App's private key | All tokens it could mint |
| GPG/SSH signing key | Publish a **revocation certificate** (GPG) / remove the key | Future signatures distrusted; past ones flagged |

Generate a **GPG revocation certificate at key-creation time** and store it offline — you will need it if the key is lost or compromised and you no longer control the secret:

```bash
gpg --output revoke-3AA5C34371567BD2.asc --gen-revoke 3AA5C34371567BD2
```

### Rotation

Rotate on a schedule and on every suspicion:

- Prefer **expiring** credentials (fine-grained PATs, App tokens) so rotation is automatic.
- Keep **one credential per consumer** so rotating one does not break others.
- For shared deploy keys, rotate by adding the new key, switching the consumer, then removing the old — zero downtime.

### Audit

- **Host audit logs** record key/token creation, pushes, force-pushes, permission changes, and SSO events — review them and ship them to your SIEM.
- **`git log --show-signature`** and required-signed-commit rules give you a cryptographic audit of *authorship*, complementing the host's transport audit of *who pushed*.
- The **reflog** (`git reflog`) records local ref movements and is invaluable when investigating an unexpected force-push.

## Protecting Against Credential Leaks

The most common Git security incident is not a broken crypto primitive — it is a **secret committed into the repository** (an API key, a `.env`, a private key) and pushed where it is now in history forever and possibly public.

### Prevention

```gitignore
# .gitignore — keep secrets out in the first place
.env
.env.*
*.pem
*.key
id_rsa
id_ed25519
**/secrets/**
*.p12
credentials.json
```

- **Never commit secrets.** Use environment variables, a secrets manager (Vault, cloud KMS/Secrets Manager), or CI secret stores — see [Cybersecurity](../cybersecurity/).
- **Pre-commit secret scanning** stops leaks before they enter history. Install a hook that runs a scanner (`gitleaks`, `detect-secrets`, `trufflehog`) on staged content:

```bash
# .git/hooks/pre-commit (or via the pre-commit framework)
#!/usr/bin/env bash
if ! gitleaks protect --staged --redact -v; then
  echo "Potential secret detected in staged changes — commit aborted."
  exit 1
fi
```

- **Enable host-side secret scanning / push protection** so the server itself rejects a push containing a recognizable secret pattern, catching what local hooks miss.

### Containment after a leak

A leaked secret in history is **compromised the instant it was pushed**, especially on a public repo where bots scrape commits within seconds. The order of operations matters:

1. **Revoke and rotate the secret first.** Removing it from history does nothing if it is already harvested — invalidate it at the source (regenerate the key/token, disable the credential). This is the single most important step.
2. **Then purge it from history** so it does not leak again. Rewriting history changes every descendant commit's hash:

```bash
# Preferred modern tool: git-filter-repo
git filter-repo --invert-paths --path config/secrets.yml      # drop a file from all history
git filter-repo --replace-text expressions.txt                # redact matched strings

# Force-push the rewritten history (coordinate with collaborators!)
git push --force-with-lease --all
git push --force-with-lease --tags
```

3. **Coordinate the rewrite.** History rewriting changes commit hashes; every collaborator must re-clone or hard-reset, and any fork/PR may still retain the old objects. On a public host, cached views and forks may keep copies — which is exactly why step 1 (revocation) is non-negotiable.
4. **Audit usage.** Check host and cloud audit logs for any use of the leaked credential during the exposure window.

<div class="tip-card">
  <h4>Why revoke before you scrub</h4>
  <p>History rewriting is slow, disruptive, and never perfectly complete (forks, mirrors, caches, backups, the attacker's local copy). Revocation is instant and total. Always invalidate the credential first; treat the history purge as cleanup, not as the fix.</p>
</div>

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Auth lives in the transport</h4>
    <p>Git stores no credentials. SSH keys, tokens, and the credential helper authenticate the connection; the host enforces permissions.</p>
  </div>
  <div class="takeaway-card">
    <h4>Signing is independent of transport</h4>
    <p>GPG/SSH signing proves authorship and integrity of a commit, verifiable by anyone long after the push — separate from who pushed it.</p>
  </div>
  <div class="takeaway-card">
    <h4>Least privilege, narrowest scope</h4>
    <p>Deploy keys per repo, fine-grained expiring tokens per consumer, OAuth scopes minimized — shrink the blast radius of every credential.</p>
  </div>
  <div class="takeaway-card">
    <h4>Prefer short-lived credentials</h4>
    <p>App installation tokens and OAuth flows beat long-lived PATs; expiry makes rotation automatic and limits exposure.</p>
  </div>
  <div class="takeaway-card">
    <h4>SSO gates the credential too</h4>
    <p>Under SAML SSO, a valid key or token must still be explicitly authorized for the org before Git operations succeed.</p>
  </div>
  <div class="takeaway-card">
    <h4>Revoke first, scrub second</h4>
    <p>A leaked secret is compromised on push. Rotate/revoke it immediately; rewriting history is cleanup, not the fix.</p>
  </div>
</div>

## See Also

- [Object Model &amp; Storage](object-model.html) — why content-addressing makes a single commit signature certify all reachable history.
- [Algorithms &amp; Advanced Operations](algorithms-and-operations.html) — server-side hooks that enforce push policy, and the reflog for auditing ref movement.
- [Protocols, Packs &amp; Performance](protocols-and-performance.html) — the SSH/HTTPS wire protocols these credentials authenticate.
- [Branching Strategies](../branching.html) — branch protection and required reviews in team workflow context.
- [Cybersecurity](../cybersecurity/) — secrets management, leak prevention, and defense in depth.
- [CI/CD](../ci-cd/) — using scoped, short-lived tokens and deploy keys safely in pipelines.
- [Git Command Reference](../git-reference.html) — `remote`, `config`, `verify-commit`, and signing command syntax.
