---
layout: docs
title: "Git: Authentication & Access Control"
description: "How Git authenticates to remotes (SSH keys, deploy keys, tokens, credential helpers), how commits are signed with GPG, SSH or Sigstore, how forges authorize access, and how to respond to leaked credentials."
permalink: /docs/technology/git/auth-and-access-control.html
toc: true
toc_sticky: true
---

[Git Internals](./) ›

Git has no user accounts and enforces no permissions of its own: anyone with a copy of a repository can read and rewrite it. **Authentication** (who is connecting) and **authorization** (what they may do) are handled by the transport, SSH or HTTPS, and by the hosting server or forge. A separate layer, **commit and tag signing**, records who created an object in a way anyone can verify later, independently of how it was transported. This page covers both layers, Git's own local trust settings, and how to contain a leaked credential.

## Three separate questions

| Question | Mechanism | Checked by | When |
|----------|-----------|------------|------|
| May this connection fetch or push? | SSH key, token, OAuth, SSO | The server | At each fetch, clone and push |
| What may this identity do to this repository and ref? | Roles, branch protection, rulesets, server hooks | The server | At each push (and in the web UI) |
| Who created this commit or tag, and is it unaltered? | GPG, SSH or X.509 signature | Anyone | At any time after creation |

The answers are independent. A commit can be pushed by an authenticated user but be unsigned, or be validly signed but pushed with a stolen token. Author and committer names in a commit are plain text that anyone can set (`git -c user.email=ceo@example.com commit`), so only a signature says anything about authorship.

```mermaid
flowchart LR
    Dev["Developer"] -- "SSH key or token" --> Auth["Transport authentication"]
    Auth --> SSO["Organisation policy<br/>SSO, 2FA, IP allow list"]
    SSO --> Authz["Authorization<br/>roles, rulesets, CODEOWNERS"]
    Authz --> Hooks["Push checks<br/>pre-receive hooks, secret scanning,<br/>required signatures"]
    Hooks --> Repo[("Repository")]
    Repo -. "anyone can verify later" .-> Verify["Signature verification"]
```

## Choosing a credential

| Consumer | Recommended credential | Why |
|----------|------------------------|-----|
| Developer workstation | SSH key (hardware-backed if possible) or HTTPS through Git Credential Manager with OAuth | No long-lived secret to paste; keys can require a touch per use |
| CI job on the forge's own runners | The job's built-in, per-run token (for example GitHub Actions' `GITHUB_TOKEN`) | Minted per job, scoped to the repository, expires automatically |
| Automation across several repositories | App installation token (GitHub App, GitLab group or project access token) | Short-lived, scoped, not tied to a person, individually revocable |
| A single server that pulls one repository | Read-only deploy key | Blast radius of one repository |
| Ad-hoc scripts using the API | Fine-grained personal access token with an expiry | Scoped to specific repositories and permissions |
| Anything else | Avoid classic tokens and shared human accounts | Account-wide scope and poor attribution |

## Transports

The remote URL's scheme selects the transport and therefore the authentication method:

| URL form | Transport | Authenticates with |
|----------|-----------|--------------------|
| `git@github.com:org/repo.git` (scp-like) | SSH | Key pair |
| `ssh://git@host:2222/org/repo.git` | SSH | Key pair |
| `https://github.com/org/repo.git` | HTTPS | Username plus token or OAuth token, via a credential helper |
| `git://host/repo.git` | Git daemon | Nothing: anonymous, unencrypted, read-only |
| `/srv/repo.git`, `file://` | Local | Filesystem permissions |

The unauthenticated `git://` protocol offers no integrity protection against network attackers, and GitHub disabled it in 2022. In practice the choice is between SSH and HTTPS.

```bash
git remote -v
git remote set-url origin git@github.com:org/repo.git       # switch to SSH
git remote set-url origin https://github.com/org/repo.git   # switch to HTTPS
```

**SSH or HTTPS?** SSH suits developer machines (a key plus an agent, nothing to paste) and fixed hosts with deploy keys. HTTPS passes more easily through corporate proxies on port 443 and fits OAuth, SSO and short-lived tokens. Both are secure when configured as below. `url.<base>.insteadOf` rewrites URLs transparently, so you can use SSH even when a project's submodules or scripts use HTTPS URLs:

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

## SSH keys

SSH uses public-key authentication: the server holds your public key and challenges the client to prove possession of the matching private key, which never leaves your machine.

### Generating a key

```bash
# Default choice
ssh-keygen -t ed25519 -C "you@example.com"

# FIDO2 hardware key (OpenSSH 8.2+): the private key never leaves the device
ssh-keygen -t ed25519-sk -O resident -O verify-required -C "you@example.com"

# Only for servers that cannot use Ed25519
ssh-keygen -t rsa -b 4096 -C "you@example.com"
```

With `-sk` key types the file on disk is only a handle; signing requires the physical authenticator and a touch (plus a PIN with `verify-required`). A stolen laptop or a copied `~/.ssh` directory is then not enough to push. `-O resident` stores the handle on the device so `ssh-keygen -K` can recreate it on another machine.

Protect software keys with a passphrase and let `ssh-agent` (or the OS keychain) cache the decrypted key:

```bash
eval "$(ssh-agent -s)"
ssh-add -t 8h ~/.ssh/id_ed25519    # forget it after 8 hours
ssh-add -l                         # list loaded keys
```

### Registering and testing

Upload the public half (`~/.ssh/id_ed25519.pub`) to your account, then test:

```bash
ssh -T git@github.com              # greets you by username
ssh -vT git@github.com             # shows which keys were offered and accepted
```

### Several identities on one host

A host such as `github.com` identifies you by the key you present, so work and personal accounts need separate keys and a way to select them. Define host aliases in `~/.ssh/config`:

```sshconfig
Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
    IdentitiesOnly yes

Host github.com
    IdentityFile ~/.ssh/id_ed25519_personal
    IdentitiesOnly yes
```

Clone work repositories as `git@github-work:org/repo.git`. `IdentitiesOnly yes` stops the client from offering every key the agent holds, which otherwise can pick the wrong account or hit the server's limit on authentication attempts. An alternative that avoids aliases is Git's `core.sshCommand`, set per repository or through a conditional include:

```ini
# ~/.gitconfig
[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work

# ~/.gitconfig-work
[core]
    sshCommand = ssh -i ~/.ssh/id_ed25519_work -o IdentitiesOnly=yes
[user]
    email = you@company.example
```

### Verifying the server

Authentication is mutual: the server presents a host key, and trusting the wrong one allows a man-in-the-middle. On first connection compare the fingerprint with the one the host publishes (GitHub lists its keys in its documentation and at `https://api.github.com/meta`) rather than accepting blindly:

```bash
ssh-keygen -lF github.com          # fingerprint(s) already in known_hosts
```

A "REMOTE HOST IDENTIFICATION HAS CHANGED" warning for an established host deserves investigation. It can be legitimate (GitHub replaced its RSA host key in March 2023 after the private key was briefly exposed), but confirm against the host's announcement before editing `known_hosts`.

Some forges (GitHub Enterprise, GitLab self-managed) also accept **SSH certificates** signed by an organisation's certificate authority, which lets an organisation issue short-lived SSH credentials centrally instead of collecting individual public keys.

## Deploy keys

A deploy key is an SSH public key attached to one repository rather than to a user.

| Property | User SSH key | Deploy key |
|----------|--------------|------------|
| Scope | Every repository the user can access | One repository |
| Access | Follows the user's role | Read-only unless write is explicitly granted |
| Tied to | A person | A repository |
| Reuse | One key, many repositories | A key can be registered on only one repository |
| Leak impact | Everything the user can reach | That repository |

```bash
ssh-keygen -t ed25519 -C "deploy: example-service" -f ~/.ssh/deploy_example -N ""
# Register deploy_example.pub under the repository's Deploy keys;
# enable write access only if the job must push (tags, a gh-pages branch).
```

Unattended keys have no passphrase, so compensate with narrow scope and rotation. When one server needs several repositories, a machine identity (a GitHub App, or GitLab group or project access tokens) scales better than many deploy keys.

## HTTPS, tokens and credential helpers

### Tokens instead of passwords

Major forges no longer accept account passwords for Git over HTTPS; you authenticate with a token that can be scoped, given an expiry, and revoked individually.

On GitHub:

| Token | Scope | Notes |
|-------|-------|-------|
| **Fine-grained personal access token** | One user or organisation; selected repositories; per-permission read or write | Recommended by GitHub where possible. Cannot span multiple organisations or act as an outside collaborator; at most 50 per user. Organisations can require approval. |
| **Classic personal access token** | Coarse scopes such as `repo`, which covers every repository you can access | Needed only for the cases fine-grained tokens do not support. Unused classic tokens are removed automatically after a year. |
| **GitHub App installation token** | The repositories and permissions the App was granted | Expires after one hour; the App's identity, not a person's, appears in audit logs |
| **`GITHUB_TOKEN` in Actions** | The workflow's repository, with permissions set by the workflow's `permissions:` block | Created per job and revoked when the job ends |

GitLab offers the equivalent personal, project and group access tokens plus CI job tokens; Bitbucket uses API tokens and repository or workspace access tokens.

Good practice for any token:

- Grant the minimum permissions and repositories, and set the shortest workable expiry.
- Use one token per consumer so a revocation affects one thing and audit logs show which consumer acted.
- Never put a token in a remote URL (`https://user:TOKEN@host/...`). It is written to `.git/config`, printed by `git remote -v`, and easily ends up in shell history and CI logs. Use a credential helper.

### Credential helpers

A credential helper supplies HTTPS credentials to Git so you are not prompted on every operation.

| Helper | Storage | Notes |
|--------|---------|-------|
| `manager` (Git Credential Manager) | OS keystore | Cross-platform; performs OAuth in the browser, including device-code flow, and refreshes tokens; handles GitHub, GitLab, Bitbucket and Azure Repos |
| `osxkeychain` | macOS Keychain | Ships with Git on macOS |
| `wincred` | Windows Credential Manager | Superseded by GCM, which Git for Windows installs |
| `libsecret` | Secret Service (GNOME Keyring, KWallet) | Linux; often packaged separately |
| `cache` | Memory of a background daemon | Nothing on disk; expires after `--timeout` seconds |
| `store` | Plaintext file `~/.git-credentials` | Avoid: the secret is stored unencrypted |

```bash
git config --global credential.helper manager
git config --global credential.helper 'cache --timeout=3600'

# Different settings per host
git config --global credential.https://git.example.com.username alice
git config --global credential.https://dev.azure.com.useHttpPath true   # one credential per repository path
```

The helper protocol is plain key/value text on standard input and output. Git asks a helper to `get` a credential before a request, then tells it to `store` it after success or `erase` it after rejection. Any program implementing those verbs can act as a helper, including one that mints a short-lived token on demand. Recent Git versions also pass the credential's expiry (`password_expiry_utc`) and an OAuth refresh token (`oauth_refresh_token`) through this protocol, so helpers can discard expired tokens and refresh them. You can see what Git would send with:

```bash
printf 'protocol=https\nhost=github.com\n\n' | git credential fill
```

With GCM or the forge's CLI acting as helper (`gh auth setup-git` configures the GitHub CLI this way), you never handle a long-lived token for interactive work.

## Commit and tag signing

### What a signature covers

A commit object is a short text record: tree hash, parent hash(es), author, committer and message. Signing produces a signature over exactly those bytes and stores it in the commit's `gpgsig` header (`gpgsig-sha256` in SHA-256 repositories). Because the commit includes its tree and parent hashes, and each of those commits to its own contents (the Merkle property described in [Object Model & Storage](object-model.html)), a valid signature fixes the entire tree and history reachable from that commit. Changing any ancestor changes every descendant hash and invalidates the signature. Annotated tags are signed the same way, which is how release tags are usually vouched for.

A signature proves that the key holder signed *these bytes*. It does not prove the key holder wrote the code, and any rewrite (rebase, amend, `filter-repo`, a forge's "rebase and merge" button) produces new commits that are unsigned or signed by whoever rewrote them.

```mermaid
sequenceDiagram
    participant D as Developer
    participant G as git
    participant K as Signer (gpg, ssh-keygen, gitsign)
    participant V as Verifier (forge, CI, git log)
    D->>G: git commit -S
    G->>K: commit bytes without signature
    K-->>G: detached signature
    G->>G: store signature in gpgsig header, hash the object
    V->>G: git verify-commit
    G->>K: bytes + signature + trusted keys
    K-->>V: good / bad / unknown key
```

### Signing formats

| Format (`gpg.format`) | Keys | Trust model | Notes |
|-----------------------|------|-------------|-------|
| `openpgp` (default) | GnuPG keys | Web of trust or keys registered on the forge | Mature; key management is the main burden |
| `ssh` (Git 2.34+) | Any SSH key, including `-sk` hardware keys | An `allowed_signers` file, or keys registered on the forge | Simplest to adopt; the forge must know the key as a *signing* key |
| `x509` | X.509 certificates (S/MIME via `gpgsm`, or Sigstore's `gitsign`) | Certificate authorities | `gitsign` issues short-lived certificates bound to an OIDC identity and logs signatures in the public Rekor transparency log |

### SSH signing

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519_signing.pub
git config --global commit.gpgSign true
git config --global tag.gpgSign true
```

Local verification needs an allowed-signers file mapping identities to keys:

```bash
# ~/.config/git/allowed_signers
you@example.com namespaces="git" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA...
```

```bash
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
```

For a team, a tracked `allowed_signers` file in the repository gives every clone the same verification policy. Forges verify against the signing keys registered on each user's account.

### GPG signing

```bash
gpg --quick-generate-key "Your Name <you@example.com>" ed25519 sign 2y
gpg --list-secret-keys --keyid-format=long      # find the key ID
git config --global user.signingkey 3AA5C34371567BD2
git config --global commit.gpgSign true
gpg --armor --export 3AA5C34371567BD2           # public key to upload to the forge
```

The email in the key's user ID must match the commit's committer email and a verified email on the forge account for the commit to be shown as verified. GnuPG 2.1 and later write a revocation certificate to `~/.gnupg/openpgp-revocs.d/` when a key is created; keep a copy offline.

### Sigstore gitsign

`gitsign` removes long-lived signing keys entirely: at signing time it opens an OIDC login (or uses a CI workload identity), obtains a certificate valid for minutes from the Fulcio CA, signs, and records the signature in the Rekor transparency log.

```bash
git config --local gpg.format x509
git config --local gpg.x509.program gitsign
git config --local commit.gpgSign true
```

Verification checks the identity and issuer embedded in the certificate (`gitsign verify --certificate-identity=... --certificate-oidc-issuer=...`) rather than a key. Forge support for displaying such commits as verified varies.

### Separate authentication and signing keys

An authentication key is used against many servers and may be forwarded through agents; a signing key makes durable statements that others rely on long after the fact. Keeping them distinct means one can be rotated without the other, and a forwarded authentication key cannot be used to sign. Forges register the two roles separately even when the key bytes are the same. Hardware-backed keys are well suited to signing.

### Verifying signatures

```bash
git log --show-signature -3
git verify-commit HEAD
git verify-tag v1.2.0
git log --format='%h %G? %GS %s'
```

| `%G?` | Meaning |
|:-----:|---------|
| `G` | Good signature from a trusted key |
| `U` | Good signature, key of unknown or undefined trust |
| `X` | Good signature that has expired |
| `Y` | Good signature made by a key that has since expired |
| `R` | Good signature made by a key that has since been revoked |
| `B` | Bad signature: the content does not match |
| `E` | Cannot be checked, for example the key is missing |
| `N` | No signature |

### Enforcing signatures

Local configuration is advisory. Enforcement belongs on the server: forge rulesets and branch protection can require signed commits, and GitHub's *vigilant mode* marks a user's unsigned commits as unverified. A CI check adds defence in depth:

```bash
# Fail if any commit on this branch is unsigned or has a bad signature
set -eu
for c in $(git rev-list origin/main..HEAD); do
  case "$(git log -1 --format=%G? "$c")" in
    G) ;;                                   # accept only fully trusted signatures
    *) echo "Unverified commit: $c" >&2; exit 1 ;;
  esac
done
```

For this to pass with SSH signing, the CI job must configure `gpg.ssh.allowedSignersFile` with the team's keys; otherwise every signature is reported as unverifiable.

## Authorization on the server

### Organisation identity: SSO

Enterprises connect the forge to an identity provider (Okta, Microsoft Entra ID, Google Workspace) with **SAML** or **OpenID Connect**, making it the single source of truth for who exists, which groups they belong to, and when access ends. Offboarding a person in the identity provider then removes their forge access.

SSO governs web sessions, but Git operations use keys and tokens. Under GitHub's SAML enforcement, an existing personal access token or SSH key must be explicitly authorized for the organisation before it works against that organisation's repositories; until then a push or fetch fails with a message saying the organisation has enabled or enforced SAML SSO. The fix is to open the token or key in account settings and authorize it for the organisation. With Enterprise Managed Users, accounts are created and controlled by the identity provider itself.

### OAuth and App grants

Third-party tools (CI services, review bots, editor integrations) receive access through an OAuth consent screen or by being installed as an App. Grant the narrowest scopes offered, review grants periodically in account and organisation settings, and treat an integration asking for organisation-admin or all-repository write access without an obvious need as a risk.

### Repository and ref rules

Independently of how a user signed in, the forge enforces:

- **Roles and teams**, for example read, triage, write, maintain and admin on GitHub, or guest through owner on GitLab.
- **Branch protection and rulesets**: required pull requests and reviews, required status checks, required signed commits, linear history, and blocking force-pushes and deletions on protected branches and tags.
- **CODEOWNERS**: required review from the owners of the paths a change touches.
- **Push checks**: secret-scanning push protection, file-size and path restrictions, and on self-hosted servers `pre-receive` and `update` hooks (see [Hooks](algorithms-and-operations.html#hooks)).

## Git's own local trust boundaries

Although Git has no access control, it does decide which repositories it trusts on the local machine, because a repository's configuration and hooks can run arbitrary commands.

| Setting | Default | Purpose |
|---------|---------|---------|
| `safe.directory` | Only repositories owned by the current user | Refuses to operate on a repository owned by another user (for example in a shared `/tmp` or a mounted volume), since its config could run code as you. Add trusted paths explicitly; container builds often need `git config --global --add safe.directory /workspace`. |
| `safe.bareRepository` | `all` (planned to become `explicit` in Git 3.0) | With `explicit`, Git ignores bare repositories it discovers inside a working tree, closing an attack where a cloned project embeds a bare repository with malicious hooks. |
| `protocol.allow`, `protocol.<name>.allow` | `file` is limited to user-initiated operations | Restricts which transports submodules and other indirect operations may use. |
| `transfer.fsckObjects` | `false` | Validate every object received on fetch and push. |

Hooks in `.git/hooks` are never cloned, but project-provided tooling such as pre-commit configurations runs code from the repository once installed, so install it only for repositories you trust.

## Revocation, rotation and audit

Assume every credential will eventually leak and plan for fast revocation, routine rotation and useful audit trails.

| Credential | Revoke by | Effect |
|------------|-----------|--------|
| User SSH key | Deleting it from the account | Every connection using it fails immediately |
| Deploy key | Removing it from the repository | That repository only |
| Personal access token | Revoking it | That token only |
| OAuth grant | Revoking the application's authorization | That application's tokens |
| App | Rotating its private key or uninstalling it | All tokens it could mint |
| GPG signing key | Publishing the revocation certificate | Signatures verify as `R` (revoked key) |
| SSH signing key | Removing it from the account and `allowed_signers`, or marking it revoked (`gpg.ssh.revocationFile`) | Signatures no longer verify |

**Rotation.** Prefer credentials that expire on their own (fine-grained tokens, App and job tokens). For long-lived keys, rotate without downtime by adding the new key, switching the consumer, then removing the old one.

**Audit.** Forge audit logs record key and token creation, SSO events, permission changes and pushes, including force-pushes; stream them to a SIEM. Signatures provide a separate, cryptographic record of authorship. Reflogs of a local or mirror clone help reconstruct what a force-push replaced.

## Leaked secrets

The most common Git security incident is not a broken credential but a secret (an API key, a `.env` file, a private key) committed to a repository and pushed, after which it lives in history and possibly on a public server.

### Prevention

```gitignore
.env
.env.*
!.env.example
*.pem
*.key
*.p12
id_rsa
id_ed25519
credentials.json
**/secrets/
```

- Keep secrets in environment variables, a secrets manager (Vault, a cloud secrets service) or the CI system's secret store; see [Cybersecurity](../cybersecurity/).
- Scan staged changes before each commit. With the pre-commit framework, add the `gitleaks` hook; as a plain hook (gitleaks 8.19 and later replaced `protect --staged` with `git --pre-commit --staged`):

```bash
#!/bin/sh
# .git/hooks/pre-commit
exec gitleaks git --pre-commit --staged --redact --verbose
```

- Turn on server-side secret scanning and **push protection**, which rejects pushes containing recognisable credentials. GitHub enables push protection by default for public repositories; other forges offer equivalents.

### Response

A secret must be considered compromised from the moment it was pushed; automated scanners harvest public commits within minutes.

```mermaid
flowchart LR
    A["1. Revoke and rotate<br/>the secret at its source"] --> B["2. Check audit logs<br/>for use during exposure"]
    B --> C["3. Purge from history<br/>git filter-repo"]
    C --> D["4. Force-push, have<br/>collaborators re-clone"]
    D --> E["5. Ask the forge to purge<br/>cached views if needed"]
```

1. **Revoke and rotate first.** Removing a secret from history does nothing about copies already taken. Invalidate it at the issuing service.
2. **Audit** the issuing service's logs for use during the exposure window.
3. **Purge it from history** with `git filter-repo`, the tool the Git project recommends in place of `git filter-branch`:

```bash
git filter-repo --invert-paths --path config/secrets.yml   # remove a file from all history
git filter-repo --replace-text expressions.txt             # redact matching strings
git remote add origin <url>                                # filter-repo removes the remote as a safety measure
git push origin --force --all && git push origin --force --tags
```

4. **Coordinate.** Every commit hash after the first affected commit changes. Collaborators must re-clone or hard-reset, and forks, open pull requests and existing clones retain the old objects.
5. **Clean up the host.** Forges may keep unreferenced commits reachable by SHA and in pull-request views; GitHub's documentation on removing sensitive data describes contacting support to purge them.

Revocation is immediate and complete; history rewriting is slow, disruptive and never reaches every copy. Treat the purge as cleanup, not as the fix.

## See also

- [Object Model & Storage](object-model.html): why one signature covers all reachable history
- [Algorithms & Advanced Operations](algorithms-and-operations.html#hooks): client and server hooks
- [Protocols, Packs & Performance](protocols-and-performance.html): the SSH and HTTPS wire protocol these credentials authenticate
- [Git Command Reference](../git-reference.html#signing-commits-and-tags): signing and [removing sensitive data](../git-reference.html#removing-sensitive-data) command syntax
- [Branching Strategies](../branching.html): branch protection and review in team workflows
- [Cybersecurity](../cybersecurity/): secrets management and defence in depth
- [CI/CD](../ci-cd/): short-lived credentials in pipelines

**Previous:** [Conflict Resolution & Recovery](conflict-and-recovery.html) · **Back to** [Git Internals](./)
