---
layout: docs
title: "Cybersecurity: Cloud & Container Security"
permalink: /docs/technology/cybersecurity/cloud-and-container-security.html
toc: true
toc_sticky: true
hide_title: true
---

<p class="breadcrumb"><a href="./">Cybersecurity</a> › Cloud &amp; Container Security</p>

# Cloud & Container Security

The cloud revolutionized how we build and deploy applications, but it moved the most common breach vector from the network perimeter to identity and configuration. This page covers the shared-responsibility model, cloud IAM and least privilege, cloud security posture management (CSPM), container image hardening, container and Kubernetes runtime security (Falco, AppArmor, SELinux, seccomp), and admission-control policy engines (OPA, Kyverno).

## Cloud Security: New Challenges, New Solutions

The cloud revolutionized how we build and deploy applications, but it also introduced new security challenges. You're no longer protecting a physical server in your data center—you're securing resources that exist "somewhere" in someone else's infrastructure.

The headline statistic of the cloud era is that the overwhelming majority of cloud breaches are caused not by sophisticated zero-days but by **customer misconfiguration**: a storage bucket left public, an over-permissive IAM role, a security group open to `0.0.0.0/0`. The cloud provider's infrastructure is rarely the weak link — your configuration of it is. Everything below follows from that observation.

### The Shared Responsibility Model

Understanding who secures what is crucial. Cloud providers operate a **shared responsibility model**: they secure the cloud itself, while you secure what you put *in* the cloud. The exact dividing line shifts depending on the service model (IaaS, PaaS, SaaS), and misunderstanding where it falls is itself a leading cause of breaches.

```python
# Cloud Provider Secures ("security OF the cloud"):
# - Physical data centers and badge access
# - Network infrastructure and backbone
# - Hypervisor / virtualization layer
# - Physical storage media and disposal

# You Secure ("security IN the cloud"):
# - Your data (classification, encryption, retention)
# - Identity and access management (who can do what)
# - Application code and dependencies
# - Operating system + patches (in IaaS)
# - Network traffic controls (security groups, NACLs)
# - Encryption keys and secrets management
```

The boundary moves with the service model:

| Layer | On-Prem | IaaS (EC2/VMs) | PaaS (App Engine, RDS) | SaaS (Workspace, M365) |
|-------|---------|----------------|------------------------|------------------------|
| Data & access | You | You | You | You |
| Application | You | You | You | Provider |
| Runtime / middleware | You | You | Provider | Provider |
| OS & patching | You | **You** | Provider | Provider |
| Virtualization | You | Provider | Provider | Provider |
| Physical | You | Provider | Provider | Provider |

The two rows that *never* shift to the provider — **data** and **identity/access** — are exactly where most real-world incidents occur. No matter how much you outsource, you always own who can reach your data.

### IAM: The Keys to Your Kingdom

In the cloud, Identity and Access Management (IAM) is your most critical security control. A misconfigured IAM policy can expose your entire infrastructure. Where a traditional network had a firewall as its perimeter, a cloud account's perimeter *is* its identity layer — an attacker with valid credentials walks straight in, no exploit required.

**The principle of least privilege** is the foundation: every identity (user, role, service account, workload) should hold the *minimum* permissions needed to do its job, and nothing more. In practice this means starting from zero and granting narrowly, rather than starting broad and trying to claw permissions back.

**Modern IAM Practices**:
- **Zero Trust Architecture**: Never trust, always verify (covered in full under [Attacks & Network Defense](attacks-and-defense.html#zero-trust-never-trust-always-verify))
- **Just-In-Time (JIT) Access**: Temporary elevated privileges granted on request and automatically revoked, so no standing admin access exists to steal
- **Passwordless Authentication**: Passkeys and FIDO2 standards remove phishable shared secrets
- **Policy Intelligence**: Provider tooling (AWS IAM Access Analyzer, GCP Policy Analyzer) that flags unused permissions and recommends tightening
- **CNAPP**: Cloud-Native Application Protection Platforms that unify posture, identity, and workload security

The classic worked example is an S3 bucket policy. The first version below is the single most common cloud misconfiguration in the world; the second shows what least privilege actually looks like:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",      // DANGER: Anyone on the internet can access!
    "Action": "s3:*",       // DANGER: All permissions, including delete!
    "Resource": "arn:aws:s3:::my-bucket/*"
  }]
}
```

```json
// Secure version: a specific principal, one action, scoped resource,
// and a network condition.
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

**Roles over long-lived keys.** The most resilient pattern is to avoid static credentials entirely. Instead of embedding an access key in an application, attach a *role* to the compute (an EC2 instance profile, an EKS IRSA service account, a GCP workload identity). The platform mints short-lived credentials automatically, so there is no long-lived secret to leak. A leaked role session token expires in minutes; a leaked access key lives until someone notices.

**Permission boundaries and SCPs.** Even least-privilege grants can drift. Guardrails like AWS *permission boundaries* (a ceiling on what a role can be granted) and *Service Control Policies* (account-wide deny rules across an organization) ensure that no individual misconfiguration can exceed an organizational limit — for example, "no one, ever, can disable CloudTrail" or "no resources outside approved regions."

### Cloud Security Posture Management (CSPM)

Least privilege is a goal; **posture management** is how you continuously verify you are still meeting it. A cloud account is not a static artifact — engineers create resources hourly, and any one of them can introduce a public bucket, an unencrypted database, or an open security group. CSPM tools (AWS Security Hub, GCP Security Command Center, Wiz, Prisma Cloud, the open-source Prowler/ScoutSuite) continuously scan your accounts and compare the live configuration against a baseline of best practices and compliance frameworks (CIS Benchmarks, PCI-DSS, SOC 2).

CSPM answers questions like:
- Which storage buckets are publicly readable or writable?
- Which IAM roles grant `*:*` or have not been used in 90 days?
- Which databases or volumes are unencrypted at rest?
- Which security groups expose SSH (22) or RDP (3389) to the entire internet?
- Is audit logging (CloudTrail / Cloud Audit Logs) enabled and immutable?

The strongest posture management is **preventive**, not just detective. Detecting a public bucket after it has been created still leaves a window of exposure. Embedding policy checks into the deployment pipeline — scanning Terraform/CloudFormation with tools like `checkov`, `tfsec`, or `terrascan` *before* `apply` — stops the misconfiguration from ever reaching production:

```bash
# Shift posture checks left: scan infrastructure-as-code in CI before deploy
checkov -d ./terraform --quiet --compact
# Example finding:
#   CKV_AWS_18: "Ensure the S3 bucket has access logging enabled"
#   CKV_AWS_53: "Ensure S3 bucket has block public ACLs enabled"
# Fail the pipeline so the misconfiguration never reaches the cloud.
```

This is the same "policy as code" philosophy that, as we will see, governs Kubernetes admission — catch the bad configuration at the gate rather than chasing it at runtime.

## Container Security: Shipping Code Safely

Containers add another layer of complexity. You're not just securing an application—you're securing the entire environment it runs in: the base image, every OS package and language dependency baked into it, the runtime that executes it, and the orchestrator that schedules it. Security spans the whole lifecycle, conventionally split into **build-time** (what goes into the image) and **runtime** (what the container is allowed to do once running).

### Image Hardening: Shrinking the Attack Surface

Every binary, library, and tool inside an image is a potential vulnerability and a potential tool for an attacker who gets a shell. The guiding principle is *minimalism*: include only what the application needs to run.

**Container Security Best Practices**:
- **Run as non-root**: A container process running as root that escapes the container is root on the host. Always create and switch to an unprivileged user.
- **Pin specific versions**: `:latest` is non-reproducible and silently pulls in new, unvetted packages. Pin a tag or, better, a digest.
- **Minimal / distroless base images**: Distroless and `scratch` images contain no shell, no package manager, no `curl` — dramatically reducing what an attacker can do after a breakout.
- **Software Bill of Materials (SBOM)**: A machine-readable inventory of every component in the image (generated by `syft`, `trivy`), increasingly required by regulation and essential for responding to the next Log4Shell-style disclosure.
- **Image scanning**: Tools like `trivy`, `grype`, and Clair scan images for known CVEs in OS packages and language dependencies, run in CI to block vulnerable images.
- **Sigstore / image signing**: Sign images (`cosign`) and verify the signature at deploy time so only images your pipeline actually built can run.

```dockerfile
# Insecure Dockerfile
FROM ubuntu:latest          # Non-reproducible "latest" tag
USER root                    # Running as root!
RUN apt-get update && apt-get install -y curl
COPY app /app
CMD ["/app"]

# Secure Dockerfile
FROM ubuntu:22.04           # Specific version
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*  # Clean up package metadata
RUN useradd -m appuser      # Create non-root user
USER appuser                # Switch to non-root
COPY --chown=appuser:appuser app /app
CMD ["/app"]
```

For compiled languages, a **multi-stage build** lets you compile in a full toolchain image and then copy only the resulting binary into a distroless or `scratch` final image, so the build tools never ship to production:

```dockerfile
# Stage 1: build with the full toolchain
FROM golang:1.22 AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -o /app ./cmd/server

# Stage 2: ship only the static binary on a distroless base
FROM gcr.io/distroless/static:nonroot
COPY --from=build /app /app
USER nonroot:nonroot        # distroless "nonroot" runs as UID 65532
ENTRYPOINT ["/app"]
# No shell, no apt, no curl — almost nothing for an attacker to use.
```

```bash
# Scan the finished image before it can be deployed
trivy image --severity HIGH,CRITICAL --exit-code 1 myregistry/app:1.4.2
# Generate an SBOM for incident response and supply-chain tracking
syft myregistry/app:1.4.2 -o spdx-json > app.sbom.json
# Sign and later verify, so only pipeline-built images run
cosign sign myregistry/app:1.4.2
cosign verify --key cosign.pub myregistry/app:1.4.2
```

### Runtime Security: Falco, seccomp, AppArmor & SELinux

Hardening the image reduces what *can* go wrong; runtime security controls what a container is *allowed* to do once it is running, and detects when it does something it shouldn't. These two halves — *confinement* (Linux kernel controls) and *detection* (behavioral monitoring) — work together.

**Linux confinement primitives** shrink the container's slice of the kernel:

- **seccomp** (secure computing mode) filters which **syscalls** a process may make. The default Docker/Kubernetes seccomp profile blocks roughly 40+ dangerous syscalls (such as `mount`, `ptrace`, `keyctl`). A tightly scoped custom profile can allow only the handful of syscalls an application actually uses, so a breakout cannot invoke the rest of the kernel API.
- **AppArmor** is a *path-based* Mandatory Access Control system (default on Debian/Ubuntu). A profile declares exactly which files a container may read or write and which capabilities it holds — e.g., deny all writes outside `/tmp` and `/app/data`.
- **SELinux** is a *label-based* MAC system (default on RHEL/Fedora). Every process and file carries a security label, and policy governs which label may access which — the model behind container isolation on OpenShift. It is more granular and more complex than AppArmor but enforces the same idea: confine the container to a least-privilege policy independent of Unix file permissions.
- **Linux capabilities**: Rather than the all-or-nothing root, drop every capability and add back only what is needed (`--cap-drop=ALL --cap-add=NET_BIND_SERVICE`).

```yaml
# Kubernetes securityContext applying several runtime controls at once
securityContext:
  runAsNonRoot: true
  runAsUser: 65532
  allowPrivilegeEscalation: false   # block setuid escalation
  readOnlyRootFilesystem: true      # container cannot modify its own image
  capabilities:
    drop: ["ALL"]                   # start from zero
    add: ["NET_BIND_SERVICE"]       # add back only what is needed
  seccompProfile:
    type: RuntimeDefault            # apply the default syscall filter
```

**Behavioral detection** watches what containers actually do at runtime. **Falco** (a CNCF project, the de-facto standard) taps the kernel via eBPF to observe syscalls and raises alerts when behavior violates a rule — for example, "a shell was spawned inside a container," "a sensitive file like `/etc/shadow` was read," or "an unexpected outbound network connection was made." Where seccomp/AppArmor/SELinux *prevent* actions, Falco *detects and alerts* on the actions that policy did not anticipate:

```yaml
# Falco rule: alert when an interactive shell starts inside a container
- rule: Terminal shell in container
  desc: A shell was spawned in a container — often a sign of an intrusion
  condition: >
    container.id != host and proc.name in (bash, sh, zsh)
    and proc.tty != 0
  output: >
    Shell opened in container (user=%user.name container=%container.name
    command=%proc.cmdline)
  priority: WARNING
```

Together these form defense in depth: image hardening removes the tools, capabilities/seccomp/AppArmor/SELinux remove the privileges, and Falco catches whatever still slips through.

## Kubernetes Security

Kubernetes magnifies every container concern because it adds a powerful control plane — the API server — that schedules workloads, holds secrets, and grants access across the whole cluster. Securing Kubernetes means securing both the workloads *and* the orchestrator that commands them. The major control surfaces are:

- **RBAC (Role-Based Access Control)**: The same least-privilege discipline as cloud IAM, applied to the Kubernetes API. Bind `Roles`/`ClusterRoles` narrowly; never grant `cluster-admin` to application service accounts. Disable automatic mounting of the service-account token in pods that do not call the API.
- **Network Policies**: By default every pod can talk to every other pod. A `NetworkPolicy` enforces a default-deny posture and whitelists only the connections each workload legitimately needs — micro-segmentation inside the cluster.
- **Secrets management**: Kubernetes `Secret` objects are only base64-encoded, not encrypted, by default. Enable encryption-at-rest for etcd and prefer an external store (Vault, cloud secret managers, External Secrets Operator) so secrets are never sitting in plaintext in etcd.
- **Pod Security Standards**: The successor to the deprecated PodSecurityPolicy. The built-in admission controller enforces `baseline` or `restricted` profiles per namespace — for instance, the `restricted` profile forbids privileged pods, host networking, and running as root.
- **Securing the control plane**: Restrict access to the API server and etcd, rotate certificates, and audit-log every API request.

### Admission Control & Policy as Code (OPA & Kyverno)

The control point that ties cloud posture and Kubernetes security together is the **admission controller** — a webhook the API server consults *before* persisting any object. Admission control is the cluster's gate: it can reject or mutate a resource at creation time, which is exactly the "prevent, don't just detect" principle of CSPM applied to Kubernetes. This is **policy as code**: rules live in version control, are reviewed like any other change, and are enforced automatically.

Two policy engines dominate:

- **OPA / Gatekeeper**: The **Open Policy Agent** is a general-purpose policy engine; **Gatekeeper** is its Kubernetes admission integration. Policies are written in OPA's declarative **Rego** language and can express arbitrarily complex logic. OPA is widely used beyond Kubernetes too (API authorization, Terraform validation), so the same engine governs many layers.
- **Kyverno**: A Kubernetes-native engine whose policies are written as ordinary Kubernetes YAML resources — no new language to learn. Kyverno can `validate` (reject), `mutate` (inject defaults like a `securityContext`), and `generate` (auto-create resources such as default NetworkPolicies in new namespaces).

A representative Kyverno policy that blocks privileged containers cluster-wide:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-privileged-containers
spec:
  validationFailureAction: Enforce   # reject violating pods at admission
  rules:
    - name: privileged-containers
      match:
        any:
          - resources:
              kinds: ["Pod"]
      validate:
        message: "Privileged containers are not allowed."
        pattern:
          spec:
            containers:
              - =(securityContext):
                  =(privileged): "false"
```

With this policy admitted, any attempt to create a privileged pod — whether by an honest mistake or a compromised CI account — is rejected by the API server before it ever runs. Combined with image scanning and signing in CI, runtime confinement (seccomp/AppArmor/SELinux), and Falco detection, the cluster has controls at every stage: what you build, what you admit, what you run, and what you observe.

---

<div class="page-nav">
  <span class="page-nav-prev"><a href="application-and-cloud-security.html">← Web Application Security</a></span>
  <span class="page-nav-next"><a href="attacks-and-defense.html">Attacks &amp; Network Defense →</a></span>
</div>

## See Also

- [Web Application Security](application-and-cloud-security.html) — injection, XSS, authentication, and JWTs in the app layer
- [Cryptography](cryptography.html) — the encryption and signing that underpins image signing and secrets-at-rest
- [Attacks & Network Defense](attacks-and-defense.html) — firewalls, VPNs, supply-chain attacks, and Zero Trust in full
- [Operations & Incident Response](operations-and-response.html) — detecting and responding when a runtime alert fires
- [AWS](../aws/) — cloud IAM and the shared-responsibility model in practice
- [Docker](../docker/) — container fundamentals and image building
- [Kubernetes](../kubernetes/) — orchestrating and securing containers at scale
