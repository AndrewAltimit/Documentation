---
layout: docs
title: "Docker: Registries & Supply-Chain Security"
permalink: /docs/technology/docker/registry.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #00aaff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Docker: Registries &amp; Supply-Chain Security</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Distribute images through registries, pin them by digest, and prove what you ship with signing, SBOMs, scanning, and provenance.</p>
</div>

[Docker](./) &raquo; Registries &amp; Supply-Chain Security

A registry is where images live between `docker build` and `docker run`. Everything downstream — what runs in production, what a vulnerability scanner reports, what an auditor trusts — depends on the integrity of that distribution step. This page covers how registries store and address images, how tags differ from immutable digests, and the modern supply-chain controls (signing, SBOMs, scanning, and provenance) that let you prove an image is the one you built from source you control.

## How a Registry Works

A registry is an HTTP service that implements the [OCI Distribution Specification](https://github.com/opencontainers/distribution-spec) (the open successor to the Docker Registry HTTP API v2). When you `docker push`, the client uploads two kinds of objects:

- **Blobs** — the compressed layer tarballs and the image config JSON. Each is content-addressed by the SHA-256 digest of its bytes.
- **Manifests** — small JSON documents that list the config blob and the ordered layer blobs (by digest and size) that make up an image. Manifests are themselves content-addressed.

A **tag** is just a mutable human-friendly name (a pointer) stored in the registry that resolves to a manifest digest. A **multi-arch image** uses a *manifest list* (also called an image index): one top-level manifest whose entries are per-platform manifests (`linux/amd64`, `linux/arm64`, …). The client picks the entry matching its platform.

<div class="docker-section">
  <div class="docker-intro">
    <p>Pushing and pulling both reduce to "exchange blobs and manifests by digest." Because every object is named by the hash of its content, the registry can deduplicate shared layers across images and the client can verify integrity end-to-end.</p>

    <div class="docker-workflow">
      <h3><i class="fas fa-sitemap"></i> Registry Object Model</h3>
      <svg viewBox="0 0 700 260" class="workflow-diagram">
        <!-- Tag -->
        <rect x="30" y="20" width="150" height="50" fill="#3498db" opacity="0.3" stroke="#2980b9" stroke-width="2" />
        <text x="105" y="42" text-anchor="middle" font-size="12" font-weight="bold">Tag</text>
        <text x="105" y="58" text-anchor="middle" font-size="10">myapp:1.4.2 (mutable)</text>

        <path d="M 105 70 L 105 100" stroke="#34495e" stroke-width="2" marker-end="url(#arrow)" />
        <text x="170" y="90" text-anchor="middle" font-size="9">resolves to</text>

        <!-- Manifest list -->
        <rect x="20" y="100" width="170" height="60" fill="#9b59b6" opacity="0.25" stroke="#8e44ad" stroke-width="2" />
        <text x="105" y="122" text-anchor="middle" font-size="11" font-weight="bold">Manifest List</text>
        <text x="105" y="138" text-anchor="middle" font-size="9">sha256:a1b2... (immutable)</text>
        <text x="105" y="151" text-anchor="middle" font-size="8">amd64 + arm64 entries</text>

        <!-- Platform manifests -->
        <rect x="250" y="40" width="150" height="50" fill="#e67e22" opacity="0.25" stroke="#d35400" stroke-width="2" />
        <text x="325" y="60" text-anchor="middle" font-size="10" font-weight="bold">Manifest (amd64)</text>
        <text x="325" y="76" text-anchor="middle" font-size="8">config + layer digests</text>

        <rect x="250" y="120" width="150" height="50" fill="#e67e22" opacity="0.25" stroke="#d35400" stroke-width="2" />
        <text x="325" y="140" text-anchor="middle" font-size="10" font-weight="bold">Manifest (arm64)</text>
        <text x="325" y="156" text-anchor="middle" font-size="8">config + layer digests</text>

        <path d="M 190 125 L 248 70" stroke="#34495e" stroke-width="1.5" marker-end="url(#arrow)" />
        <path d="M 190 135 L 248 140" stroke="#34495e" stroke-width="1.5" marker-end="url(#arrow)" />

        <!-- Blobs -->
        <rect x="470" y="30" width="200" height="200" fill="#27ae60" opacity="0.15" stroke="#229954" stroke-width="2" />
        <text x="570" y="50" text-anchor="middle" font-size="11" font-weight="bold">Content-Addressed Blobs</text>
        <rect x="490" y="65" width="160" height="26" fill="#229954" opacity="0.5" />
        <text x="570" y="82" text-anchor="middle" font-size="8" fill="white">config json (sha256:...)</text>
        <rect x="490" y="100" width="160" height="26" fill="#229954" opacity="0.5" />
        <text x="570" y="117" text-anchor="middle" font-size="8" fill="white">layer 1 (sha256:...)</text>
        <rect x="490" y="135" width="160" height="26" fill="#229954" opacity="0.5" />
        <text x="570" y="152" text-anchor="middle" font-size="8" fill="white">layer 2 (shared, dedup)</text>
        <rect x="490" y="170" width="160" height="26" fill="#229954" opacity="0.5" />
        <text x="570" y="187" text-anchor="middle" font-size="8" fill="white">layer 3 (sha256:...)</text>

        <path d="M 400 65 L 468 100" stroke="#34495e" stroke-width="1.5" marker-end="url(#arrow)" />
        <path d="M 400 145 L 468 140" stroke="#34495e" stroke-width="1.5" marker-end="url(#arrow)" />
      </svg>
    </div>
  </div>
</div>

The digest you see in `docker images --digests` or `docker buildx imagetools inspect` is the SHA-256 of the manifest bytes. That single hash transitively pins the config and every layer, because the manifest references them by their own digests. This is the property that makes digests trustworthy: change anything inside the image and the top-level digest changes.

### Image References Anatomy

```
ghcr.io/acme/web-api:1.4.2@sha256:9f86d08...
└──┬──┘ └───┬────┘ └─┬─┘ └────────┬────────┘
 host    repository  tag        digest
```

| Part | Example | Notes |
|------|---------|-------|
| Host (registry) | `ghcr.io` | Omitted defaults to Docker Hub (`registry-1.docker.io`) |
| Repository (namespace + name) | `acme/web-api` | On Docker Hub, a bare name like `nginx` means `library/nginx` |
| Tag | `1.4.2` | Mutable pointer; defaults to `latest` if omitted |
| Digest | `sha256:9f86d08...` | Immutable; when present, overrides the tag |

When both a tag and a digest are present, the digest wins — the tag is informational. Pulling `image:tag@sha256:...` fetches exactly that manifest and errors if the tag no longer points there.

## Choosing a Registry

Most teams use one of a few registries. They all speak the OCI distribution protocol, so `docker`, `buildx`, `crane`, and `cosign` work against any of them; they differ in hosting model, authentication, and integration.

<div class="storage-section">
  <h3><i class="fas fa-warehouse"></i> Registry Options</h3>

  <div class="storage-comparison">
    <table class="storage-table">
      <thead>
        <tr>
          <th>Registry</th>
          <th>Hosting</th>
          <th>Auth Model</th>
          <th>Best For</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Docker Hub</strong></td>
          <td>SaaS (Docker, Inc.)</td>
          <td>Docker ID / PAT</td>
          <td>Public images, official base images, getting started</td>
        </tr>
        <tr>
          <td><strong>GitHub Container Registry (GHCR)</strong></td>
          <td>SaaS (GitHub)</td>
          <td>GitHub token / PAT</td>
          <td>Repos already on GitHub; tight Actions integration</td>
        </tr>
        <tr>
          <td><strong>Amazon ECR</strong></td>
          <td>SaaS (AWS)</td>
          <td>IAM (short-lived tokens)</td>
          <td>Workloads on ECS/EKS; IAM-native access control</td>
        </tr>
        <tr>
          <td><strong>Google Artifact Registry (GAR/GCR)</strong></td>
          <td>SaaS (Google Cloud)</td>
          <td>IAM / service accounts</td>
          <td>GKE and Cloud Run deployments</td>
        </tr>
        <tr>
          <td><strong>Harbor</strong></td>
          <td>Self-hosted (CNCF)</td>
          <td>Local / OIDC / LDAP</td>
          <td>On-prem, air-gapped, policy-driven enterprises</td>
        </tr>
        <tr>
          <td><strong>Azure Container Registry (ACR)</strong></td>
          <td>SaaS (Azure)</td>
          <td>Entra ID / tokens</td>
          <td>AKS and Azure-native pipelines</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

### Authenticating and Pushing

The mechanics are identical across providers — only the login step differs because each has its own credential source.

```bash
# Docker Hub (default registry; tag with your namespace)
docker login -u myuser
docker tag myapp:1.4.2 myuser/myapp:1.4.2
docker push myuser/myapp:1.4.2

# GitHub Container Registry (use a PAT or the Actions GITHUB_TOKEN)
echo "$CR_PAT" | docker login ghcr.io -u my-gh-user --password-stdin
docker tag myapp:1.4.2 ghcr.io/acme/myapp:1.4.2
docker push ghcr.io/acme/myapp:1.4.2

# Amazon ECR (short-lived token from the AWS CLI, repo must exist)
aws ecr get-login-password --region us-east-1 \
  | docker login --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:1.4.2

# Google Artifact Registry (helper wires gcloud creds into the Docker client)
gcloud auth configure-docker us-docker.pkg.dev
docker push us-docker.pkg.dev/my-project/my-repo/myapp:1.4.2
```

ECR tokens expire after 12 hours, which is why CI authenticates on every run rather than storing a static password. GHCR inside GitHub Actions can authenticate with the automatically provisioned `GITHUB_TOKEN`, scoped to the repository — no long-lived secret required.

<div class="security-section">
  <div class="security-principles">
    <div class="principle-grid">
      <div class="principle-card">
        <i class="fas fa-server"></i>
        <h5>Self-Host vs SaaS</h5>
        <p>Harbor gives you on-prem control, replication, and air-gap support; managed registries remove the ops burden.</p>
      </div>
      <div class="principle-card">
        <i class="fas fa-id-badge"></i>
        <h5>Identity-Native Auth</h5>
        <p>Cloud registries fold into IAM, so access follows your existing roles instead of separate registry credentials.</p>
      </div>
      <div class="principle-card">
        <i class="fas fa-copy"></i>
        <h5>Replication &amp; Caching</h5>
        <p>Harbor and pull-through caches mirror upstream images to cut egress and survive upstream outages.</p>
      </div>
      <div class="principle-card">
        <i class="fas fa-trash-alt"></i>
        <h5>Retention &amp; GC</h5>
        <p>Tag retention rules plus garbage collection reclaim space from untagged, content-addressed blobs.</p>
      </div>
    </div>
  </div>
</div>

### Harbor: a Self-Hosted Registry

[Harbor](https://goharbor.io/) is the CNCF-graduated open-source registry. Beyond storing images it adds project-level RBAC, built-in Trivy scanning, image signing enforcement, replication between Harbor instances (and to/from Docker Hub or ECR), and a pull-through proxy cache. It is the common choice for air-gapped or compliance-driven environments where images cannot leave the perimeter. You point the Docker client at it exactly like any other registry:

```bash
docker login harbor.internal.acme.com
docker push harbor.internal.acme.com/platform/myapp:1.4.2
```

## Tagging Strategy and Immutability

Tags are mutable by default, and that mutability is the single largest source of "works in CI, breaks in prod" incidents. Two builds can both be tagged `latest`; whichever pushed last wins, and a pull tomorrow may silently get a different image than a pull today.

### Why `latest` Is a Trap

`latest` is not special to the registry — it is just the tag applied when you omit one. It carries no guarantee of being the newest or most stable build. Deploying `myapp:latest` means your running version depends on push timing and pull caching, which is irreproducible. **Pin deployments to a specific tag, and verify with a digest.**

### A Workable Tagging Scheme

Apply *multiple* tags to the same image so humans and machines each get what they need:

```bash
# One build, several tags pointing at the same digest
docker build -t myapp:1.4.2 .
docker tag myapp:1.4.2 myapp:1.4
docker tag myapp:1.4.2 myapp:1
docker tag myapp:1.4.2 myapp:sha-$(git rev-parse --short HEAD)
docker push --all-tags myapp
```

| Tag style | Example | Mutability | Use |
|-----------|---------|------------|-----|
| Immutable build ID | `sha-a1b2c3d` | Effectively immutable | Deployments, rollbacks, audit trails |
| Exact semver | `1.4.2` | Should be immutable | Release pinning |
| Rolling minor/major | `1.4`, `1` | Moves forward | "Latest patch of 1.4" convenience |
| Channel | `stable`, `edge` | Moves forward | Human-facing convenience only |

The `sha-<commit>` tag ties an image back to the exact source commit that produced it — invaluable when correlating a production incident with a code change.

### Enforcing Immutability at the Registry

Convention is not enough; enforce it server-side so a tag can never be overwritten once published:

- **Amazon ECR** — set the repository to `IMMUTABLE` tag mutability; re-pushing an existing tag is rejected.
- **Harbor** — define *immutability rules* per project matching tag patterns (e.g. `v*`).
- **Google Artifact Registry** — enable immutable tags on the repository.
- **Universal** — deploy by digest (`image@sha256:...`) so the tag is irrelevant to what actually runs.

```bash
# ECR: make a repository tag-immutable
aws ecr create-repository --repository-name myapp \
  --image-tag-mutability IMMUTABLE

# Resolve a tag to its digest once, then deploy the digest forever
docker buildx imagetools inspect ghcr.io/acme/myapp:1.4.2 \
  --format '{% raw %}{{.Manifest.Digest}}{% endraw %}'
# -> sha256:9f86d081884c7d65...  (use this in your manifests)
```

## The Software Supply Chain Problem

Pinning a digest proves an image has not changed since you recorded that digest. It does **not** prove *who built it*, *from what source*, or *whether it contains known-vulnerable components*. Supply-chain security adds those guarantees. The mental model is a chain of attestations, each answering a different question:

| Control | Question it answers | Tool(s) |
|---------|---------------------|---------|
| **Digest pinning** | Did the bytes change? | Registry / OCI digests |
| **Signing** | Did *we* publish this, with our key? | cosign, Notation, DCT |
| **SBOM** | What components are inside? | Syft, BuildKit |
| **Vulnerability scan** | Do those components have known CVEs? | Trivy, Grype, Scout |
| **Provenance** | How and from what source was it built? | SLSA, BuildKit attestations |

These are independent and composable. A mature pipeline produces all of them as OCI artifacts stored *next to* the image in the same registry, then enforces them at deploy time with an admission policy.

## Image Signing

Signing binds a cryptographic identity to an image digest, so a verifier can reject anything not signed by an expected key or identity. There are three approaches you will encounter.

### Docker Content Trust (legacy)

The original mechanism, built on [The Update Framework (TUF)](https://theupdateframework.io/) via Notary v1. Setting `DOCKER_CONTENT_TRUST=1` makes the Docker client sign on push and verify on pull:

```bash
export DOCKER_CONTENT_TRUST=1
docker push myorg/myapp:1.4.2   # prompts to create/use signing keys
docker pull myorg/myapp:1.4.2   # fails if the tag is unsigned
```

DCT works but is tied to the Docker CLI, uses tag-based (not digest-based) trust, and the Notary v1 ecosystem is effectively in maintenance. New designs should prefer cosign or Notation, which store signatures as standard OCI artifacts any tool can read.

### cosign (Sigstore)

[cosign](https://docs.sigstore.dev/cosign/overview/), part of the Sigstore project, is the de facto standard. It signs the image *digest* and stores the signature as an OCI artifact alongside the image. It supports two modes:

**Key-based** — you hold a key pair:

```bash
cosign generate-key-pair                       # cosign.key + cosign.pub
cosign sign --key cosign.key ghcr.io/acme/myapp@sha256:9f86d08...
cosign verify --key cosign.pub ghcr.io/acme/myapp@sha256:9f86d08...
```

**Keyless** — the headline feature. Instead of managing long-lived keys, cosign requests a short-lived certificate from the Sigstore [Fulcio](https://docs.sigstore.dev/certificate_authority/overview/) CA, bound to an OIDC identity (a GitHub Actions workload, a Google account, etc.). The signing event is recorded in the [Rekor](https://docs.sigstore.dev/logging/overview/) transparency log. There is no key to leak or rotate:

```bash
# In CI, COSIGN_EXPERIMENTAL not needed on modern cosign; uses ambient OIDC
cosign sign ghcr.io/acme/myapp@sha256:9f86d08...

# Verify that the signer was a specific GitHub Actions workflow
cosign verify ghcr.io/acme/myapp@sha256:9f86d08... \
  --certificate-identity-regexp 'https://github.com/acme/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Keyless verification checks two things: the certificate's identity (who) and its OIDC issuer (where the identity came from). That pair is far harder to forge than possession of a static key.

### Notation (Notary v2 / CNCF)

[Notation](https://notaryproject.dev/) is the CNCF Notary Project's signer. Like cosign it produces OCI-native signatures, but it is built around x.509 PKI and trust policies rather than Sigstore's keyless model — a better fit for enterprises with an existing certificate authority. Azure Container Registry and others integrate it natively.

```bash
notation sign $REGISTRY/myapp@sha256:9f86d08...
notation verify $REGISTRY/myapp@sha256:9f86d08...   # uses configured trust policy
```

<div class="security-section">
  <h3><i class="fas fa-shield-alt"></i> Choosing a Signer</h3>

  <div class="storage-comparison">
    <table class="storage-table">
      <thead>
        <tr><th>Tool</th><th>Trust root</th><th>Signature storage</th><th>Pick it when</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>cosign</strong></td>
          <td>Keys or keyless (Fulcio + OIDC)</td>
          <td>OCI artifact</td>
          <td>You want keyless CI signing and the broadest ecosystem</td>
        </tr>
        <tr>
          <td><strong>Notation</strong></td>
          <td>x.509 / your own CA</td>
          <td>OCI artifact</td>
          <td>You already run PKI and want CA-rooted trust policies</td>
        </tr>
        <tr>
          <td><strong>Docker Content Trust</strong></td>
          <td>TUF / Notary v1</td>
          <td>Separate Notary server</td>
          <td>Maintaining existing DCT setups only; avoid for new work</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

## Software Bill of Materials (SBOM)

An SBOM is a machine-readable inventory of everything inside an image: OS packages, language dependencies, versions, and licenses. It is the difference between "we think we use log4j somewhere" and "image `sha256:9f86…` contains `log4j-core 2.14.1` in `/app/lib`." When a new CVE lands, you query SBOMs instead of rebuilding and rescanning the world.

Two standard formats dominate: **SPDX** (Linux Foundation / ISO standard) and **CycloneDX** (OWASP). Tools emit either.

### Generating an SBOM with Syft

[Syft](https://github.com/anchore/syft) (by Anchore) scans an image's filesystem and catalogs its components:

```bash
# Human-readable table
syft ghcr.io/acme/myapp:1.4.2

# CycloneDX JSON (good input for scanners and dependency tooling)
syft ghcr.io/acme/myapp:1.4.2 -o cyclonedx-json=sbom.cdx.json

# SPDX JSON (widely required for compliance / NTIA minimum elements)
syft ghcr.io/acme/myapp:1.4.2 -o spdx-json=sbom.spdx.json
```

BuildKit can also generate an SBOM *at build time* and attach it as an attestation, which is more accurate than scanning the finished image because the builder knows exactly what each stage installed:

```bash
docker buildx build --sbom=true --provenance=true \
  -t ghcr.io/acme/myapp:1.4.2 --push .
```

Attaching the SBOM to the image (rather than keeping it in a CI artifact bucket) means anyone who can pull the image can also fetch its bill of materials with `docker buildx imagetools inspect --format '{% raw %}{{ json .SBOM }}{% endraw %}'` or `cosign download sbom`.

## Vulnerability Scanning

A scanner cross-references the components in an image (from its SBOM or by introspecting the filesystem) against vulnerability databases such as the NVD, GitHub Advisory Database, and distro security trackers, then reports matching CVEs by severity.

### Trivy and Grype

[Trivy](https://github.com/aquasecurity/trivy) (Aqua) and [Grype](https://github.com/anchore/grype) (Anchore) are the two leading open-source scanners. Both scan an image directly or an SBOM you already produced.

```bash
# Trivy: scan an image, fail CI on High/Critical only
trivy image --severity HIGH,CRITICAL --exit-code 1 ghcr.io/acme/myapp:1.4.2

# Trivy: scan the SBOM you already built (faster, no re-pull)
trivy sbom sbom.cdx.json

# Grype: scan an image and gate on a severity threshold
grype ghcr.io/acme/myapp:1.4.2 --fail-on high

# Grype: scan a Syft SBOM directly
grype sbom:./sbom.spdx.json
```

`docker scout cves` (built into Docker Desktop and the CLI) is a convenient third option that also surfaces remediation advice — "upgrade base image from `3.18` to `3.19` to drop 7 CVEs."

### Making Scan Results Actionable

Raw CVE counts are noise without policy. Practical scanning means:

- **Gate on severity, not totals.** Failing on every `LOW` blocks releases for unfixable noise; gate on `HIGH`/`CRITICAL` with a fix available.
- **Distinguish fixable from unfixable.** A CVE with no upstream patch cannot be remediated by you today; Trivy's `--ignore-unfixed` filters these out of the gate.
- **Suppress with justification, not silence.** Use a `.trivyignore` (or VEX document) to waive specific CVEs *with a reason and an expiry*, rather than lowering the global threshold.
- **Re-scan continuously.** An image clean at build time accumulates CVEs as new ones are disclosed. Scan on a schedule, not just at push.

```bash
# Only fail on fixable High/Critical issues
trivy image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 \
  ghcr.io/acme/myapp:1.4.2
```

This is also where [minimal base images](dockerfiles.html) pay off directly: a distroless or `alpine` base ships far fewer packages, so there is simply less surface for a scanner to flag.

## Provenance and SLSA

Signing proves *who* published an image; provenance proves *how it was built*. **Build provenance** is a signed statement describing the build: which source repository and commit, which builder, which build parameters, and which materials (dependencies) went in. It is the attestation that lets a verifier reject an image that did not come from your trusted, hermetic pipeline — defending against a compromised CI runner or a tampered build.

### SLSA

[SLSA](https://slsa.dev/) ("Supply-chain Levels for Software Artifacts") is a framework that grades build integrity in levels. Roughly:

| SLSA Build Level | Guarantee |
|------------------|-----------|
| **L0** | No guarantees |
| **L1** | Provenance exists and describes how the artifact was built |
| **L2** | Provenance is signed and generated by a hosted build service |
| **L3** | Build runs on a hardened, isolated platform; provenance is unforgeable by the build itself |

Higher levels make provenance progressively harder to forge — at L3 the build platform generates and signs provenance in a way the build steps themselves cannot tamper with.

### Generating Provenance

BuildKit emits in-toto provenance attestations and attaches them to the image:

```bash
# mode=max records the full build: source, build args, base images, etc.
docker buildx build \
  --provenance=mode=max \
  --sbom=true \
  -t ghcr.io/acme/myapp:1.4.2 --push .

# Inspect the attached attestations
docker buildx imagetools inspect ghcr.io/acme/myapp:1.4.2 \
  --format '{% raw %}{{ json .Provenance }}{% endraw %}'
```

GitHub's [official SLSA provenance action](https://github.com/slsa-framework/slsa-github-generator) and the built-in [artifact attestations](https://docs.github.com/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds) produce signed, SLSA-aligned provenance directly from Actions, with the signing identity rooted in the workflow itself.

## Putting It Together: a Hardened Pipeline

A complete supply-chain-aware build does six things in sequence, producing artifacts that are stored beside the image and enforced at deploy time. The example below uses GitHub Actions with GHCR, keyless cosign, and BuildKit attestations.

```yaml
# .github/workflows/release.yml
name: build-and-attest
on:
  push:
    tags: ["v*"]

permissions:
  contents: read
  packages: write        # push to GHCR
  id-token: write        # OIDC token for keyless cosign

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${% raw %}{{ github.actor }}{% endraw %}
          password: ${% raw %}{{ secrets.GITHUB_TOKEN }}{% endraw %}

      # 1. Build with provenance + SBOM attestations attached
      - id: build
        uses: docker/build-push-action@v6
        with:
          push: true
          tags: ghcr.io/${% raw %}{{ github.repository }}{% endraw %}:${% raw %}{{ github.ref_name }}{% endraw %}
          provenance: mode=max
          sbom: true

      # 2. Scan the freshly pushed image; fail on fixable High/Critical
      - uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/${% raw %}{{ github.repository }}{% endraw %}:${% raw %}{{ github.ref_name }}{% endraw %}
          severity: HIGH,CRITICAL
          ignore-unfixed: true
          exit-code: "1"

      # 3. Keyless-sign the image digest (identity = this workflow)
      - uses: sigstore/cosign-installer@v3
      - run: cosign sign --yes ghcr.io/${% raw %}{{ github.repository }}{% endraw %}@${% raw %}{{ steps.build.outputs.digest }}{% endraw %}
```

At deploy time, an admission controller (for example [Kyverno](../kubernetes/) or Sigstore's policy-controller) verifies the signature identity and the presence of provenance before the image is allowed to run:

```bash
# What a deploy-time gate effectively checks
cosign verify ghcr.io/acme/myapp@sha256:9f86d08... \
  --certificate-identity-regexp 'https://github.com/acme/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

The result is an unbroken chain: a digest no one can alter, a signature only your pipeline can produce, an SBOM listing every component, a scan gating on real risk, and provenance tying the artifact back to a specific commit and builder.

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Deploy by Digest</h4>
    <p>Tags are mutable pointers; the <code>sha256:</code> digest immutably pins the config and every layer. Resolve tags to digests once and deploy the digest.</p>
  </div>
  <div class="takeaway-card">
    <h4>Enforce Immutability Server-Side</h4>
    <p>Set ECR/GAR repos or Harbor projects to reject tag overwrites, and never deploy <code>latest</code> — convention alone is not enough.</p>
  </div>
  <div class="takeaway-card">
    <h4>Sign with cosign</h4>
    <p>Keyless cosign signing binds a workflow identity to an image digest with no long-lived keys to leak. Verify identity <em>and</em> issuer.</p>
  </div>
  <div class="takeaway-card">
    <h4>SBOM + Scan + Provenance</h4>
    <p>Generate an SBOM (Syft/BuildKit), gate on fixable High/Critical CVEs (Trivy/Grype), and attach SLSA provenance so you can prove what shipped and how.</p>
  </div>
</div>

---

## See Also

- [Fundamentals](fundamentals.html) - Images, layers, and the content-addressed model behind digests
- [Storage &amp; Security](storage-security.html) - Runtime hardening, secrets, and the scanning basics expanded here
- [Dockerfiles &amp; CI/CD](dockerfiles.html) - Minimal base images and build pipelines that produce these artifacts
- [Advanced Patterns](advanced.html) - Production deployment architectures
- [Kubernetes](../kubernetes/) - Admission control and verifying signatures at deploy time
- [Cybersecurity](../cybersecurity/) - Broader security fundamentals
