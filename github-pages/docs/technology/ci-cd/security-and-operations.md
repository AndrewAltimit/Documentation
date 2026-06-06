---
layout: docs
title: "CI/CD: Security, GitOps & Operations"
permalink: /docs/technology/ci-cd/security-and-operations.html
toc: true
toc_sticky: true
hide_title: true
---

[CI/CD](./) ›

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Security, GitOps & Operations</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Securing pipelines, GitOps and Infrastructure as Code, observability, and the advanced topics that keep delivery reliable at scale.</p>
</div>

## Security in CI/CD

### Secrets Management

**Bad Practice**:
```yaml
deploy:
  script:
    - API_KEY="sk-1234567890" npm run deploy  # Never do this!
```

**Good Practice**:
```yaml
deploy:
  script:
    - npm run deploy
  environment:
    name: production
  secrets:
    API_KEY:
      from_secret: api_key_production
```

### Security Scanning Pipeline

```yaml
security-scan:
  stage: security
  parallel:
    - dependency-check:
        script:
          - npm audit
          - pip-audit  # Python (alternative to safety)
          - bundle audit  # Ruby
          - osv-scanner --recursive .  # Google's OSV Scanner

    - sast:  # Static Application Security Testing
        script:
          - semgrep --config=auto
          - bandit -r src/  # Python
          - snyk code test

    - container-scan:
        script:
          - trivy image myapp:latest
          - grype myapp:latest  # Anchore scanner
          - docker scout cves myapp:latest  # Docker's native scanner

    - secrets-scan:
        script:
          - gitleaks detect --source=.
          - trufflehog filesystem . --json
```

### Security Best Practices

1. **Rotate Secrets Regularly**
   ```yaml
   - name: Check secret age
     run: |
       if [ $(secret_age $SECRET_NAME) -gt 90 ]; then
         echo "::error::Secret older than 90 days!"
         exit 1
       fi
   ```

2. **Least Privilege Access**
   ```yaml
   deploy:
     permissions:
       contents: read
       deployments: write
       # Only what's needed, nothing more
   ```

3. **Audit Logs**
   ```yaml
   after_script:
     - echo "Deployed by $CI_USER at $CI_TIMESTAMP" >> audit.log
   ```

## Software Supply-Chain Security

The classic threat model assumes attackers come through the front door — the running application. Supply-chain attacks come through the *build*: a compromised dependency (SolarWinds, `event-stream`, the `xz` backdoor), a poisoned base image, a stolen signing key, or a tampered build step that injects malware into an otherwise-legitimate artifact. Because the resulting binary is signed and shipped through normal channels, downstream consumers trust it implicitly. Supply-chain security closes this gap by making the **provenance** of every artifact verifiable: *what* source produced it, *which* builder ran, and *what* it depends on — all cryptographically attested and checked before deployment.

The pieces fit together as a chain of evidence:

```
source ──build──▶ artifact ──sign──▶ signature
   │                  │                   │
   └─ dependency      └─ SBOM             └─ provenance attestation
      scanning           (bill of            (SLSA: who/how/from-what)
                          materials)
                              │
                       admission/verify gate ──▶ deploy
```

### SLSA: Levels of Build Integrity

**SLSA** (Supply-chain Levels for Software Artifacts, pronounced "salsa") is a framework that grades how trustworthy an artifact's build process is. It is *not* a tool — it is a set of requirements you satisfy by hardening your pipeline. The current v1.0 track focuses on **build provenance**:

| Level | Requirement | What it stops |
|-------|-------------|---------------|
| **L0** | No guarantees | Nothing |
| **L1** | Provenance exists — build emits a signed record of how the artifact was produced | Mistakes; "where did this binary come from?" |
| **L2** | Provenance is **signed** by a hosted build platform; source and build are version-controlled | Tampering with provenance after the fact |
| **L3** | Build runs on a **hardened, isolated** platform; provenance is **non-forgeable** (signed by the platform, not the build script) | A malicious build step forging its own provenance; cross-build contamination |

The key idea at L3 is **isolation**: the build environment cannot reach the signing key, so even a fully compromised build *script* cannot forge a valid provenance attestation. GitHub-hosted runners with the `slsa-framework/slsa-github-generator` reusable workflow reach L3 because the signing happens in a separate, trusted reusable workflow your job cannot tamper with.

{% raw %}
```yaml
# SLSA L3 provenance via the official GitHub generator
jobs:
  build:
    outputs:
      digest: ${{ steps.hash.outputs.digest }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make release   # produces ./dist/app
      - id: hash
        run: echo "digest=$(sha256sum dist/app | base64 -w0)" >> "$GITHUB_OUTPUT"

  provenance:
    needs: [build]
    permissions:
      actions: read       # read the workflow run
      id-token: write     # OIDC token for keyless signing
      contents: write     # attach provenance to the release
    # The reusable workflow runs isolated from your build job — this is what
    # makes the provenance non-forgeable and earns L3.
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.0.0
    with:
      base64-subjects: ${{ needs.build.outputs.digest }}
```
{% endraw %}

### Artifact Signing with Cosign and Sigstore

Signing answers "is this the exact artifact the build produced, and did *we* publish it?" Traditionally this meant long-lived GPG/PGP keys that teams had to store, rotate, and inevitably leak. **Sigstore** replaces long-lived keys with **keyless signing**: an ephemeral key pair is minted per-signature, certified against an OIDC identity (your GitHub Actions workload identity, a Google account, etc.) by the **Fulcio** CA, and the signing event is recorded in **Rekor**, a public tamper-evident transparency log. The private key is discarded seconds later — there is nothing long-lived to steal.

**cosign** is the CLI that drives this for container images and arbitrary blobs:

```yaml
sign-image:
  permissions:
    id-token: write     # Sigstore OIDC keyless signing
    packages: write     # push the signature to the registry
  steps:
    - uses: sigstore/cosign-installer@v3

    # Keyless: identity comes from the GitHub Actions OIDC token,
    # certificate from Fulcio, signature logged in Rekor.
    - name: Sign the image
      run: cosign sign --yes ghcr.io/myorg/app@${IMAGE_DIGEST}

    # Verify before deploy — pin the expected identity and issuer so a
    # signature from any other workflow is rejected.
    - name: Verify
      run: |
        cosign verify ghcr.io/myorg/app@${IMAGE_DIGEST} \
          --certificate-identity-regexp '^https://github.com/myorg/' \
          --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

> **Always sign and verify by digest (`@sha256:...`), never by tag.** Tags are mutable — `:latest` can be repointed at a malicious image after you sign the tag. The digest is the content hash, so a signature over a digest is a signature over exactly those bytes. See [Docker Registry & Distribution](../docker/registry.html#image-signing) for how digests, tags, and content trust work at the registry level.

### SBOM Generation and Attestation

A **Software Bill of Materials** is a complete, machine-readable inventory of every component in an artifact — direct and transitive dependencies, versions, licenses, and hashes. When the next `Log4Shell`-class CVE drops, the question "are we affected, and where?" becomes a query against stored SBOMs instead of a frantic codebase audit. The two dominant formats are **SPDX** (ISO standard, license-focused) and **CycloneDX** (OWASP, security-focused); most tools emit both.

```yaml
sbom:
  steps:
    # Generate an SBOM from the built image (syft works on images, dirs, archives)
    - run: |
        syft ghcr.io/myorg/app@${IMAGE_DIGEST} \
          -o spdx-json=sbom.spdx.json \
          -o cyclonedx-json=sbom.cdx.json

    # Attach the SBOM to the image as a signed attestation, not a loose file.
    # An *attestation* binds the SBOM to the image digest and signs the pair,
    # so consumers can verify "this SBOM really describes this image."
    - run: |
        cosign attest --yes \
          --predicate sbom.spdx.json \
          --type spdxjson \
          ghcr.io/myorg/app@${IMAGE_DIGEST}

    # Scan the SBOM for known vulnerabilities (decoupled from generation —
    # you can re-scan old SBOMs as new CVEs are published, without rebuilding)
    - run: grype sbom:sbom.cdx.json --fail-on high
```

The distinction between a *file* and an *attestation* matters: a `sbom.json` sitting in build artifacts is unsigned and unbound — anyone can swap it. `cosign attest` signs the SBOM-plus-digest as a unit and stores it alongside the image in the registry (and in Rekor), so `cosign verify-attestation` can later prove the SBOM is authentic and describes that exact image.

### Dependency and Secret Scanning in Pipelines

Provenance proves *who built it*; scanning proves *it isn't already known to be bad*. Both run as **gates** — failing the build rather than merely warning — so a vulnerable dependency or a leaked credential never reaches a registry.

```yaml
supply-chain-gate:
  stage: security
  parallel:
    - dependency-vulns:
        script:
          # SCA against your lockfiles + the image's OS packages
          - osv-scanner --recursive .
          - trivy image --severity HIGH,CRITICAL --exit-code 1 ${IMAGE}

    - secret-scanning:
        script:
          # Scan full git history, not just the working tree — a secret
          # committed and "removed" still lives in earlier commits.
          - gitleaks detect --source=. --redact
          - trufflehog git file://. --since-commit HEAD~50 --only-verified

    - license-policy:
        script:
          # Block copyleft/forbidden licenses pulled in transitively
          - trivy image --scanners license --severity HIGH ${IMAGE}
```

Push-time scanning is necessary but reactive; complement it with **pre-receive enforcement** (pre-commit `gitleaks` hooks so secrets are caught before they ever land in history) and **continuous re-scanning** of already-published images, since new CVEs are disclosed against artifacts that scanned clean yesterday. Registry-side scanning (Trivy/Grype integrated into the registry, or native scanners like Harbor's) provides this last line of defense — see [Docker Registry & Distribution](../docker/registry.html#vulnerability-scanning) for registry-integrated scanning and admission policy.

### Secret Rotation

Even well-managed secrets must be assumed to leak eventually; rotation bounds the blast radius by limiting how long a leaked credential is useful. The progression of maturity is:

1. **Static long-lived secrets, manually rotated** — the floor. Track each secret's age and fail the pipeline past a policy threshold (the 90-day check shown earlier).
2. **Automated rotation via a secrets manager** — Vault, AWS Secrets Manager, or GCP Secret Manager rotates the credential on a schedule and re-issues it to consumers, so no human handles the value.
3. **Short-lived dynamic secrets** — the secrets manager mints a credential *on demand* with a TTL of minutes (e.g. Vault's database secrets engine creates a per-job DB user that auto-expires). There is no standing secret to rotate.
4. **No secrets at all — workload identity / OIDC federation** — the pipeline presents a short-lived OIDC token proving its identity, and the cloud provider exchanges it for scoped, temporary credentials. This is strictly better than rotation: nothing long-lived exists to leak.

```yaml
# Keyless cloud auth via OIDC — no stored AWS keys to rotate or leak.
deploy:
  permissions:
    id-token: write     # mint the OIDC token
    contents: read
  steps:
    - uses: aws-actions/configure-aws-credentials@v4
      with:
        role-to-assume: arn:aws:iam::123456789012:role/deploy-role
        aws-region: us-east-1
        # No aws-access-key-id / aws-secret-access-key — STS issues
        # temporary credentials in exchange for the OIDC token.
```

```yaml
# Short-lived dynamic DB credential from Vault (TTL=1h, auto-revoked)
- run: |
    export VAULT_TOKEN=$(vault write -field=token \
      auth/jwt/login role=ci jwt="$CI_JOB_JWT")
    creds=$(vault read -format=json database/creds/app-readonly)
    export DB_USER=$(echo "$creds" | jq -r .data.username)
    export DB_PASS=$(echo "$creds" | jq -r .data.password)
    ./run-migrations.sh
    # Lease expires automatically; nothing persists past the job.
```

When a *static* secret must remain, rotate without downtime by overlapping validity: provision the new credential, deploy consumers configured to accept **both** old and new, then revoke the old one once every consumer has rolled. The same dual-key window applies to signing keys — though Sigstore's keyless model (above) sidesteps signing-key rotation entirely by never holding a long-lived key.

## Monitoring and Observability

### Pipeline Metrics

```yaml
collect-metrics:
  after_script:
    - |
      # Send metrics to monitoring system
      curl -X POST https://metrics.api/pipeline \
        -d '{
          "pipeline": "$CI_PIPELINE_ID",
          "duration": "$CI_PIPELINE_DURATION",
          "status": "$CI_JOB_STATUS",
          "branch": "$CI_COMMIT_BRANCH"
        }'
```

### Key Metrics to Track

1. **Lead Time**: Commit to production
2. **Deployment Frequency**: Deploys per day/week
3. **MTTR**: Mean Time To Recovery
4. **Change Failure Rate**: Failed deploys percentage

### Observability Dashboard Example

```yaml
# Grafana dashboard query
deployment_frequency:
  query: |
    count(
      ci_pipeline_status{status="success", branch="main"}
    ) by (day)

lead_time_p95:
  query: |
    histogram_quantile(0.95,
      ci_pipeline_duration_seconds_bucket
    )
```

## GitOps

With GitOps, Git is the single source of truth for both application and infrastructure state. Developers commit changes; a GitOps operator (ArgoCD, Flux) continuously reconciles the cluster to match the repository.

### The GitOps Workflow

```yaml
# 1. Developer commits code
git add .
git commit -m "feat: add payment processing"
git push origin feature/payments

# 2. CI pipeline runs tests
ci-pipeline:
  - test
  - build
  - push-image

# 3. Update deployment manifest
update-manifest:
  script:
    - git clone https://github.com/myorg/k8s-configs
    - cd k8s-configs
    - yq eval '.image.tag = "'$CI_COMMIT_SHA'"' -i app/values.yaml
    - git commit -am "Update app to $CI_COMMIT_SHA"
    - git push

# 4. GitOps operator syncs
# ArgoCD/Flux automatically deploys changes
```

The same loop works for Kubernetes manifests authored with `yq`/Helm and for declarative Infrastructure as Code (Terraform state committed to Git), so application and infrastructure changes flow through one reviewed, auditable history.

### GitOps Best Practices

1. **Separate Config Repo**
   ```
   app-code/          # Application source
   app-config/        # Kubernetes manifests
   app-secrets/       # Encrypted secrets (using Sealed Secrets/SOPS)
   ```

2. **Environment Branches**
   ```
   main     → production/
   staging  → staging/
   develop  → development/
   ```

3. **Automated Rollback**
   ```yaml
   on-failure:
     script:
       - git revert HEAD
       - git push
       # GitOps operator automatically rolls back
   ```

**Modern GitOps Tools:**
- **ArgoCD**: Most popular, great UI, multi-cluster support (with ApplicationSets)
- **Flux v2**: GitOps toolkit, native Kubernetes controller
- **Rancher Fleet**: Multi-cluster GitOps at scale
- **Weave GitOps**: Enterprise features, policy management

## Infrastructure as Code Integration

### Terraform in CI/CD

```yaml
terraform-pipeline:
  stages:
    - validate
    - plan
    - apply

  validate:
    script:
      - terraform init
      - terraform validate
      - terraform fmt -check

  plan:
    script:
      - terraform plan -out=tfplan
    artifacts:
      paths:
        - tfplan

  apply:
    script:
      - terraform apply tfplan
    when: manual
    only:
      - main
```

### Ansible Integration

```yaml
ansible-deploy:
  script:
    - ansible-playbook -i inventory/production deploy.yml
  before_script:
    - ansible-galaxy install -r requirements.yml
    - ansible-lint playbooks/
```

## Common Pitfalls and Troubleshooting

### 1. Flaky Tests

**Problem**: Tests pass locally but fail in CI

**Solutions**:
```javascript
// Bad: Time-dependent test
it('expires after 1 hour', async () => {
  await sleep(3600000);  // Don't do this!
  expect(isExpired()).toBe(true);
});

// Good: Mock time
it('expires after 1 hour', async () => {
  const clock = sinon.useFakeTimers();
  clock.tick(3600000);
  expect(isExpired()).toBe(true);
  clock.restore();
});
```

### 2. Secret Leaks

**Problem**: Accidentally committed secrets

**Prevention**:
```yaml
pre-commit-check:
  script:
    - gitleaks detect --source=. --verbose
    - detect-secrets scan --all-files
```

### 3. Long Build Times

**Problem**: Pipeline takes hours

**Solutions**:
```yaml
# Cache dependencies
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - node_modules/
    - .npm/

# Parallel jobs
test:
  parallel: 4
  script:
    - npm run test:chunk:${CI_NODE_INDEX}

# Incremental builds
build:
  script:
    - npm run build --since=$CI_COMMIT_BEFORE_SHA
```

### 4. Environment Drift

**Problem**: "Works in staging, breaks in production"

**Solution**:
```yaml
# Use identical environments
.deploy_template: &deploy_template
  image: deploy:v1.2.3
  variables:
    TERRAFORM_VERSION: "1.5.0"
    KUBECTL_VERSION: "1.27.0"

deploy_staging:
  <<: *deploy_template
  environment: staging

deploy_production:
  <<: *deploy_template
  environment: production
```

## Real-World Examples

### Example 1: E-commerce Platform

**Challenge**: Deploy updates without affecting active shoppers

**Solution**:
```yaml
name: E-commerce Deployment

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      # 1. Build and test
      - uses: actions/checkout@v4
      - run: |
          docker build -t shop:$GITHUB_SHA .
          docker run shop:$GITHUB_SHA npm test

      # 2. Deploy to canary (5% traffic)
      - name: Canary Deploy
        run: |
          kubectl set image deployment/shop-canary \
            shop=shop:$GITHUB_SHA -n production

      # 3. Monitor metrics
      - name: Monitor Canary
        run: |
          ./scripts/monitor-canary.sh --duration=30m \
            --error-threshold=1% \
            --latency-p99=200ms

      # 4. Full rollout
      - name: Production Deploy
        run: |
          kubectl set image deployment/shop \
            shop=shop:$GITHUB_SHA -n production
```

### Example 2: Microservices Platform

**Challenge**: Coordinate deployment of 50+ services

**Solution**:
{% raw %}
```yaml
# Monorepo CI/CD
name: Microservices Pipeline

on:
  push:
    branches: [main]

jobs:
  detect-changes:
    outputs:
      services: ${{ steps.filter.outputs.changes }}
    steps:
      - uses: dorny/paths-filter@v2
        id: filter
        with:
          filters: |
            auth: services/auth/**
            payment: services/payment/**
            inventory: services/inventory/**
            # ... 47 more services

  build-and-deploy:
    needs: detect-changes
    strategy:
      matrix:
        service: ${{ fromJson(needs.detect-changes.outputs.services) }}
    steps:
      - name: Build Service
        run: |
          cd services/${{ matrix.service }}
          docker build -t ${{ matrix.service }}:$GITHUB_SHA .

      - name: Deploy Service
        run: |
          helm upgrade --install ${{ matrix.service }} \
            ./charts/${{ matrix.service }} \
            --set image.tag=$GITHUB_SHA \
            --wait --timeout=5m
```
{% endraw %}

### Example 3: Mobile App Deployment

**Challenge**: Deploy to multiple app stores with different requirements

**Solution**:
{% raw %}
```yaml
name: Mobile App Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        platform: [ios, android]

    steps:
      - uses: actions/checkout@v4

      - name: Build ${{ matrix.platform }}
        run: |
          if [ "${{ matrix.platform }}" == "ios" ]; then
            fastlane ios build
          else
            fastlane android build
          fi

      - name: Run Tests
        run: |
          if [ "${{ matrix.platform }}" == "ios" ]; then
            fastlane ios test
          else
            fastlane android test
          fi

  deploy:
    needs: build
    steps:
      - name: Deploy to App Store
        run: fastlane ios release

      - name: Deploy to Play Store
        run: fastlane android release

      - name: Notify Team
        run: |
          curl -X POST $SLACK_WEBHOOK \
            -d '{"text":"Version ${{ github.ref }} released to app stores!"}'
```
{% endraw %}

## Advanced Topics

### Self-Hosted Runners

```yaml
# Setup for high-security environments
self-hosted-runner:
  runs-on: [self-hosted, linux, x64, gpu]
  container:
    image: custom-runner:latest
    options: --gpus all
  steps:
    - name: ML Model Training
      run: python train.py --gpu --distributed
```

### Pipeline as Code Libraries

```groovy
// Jenkins Shared Library
@Library('company-pipeline-lib') _

companyPipeline {
  language = 'java'
  type = 'microservice'
  deployEnvironments = ['dev', 'staging', 'prod']
  slackChannel = '#deployments'
}
```

### Multi-Cloud Deployments

{% raw %}
```yaml
multi-cloud-deploy:
  strategy:
    matrix:
      cloud: [aws, azure, gcp]
  steps:
    - name: Deploy to ${{ matrix.cloud }}
      run: |
        case "${{ matrix.cloud }}" in
          aws)
            terraform apply -var-file=aws.tfvars
            ;;
          azure)
            terraform apply -var-file=azure.tfvars
            ;;
          gcp)
            terraform apply -var-file=gcp.tfvars
            ;;
        esac
```
{% endraw %}

## Emerging Trends in CI/CD

1. **AI-Powered CI/CD**
   - Predictive test selection
   - Automated flaky test detection
   - AI-generated pipeline optimizations
   - Smart deployment timing

2. **Supply Chain Security**
   - SBOM (Software Bill of Materials) generation
   - SLSA compliance automation
   - Sigstore for artifact signing
   - Dependency attestation

3. **Platform Engineering**
   - Internal Developer Platforms (IDPs)
   - Golden paths for deployment
   - Self-service infrastructure
   - Developer experience metrics

4. **Green CI/CD**
   - Carbon-aware computing
   - Energy-efficient build scheduling
   - Resource optimization
   - Sustainability metrics

## Resources and Further Learning

### Essential Tools
- **Pipeline Syntax Validators**:
  - GitHub Actions playground
  - GitLab CI Lint
  - CircleCI Config Validator
- **Security Scanners**:
  - Snyk (now with AI-powered fixes)
  - SonarQube/SonarCloud
  - Checkmarx
  - GitHub Advanced Security
- **Monitoring**:
  - Datadog CI Visibility
  - New Relic CodeStream
  - Grafana Cloud
  - OpenTelemetry (standard for observability)
- **GitOps Operators**:
  - ArgoCD (with ApplicationSets)
  - Flux v2
  - Crossplane (infrastructure composition)

### Books and Courses
- "Continuous Delivery" by Jez Humble (Classic)
- "The DevOps Handbook" by Gene Kim et al.
- "Accelerate" by Nicole Forsgren et al.
- "Modern Software Engineering" by David Farley (2022)
- "The Phoenix Project" & "The Unicorn Project" by Gene Kim

### Online Learning
- **DevOps with GitLab CI** - GitLab's official course
- **GitHub Actions Deep Dive** - A Cloud Guru
- **Jenkins 2023 Masterclass** - Udemy
- **CNCF CI/CD with Tekton** - Linux Foundation

### Community Resources
- CNCF CI/CD Landscape
- DevOps Weekly Newsletter
- CI/CD Collective Forum

Adopting CI/CD is incremental work: start simple, measure everything, and refine the pipeline based on what the metrics show.

---

<nav class="page-nav">
  <a href="deployment.html">⬅ Deployment Strategies</a>
  <a href="./">CI/CD Hub ➡</a>
</nav>

<div class="see-also-card">
  <h4>See Also</h4>
  <ul>
    <li><a href="platforms-and-pipelines.html">Platforms & Pipeline Design</a> — choosing a platform and structuring pipelines</li>
    <li><a href="deployment.html">Deployment Strategies</a> — blue-green, canary, and rolling rollouts</li>
    <li><a href="../terraform/">Terraform</a> — infrastructure as code for the IaC pipelines above</li>
    <li><a href="../cybersecurity/">Cybersecurity</a> — securing the pipeline and its secrets</li>
    <li><a href="../docker/registry.html">Docker Registry &amp; Distribution</a> — image signing, SBOMs, and provenance at the registry level</li>
  </ul>
</div>
