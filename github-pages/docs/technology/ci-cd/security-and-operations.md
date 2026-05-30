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
  </ul>
</div>
