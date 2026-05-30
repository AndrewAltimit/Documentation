---
layout: docs
title: CI/CD
permalink: /docs/technology/ci-cd/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">CI/CD</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Continuous Integration & Continuous Deployment: From Code to Production</p>
</div>

<div class="intro-card">
  <p class="lead-text">CI/CD transforms software delivery from a risky, manual process into an automated, reliable pipeline. By automatically building, testing, and deploying code changes, teams can release features faster, catch bugs earlier, and deliver value to users continuously. This automation isn't just about speed—it's about creating a safety net that gives developers confidence to innovate without fear of breaking production.</p>

  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-rocket"></i>
      <h4>Rapid Delivery</h4>
      <p>Deploy changes multiple times per day with confidence</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-shield-alt"></i>
      <h4>Early Bug Detection</h4>
      <p>Catch issues in minutes, not days or weeks</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-users"></i>
      <h4>Team Efficiency</h4>
      <p>Free developers from manual deployment tasks</p>
    </div>
  </div>
</div>

## What is CI/CD?

### The Restaurant Kitchen Analogy

Imagine a busy restaurant kitchen:

**Without CI/CD** (Traditional Approach):
- Chef prepares entire meal alone
- No one tastes until it reaches the customer
- If something's wrong, the whole meal is remade
- One chef = one meal at a time

**With CI/CD** (Modern Approach):
- Multiple chefs work on different dishes
- Each component is tasted immediately (CI)
- Approved dishes go straight to customers (CD)
- Kitchen runs continuously, serving many orders

### Breaking It Down

**Continuous Integration (CI)**: Developers merge code changes frequently (usually several times per day), with each merge triggering automated builds and tests.

**Continuous Deployment (CD)**: Code that passes all tests is automatically deployed to production without manual intervention.

**Continuous Delivery**: A variation where code is automatically prepared for release but requires manual approval to deploy.

### The Pipeline at a Glance

A commit flows through automated stages, each a gate that must pass before the next runs. The split between Delivery and Deployment is simply whether the final promotion to production is manual or automatic.

```mermaid
flowchart LR
    COMMIT["git push / PR"] --> BUILD["Build"]
    BUILD --> TEST["Test and Lint"]
    TEST --> SCAN["Security Scan"]
    SCAN --> STAGE["Deploy to Staging"]
    STAGE --> GATE{"Approval?"}
    GATE -->|manual = Delivery| PROD["Deploy to Production"]
    GATE -->|automatic = Deployment| PROD
    PROD --> MON["Monitor and Rollback if needed"]
    style BUILD fill:#e3f2fd,stroke:#1565c0
    style TEST fill:#e3f2fd,stroke:#1565c0
    style SCAN fill:#e3f2fd,stroke:#1565c0
    style STAGE fill:#e8f5e9,stroke:#2e7d32
    style PROD fill:#e8f5e9,stroke:#2e7d32
```

---

## Getting Started

You can stand up a working pipeline in well under an hour, then deepen it over the following weeks.

### Your First Pipeline (30 Minutes)

Build a simple CI/CD pipeline for a Node.js application using GitHub Actions:

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  # Job 1: Continuous Integration
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run linter
      run: npm run lint

    - name: Run tests
      run: npm test

    - name: Build application
      run: npm run build

  # Job 2: Continuous Deployment
  deploy:
    needs: test  # Only run if tests pass
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to production
      run: |
        # Your deployment commands here
        echo "Deploying to production..."
```

**Understanding the pipeline flow:**

1. **Trigger**: Code pushed to main/develop or PR opened
2. **Checkout**: Pipeline gets latest code
3. **Setup**: Install required tools (Node.js)
4. **Dependencies**: Install project dependencies
5. **Quality Checks**: Run linter for code standards
6. **Tests**: Execute automated tests
7. **Build**: Compile/bundle the application
8. **Deploy**: If on main branch and tests pass, deploy

**Quick-start checklist:**

- [ ] Create `.github/workflows/` directory
- [ ] Add workflow YAML file
- [ ] Define trigger events (push, PR, schedule)
- [ ] Set up build environment
- [ ] Add test commands
- [ ] Configure deployment (if ready)
- [ ] Commit and push to see it run

### A Four-Week Path to Production

Once the first pipeline runs, expand it incrementally:

| Week | Focus | Tasks |
|------|-------|-------|
| **1 — Foundation** | Get a pipeline running | Choose a platform, create a "Hello World" pipeline, add basic tests, set up notifications |
| **2 — Expansion** | Add quality gates | Add linting, enable branch protection, create a staging deployment, add security scanning |
| **3 — Optimization** | Make it fast and safe | Implement caching, parallelize tests, add performance tests, build deployment rollback |
| **4 — Production Ready** | Operate with confidence | Set up monitoring, implement blue-green deployment, add compliance checks, document runbooks |

---

## Explore the Guides

<div class="command-grid">
  <div class="nav-card">
    <h4><i class="fas fa-sitemap"></i> <a href="platforms-and-pipelines.html">Platforms & Pipeline Design</a></h4>
    <p>Popular CI/CD platforms compared, pipeline design patterns, and testing strategies.</p>
  </div>
  <div class="nav-card">
    <h4><i class="fas fa-exchange-alt"></i> <a href="deployment.html">Deployment Strategies</a></h4>
    <p>Blue-green, canary, rolling deployments, and feature flags for releasing safely.</p>
  </div>
  <div class="nav-card">
    <h4><i class="fas fa-shield-alt"></i> <a href="security-and-operations.html">Security, GitOps & Operations</a></h4>
    <p>Securing pipelines, monitoring and observability, GitOps, IaC integration, and advanced topics.</p>
  </div>
</div>

---

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>CI and CD Are Distinct</h4>
    <p>CI integrates and tests every change automatically; CD takes passing builds the rest of the way to staging or production. You can adopt CI long before full CD.</p>
  </div>
  <div class="takeaway-card">
    <h4>Fast Feedback Is the Point</h4>
    <p>The value is catching problems in minutes, not days. Parallelize tests, fail fast, and keep pipelines quick enough that developers trust them.</p>
  </div>
  <div class="takeaway-card">
    <h4>Deploy Strategies Manage Risk</h4>
    <p>Blue-green, canary, and rolling deployments trade speed for safety in different ways. Pair them with automated rollback so a bad release is reversible.</p>
  </div>
  <div class="takeaway-card">
    <h4>Secure the Supply Chain</h4>
    <p>Pipelines hold secrets and ship artifacts. Scan dependencies, sign artifacts, generate SBOMs, and grant least-privilege credentials to runners.</p>
  </div>
</div>

---

<div class="see-also-card">
  <h4>See Also</h4>
  <ul>
    <li><a href="../git/">Git Version Control</a> — the commits that trigger every pipeline</li>
    <li><a href="../branching.html">Branching Strategies</a> — workflow patterns that shape your pipeline</li>
    <li><a href="../docker/">Docker</a> — containerization for consistent build environments</li>
    <li><a href="../kubernetes/">Kubernetes</a> — orchestration and automated deployments</li>
    <li><a href="../terraform/">Terraform</a> — infrastructure as code for automated provisioning</li>
    <li><a href="../cybersecurity/">Cybersecurity</a> — securing the pipeline and its secrets</li>
  </ul>
</div>
