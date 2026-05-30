---
layout: docs
title: "CI/CD: Platforms & Pipeline Design"
permalink: /docs/technology/ci-cd/platforms-and-pipelines.html
toc: true
toc_sticky: true
hide_title: true
---

[CI/CD](./) ›

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Platforms & Pipeline Design</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Choosing a CI/CD platform, structuring pipelines, and testing strategies that keep feedback fast.</p>
</div>

## Popular CI/CD Platforms

### GitHub Actions
**Best for**: GitHub-hosted projects, easy integration
```yaml
# Example: Python app with GitHub Actions
name: Python CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    - run: |
        pip install -r requirements.txt
        pytest
```

**Recent GitHub Actions Features:**
- **Larger runners**: Up to 64 vCPUs and 256 GB RAM
- **GPU runners**: NVIDIA GPU support for ML workloads
- **Arm64 runners**: Native ARM architecture support
- **Deployment protection rules**: Environment-specific approvals

### GitLab CI/CD
**Best for**: GitLab users, built-in DevOps features
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - npm install
    - npm test

deploy:
  stage: deploy
  script:
    - npm run build
    - npm run deploy
  only:
    - main
```

### Jenkins
**Best for**: Complex workflows, self-hosted, plugins
```groovy
// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'mvn clean package'
            }
        }
        stage('Test') {
            steps {
                sh 'mvn test'
            }
        }
        stage('Deploy') {
            steps {
                sh './deploy.sh'
            }
        }
    }
}
```

### CircleCI
**Best for**: Fast builds, Docker support
```yaml
# .circleci/config.yml
version: 2.1
jobs:
  build-and-test:
    docker:
      - image: cimg/node:18.0
    steps:
      - checkout
      - run: npm install
      - run: npm test
      - run: npm run build

workflows:
  main:
    jobs:
      - build-and-test
```

### Platform Comparison

| Platform | Pros | Cons | Best Use Case |
|----------|------|------|---------------|
| GitHub Actions | Free for public repos, great integration | Limited free minutes for private | GitHub projects |
| GitLab CI | Complete DevOps platform | Can be complex | Full DevOps lifecycle |
| Jenkins | Highly customizable, mature | Requires maintenance | Enterprise, complex needs |
| CircleCI | Fast, good caching | Can get expensive | Speed-critical projects |
| Travis CI | Simple setup | Less popular now | Legacy projects |
| Azure DevOps | Great for .NET | Microsoft-centric | Windows/Azure shops |
| Buildkite | Hybrid model, scalable | Requires own runners | High-security needs |
| Drone | Container-native, simple | Smaller community | Kubernetes environments |

## Pipeline Design Patterns

### 1. Simple Linear Pipeline
```yaml
stages:
  - build
  - test
  - deploy
```
**Use when**: Starting out, simple projects

### 2. Parallel Execution
```yaml
test:
  parallel:
    - unit-tests:
        script: npm run test:unit
    - integration-tests:
        script: npm run test:integration
    - lint:
        script: npm run lint
```
**Use when**: Tests are independent, need speed

### 3. Matrix Builds
```yaml
strategy:
  matrix:
    node-version: [14, 16, 18]
    os: [ubuntu-latest, windows-latest]
```
**Use when**: Testing across multiple environments

### 4. Fan-out/Fan-in
```yaml
stages:
  - build
  - parallel-tests  # Fan-out: multiple parallel jobs
  - aggregate       # Fan-in: collect results
  - deploy
```
**Use when**: Complex test suites, need aggregated results

### 5. Blue-Green Pipeline
```yaml
deploy-blue:
  script: deploy_to_blue_environment()

smoke-test:
  script: test_blue_environment()

switch-traffic:
  script: route_traffic_to_blue()

cleanup-green:
  script: teardown_green_environment()
```
**Use when**: Zero-downtime deployments critical (see [Deployment Strategies](deployment.html) for the full pattern)

## Testing Strategies in CI/CD

### The Testing Pyramid

```
        /\
       /  \  E2E Tests (Slow, Few)
      /    \
     /------\ Integration Tests (Medium)
    /        \
   /----------\ Unit Tests (Fast, Many)
```

### Unit Tests in CI
```javascript
// Fast, isolated tests
describe('Calculator', () => {
  it('adds two numbers', () => {
    expect(add(2, 3)).toBe(5);
  });
});
```
**Run on**: Every commit
**Duration**: < 5 minutes

### Integration Tests
```javascript
// Test component interactions
describe('API Integration', () => {
  it('creates user and sends email', async () => {
    const user = await createUser(data);
    expect(emailService.sent).toBe(true);
  });
});
```
**Run on**: Pull requests
**Duration**: 5-15 minutes

### End-to-End Tests
```javascript
// Test complete user flows
describe('Checkout Flow', () => {
  it('completes purchase', async () => {
    await login();
    await addToCart();
    await checkout();
    expect(orderConfirmation).toBeVisible();
  });
});
```
**Run on**: Pre-deployment
**Duration**: 15-60 minutes

### Performance Tests
```yaml
performance-test:
  script:
    - k6 run load-test.js
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
```

---

<nav class="page-nav">
  <a href="./">⬅ CI/CD Hub</a>
  <a href="deployment.html">Deployment Strategies ➡</a>
</nav>

<div class="see-also-card">
  <h4>See Also</h4>
  <ul>
    <li><a href="deployment.html">Deployment Strategies</a> — blue-green, canary, rolling, and feature flags</li>
    <li><a href="security-and-operations.html">Security, GitOps & Operations</a> — securing pipelines and GitOps</li>
    <li><a href="../docker/">Docker</a> — consistent build environments</li>
  </ul>
</div>
