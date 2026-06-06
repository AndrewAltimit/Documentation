---
layout: docs
title: "CI/CD: Deployment Strategies"
permalink: /docs/technology/ci-cd/deployment.html
toc: true
toc_sticky: true
---

[CI/CD](./) ›

Blue-green, canary, rolling, and feature-flag deployments release changes without taking the system down. Each strategy below trades speed for safety in a different way. Pair any of them with automated rollback so a bad release is always reversible.

## 1. Blue-Green Deployment

Run two identical environments ("blue" and "green"); deploy to the inactive one, verify it, then switch traffic over. The old environment stays available as an instant rollback target.

```yaml
deploy-blue-green:
  steps:
    # Deploy to inactive environment
    - name: Deploy to Blue
      run: |
        kubectl set image deployment/app app=myapp:$VERSION -n blue
        kubectl wait --for=condition=ready pod -l app=myapp -n blue

    # Test new version
    - name: Smoke Test Blue
      run: ./smoke-test.sh https://blue.example.com

    # Switch traffic
    - name: Switch to Blue
      run: kubectl patch service app -p '{"spec":{"selector":{"env":"blue"}}}'

    # Keep green as rollback option
    - name: Tag Green for Rollback
      run: kubectl label deployment/app version=previous -n green
```

**Use when**: Zero-downtime deployments are critical and you can afford to run two full environments.

## 2. Canary Deployment

```yaml
canary-deploy:
  steps:
    # Deploy canary (10% traffic)
    - name: Deploy Canary
      run: |
        kubectl apply -f canary-deployment.yaml
        kubectl set image deployment/app-canary app=myapp:$VERSION

    # Monitor metrics
    - name: Monitor Canary
      run: |
        for i in {1..10}; do
          ERROR_RATE=$(curl -s https://metrics.api/error-rate)
          if [ $ERROR_RATE -gt 5 ]; then
            echo "High error rate detected!"
            exit 1
          fi
          sleep 60
        done

    # Promote canary
    - name: Promote Canary
      run: kubectl set image deployment/app app=myapp:$VERSION
```

## 3. Rolling Deployment

```yaml
rolling-deploy:
  steps:
    - name: Configure Rolling Update
      run: |
        kubectl patch deployment app -p '{
          "spec": {
            "strategy": {
              "type": "RollingUpdate",
              "rollingUpdate": {
                "maxSurge": 1,
                "maxUnavailable": 0
              }
            }
          }
        }'

    - name: Update Image
      run: kubectl set image deployment/app app=myapp:$VERSION

    - name: Monitor Rollout
      run: kubectl rollout status deployment/app --timeout=10m
```

## 4. Feature Flags Deployment

```javascript
// Deploy code but control feature activation
if (featureFlag.isEnabled('new-checkout-flow')) {
  return renderNewCheckout();
} else {
  return renderOldCheckout();
}
```

```yaml
feature-flag-deploy:
  steps:
    - name: Deploy with Feature Off
      run: |
        FEATURE_FLAGS='{"new-checkout-flow": false}' \
        npm run deploy

    - name: Gradual Rollout
      run: |
        for percentage in 10 25 50 100; do
          ./set-feature-flag.sh new-checkout-flow $percentage
          sleep 3600  # Monitor for 1 hour
        done
```

---

<nav class="page-nav">
  <a href="platforms-and-pipelines.html">⬅ Platforms & Pipeline Design</a>
  <a href="security-and-operations.html">Security, GitOps & Operations ➡</a>
</nav>

## See Also

- [Platforms & Pipeline Design](platforms-and-pipelines.html) — the blue-green pipeline pattern
- [Security, GitOps & Operations](security-and-operations.html) — monitoring rollouts and GitOps
- [Kubernetes](../kubernetes/) — the orchestration behind these rollout patterns
