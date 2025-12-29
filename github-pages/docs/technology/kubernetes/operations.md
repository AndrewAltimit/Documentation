---
layout: docs
title: "Kubernetes: Operations"
permalink: /docs/technology/kubernetes/operations.html
toc: true
toc_sticky: true
---

## kubectl Mastery: Command-Line Kubernetes

kubectl is your primary interface for managing Kubernetes clusters. This section covers the commands you will use most often, organized by task.

**Before you begin**: kubectl needs to know which cluster to talk to. This is configured through contexts, which combine a cluster, user, and namespace.

### Essential kubectl Commands

| Task | Command |
|------|---------|
| See what is running | `kubectl get pods` |
| Get more details | `kubectl describe pod <name>` |
| View logs | `kubectl logs <pod-name>` |
| Execute in container | `kubectl exec -it <pod> -- /bin/sh` |
| Apply configuration | `kubectl apply -f manifest.yaml` |
| Delete resources | `kubectl delete -f manifest.yaml` |

### Working with Multiple Clusters

```bash
# List available contexts
kubectl config get-contexts

# Switch to a different cluster
kubectl config use-context production-cluster

# Set default namespace for current context
kubectl config set-context --current --namespace=production
```

### Viewing Resources

```bash
# List resources with extra info
kubectl get pods -o wide
kubectl get all -n production

# Watch for changes in real-time
kubectl get pods -w

# Filter by labels
kubectl get pods -l environment=production
kubectl get pods -l 'tier in (frontend, backend)'
```

### Creating and Updating Resources

```bash
# Apply configuration (create or update)
kubectl apply -f deployment.yaml

# Quick edits via patch
kubectl patch deployment nginx -p '{"spec":{"replicas":5}}'

# Edit in your default editor
kubectl edit deployment nginx
```

### Debugging and Troubleshooting

```bash
# View pod logs
kubectl logs mypod
kubectl logs mypod --previous  # crashed container
kubectl logs -f -l app=nginx   # follow all matching pods

# Execute commands in container
kubectl exec -it mypod -- /bin/sh

# Port forward for local testing
kubectl port-forward svc/myservice 8080:80

# Check resource usage
kubectl top pods --sort-by=memory
kubectl top nodes
```

### Secrets and ConfigMaps

```bash
# Create secret from literal value
kubectl create secret generic db-creds --from-literal=password=mypass

# Decode a secret value
kubectl get secret db-creds -o jsonpath="{.data.password}" | base64 -d

# Create configmap from file
kubectl create configmap app-config --from-file=config.yaml
```

### Power User Tips

**Useful aliases** (add to your shell config):
```bash
alias k='kubectl'
alias kgp='kubectl get pods'
alias kaf='kubectl apply -f'
```

**JSONPath for extracting data**:
```bash
# Get all pod names
kubectl get pods -o jsonpath='{.items[*].metadata.name}'
```

**Node management**:
```bash
# Prepare node for maintenance
kubectl drain node1 --ignore-daemonsets --delete-emptydir-data
kubectl uncordon node1  # make schedulable again

# Mark node unschedulable (no drain)
kubectl cordon node1
```

### kubectl Plugins with Krew

Krew is a plugin manager for kubectl. Popular plugins:

| Plugin | Purpose |
|--------|---------|
| `ctx` | Quickly switch contexts |
| `ns` | Quickly switch namespaces |
| `tree` | Show resource hierarchy |
| `neat` | Clean up YAML output |

Install krew and a plugin:
```bash
kubectl krew install ctx
kubectl ctx production  # switch context
```

## Best Practices: Production Checklist

Before going to production, verify your setup against these categories:

### Resource Management

| Practice | Why It Matters |
|----------|----------------|
| Set resource requests/limits | Prevents resource starvation and runaway costs |
| Use namespaces | Isolate environments and teams |
| Label everything consistently | Enables filtering, monitoring, and cost allocation |
| Implement ResourceQuotas | Prevents one team from consuming all resources |

### High Availability

| Practice | Why It Matters |
|----------|----------------|
| Run 3+ replicas | Survives node failures |
| Use pod anti-affinity | Spreads pods across nodes/zones |
| Define PodDisruptionBudgets | Controls how many pods can be down during updates |
| Implement health probes | Ensures traffic only goes to healthy pods |

### Security

| Practice | Why It Matters |
|----------|----------------|
| Enable RBAC with least privilege | Limits blast radius of compromised accounts |
| Use NetworkPolicies | Prevents lateral movement between services |
| Run as non-root | Reduces container escape impact |
| Scan images for vulnerabilities | Catches known issues before deployment |

### Observability

| Practice | Why It Matters |
|----------|----------------|
| Centralize logs | Enables debugging after pod deletion |
| Expose metrics | Enables alerting and capacity planning |
| Implement distributed tracing | Debugs latency across services |
| Set up alerts | Catches issues before users notice |

## Helm: Kubernetes Package Manager

Managing dozens of YAML files for a single application becomes unwieldy. Helm solves this by packaging related resources into **charts** that can be versioned, shared, and customized.

**When to use Helm**:
- Deploying complex applications with many resources
- Sharing application configurations across teams
- Managing different configurations for different environments
- Installing third-party applications (databases, monitoring tools)

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Chart** | Package of Kubernetes resources |
| **Release** | An installed instance of a chart |
| **Values** | Configuration that customizes a chart |
| **Repository** | Collection of charts |

### Chart Structure
```
mychart/
├── Chart.yaml      # Metadata (name, version)
├── values.yaml     # Default configuration
├── templates/      # Kubernetes manifests with templating
│   ├── deployment.yaml
│   ├── service.yaml
│   └── _helpers.tpl
└── charts/         # Dependencies
```

### Common Helm Commands

```bash
# Install a chart
helm install myrelease ./mychart

# Install with custom values
helm install myrelease ./mychart -f production-values.yaml

# Preview what would be installed
helm install myrelease ./mychart --dry-run

# Upgrade an existing release
helm upgrade myrelease ./mychart

# Rollback to previous version
helm rollback myrelease 1

# List installed releases
helm list

# Uninstall a release
helm uninstall myrelease
```

### Using Public Charts

```bash
# Add a chart repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Search for charts
helm search repo postgresql

# Install from repository
helm install mydb bitnami/postgresql -f values.yaml
```

### Helm Best Practices

| Practice | Benefit |
|----------|---------|
| Use `--dry-run` before install | Catch errors early |
| Keep values in version control | Track configuration changes |
| Use separate values files per environment | Clean separation of concerns |
| Run `helm lint` before commits | Validate chart syntax |

## Common Patterns: Proven Architectural Approaches

These patterns appear repeatedly in successful Kubernetes deployments. Understanding when to use each helps you design better systems.

### Multi-Container Pod Patterns

| Pattern | Purpose | Example Use Case |
|---------|---------|------------------|
| **Sidecar** | Extend/enhance main container | Log forwarding, service mesh proxy |
| **Ambassador** | Proxy outbound connections | Database proxy, API gateway |
| **Adapter** | Standardize output format | Convert logs to Prometheus metrics |
| **Init Container** | Run setup before main container | Database migrations, wait for dependencies |

### Sidecar Pattern

A helper container runs alongside your application, sharing storage or network:

```yaml
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
  - name: log-forwarder
    image: fluentbit:latest
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
  volumes:
  - name: logs
    emptyDir: {}
```

**Common sidecars**: Logging agents, service mesh proxies (Envoy), security agents.

### Init Containers

Init containers run to completion before the main container starts:

```yaml
spec:
  initContainers:
  - name: wait-for-db
    image: busybox
    command: ['sh', '-c', 'until nc -z db 5432; do sleep 2; done']
  - name: migrate
    image: myapp:latest
    command: ['./migrate.sh']
  containers:
  - name: app
    image: myapp:latest
```

**Common uses**: Database migrations, waiting for dependencies, fetching configuration.


## Troubleshooting: When Things Go Wrong

When something breaks, a systematic approach saves time. Start broad, then narrow down.

### Quick Diagnostic Commands

```bash
# What is unhealthy?
kubectl get pods --all-namespaces | grep -v Running
kubectl get events --sort-by='.lastTimestamp' -A

# Why is this pod unhealthy?
kubectl describe pod <pod-name>
kubectl logs <pod-name> --previous
```

### Common Issues Quick Reference

| Issue | Symptom | First Command | Likely Cause |
|-------|---------|---------------|--------------|
| **ImagePullBackOff** | Pod stuck pulling image | `kubectl describe pod <name>` | Wrong image name, missing credentials |
| **CrashLoopBackOff** | Pod keeps restarting | `kubectl logs <pod> --previous` | App crash, OOM, bad config |
| **Pending** | Pod not scheduling | `kubectl describe pod <name>` | Insufficient resources, no matching nodes |
| **OOMKilled** | Container killed | `kubectl describe pod <name>` | Memory limit too low |

### ImagePullBackOff

The image cannot be pulled. Check:
1. Is the image name correct?
2. Does the registry require authentication?

```bash
# Create registry credentials
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<user> \
  --docker-password=<pass>
```

### CrashLoopBackOff

The container starts but crashes. Debug with:

```bash
kubectl logs <pod-name> --previous
kubectl describe pod <pod-name>
```

Common fixes:
- Increase memory limits if OOMKilled
- Extend `initialDelaySeconds` on probes if app needs time to start
- Check environment variables and config

### Pending Pods

Pod cannot be scheduled. Check events in:

```bash
kubectl describe pod <pod-name>
```

Common causes:
- **Insufficient resources**: Scale down other workloads or add nodes
- **Node selector/affinity**: No matching nodes exist
- **PVC pending**: Storage class or capacity issue

### Service Not Reachable

Debug network issues:

```bash
# Check if service has endpoints
kubectl get endpoints <service-name>

# Test from inside cluster
kubectl run debug --rm -it --image=busybox -- wget -O- <service>:<port>
```

If endpoints are empty, the service selector does not match any pod labels.

## Certification Path

If you want to validate your Kubernetes skills, consider these certifications:

| Certification | Focus | Prerequisites |
|---------------|-------|---------------|
| **CKA** | Cluster administration, troubleshooting | None |
| **CKAD** | Application development, configuration | None |
| **CKS** | Security hardening, runtime security | CKA required |

All exams are hands-on, performance-based tests where you solve real Kubernetes problems in a live environment.

## Summary

This guide covered the operational aspects of running Kubernetes:

- **kubectl**: Your primary tool for cluster interaction
- **Helm**: Package management for complex applications
- **Patterns**: Proven approaches like sidecars and init containers
- **Troubleshooting**: Systematic debugging of common issues
- **Best Practices**: Production-ready configurations

The key to Kubernetes mastery is practice. Start with simple deployments, gradually add complexity, and always follow the principle of declarative configuration: describe what you want, and let Kubernetes make it happen.
