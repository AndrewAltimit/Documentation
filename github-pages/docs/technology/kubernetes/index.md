---
layout: docs
title: Kubernetes
permalink: /docs/technology/kubernetes/
toc: false
---

<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">Kubernetes</h1>
    <p class="hero-subtitle">Container Orchestration at Scale</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. Originally developed by Google and now maintained by the Cloud Native Computing Foundation (CNCF), Kubernetes has become the de facto standard for container orchestration in production environments.</p>
</div>

## Why Kubernetes?

Running containers on a single machine is straightforward. But what happens when you need to run hundreds of containers across dozens of servers, ensure they stay healthy, handle traffic spikes, and deploy updates without downtime? This is where Kubernetes becomes essential.

**Consider the following scenario**: Your web application runs in containers and traffic has grown beyond what a single server can handle. You need to:
- Run multiple copies of your application across different servers
- Automatically restart crashed containers
- Route user requests to healthy instances
- Deploy new versions without interrupting service
- Scale up during peak hours and down during quiet periods

Without Kubernetes, you would need custom scripts, manual intervention, and constant monitoring. Kubernetes handles all of this automatically through a declarative approach: you describe what you want, and Kubernetes makes it happen.

## Quick Navigation

### [Fundamentals](fundamentals.html)
**Start here if you are new to Kubernetes.** Learn the building blocks that everything else depends on.

- Quick start guide to deploy your first application
- Core architecture: how Kubernetes works under the hood
- Pods, Deployments, and ReplicaSets explained
- Services: giving your applications stable network addresses
- Labels, selectors, and namespaces for organization

### [Workloads & Storage](workloads.html)
**Move here once you understand the basics.** Learn how to handle real-world requirements like persistent data and specialized workloads.

- StatefulSets for databases and stateful applications
- DaemonSets for cluster-wide agents (monitoring, logging)
- Jobs and CronJobs for batch processing
- Persistent storage that survives pod restarts
- Configuration and secrets management
- Security hardening for production

### [Operations](operations.html)
**Essential for anyone managing Kubernetes clusters.** Day-to-day tools and techniques for running reliable systems.

- kubectl command reference and power-user tips
- Helm: package management for Kubernetes
- Proven architectural patterns (sidecar, ambassador, init containers)
- Troubleshooting guide for common issues
- Production best practices

### [Advanced Topics](advanced.html)
**For experienced practitioners.** Deep dives into production-grade deployments and the broader ecosystem.

- Real-world case studies from production environments
- Certification paths (CKA, CKAD, CKS)
- Performance tuning and optimization
- Ecosystem tools and integrations

---

## Key Capabilities

<div class="key-insights">
  <div class="insight-card">
    <h4>Container Orchestration</h4>
    <p>Automated deployment and management</p>
  </div>
  <div class="insight-card">
    <h4>Auto-scaling</h4>
    <p>Dynamic resource allocation</p>
  </div>
  <div class="insight-card">
    <h4>Self-healing</h4>
    <p>Automatic recovery and rollbacks</p>
  </div>
</div>

## When to Use Kubernetes

Kubernetes adds complexity, so it is important to understand when it provides value:

| Scenario | Kubernetes? | Why |
|----------|-------------|-----|
| Single application on one server | No | Docker Compose is simpler |
| Multiple services, need scaling | Yes | Automated scaling and load balancing |
| Microservices architecture | Yes | Service discovery and networking built-in |
| Need zero-downtime deployments | Yes | Rolling updates are native |
| Consistent dev/staging/prod | Yes | Same configuration across environments |
| Team needs self-service deployment | Yes | Declarative configs enable GitOps |

**Not ready for Kubernetes yet?** Start with [Docker](../docker/) to learn container fundamentals first.

---

## See Also

- [Docker](../docker/) - Container fundamentals
- [AWS EKS](../aws/compute.html) - Managed Kubernetes on AWS
- [Terraform](../terraform/) - Infrastructure as code for K8s
- [CI/CD](../ci-cd.html) - Continuous deployment pipelines
