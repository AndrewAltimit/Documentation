---
layout: docs
title: Kubernetes
permalink: /docs/technology/kubernetes/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #326ce5 0%, #54a3ff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Kubernetes</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Container Orchestration at Scale</p>
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

## Learning Path

The four guides build on each other. Start at Fundamentals if you are new; jump ahead if you already run clusters.

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="fundamentals.html">1. Fundamentals</a></h4>
    <p>Start here. Cluster architecture, the apply control flow, Pods, Deployments, Services, Ingress, labels, and namespaces.</p>
  </div>
  <div class="nav-card">
    <h4><a href="workloads.html">2. Workloads &amp; Storage</a></h4>
    <p>StatefulSets, DaemonSets, Jobs/CronJobs, persistent volumes, autoscaling, ConfigMaps/Secrets, RBAC, and Pod Security.</p>
  </div>
  <div class="nav-card">
    <h4><a href="operations.html">3. Operations</a></h4>
    <p>kubectl power use, Helm, sidecar/init-container patterns, a systematic troubleshooting guide, and a production checklist.</p>
  </div>
  <div class="nav-card">
    <h4><a href="advanced.html">4. Advanced Topics</a></h4>
    <p>CRDs and Operators, service mesh, GitOps, performance tuning, certifications (CKA/CKAD/CKS), and the ecosystem.</p>
  </div>
</div>

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
