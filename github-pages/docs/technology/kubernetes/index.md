---
layout: docs
title: Kubernetes
permalink: /docs/technology/kubernetes/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #326ce5 0%, #54a3ff 100%); color: white; padding: 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.25rem;">Kubernetes</h1>
  <p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.9;">Container orchestration at scale</p>
</div>

Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. Originally built at Google and now maintained by the Cloud Native Computing Foundation (CNCF), it is the de facto standard for running containers in production.

## Why Kubernetes?

Running containers on a single machine is straightforward. The hard part is running hundreds of them across dozens of servers while keeping them healthy, absorbing traffic spikes, and deploying updates without downtime. When a web application outgrows a single server, you need to run multiple copies across servers, restart crashed containers automatically, route requests to healthy instances, roll out new versions without interrupting service, and scale up at peak and down at quiet times. Without Kubernetes that means custom scripts, manual intervention, and constant monitoring. Kubernetes handles it through a declarative model: you describe the desired state and it continuously reconciles reality toward it.

## Learning Path

The guides build on each other. Start with the Fundamentals track if you are new; jump ahead if you already run clusters.

### Fundamentals

Three focused pages covering everything you need to run real workloads.

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="fundamentals.html">Part I: Core Concepts</a></h4>
    <p>Start here. Cluster architecture, the apply control flow, Pods, Deployments, labels, and namespaces.</p>
  </div>
  <div class="nav-card">
    <h4><a href="fundamentals-networking.html">Networking &amp; Configuration</a></h4>
    <p>Services and kube-proxy, Ingress, NetworkPolicies, ConfigMaps and Secrets, and RBAC.</p>
  </div>
  <div class="nav-card">
    <h4><a href="fundamentals-resources.html">Health &amp; Resource Management</a></h4>
    <p>Liveness/readiness probes, requests and limits, QoS classes, scheduling, and horizontal autoscaling.</p>
  </div>
</div>

### Stateful Workloads & Operations

Going beyond stateless apps: storage, controllers, and day-two operations.

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="workloads.html">Workloads</a></h4>
    <p>StatefulSets, DaemonSets, Jobs/CronJobs, autoscaling, and Pod Security for managing diverse application types.</p>
  </div>
  <div class="nav-card">
    <h4><a href="persistence.html">Stateful Workloads &amp; Persistence</a></h4>
    <p>Persistent volumes, dynamic provisioning, StatefulSet ordering, backup and disaster recovery, and database patterns.</p>
  </div>
  <div class="nav-card">
    <h4><a href="operations.html">Operations</a></h4>
    <p>kubectl power use, Helm, sidecar/init-container patterns, a systematic troubleshooting guide, and a production checklist.</p>
  </div>
</div>

### Going Further

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="advanced.html">Advanced Topics</a></h4>
    <p>CRDs and Operators, service mesh, GitOps, performance tuning, certifications (CKA/CKAD/CKS), and the ecosystem.</p>
  </div>
</div>

---

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
- [CI/CD](../ci-cd/) - Continuous deployment pipelines
