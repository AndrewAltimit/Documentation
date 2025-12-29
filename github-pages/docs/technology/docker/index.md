---
layout: docs
title: Docker
permalink: /docs/technology/docker/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #00aaff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Containers</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Build, Ship, and Run Anywhere</p>
</div>

<div class="intro-card">
  <p class="lead-text">Docker revolutionizes application deployment by solving the "it works on my machine" problem. Containers package applications with all their dependencies into lightweight, portable units that run identically across development, testing, and production environments.</p>
</div>

## Why Learn Docker?

Before diving into containers, consider what problems they solve:

- **Environment consistency**: Your application behaves the same way on every machine, from your laptop to production servers
- **Simplified onboarding**: New team members can start contributing within minutes instead of spending days configuring their environment
- **Efficient resource usage**: Run more applications on the same hardware compared to traditional virtual machines
- **Streamlined deployments**: Package once, deploy anywhere with confidence

Whether you are a developer tired of debugging environment-specific issues or an operations engineer looking to standardize deployments, Docker provides the foundation for modern application delivery.

---

## Quick Navigation

### [Fundamentals](fundamentals.html)
Start here to understand what containers are and how to use them.

- Why Docker? The real problems it solves for teams
- Core concepts: images, containers, and registries
- Essential commands for daily work
- How containers differ from virtual machines
- Docker network basics

### [Storage & Security](storage-security.html)
Learn how to persist data and protect your containers.

- Choosing between volumes, bind mounts, and tmpfs
- Network drivers and when to use each
- Security hardening for production
- Troubleshooting common issues
- Installation and configuration

### [Dockerfiles & CI/CD](dockerfiles.html)
Build custom images and integrate with your development workflow.

- Writing effective Dockerfiles from scratch
- Optimization techniques for smaller, faster images
- Multi-stage builds for production
- Docker Swarm for orchestration
- CI/CD integration with GitHub Actions, GitLab, and Jenkins

### [Advanced Patterns](advanced.html)
Real-world examples and techniques for complex deployments.

- Production-ready patterns and architectures
- Case studies from real applications
- WebAssembly and the future of containers
- Advanced multi-stage build strategies
- Deployment strategies for zero-downtime releases

---

## Key Capabilities

Understanding what makes containers different helps you appreciate when to use them.

<div class="key-insights">
  <div class="insight-card">
    <h4>Lightweight</h4>
    <p>Share host OS kernel</p>
  </div>
  <div class="insight-card">
    <h4>Fast Startup</h4>
    <p>Seconds vs minutes</p>
  </div>
  <div class="insight-card">
    <h4>Portable</h4>
    <p>Run anywhere consistently</p>
  </div>
</div>

| Capability | Containers | Virtual Machines |
|------------|------------|------------------|
| Startup time | Seconds | Minutes |
| Memory overhead | Minimal (shared kernel) | High (full OS per VM) |
| Disk usage | MBs | GBs |
| Isolation level | Process-level | Hardware-level |
| Best for | Microservices, CI/CD | Legacy apps, different OS |

---

## Quick Reference

Looking for a quick command reference? See [Docker Essentials](../docker-essentials.html) for commonly used commands.

---

## See Also

- [Docker Essentials](../docker-essentials.html) - Quick command reference
- [Kubernetes](../kubernetes/) - Container orchestration
- [AWS ECS](../aws/compute.html) - Managed container service
- [CI/CD](../ci-cd.html) - Continuous deployment
