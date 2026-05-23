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

## Learning Path

Work through the four guides in order, or jump to the one matching your task.

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="fundamentals.html">1. Fundamentals</a></h4>
    <p>Start here. Images vs containers, essential commands, image layering, containers vs VMs, and bridge networking basics.</p>
  </div>
  <div class="nav-card">
    <h4><a href="storage-security.html">2. Storage &amp; Security</a></h4>
    <p>Persist data with volumes, bind mounts, and tmpfs; choose network drivers; harden containers; install and troubleshoot.</p>
  </div>
  <div class="nav-card">
    <h4><a href="dockerfiles.html">3. Dockerfiles &amp; CI/CD</a></h4>
    <p>Write and optimize Dockerfiles, multi-stage builds, Docker Swarm, and pipelines with GitHub Actions and GitLab.</p>
  </div>
  <div class="nav-card">
    <h4><a href="advanced.html">4. Advanced Patterns</a></h4>
    <p>Production architectures, real case studies, design patterns, and WebAssembly as a next-gen container runtime.</p>
  </div>
</div>

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
