---
layout: docs
title: Docker
permalink: /docs/technology/docker/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #00aaff 100%); color: white; padding: 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.25rem;">Containers</h1>
  <p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.9;">Build, ship, and run anywhere</p>
</div>

Docker solves the "it works on my machine" problem by packaging applications with all their dependencies into lightweight, portable units that run identically across development, testing, and production.

## Why Learn Docker?

Containers buy four things: **environment consistency** (the same behavior on a laptop and in production), **fast onboarding** (a new contributor runs one `docker run` instead of spending days configuring tools), **efficient resource use** (higher density than virtual machines), and **simple deployments** (package once, deploy anywhere). They are the foundation of modern application delivery for developers and operations engineers alike.

---

## Learning Path

Work through the guides in order, or jump to the one matching your task.

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
    <h4><a href="docker-networking.html">4. Networking</a></h4>
    <p>Network drivers, DNS-based service discovery, port publishing, multi-host overlays, and locking the network down.</p>
  </div>
  <div class="nav-card">
    <h4><a href="registry.html">5. Registries &amp; Supply Chain</a></h4>
    <p>Distribute images through registries, pin by digest, and prove what you ship with signing, SBOMs, scanning, and provenance.</p>
  </div>
  <div class="nav-card">
    <h4><a href="docker-design-patterns.html">6. Design Patterns</a></h4>
    <p>Multi-container composition patterns, hardened images, and runtime security for production workloads.</p>
  </div>
  <div class="nav-card">
    <h4><a href="advanced.html">7. Advanced Patterns</a></h4>
    <p>Production architectures, real case studies, design patterns, and WebAssembly as a next-gen container runtime.</p>
  </div>
  <div class="nav-card">
    <h4><a href="../container-runtimes.html">8. Container Runtimes</a></h4>
    <p>Beyond Docker: the OCI spec, runc/containerd/CRI-O, sandboxed runtimes (gVisor, Kata), Firecracker microVMs, and Wasm.</p>
  </div>
</div>

<p style="margin-top: 1.5rem; padding: 0.75rem 1rem; border-left: 4px solid #0066cc; background: rgba(0, 102, 204, 0.08); border-radius: 4px;"><strong><span style="display: inline-block; background: #0066cc; color: white; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em; padding: 0.15rem 0.5rem; border-radius: 3px; vertical-align: middle;">QUICK REFERENCE</span> <a href="../docker-essentials.html">Docker Essentials</a></strong> is a side-by-side command cheat sheet, <em>not</em> a step in the sequential path above. Keep it open while you work through the guides.</p>

---

## Key Capabilities

Containers are lightweight (they share the host kernel), start in seconds rather than minutes, and run consistently anywhere. The contrast with virtual machines makes the trade-offs concrete:

| Capability | Containers | Virtual Machines |
|------------|------------|------------------|
| Startup time | Seconds | Minutes |
| Memory overhead | Minimal (shared kernel) | High (full OS per VM) |
| Disk usage | MBs | GBs |
| Isolation level | Process-level | Hardware-level |
| Best for | Microservices, CI/CD | Legacy apps, different OS |

---

## See Also

- [Docker Essentials](../docker-essentials.html) - Quick command reference
- [Container Runtimes](../container-runtimes.html) - Runtimes beyond Docker
- [Kubernetes](../kubernetes/) - Container orchestration
- [AWS ECS](../aws/compute.html) - Managed container service
- [CI/CD](../ci-cd/) - Continuous deployment
