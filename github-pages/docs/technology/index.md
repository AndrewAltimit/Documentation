---
layout: docs
title: Technology Documentation Hub
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Technology Documentation Hub</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">A reference library for DevOps, cloud, data, and modern software infrastructure</p>
</div>

<div class="intro-card">
  <p class="lead-text">This is a practical, reference-oriented knowledge base spanning the full software delivery stack — from the network packets and database rows beneath an application, through the version control and CI/CD that ship it, up to the containers, orchestration, and cloud platforms that run it in production. Pages favor concrete commands, comparison tables, and decision guidance over tutorials.</p>
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-layer-group"></i>
    <h4>Full Stack</h4>
    <p>Foundations, tooling, data, and infrastructure in one place</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-terminal"></i>
    <h4>Command-First</h4>
    <p>Copy-paste examples, cheat sheets, and config snippets</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-route"></i>
    <h4>Decision Guidance</h4>
    <p>When to use X vs Y, with comparison tables and trade-offs</p>
  </div>
</div>

## Browse by Area

### Infrastructure & DevOps

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="docker/">Docker</a></h4>
    <p>Containerization fundamentals, Dockerfiles, storage, and security. Start here for images and containers.</p>
  </div>
  <div class="nav-card">
    <h4><a href="docker-essentials.html">Docker Essentials</a></h4>
    <p>A daily-driver command cheat sheet — run, build, compose, network, and clean up.</p>
  </div>
  <div class="nav-card">
    <h4><a href="kubernetes/">Kubernetes</a></h4>
    <p>Container orchestration at scale: pods, deployments, services, and production patterns.</p>
  </div>
  <div class="nav-card">
    <h4><a href="terraform/">Terraform</a></h4>
    <p>Infrastructure as Code for multi-cloud provisioning, state, and modules.</p>
  </div>
  <div class="nav-card">
    <h4><a href="aws/">AWS</a></h4>
    <p>Core cloud services — compute, storage, databases, and networking on Amazon Web Services.</p>
  </div>
  <div class="nav-card">
    <h4><a href="ci-cd.html">CI/CD</a></h4>
    <p>Pipelines from code to production: testing strategies, deployment patterns, and GitOps.</p>
  </div>
</div>

### Development & Tools

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="git-crash-course.html">Git Crash Course</a></h4>
    <p>Zero-to-productive in version control. The fastest on-ramp if you are new to Git.</p>
  </div>
  <div class="nav-card">
    <h4><a href="git.html">Git Version Control</a></h4>
    <p>Architecture and internals: the object model, the DAG, and how Git actually works.</p>
  </div>
  <div class="nav-card">
    <h4><a href="git-reference.html">Git Command Reference</a></h4>
    <p>The lookup cheat sheet — every common command with syntax and examples.</p>
  </div>
  <div class="nav-card">
    <h4><a href="branching.html">Branching Strategies</a></h4>
    <p>Git Flow vs GitHub Flow vs trunk-based development, with a decision matrix.</p>
  </div>
  <div class="nav-card">
    <h4><a href="please-build.html">Please Build</a></h4>
    <p>A high-performance, Bazel-style build system for polyglot monorepos.</p>
  </div>
  <div class="nav-card">
    <h4><a href="unreal.html">Unreal Engine</a></h4>
    <p>UE5 real-time 3D: Nanite, Lumen, and Blueprints for games and beyond.</p>
  </div>
</div>

### Data

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="database-crash-course.html">Database Crash Course</a></h4>
    <p>Core concepts and SQL basics — the quick on-ramp to working with databases.</p>
  </div>
  <div class="nav-card">
    <h4><a href="database-design.html">Database Design</a></h4>
    <p>Deep dive: normalization, indexing, query execution, distributed databases, and NoSQL.</p>
  </div>
  <div class="nav-card">
    <h4><a href="networking.html">Networking</a></h4>
    <p>TCP/IP, routing, congestion control, and modern network architecture.</p>
  </div>
</div>

### Security

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="cybersecurity.html">Cybersecurity</a></h4>
    <p>Cryptography, web/cloud security, attack techniques, and incident response.</p>
  </div>
</div>

### Advanced & Emerging

<div class="command-grid">
  <div class="nav-card">
    <h4><a href="ai.html">AI Fundamentals</a></h4>
    <p>Comprehensive technical overview of modern AI and large language models.</p>
  </div>
  <div class="nav-card">
    <h4><a href="quantumcomputing.html">Quantum Computing</a></h4>
    <p>Quantum algorithms, the NISQ era, and quantum programming platforms.</p>
  </div>
</div>

> **Note on layout:** The platform topics — **Docker**, **Kubernetes**, **AWS**, and **Terraform** — are multi-page sections living in their own subdirectories (e.g. `docker/`, `kubernetes/`). Their links above point to each section's landing page. Everything else is a single reference page (`.html`).

## How These Topics Connect

A typical web application sits on top of the foundations and is delivered by the tooling and infrastructure layers below:

```mermaid
flowchart TD
    subgraph Foundations
        NET[Networking]
        DB[Databases]
        SEC[Cybersecurity]
    end
    subgraph Tooling
        GIT[Git]
        CI[CI/CD]
        BUILD[Please Build]
    end
    subgraph Infrastructure
        DOCKER[Docker]
        K8S[Kubernetes]
        TF[Terraform]
        AWS[AWS]
    end
    APP([Application])
    NET --> APP
    DB --> APP
    GIT --> CI
    BUILD --> CI
    CI --> DOCKER
    DOCKER --> K8S
    TF --> AWS
    K8S --> AWS
    APP --> DOCKER
    SEC -.guards.-> APP
    SEC -.guards.-> K8S
    SEC -.guards.-> NET
```

Each layer builds on the ones beneath it: code lives in **Git**, is built and tested by **CI/CD**, packaged into **Docker** images, orchestrated by **Kubernetes**, and runs on infrastructure provisioned with **Terraform** on a cloud like **AWS** — all underpinned by **networking**, **databases**, and **security**.

## Suggested Learning Paths

<div class="command-grid">
  <div class="step-card">
    <h4>New to the field</h4>
    <p>Build foundations first: <a href="networking.html">Networking</a> → <a href="database-crash-course.html">Database Crash Course</a> → <a href="git-crash-course.html">Git Crash Course</a>.</p>
  </div>
  <div class="step-card">
    <h4>Learning DevOps</h4>
    <p><a href="git.html">Git</a> → <a href="ci-cd.html">CI/CD</a> → <a href="docker/">Docker</a> → <a href="kubernetes/">Kubernetes</a> → <a href="terraform/">Terraform</a>.</p>
  </div>
  <div class="step-card">
    <h4>Cloud / platform engineer</h4>
    <p>Focus on <a href="aws/">AWS</a>, <a href="terraform/">Terraform</a>, and <a href="kubernetes/">Kubernetes</a>, with <a href="cybersecurity.html">Cybersecurity</a> throughout.</p>
  </div>
  <div class="step-card">
    <h4>Backend / data</h4>
    <p><a href="database-design.html">Database Design</a> for modeling and scaling, plus <a href="networking.html">Networking</a> for performance.</p>
  </div>
</div>

## Related Documentation

<div class="see-also-card">
  <h4>See Also</h4>
  <ul>
    <li><a href="../ai-ml/">AI/ML Hub</a> — Stable Diffusion, ComfyUI, LoRA training, and generative AI</li>
    <li><a href="../quantum-computing/">Quantum Computing Hub</a> — from quantum theory to programming</li>
    <li><a href="../distributed-systems/">Distributed Systems</a> — consensus, replication, and architecture patterns</li>
    <li><a href="../reference/">Reference Sheets</a> — quick command and configuration cheat sheets</li>
    <li><a href="../physics/">Physics Documentation</a> — quantum mechanics underlying quantum computing</li>
  </ul>
</div>

---

*This documentation combines reference depth with practical examples. For corrections or suggestions, visit the [GitHub repository](https://github.com/AndrewAltimit/Documentation).*
