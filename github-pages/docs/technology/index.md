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

This is a practical, reference-oriented knowledge base spanning the full software delivery stack — from the network packets and database rows beneath an application, through the version control and CI/CD that ship it, up to the containers, orchestration, and cloud platforms that run it in production. Pages favor concrete commands, comparison tables, and decision guidance over tutorials: the **full stack** in one place, **command-first** examples and cheat sheets, and **decision guidance** on when to use X vs Y.

## Browse by Area

### Infrastructure & DevOps

| Topic | Covers |
|-------|--------|
| [Docker](docker/) | Containerization fundamentals, Dockerfiles, storage, and security. Start here for images and containers. |
| [Docker Essentials](docker-essentials.html) | A daily-driver command cheat sheet — run, build, compose, network, and clean up. |
| [Kubernetes](kubernetes/) | Container orchestration at scale: pods, deployments, services, and production patterns. |
| [Terraform](terraform/) | Infrastructure as Code for multi-cloud provisioning, state, and modules. |
| [AWS](aws/) | Core cloud services — compute, storage, databases, and networking on Amazon Web Services. |
| [CI/CD](ci-cd/) | Pipelines from code to production: testing strategies, deployment patterns, and GitOps. |

### Development & Tools

| Topic | Covers |
|-------|--------|
| [Git Crash Course](git-crash-course.html) | Zero-to-productive in version control. The fastest on-ramp if you are new to Git. |
| [Git Version Control](git/) | Architecture and internals: the object model, the DAG, and how Git actually works. |
| [Git Command Reference](git-reference.html) | The lookup cheat sheet — every common command with syntax and examples. |
| [Branching Strategies](branching.html) | Git Flow vs GitHub Flow vs trunk-based development, with a decision matrix. |
| [Please Build](please-build.html) | A high-performance, Bazel-style build system for polyglot monorepos. |
| [Unreal Engine](unreal.html) | UE5 real-time 3D: Nanite, Lumen, and Blueprints for games and beyond. |

### Data

| Topic | Covers |
|-------|--------|
| [Database Crash Course](database-crash-course.html) | Core concepts and SQL basics — the quick on-ramp to working with databases. |
| [Database Design](database-design/) | Deep dive: normalization, indexing, query execution, distributed databases, and NoSQL. |
| [Networking](networking/) | TCP/IP, routing, congestion control, and modern network architecture. |

### Security

| Topic | Covers |
|-------|--------|
| [Cybersecurity](cybersecurity/) | Cryptography, web/cloud security, attack techniques, and incident response. |

### Advanced & Emerging

| Topic | Covers |
|-------|--------|
| [AI Fundamentals](ai/) | Comprehensive technical overview of modern AI and large language models. |
| [Quantum Computing](quantumcomputing.html) | Quantum algorithms, the NISQ era, and quantum programming platforms. |

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

- **New to the field** — build foundations first: [Networking](networking/) → [Database Crash Course](database-crash-course.html) → [Git Crash Course](git-crash-course.html).
- **Learning DevOps** — [Git](git/) → [CI/CD](ci-cd/) → [Docker](docker/) → [Kubernetes](kubernetes/) → [Terraform](terraform/).
- **Cloud / platform engineer** — focus on [AWS](aws/), [Terraform](terraform/), and [Kubernetes](kubernetes/), with [Cybersecurity](cybersecurity/) throughout.
- **Backend / data** — [Database Design](database-design/) for modeling and scaling, plus [Networking](networking/) for performance.

## Related Documentation

- [AI/ML Hub](../ai-ml/) — Stable Diffusion, ComfyUI, LoRA training, and generative AI
- [Quantum Computing Hub](../quantum-computing/) — from quantum theory to programming
- [Distributed Systems](../distributed-systems/) — consensus, replication, and architecture patterns
- [Reference Sheets](../reference/) — quick command and configuration cheat sheets
- [Physics Documentation](../physics/) — quantum mechanics underlying quantum computing

---

*This documentation combines reference depth with practical examples. For corrections or suggestions, visit the [GitHub repository](https://github.com/AndrewAltimit/Documentation).*
