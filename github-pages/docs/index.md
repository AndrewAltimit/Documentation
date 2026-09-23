---
layout: docs
title: Complete Documentation Index
description: Every page in Andrew's Notebook, grouped by section and topic, with the overview page for each topic and all of its sub-pages.
toc: false  # Index pages typically don't need TOC
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #4facfe 100%);">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Documentation Index</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.92;">Every page on the site, grouped by section and topic.</p>
</div>

This is the complete listing of the site's content. Each table row is one topic: the **Overview** column links the page that introduces the topic and orients you, and the **Sub-pages** column lists every focused page that goes deeper. Rows run roughly from introductory to advanced within each section.

For guided entry points, see [Getting Started](../getting-started.html); for a picture of how topics depend on one another, see the [Topic Map](topic-map.html). The site [search](../search.html) indexes the main hub and overview pages, so this index is the reliable way to find a specific sub-page.

| Section | What it covers | Hub |
|---------|----------------|-----|
| [Technology](#technology) | Containers, orchestration, cloud, IaC, version control, databases, networking, security | [Technology Hub](technology/) |
| [Architecture & Operations](#architecture--operations) | Distributed systems, APIs, event-driven design, observability, testing, performance | [Distributed Systems Hub](distributed-systems/) |
| [AI & Machine Learning](#ai--machine-learning) | ML theory and architectures, plus hands-on generative image models | [AI Hub](artificial-intelligence/) |
| [Game Development & Graphics](#game-development--graphics) | Engines, rendering, shaders, multiplayer, XR | [Game Development](gamedev/) |
| [Physics](#physics) | Classical mechanics through quantum field theory, string theory, and computation | [Physics Hub](physics/) |
| [Theory & Research](#theory--research) | Graduate-level computer science and mathematics | [Advanced Topics Hub](advanced/) |
| [Reference](#reference--navigation) | Cheat sheets and navigation aids | [Quick Reference](reference/) |

## Technology

The software delivery stack, from the packets and rows beneath an application to the pipelines and clouds that run it. The [Technology Hub](technology/) suggests reading orders.

### Infrastructure and DevOps

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Docker | [Docker](technology/docker/) · [Docker Essentials](technology/docker-essentials.html) (command cheat sheet) | [Fundamentals](technology/docker/fundamentals.html) · [Dockerfiles & CI/CD](technology/docker/dockerfiles.html) · [Networking](technology/docker/docker-networking.html) · [Storage & Security](technology/docker/storage-security.html) · [Registries & Supply-Chain Security](technology/docker/registry.html) · [Design Patterns](technology/docker/docker-design-patterns.html) · [Production Patterns](technology/docker/advanced.html) · [Container Runtimes & Alternatives](technology/container-runtimes.html) |
| Kubernetes | [Kubernetes](technology/kubernetes/) | [Fundamentals: Architecture & Core Objects](technology/kubernetes/fundamentals.html) · [Networking & Configuration](technology/kubernetes/fundamentals-networking.html) · [Health & Resource Management](technology/kubernetes/fundamentals-resources.html) · [Workloads & Storage](technology/kubernetes/workloads.html) · [Stateful Workloads & Persistence](technology/kubernetes/persistence.html) · [Operations](technology/kubernetes/operations.html) · [Advanced Topics](technology/kubernetes/advanced.html) |
| Terraform | [Terraform](technology/terraform/) | [Core Concepts](technology/terraform/core-concepts.html) · [State & Modules](technology/terraform/state-modules.html) · [Enterprise Patterns](technology/terraform/patterns.html) · [Advanced Topics & Troubleshooting](technology/terraform/advanced.html) |
| AWS | [AWS Cloud Services](technology/aws/) | [Compute](technology/aws/compute.html) · [Storage](technology/aws/storage.html) · [Databases](technology/aws/databases.html) · [Networking & Content Delivery](technology/aws/networking.html) · [Security & Identity](technology/aws/security.html) · [Monitoring & Messaging](technology/aws/monitoring.html) · [Infrastructure as Code](technology/aws/iac.html) · [Cost Optimization](technology/aws/cost.html) · [Architecture Patterns](technology/aws/architecture.html) · [Troubleshooting](technology/aws/troubleshooting.html) |
| CI/CD | [CI/CD](technology/ci-cd/) | [Platforms & Pipeline Design](technology/ci-cd/platforms-and-pipelines.html) · [Deployment Strategies](technology/ci-cd/deployment.html) · [Security, GitOps & Operations](technology/ci-cd/security-and-operations.html) |
| Build systems and monorepos | [Monorepo Strategies](advanced/monorepo/) | [Monorepo Tooling & Build Systems](advanced/monorepo-tooling/) · [Scaling & Engineering](advanced/monorepo-scaling/) · [Please Build](technology/please-build.html) |

### Version control

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Git usage | [Git Crash Course](technology/git-crash-course.html) (learning path) · [Git Command Reference](technology/git-reference.html) (lookup) | [Branching Strategies](technology/branching.html) · [Advanced Branching Techniques](technology/advanced-branching-techniques.html) · [Conflict Resolution & Recovery](technology/git/conflict-and-recovery.html) · [Authentication & Access Control](technology/git/auth-and-access-control.html) |
| Git internals | [Git Internals](technology/git/) | [Object Model & Storage](technology/git/object-model.html) · [Algorithms & Advanced Operations](technology/git/algorithms-and-operations.html) · [Protocols, Packs & Performance](technology/git/protocols-and-performance.html) |

### Data

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Databases | [Database Crash Course](technology/database-crash-course.html) · [Database Design](technology/database-design/) | [Data Modeling & Normalization](technology/database-design/modeling.html) · [Indexing & Query Execution](technology/database-design/indexing-and-queries.html) · [Transactions & Concurrency](technology/database-design/transactions-and-concurrency.html) · [Storage Engines & Recovery](technology/database-design/storage-internals.html) · [ORMs & Data-Access Patterns](technology/database-design/orm-patterns.html) · [Schema Evolution & Migrations](technology/database-design/schema-evolution-and-migrations.html) · [Operations & Monitoring](technology/database-design/operations-and-monitoring.html) |
| Distributed and NoSQL data | [Distributed & NoSQL Databases](technology/database-design/distributed-and-nosql.html) | [NoSQL Data Models](technology/database-design/nosql-data-models.html) · [Replication & Consensus](technology/database-design/replication-and-consensus.html) · [Distributed Transactions](technology/database-design/distributed-transactions.html) |

### Networking and security

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Networking | [Networking](technology/networking/) | [Layers & Addressing](technology/networking/fundamentals.html) · [Routing & Switching](technology/networking/routing.html) · [Transport & Application Protocols](technology/networking/transport-and-protocols.html) · [Wireless & Mobile](technology/networking/wireless-and-mobile.html) · [Performance, QoS & Security](technology/networking/performance-and-security.html) · [Programmable Networks (SDN/NFV/P4)](technology/networking/programmable-networks.html) · [Cloud Networking](technology/networking/cloud-networking.html) · [Modern Architecture & Frontiers](technology/networking/modern-architecture.html) |
| Cybersecurity | [Cybersecurity](technology/cybersecurity/) | [Cryptography](technology/cybersecurity/cryptography.html) · [Attacks & Network Defense](technology/cybersecurity/attacks-and-defense.html) · [Application Security](technology/cybersecurity/application-and-cloud-security.html) · [Cloud & Container Security](technology/cybersecurity/cloud-and-container-security.html) · [Security Operations](technology/cybersecurity/security-operations.html) · [Incident Response & Forensics](technology/cybersecurity/incident-response.html) · [Foundations, Operations & Research](technology/cybersecurity/operations-and-response.html) · [Privacy Engineering](technology/cybersecurity/privacy-engineering.html) · [Compliance & Governance](technology/cybersecurity/compliance-and-governance.html) |

### Quantum computing

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Quantum computing | [Quantum Computing Hub](quantum-computing/) | [Quantum Computing](technology/quantumcomputing.html) (qubits to hardware) · [QM: Quantum Computing](physics/quantum-mechanics/qm-computing.html) (the physics) · [Quantum Algorithms Research](advanced/quantum-algorithms-research/) (complexity and error correction) |

## Architecture & Operations

How services communicate, stay correct under failure, and remain observable and fast in production.

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Distributed systems | [Distributed Systems Hub](distributed-systems/) | [Consensus & Coordination](distributed-systems/consensus-and-coordination.html) · [Replication Strategies](distributed-systems/replication-strategies.html) · [Failure Detection & Gossip](distributed-systems/failure-detection.html) · [Service Discovery & Configuration](distributed-systems/service-discovery.html) · [Resilience Patterns](distributed-systems/resilience-patterns.html) · [Client-Side Consistency & Sync](distributed-systems/client-side-consistency.html) · [Microservices & Event-Driven Architecture](distributed-systems/microservices-and-event-driven.html) · [Observability](distributed-systems/observability.html) · [Testing & Chaos Engineering](distributed-systems/testing-distributed-systems.html) · [Distributed Systems Theory](advanced/distributed-systems-theory/) |
| API design | [API Design & Communication](api-design/) | [REST](api-design/rest.html) · [GraphQL](api-design/graphql.html) · [gRPC & Protocol Buffers](api-design/grpc-and-protobuf.html) · [Async & Event-Driven APIs](api-design/async-and-events.html) |
| Event-driven architecture | [Event-Driven Architecture](event-driven/) | [Message Brokers & Streaming](event-driven/message-brokers.html) · [Patterns: Sagas, CQRS & Event Sourcing](event-driven/patterns.html) |
| Observability | [Observability](observability/) | [Metrics & Monitoring](observability/metrics.html) · [Logging](observability/logging.html) · [Distributed Tracing](observability/tracing.html) |
| Software testing | [Software Testing & QA](testing/) | [Unit & Integration](testing/unit-and-integration.html) · [Advanced Strategies: Property-Based, Fuzz & Chaos](testing/advanced-testing.html) |
| Performance | [Performance Optimization](optimization/) | [Algorithmic Optimization](optimization/algorithmic-optimization.html) · [CPU Profiling & Tuning](optimization/cpu-optimization.html) · [Memory Optimization](optimization/memory-optimization.html) · [GPU Optimization](optimization/gpu-optimization.html) · [Network & I/O Optimization](optimization/network-io-optimization.html) · [Platform-Specific Tuning](optimization/platform-tuning.html) |

## AI & Machine Learning

Two complementary tracks. The **conceptual track** under Technology explains how machine learning works, from a no-math introduction to learning theory. The **generative AI track** is hands-on: diffusion image models, the tools that drive them, and how to run them in production. The [Artificial Intelligence Hub](artificial-intelligence/) connects both.

### Machine learning concepts

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Introductions | [AI Fundamentals](technology/ai-fundamentals-simple.html) (no math) | [AI Deep Dive: Language Models & Transformers](technology/ai-lecture-2023.html) |
| Machine learning | [AI & Machine Learning](technology/ai/) | [ML Foundations](technology/ai/ml-foundations.html) · [Core ML Algorithms](technology/ai/core-ml-algorithms.html) · [Loss Functions & Objectives](technology/ai/loss-functions.html) · [ML & Deep Learning](technology/ai/architectures.html) · [Deep Learning Architectures](technology/ai/deep-learning-architectures.html) · [Deep Learning Theory](technology/ai/deep-learning-theory.html) |
| Beyond supervised learning | [Generative Models](technology/ai/generative-models.html) | [Reinforcement Learning](technology/ai/reinforcement-learning.html) · [Fine-Tuning & Transfer Learning](technology/ai/fine-tuning.html) · [Frontier Research & Ethics](technology/ai/frontier-and-ethics.html) · [Advanced AI Mathematics](advanced/ai-mathematics/) |

### Generative AI (diffusion models)

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Foundations | [AI/ML Hub](ai-ml/) | [Stable Diffusion Fundamentals](ai-ml/stable-diffusion-fundamentals.html) · [Model Types Explained](ai-ml/model-types.html) (checkpoints, LoRAs, VAEs, embeddings) · [Output Formats: Image to 3D](ai-ml/output-formats.html) |
| Base models | [Base Models Comparison](ai-ml/base-models-comparison.html) | [SDXL](ai-ml/sdxl-guide.html) · [Stable Diffusion 3](ai-ml/sd3-guide.html) · [FLUX](ai-ml/flux-guide.html) · [Pony & Community Fine-Tunes](ai-ml/pony-and-finetunes.html) |
| Tools and control | [ComfyUI Guide](ai-ml/comfyui-guide.html) | [ControlNet](ai-ml/controlnet.html) · [Inpainting & Image Editing](ai-ml/inpainting-editing.html) · [LoRA Training](ai-ml/lora-training.html) · [Advanced Techniques & Workflows](ai-ml/advanced-techniques.html) |
| Production | [Production Pipelines & Automation](ai-ml/production-pipelines.html) | [Optimization & Performance](ai-ml/optimization-guide.html) · [Model Compression](ai-ml/model-compression.html) · [MLOps & Production](ai-ml/mlops-production.html) |

## Game Development & Graphics

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Game development | [Game Development](gamedev/) | [Multiplayer Networking](gamedev/multiplayer-networking.html) · [Procedural Content Generation](gamedev/procedural-generation.html) · [Game AI](ai-ml/game-ai.html) · [UI/UX & Menu Architecture](gamedev/ui-design.html) · [Audio Design](gamedev/audio-design.html) · [Save Systems & Persistence](gamedev/save-systems.html) · [Testing & QA](gamedev/testing-qa.html) · [Monetization & Business Models](gamedev/monetization.html) |
| Graphics | [3D Graphics & Rendering](graphics/3d-rendering.html) | [Shader Programming](graphics/shaders.html) · [GPU Optimization](optimization/gpu-optimization.html) |
| Engines and XR | [Unreal Engine](technology/unreal.html) | [VR & AR Development](vr-ar/) |

## Physics

First-principles treatments that pair the formalism with physical intuition. Each topic's overview page states its prerequisites; the [Physics Hub](physics/) suggests reading orders.

### Classical physics

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Classical mechanics | [Classical Mechanics](physics/classical-mechanics/) | [Newtonian Mechanics & Conservation Laws](physics/classical-mechanics/newtonian.html) · [Oscillations & Waves](physics/classical-mechanics/waves.html) · [Lagrangian & Hamiltonian Mechanics](physics/classical-mechanics/lagrangian-hamiltonian.html) · [Rigid Body Dynamics](physics/classical-mechanics/rigid-body-dynamics.html) · [Chaos & Nonlinear Dynamics](physics/classical-mechanics/chaos-and-computational.html) · [Geometric Formalism](physics/classical-mechanics/geometric-mechanics.html) · [Computational Methods](physics/classical-mechanics/computational-classical-mechanics.html) |
| Fluids | [Fluid Mechanics](physics/fluid-mechanics.html) | [Finite Elements & Fluid Dynamics](physics/computational-physics/fem-and-cfd.html) (numerical) |
| Thermodynamics | [Thermodynamics](physics/thermodynamics.html) | [Advanced Topics](physics/thermodynamics-advanced.html) |
| Statistical mechanics | [Statistical Mechanics](physics/statistical-mechanics/) | [Classical & Quantum Statistical Mechanics](physics/statistical-mechanics/classical-and-quantum.html) · [Phase Transitions & Graduate Formalism](physics/statistical-mechanics/phase-transitions-and-advanced.html) |

### Relativity and gravitation

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Relativity | [Relativity](physics/relativity/) | [Special Relativity](physics/relativity/special-relativity.html) · [General Relativity](physics/relativity/general-relativity.html) · [Tensor Formalism & the Field Equations](physics/relativity/tensor-formalism.html) · [Black Holes](physics/relativity/black-holes.html) · [Gravitational Waves](physics/relativity/gravitational-waves.html) · [Relativistic Cosmology](physics/relativity/cosmology.html) · [Graduate Topics](physics/relativity/advanced.html) · [Toward Quantum Gravity](physics/relativity/quantum-gravity.html) |

### Quantum physics

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Quantum mechanics | [Quantum Mechanics](physics/quantum-mechanics/) | [States, Operators & Dynamics](physics/quantum-mechanics/formalism.html) · [Systems & Phenomena](physics/quantum-mechanics/systems-and-phenomena.html) · [Bell's Theorem & Experimental Tests](physics/quantum-mechanics/bell-inequalities-and-tests.html) · [Computing & Advanced Topics](physics/quantum-mechanics/computing-and-advanced.html) (sub-hub) · [Advanced Formalism](physics/quantum-mechanics/qm-advanced-formalism.html) · [Computational Methods](physics/quantum-mechanics/qm-computational-methods.html) · [Quantum Computing](physics/quantum-mechanics/qm-computing.html) · [Research Frontiers](physics/quantum-mechanics/qm-research-frontiers.html) |
| Quantum field theory | [Quantum Field Theory](physics/quantum-field-theory.html) | [Canonical Quantization](physics/qft-quantization.html) · [Path Integrals & Methods](physics/qft-methods.html) · [Renormalization & the RG](physics/renormalization.html) · [Gauge Theories & the Standard Model](physics/gauge-and-standard-model.html) · [Modern Frontiers](physics/qft-frontiers.html) |
| String theory | [String Theory](physics/string-theory/) | [D-Branes, Dualities & M-Theory](physics/string-theory/dualities-and-branes.html) · [Graduate Formalism](physics/string-theory/string-theory-formalism.html) · [Criticisms & Research Frontiers](physics/string-theory/frontiers-and-formalism.html) |

### Matter and computation

| Topic | Overview | Sub-pages |
|-------|----------|-----------|
| Condensed matter | [Condensed Matter Physics](physics/condensed-matter/) | [Lattice Dynamics & Phonons](physics/condensed-matter/lattice-dynamics.html) · [Metals & Magnetism](physics/condensed-matter/metals-and-magnetism.html) · [Superconductivity, Quantum Hall & Topological Phases](physics/condensed-matter/emergent-phases.html) · [Disorder & Localization](physics/condensed-matter/disorder-and-localization.html) · [Experimental Techniques](physics/condensed-matter/experimental-techniques.html) · [Graduate-Level Formalism](physics/condensed-matter/advanced-formalism.html) |
| Computational physics | [Computational Physics](physics/computational-physics/) | [Monte Carlo & Molecular Dynamics](physics/computational-physics/monte-carlo-and-md.html) · [Finite Elements & Fluid Dynamics](physics/computational-physics/fem-and-cfd.html) · [Quantum Computational Methods](physics/computational-physics/quantum-methods.html) · [Electronic Structure Beyond DFT](physics/computational-physics/electronic-structure-beyond-dft.html) · [Parallel & High-Performance Computing](physics/computational-physics/hpc-and-ml.html) · [Machine Learning for Physics](physics/computational-physics/ml-for-physics.html) · [Visualization, Libraries & Best Practices](physics/computational-physics/tools-and-practices.html) |

## Theory & Research

Graduate-level, proof-oriented pages. Each opens with its prerequisites and the intuition behind the formalism. The [Advanced Topics Hub](advanced/) groups them into reading paths.

| Area | Pages |
|------|-------|
| Computation and complexity | [Automata Theory & Formal Languages](advanced/automata-and-formal-languages/) · [Computational Complexity Theory](advanced/complexity-theory/) · [Approximation Algorithms & Hardness](advanced/approximation-algorithms/) |
| Information and security | [Information & Coding Theory](advanced/information-coding-theory/) · [Cryptography: Foundations & Post-Quantum](advanced/cryptography/) |
| Mathematical structures | [Category Theory & Type Theory](advanced/category-and-type-theory/) · [Topology & Geometry in Computation](advanced/topology-and-geometry-in-computation/) |
| Applied theory | [Advanced AI Mathematics](advanced/ai-mathematics/) · [Distributed Systems Theory](advanced/distributed-systems-theory/) · [Quantum Algorithms Research](advanced/quantum-algorithms-research/) |

## Reference & Navigation

| Page | Use it for |
|------|------------|
| [Quick Reference Guide](reference/) | Git, Docker, kubectl, and AWS CLI commands; physics constants and equations; Big-O tables; regex and pre-flight checklists |
| [Getting Started](../getting-started.html) | How the site is organized, page types, and starting points by role |
| [Topic Map](topic-map.html) | Interactive prerequisite map and role-based reading paths |
| [Search](../search.html) | Keyword search over the main hub and overview pages |
