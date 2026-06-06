---
layout: docs
title: Complete Documentation Index
toc: false  # Index pages typically don't need TOC
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #4facfe 100%);">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Documentation Index</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.92;">Every page, organized by category &mdash; from beginner guides to graduate-level theory.</p>
</div>

The complete listing of every page on the site, grouped by category &mdash; [Technology](#technology), [Architecture & Operations](#architecture--operations), [AI / ML](#aiml---generative-ai), [Physics](#physics), and the [specialized](#specialized-hubs) and [research](#research--advanced-topics) hubs. You can also browse the [visual topic map](topic-map.html) or [search](../search.html) directly.

<div class="tip-card" markdown="1">
**New here? Start at a hub.** Each major area has a hub page that orients you and recommends a reading path: the [Quantum Computing Hub](quantum-computing/), [Distributed Systems Hub](distributed-systems/), [Performance Optimization Hub](optimization/), [AI/ML Hub](ai-ml/), and the [Physics Hub](physics/). The lists below are the full index; the hubs are the guided way in.
</div>


## Technology

Infrastructure, DevOps, cloud, security, and game/real-time development.

### Infrastructure & DevOps

<div class="command-grid">
  <a href="technology/terraform/" class="nav-card"><h4><i class="fas fa-cubes"></i> Terraform</h4><p>Infrastructure as Code for multi-cloud provisioning.</p></a>
  <a href="technology/docker/" class="nav-card"><h4><i class="fab fa-docker"></i> Docker</h4><p>Comprehensive containerization guide, images to networking.</p></a>
  <a href="technology/docker-essentials.html" class="nav-card"><h4><i class="fas fa-terminal"></i> Docker Essentials</h4><p>Quick command reference for everyday Docker.</p></a>
  <a href="technology/kubernetes/" class="nav-card"><h4><i class="fas fa-dharmachakra"></i> Kubernetes</h4><p>Production-grade container orchestration.</p></a>
  <a href="technology/aws/" class="nav-card"><h4><i class="fab fa-aws"></i> AWS</h4><p>Amazon Web Services: compute, storage, and networking.</p></a>
  <a href="technology/ci-cd/" class="nav-card"><h4><i class="fas fa-sync-alt"></i> CI/CD</h4><p>Continuous integration and deployment pipelines.</p></a>
  <a href="technology/please-build.html" class="nav-card"><h4><i class="fas fa-hammer"></i> Please Build</h4><p>High-performance polyglot build system.</p></a>
</div>

### Development & Version Control

<div class="command-grid">
  <a href="technology/git/" class="nav-card"><h4><i class="fab fa-git-alt"></i> Git Version Control</h4><p>Core concepts, the object model, and workflows.</p></a>
  <a href="technology/git-reference.html" class="nav-card"><h4><i class="fas fa-book"></i> Git Command Reference</h4><p>Comprehensive command-by-command guide.</p></a>
  <a href="technology/branching.html" class="nav-card"><h4><i class="fas fa-code-branch"></i> Branching Strategies</h4><p>Git Flow, trunk-based, and release patterns.</p></a>
  <a href="technology/database-design/" class="nav-card"><h4><i class="fas fa-database"></i> Database Design</h4><p>Relational and NoSQL modeling patterns.</p></a>
</div>

### Networking & Security

<div class="command-grid">
  <a href="technology/networking/" class="nav-card"><h4><i class="fas fa-network-wired"></i> Networking</h4><p>TCP/IP, protocols, DNS, and modern architecture.</p></a>
  <a href="technology/cybersecurity/" class="nav-card"><h4><i class="fas fa-shield-alt"></i> Cybersecurity</h4><p>Security principles and threat mitigation.</p></a>
  <a href="technology/quantumcomputing.html" class="nav-card"><h4><i class="fas fa-atom"></i> Quantum Computing</h4><p>From qubits to algorithms and hardware.</p></a>
</div>

### Game & Real-Time Development

<div class="command-grid">
  <a href="technology/unreal.html" class="nav-card"><h4><i class="fas fa-gamepad"></i> Unreal Engine</h4><p>UE5 development with Nanite, Lumen, and MetaSounds.</p></a>
  <a href="gamedev/" class="nav-card"><h4><i class="fas fa-dice-d20"></i> Game Development</h4><p>Engines, the game loop, ECS, and architecture.</p></a>
  <a href="graphics/3d-rendering.html" class="nav-card"><h4><i class="fas fa-cube"></i> 3D Graphics &amp; Rendering</h4><p>Rasterization, shading, and the GPU pipeline.</p></a>
  <a href="vr-ar/" class="nav-card"><h4><i class="fas fa-vr-cardboard"></i> VR / AR Development</h4><p>Immersive spatial computing fundamentals.</p></a>
  <a href="ai-ml/game-ai.html" class="nav-card"><h4><i class="fas fa-robot"></i> Game AI</h4><p>Pathfinding, behavior trees, and decision-making.</p></a>
</div>

### Artificial Intelligence (Conceptual)

<div class="command-grid">
  <a href="technology/ai-fundamentals-simple.html" class="nav-card"><h4><i class="fas fa-lightbulb"></i> AI Fundamentals — Simplified</h4><p>A no-math introduction to how AI works.</p></a>
  <a href="technology/ai/" class="nav-card"><h4><i class="fas fa-brain"></i> Artificial Intelligence</h4><p>Comprehensive technical overview.</p></a>
  <a href="technology/ai/deep-learning-architectures.html" class="nav-card"><h4><i class="fas fa-network-wired"></i> Deep Learning Architectures</h4><p>CNNs, transformers, and modern network design.</p></a>
  <a href="technology/ai/reinforcement-learning.html" class="nav-card"><h4><i class="fas fa-gamepad"></i> Reinforcement Learning</h4><p>Policies, value functions, and reward shaping.</p></a>
  <a href="technology/ai-lecture-2023.html" class="nav-card"><h4><i class="fas fa-graduation-cap"></i> AI Deep Dive</h4><p>Advanced concepts and research directions.</p></a>
</div>

## Architecture & Operations

How services talk to each other, prove they work, and stay observable in production. These cross-cutting areas pair with the [Distributed Systems Hub](distributed-systems/) and [Infrastructure & DevOps](#infrastructure--devops) above.

### Observability

<div class="command-grid">
  <a href="observability/" class="nav-card"><h4><i class="fas fa-eye"></i> Observability Hub</h4><p>Metrics, logs, and traces — inferring internal state from telemetry.</p></a>
  <a href="observability/metrics.html" class="nav-card"><h4><i class="fas fa-chart-line"></i> Metrics</h4><p>Time series, the RED/USE methods, and alerting.</p></a>
  <a href="observability/logging.html" class="nav-card"><h4><i class="fas fa-file-alt"></i> Logging</h4><p>Structured, high-cardinality event records.</p></a>
  <a href="observability/tracing.html" class="nav-card"><h4><i class="fas fa-route"></i> Distributed Tracing</h4><p>Following a request across service boundaries.</p></a>
</div>

### API Design & Communication

<div class="command-grid">
  <a href="api-design/" class="nav-card"><h4><i class="fas fa-file-signature"></i> API Design Hub</h4><p>Choosing and designing the contracts between services.</p></a>
  <a href="api-design/rest.html" class="nav-card"><h4><i class="fas fa-exchange-alt"></i> REST</h4><p>Resource modeling, HTTP semantics, and versioning.</p></a>
  <a href="api-design/graphql.html" class="nav-card"><h4><i class="fas fa-project-diagram"></i> GraphQL</h4><p>Schema design, resolvers, and the N+1 problem.</p></a>
  <a href="api-design/grpc-and-protobuf.html" class="nav-card"><h4><i class="fas fa-bolt"></i> gRPC &amp; Protobuf</h4><p>Binary contracts and streaming RPC.</p></a>
  <a href="api-design/async-and-events.html" class="nav-card"><h4><i class="fas fa-stream"></i> Async &amp; Events</h4><p>Asynchronous, message-based communication.</p></a>
</div>

### Software Testing & QA

<div class="command-grid">
  <a href="testing/" class="nav-card"><h4><i class="fas fa-vial"></i> Testing Hub</h4><p>The testing discipline from unit assertions to chaos.</p></a>
  <a href="testing/unit-and-integration.html" class="nav-card"><h4><i class="fas fa-layer-group"></i> Unit &amp; Integration</h4><p>The base and middle of the test pyramid.</p></a>
  <a href="testing/advanced-testing.html" class="nav-card"><h4><i class="fas fa-flask-vial"></i> Advanced Testing</h4><p>Property-based, fuzz, and chaos engineering.</p></a>
</div>

### Event-Driven Architecture

<div class="command-grid">
  <a href="event-driven/" class="nav-card"><h4><i class="fas fa-broadcast-tower"></i> Event-Driven Hub</h4><p>Systems that react to facts rather than commands.</p></a>
  <a href="event-driven/message-brokers.html" class="nav-card"><h4><i class="fas fa-inbox"></i> Message Brokers</h4><p>Kafka, queues, and log-based delivery.</p></a>
  <a href="event-driven/patterns.html" class="nav-card"><h4><i class="fas fa-sitemap"></i> Patterns</h4><p>Choreography, sagas, CQRS, and event sourcing.</p></a>
</div>

## AI/ML - Generative AI

Hands-on generative AI: diffusion models, training, and production workflows. New to this area? Begin at the [AI/ML Hub](ai-ml/).

### Getting Started

<div class="command-grid">
  <a href="ai-ml/" class="nav-card"><h4><i class="fas fa-compass"></i> AI/ML Overview</h4><p>The hub — start here for all generative-AI content.</p></a>
  <a href="ai-ml/stable-diffusion-fundamentals.html" class="nav-card"><h4><i class="fas fa-image"></i> Stable Diffusion Fundamentals</h4><p>Core diffusion concepts explained.</p></a>
  <a href="ai-ml/base-models-comparison.html" class="nav-card"><h4><i class="fas fa-balance-scale"></i> Base Models Comparison</h4><p>SD 1.5, SDXL, FLUX, and SD3 compared.</p></a>
</div>

### Tools & Workflows

<div class="command-grid">
  <a href="ai-ml/comfyui-guide.html" class="nav-card"><h4><i class="fas fa-project-diagram"></i> ComfyUI Guide</h4><p>Node-based visual workflow creation.</p></a>
  <a href="ai-ml/lora-training.html" class="nav-card"><h4><i class="fas fa-sliders-h"></i> LoRA Training</h4><p>Fine-tune your own models efficiently.</p></a>
  <a href="ai-ml/controlnet.html" class="nav-card"><h4><i class="fas fa-crosshairs"></i> ControlNet</h4><p>Precise structural control over generation.</p></a>
</div>

### Going Deeper

<div class="command-grid">
  <a href="ai-ml/model-types.html" class="nav-card"><h4><i class="fas fa-layer-group"></i> Model Types Explained</h4><p>LoRAs, embeddings, VAEs, and checkpoints.</p></a>
  <a href="ai-ml/output-formats.html" class="nav-card"><h4><i class="fas fa-photo-video"></i> Output Formats</h4><p>Image, video, and audio generation.</p></a>
  <a href="ai-ml/advanced-techniques.html" class="nav-card"><h4><i class="fas fa-magic"></i> Advanced Techniques</h4><p>Professional, production-grade workflows.</p></a>
</div>

## Physics

First-principles treatments pairing rigorous math with physical intuition. The [Physics Hub](physics/) suggests guided reading paths.

### Classical Physics

<div class="command-grid">
  <a href="physics/classical-mechanics/" class="nav-card"><h4><i class="fas fa-atom"></i> Classical Mechanics</h4><p>Newton's laws through Lagrangian and Hamiltonian formalism.</p></a>
  <a href="physics/thermodynamics.html" class="nav-card"><h4><i class="fas fa-fire"></i> Thermodynamics</h4><p>Heat, work, entropy, and the four laws.</p></a>
  <a href="physics/statistical-mechanics/" class="nav-card"><h4><i class="fas fa-dice"></i> Statistical Mechanics</h4><p>From microscopic randomness to macroscopic law.</p></a>
</div>

### Modern Physics

<div class="command-grid">
  <a href="physics/relativity/" class="nav-card"><h4><i class="fas fa-clock"></i> Relativity</h4><p>Special and general relativity, spacetime, and gravity.</p></a>
  <a href="physics/quantum-mechanics/" class="nav-card"><h4><i class="fas fa-wave-square"></i> Quantum Mechanics</h4><p>Wave functions, uncertainty, and entanglement.</p></a>
</div>

### Advanced & Computational

<div class="command-grid">
  <a href="physics/condensed-matter/" class="nav-card"><h4><i class="fas fa-cube"></i> Condensed Matter</h4><p>Solids, superconductors, and topological materials.</p></a>
  <a href="physics/quantum-field-theory.html" class="nav-card"><h4><i class="fas fa-project-diagram"></i> Quantum Field Theory</h4><p>Fields, particles, and the Standard Model.</p></a>
  <a href="physics/string-theory/" class="nav-card"><h4><i class="fas fa-infinity"></i> String Theory</h4><p>Extra dimensions and quantum gravity.</p></a>
  <a href="physics/computational-physics/" class="nav-card"><h4><i class="fas fa-laptop-code"></i> Computational Physics</h4><p>Numerical methods, Monte Carlo, and simulation.</p></a>
</div>

## Specialized Hubs

Curated landing pages that gather related material across the site and recommend a path through it.

<div class="command-grid">
  <a href="quantum-computing/" class="nav-card"><h4><i class="fas fa-atom"></i> Quantum Computing Hub</h4><p>Theory-to-hardware: qubits, algorithms, and cloud platforms.</p></a>
  <a href="distributed-systems/" class="nav-card"><h4><i class="fas fa-network-wired"></i> Distributed Systems Hub</h4><p>Consensus, consistency, and resilient architecture patterns.</p></a>
  <a href="distributed-systems/resilience-patterns.html" class="nav-card"><h4><i class="fas fa-heart-pulse"></i> Resilience Patterns</h4><p>Retries, circuit breakers, bulkheads, and backpressure.</p></a>
  <a href="optimization/" class="nav-card"><h4><i class="fas fa-tachometer-alt"></i> Performance Optimization</h4><p>Profiling, bottleneck analysis, and scaling.</p></a>
  <a href="optimization/gpu-optimization.html" class="nav-card"><h4><i class="fas fa-microchip"></i> GPU Optimization</h4><p>Kernels, memory hierarchy, and throughput tuning.</p></a>
  <a href="artificial-intelligence/" class="nav-card"><h4><i class="fas fa-brain"></i> Artificial Intelligence Hub</h4><p>Comprehensive AI resources and orientation.</p></a>
</div>

## Research & Advanced Topics

Graduate-level, proof-oriented material. Each page frames its prerequisites and intuition before the formalism.

<div class="command-grid">
  <a href="advanced/" class="nav-card"><h4><i class="fas fa-flask"></i> Advanced Topics Hub</h4><p>The research section's index and reading paths.</p></a>
  <a href="advanced/ai-mathematics/" class="nav-card"><h4><i class="fas fa-square-root-alt"></i> AI Mathematics</h4><p>Statistical learning theory and optimization landscapes.</p></a>
  <a href="advanced/distributed-systems-theory/" class="nav-card"><h4><i class="fas fa-ban"></i> Distributed Systems Theory</h4><p>Impossibility results and formal verification.</p></a>
  <a href="advanced/quantum-algorithms-research/" class="nav-card"><h4><i class="fas fa-microchip"></i> Quantum Algorithms Research</h4><p>Complexity, error correction, and NISQ algorithms.</p></a>
  <a href="advanced/monorepo/" class="nav-card"><h4><i class="fas fa-code-branch"></i> Monorepo Strategies</h4><p>Build graphs, caching, and large-repo engineering.</p></a>
</div>

## Reference & Navigation

<div class="command-grid">
  <a href="reference/" class="nav-card"><h4><i class="fas fa-list"></i> Quick Reference Guide</h4><p>CLI commands, physics constants, Big-O, regex, and checklists.</p></a>
  <a href="topic-map.html" class="nav-card"><h4><i class="fas fa-sitemap"></i> Topic Map</h4><p>Visual navigation across every knowledge domain.</p></a>
  <a href="../search.html" class="nav-card"><h4><i class="fas fa-search"></i> Search</h4><p>Full-text search across the whole site.</p></a>
</div>

---

## Where This Site Goes Deep

Most of the site is reference-grade, but a few areas go well past the usual cheat-sheet treatment. If you want to see what the documentation can do, start here:

- **Generative AI** — Stable Diffusion 3 and FLUX architectures, flow matching, and production workflows
- **Kubernetes** — current production patterns: workloads, storage, operations, and advanced scheduling
- **Git internals** — how commits, refs, and the object model actually work, plus security practices
- **Quantum** — a theory-to-hardware path spanning quantum mechanics, algorithms, and cloud platforms
- **Physics** — first-principles treatments from classical mechanics through quantum field theory
- **Research hub** — proof-oriented pages on [learning theory](advanced/ai-mathematics/), [consensus](advanced/distributed-systems-theory/), and [quantum algorithms](advanced/quantum-algorithms-research/)