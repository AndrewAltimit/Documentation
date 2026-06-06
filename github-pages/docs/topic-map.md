---
layout: docs
title: Topic Map
permalink: /docs/topic-map.html
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Interactive Learning Map</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.92;">Discover your personalized learning path through the documentation.</p>
</div>

Drag the nodes, follow the connections, and chart a route through the documentation. Prefer a plain list? See the [complete documentation index](./). Already know the topic? [Search](../search.html) jumps you straight there.

{% include topic-map.html %}

## Pick Your Level

Each track collects a handful of pages at the right depth; the map above shows how they connect.

<div class="quick-start-grid">
  <div class="quick-start-card">
    <h3>Complete Beginner</h3>
    <p>No prior experience needed &mdash; short, friendly crash courses:</p>
    <ul>
      <li><a href="technology/git-reference.html">Git Quick Start</a></li>
      <li><a href="technology/docker-essentials.html">Docker Quick Start</a></li>
      <li><a href="technology/database-design/">Database Basics</a></li>
      <li><a href="technology/ai-fundamentals-simple.html">AI for Beginners</a> (no math)</li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>Intermediate</h3>
    <p>You know the basics &mdash; build real working knowledge:</p>
    <ul>
      <li><a href="technology/branching.html">Git Branching Strategies</a></li>
      <li><a href="technology/docker/">Docker Deep Dive</a></li>
      <li><a href="technology/database-design/">Database Design Patterns</a></li>
      <li><a href="technology/ai/">AI &amp; Neural Networks</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>Advanced</h3>
    <p>Comfortable already &mdash; dive into theory and research:</p>
    <ul>
      <li><a href="technology/git/">Git Internals &amp; Theory</a></li>
      <li><a href="technology/kubernetes/">Kubernetes Architecture</a></li>
      <li><a href="advanced/distributed-systems-theory/">Distributed Systems Theory</a></li>
      <li><a href="advanced/ai-mathematics/">AI Mathematics</a></li>
    </ul>
  </div>
</div>

## How to Navigate This Map

### Interactive Features
- **Click and drag** nodes to explore the visualization
- **Click any topic** to see details and available content
- **Use difficulty filters** to focus on your level
- **Follow connections** to discover related topics
- **Zoom and pan** to explore different knowledge domains

### Understanding Connections
<div class="connection-legend">
  <div class="legend-item">
    <span class="connection-line green"></span>
    <strong>Progressive Learning</strong> - Natural path from easier to harder
  </div>
  <div class="legend-item">
    <span class="connection-line blue"></span>
    <strong>Related Topics</strong> - Similar concepts at the same level
  </div>
  <div class="legend-item">
    <span class="connection-line purple"></span>
    <strong>Cross-Domain</strong> - Interdisciplinary connections
  </div>
</div>

## Suggested Learning Paths

Three example routes through the site. Each top row is the spine; the row beneath lists the deeper companion topic for each step.

### Path 1: Full-Stack / Cloud Developer

```mermaid
flowchart LR
    G["Git Basics"] --> D["Docker"] --> DB["Databases"] --> A["AWS"] --> K["Kubernetes"]
    G -.-> Gb["Branching"]
    D -.-> Dc["Compose"]
    DB -.-> DBd["Schema design"]
    A -.-> At["Terraform"]
    K -.-> Kh["Helm / operators"]
    style G fill:#e3f2fd,stroke:#1565c0
    style D fill:#e3f2fd,stroke:#1565c0
    style DB fill:#e3f2fd,stroke:#1565c0
    style A fill:#e3f2fd,stroke:#1565c0
    style K fill:#e3f2fd,stroke:#1565c0
    style Gb fill:#fff3e0,stroke:#e65100
    style Dc fill:#fff3e0,stroke:#e65100
    style DBd fill:#fff3e0,stroke:#e65100
    style At fill:#fff3e0,stroke:#e65100
    style Kh fill:#fff3e0,stroke:#e65100
```

### Path 2: AI / ML Engineer

```mermaid
flowchart LR
    AB["AI Basics<br/>(no math)"] --> NN["Neural Networks"] --> DL["Deep Learning"] --> GEN["Generative AI"] --> TH["AI Mathematics"]
    AB -.-> M["Linear algebra<br/>& probability"]
    NN -.-> T["Transformers"]
    DL -.-> SD["Stable Diffusion"]
    GEN -.-> LoRA["LoRA / ComfyUI"]
    style AB fill:#e8f5e9,stroke:#2e7d32
    style NN fill:#e8f5e9,stroke:#2e7d32
    style DL fill:#e8f5e9,stroke:#2e7d32
    style GEN fill:#e8f5e9,stroke:#2e7d32
    style TH fill:#e8f5e9,stroke:#2e7d32
    style M fill:#fff3e0,stroke:#e65100
    style T fill:#fff3e0,stroke:#e65100
    style SD fill:#fff3e0,stroke:#e65100
    style LoRA fill:#fff3e0,stroke:#e65100
```

### Path 3: DevOps Engineer

```mermaid
flowchart LR
    GG["Git"] --> DD["Docker"] --> CI["CI/CD"] --> KK["Kubernetes"] --> OBS["Observability"]
    GG -.-> Br["Branching strategy"]
    DD -.-> Cmp["Compose"]
    CI -.-> Pipe["Pipelines"]
    KK -.-> Helm2["Helm"]
    OBS -.-> Prom["Prometheus / Grafana"]
    style GG fill:#ede7f6,stroke:#5e35b1
    style DD fill:#ede7f6,stroke:#5e35b1
    style CI fill:#ede7f6,stroke:#5e35b1
    style KK fill:#ede7f6,stroke:#5e35b1
    style OBS fill:#ede7f6,stroke:#5e35b1
    style Br fill:#fff3e0,stroke:#e65100
    style Cmp fill:#fff3e0,stroke:#e65100
    style Pipe fill:#fff3e0,stroke:#e65100
    style Helm2 fill:#fff3e0,stroke:#e65100
    style Prom fill:#fff3e0,stroke:#e65100
```

<div class="tip-card" markdown="1">
#### Getting the most from these paths
- **Start small.** Pick one spine and follow it; resist learning five things at once.
- **Follow the dotted lines later.** The companion topics deepen each step once the basics click.
- **Build as you go.** Apply each topic to a tiny real project before moving on.
- **Revisit.** The advanced pages reward a second read after you've used the basics in anger.
</div>

## Learning Paths by Role

Role-focused reading lists &mdash; the pages that matter most for each discipline, in order.

<div class="quick-start-grid">
  <div class="quick-start-card">
    <h3>Full-Stack / Cloud Developer</h3>
    <p>Ship a service end to end, from repo to cloud:</p>
    <ul>
      <li><a href="technology/git-reference.html">Git Reference</a></li>
      <li><a href="technology/docker/">Docker Deep Dive</a></li>
      <li><a href="technology/database-design/">Database Design</a></li>
      <li><a href="api-design/rest.html">REST API Design</a></li>
      <li><a href="technology/aws/">AWS Cloud Services</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>AI / ML Engineer</h3>
    <p>From neural-network basics to production models:</p>
    <ul>
      <li><a href="technology/ai-fundamentals-simple.html">AI for Beginners</a></li>
      <li><a href="technology/ai/">AI &amp; Neural Networks</a></li>
      <li><a href="ai-ml/">Generative AI &amp; Diffusion</a></li>
      <li><a href="ai-ml/mlops-production.html">MLOps in Production</a></li>
      <li><a href="advanced/ai-mathematics/">AI Mathematics</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>DevOps Engineer</h3>
    <p>Build, ship, and run software reliably:</p>
    <ul>
      <li><a href="technology/branching.html">Git Branching Strategies</a></li>
      <li><a href="technology/docker/">Docker</a></li>
      <li><a href="technology/ci-cd/">CI/CD Pipelines</a></li>
      <li><a href="technology/kubernetes/">Kubernetes</a></li>
      <li><a href="observability/">Observability</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>Game Developer</h3>
    <p>From engine fundamentals to shipping a game:</p>
    <ul>
      <li><a href="gamedev/">Game Development Overview</a></li>
      <li><a href="graphics/3d-rendering.html">3D Rendering</a></li>
      <li><a href="graphics/shaders.html">Shaders</a></li>
      <li><a href="gamedev/multiplayer-networking.html">Multiplayer Networking</a></li>
      <li><a href="gamedev/procedural-generation.html">Procedural Generation</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>Data Engineer</h3>
    <p>Move, model, and serve data at scale:</p>
    <ul>
      <li><a href="technology/database-design/">Database Design</a></li>
      <li><a href="technology/database-design/distributed-and-nosql.html">Distributed &amp; NoSQL Stores</a></li>
      <li><a href="event-driven/">Event-Driven Architecture</a></li>
      <li><a href="api-design/grpc-and-protobuf.html">gRPC &amp; Protobuf</a></li>
      <li><a href="distributed-systems/">Distributed Systems</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>Security Engineer</h3>
    <p>Defend systems across the stack:</p>
    <ul>
      <li><a href="technology/cybersecurity/">Cybersecurity Overview</a></li>
      <li><a href="technology/cybersecurity/attacks-and-defense.html">Attacks &amp; Defense</a></li>
      <li><a href="technology/cybersecurity/cryptography.html">Applied Cryptography</a></li>
      <li><a href="technology/cybersecurity/cloud-and-container-security.html">Cloud &amp; Container Security</a></li>
      <li><a href="technology/networking/performance-and-security.html">Network Security</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>SRE / Platform Engineer</h3>
    <p>Keep distributed systems healthy and observable:</p>
    <ul>
      <li><a href="technology/kubernetes/">Kubernetes</a></li>
      <li><a href="technology/terraform/">Terraform (IaC)</a></li>
      <li><a href="observability/">Observability</a></li>
      <li><a href="distributed-systems/resilience-patterns.html">Resilience Patterns</a></li>
      <li><a href="testing/">Software Testing</a></li>
    </ul>
  </div>
</div>

---

<div class="cta-section">
  <a href="#pick-your-level" class="btn btn-primary">Choose a starting point</a>
  <a href="/" class="btn btn-secondary">Documentation home</a>
</div>

<style>
/* Page-specific styling only; grids, hero, buttons, and CTA come from global CSS. */
.connection-legend {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin: 1rem 0;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

html[data-theme="dark"] .connection-legend {
  background: #1f2733;
  color: #e6e6e6;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.connection-line {
  width: 50px;
  height: 3px;
  display: inline-block;
}

.connection-line.green {
  background: #28a745;
}

.connection-line.blue {
  background: #007bff;
}

.connection-line.purple {
  background: #6f42c1;
}

.cta-section .btn { margin: 0.25rem; }
</style>
