---
layout: docs
title: Topic Map
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Interactive Learning Map</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.92;">Discover your personalized learning path through the documentation.</p>
</div>

<div class="intro-card">
  <p class="lead-text">Every expert was once a beginner. This interactive map helps you navigate from wherever you are to wherever you want to be &mdash; drag the nodes, follow the connections, and chart your own route through the documentation. Prefer a plain list? See the <a href="./">complete documentation index</a>. Already know the topic? <a href="../search.html">Search</a> jumps you straight there.</p>
</div>

{% include topic-map.html %}

## Pick Your Level

<p>Not sure where to jump in? Each track below collects a handful of pages at the right depth. Start anywhere &mdash; the map above shows how they connect.</p>

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

---

<div class="cta-section">
  <h2>Ready to Start Your Journey?</h2>
  <p>Pick a topic that interests you and dive in. Remember, every expert started exactly where you are now.</p>
  <a href="#pick-your-level" class="btn btn-primary">Choose Your Starting Point</a>
  <a href="/" class="btn btn-secondary">Back to Documentation Home</a>
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