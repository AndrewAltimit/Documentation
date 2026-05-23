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
  <p class="lead-text">Every expert was once a beginner. This interactive map helps you navigate from wherever you are to wherever you want to be. Find your starting point, explore connections, and chart your own learning journey. Prefer a plain list? See the <a href="./">complete documentation index</a>.</p>
</div>

{% include topic-map.html %}

## Quick Start Guides

<div class="quick-start-grid">
  <div class="quick-start-card">
    <h3>Complete Beginner</h3>
    <p>Start with our 5-minute crash courses:</p>
    <ul>
      <li><a href="technology/git-reference.html">Git Quick Start</a></li>
      <li><a href="technology/docker-essentials.html">Docker Quick Start</a></li>
      <li><a href="technology/database-design.html">Database Basics</a></li>
      <li><a href="technology/ai-fundamentals-simple.html">AI for Beginners</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>Intermediate</h3>
    <p>Explore intermediate topics:</p>
    <ul>
      <li><a href="technology/branching.html">Git Branching Strategies</a></li>
      <li><a href="technology/docker/">Docker Deep Dive</a></li>
      <li><a href="technology/database-design.html">Database Design Patterns</a></li>
      <li><a href="technology/ai.html">AI & Neural Networks</a></li>
    </ul>
  </div>

  <div class="quick-start-card">
    <h3>Advanced</h3>
    <p>Dive into research-level content:</p>
    <ul>
      <li><a href="technology/git.html">Git Internals & Theory</a></li>
      <li><a href="technology/kubernetes/">Kubernetes Architecture</a></li>
      <li><a href="advanced/distributed-systems-theory/">Distributed Systems</a></li>
      <li><a href="technology/ai-lecture-2023.html">Advanced AI Theory</a></li>
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

### Path 1: Full-Stack Developer Journey
```
Git Basics → Docker → Databases → AWS → Kubernetes
     ↓           ↓          ↓        ↓         ↓
Branching    Compose    Design   Terraform  Helm
```

### Path 2: AI/ML Engineer Journey
```
AI Basics → Python → Neural Networks → Deep Learning → MLOps
     ↓         ↓            ↓               ↓           ↓
  Math    Libraries    TensorFlow       Research    Production
```

### Path 3: DevOps Engineer Journey
```
Linux → Git → Docker → CI/CD → Kubernetes → Monitoring
   ↓      ↓       ↓       ↓         ↓           ↓
Shell  Branching Compose Jenkins   Helm    Prometheus
```

## Pro Tips

1. **Start Small**: Don't try to learn everything at once
2. **Follow Interests**: Let curiosity guide your path
3. **Practice Regularly**: Apply concepts in real projects
4. **Join Community**: Share your journey with others
5. **Review Often**: Revisit topics to reinforce learning

---

<div class="cta-section">
  <h2>Ready to Start Your Journey?</h2>
  <p>Pick a topic that interests you and dive in. Remember, every expert started exactly where you are now.</p>
  <a href="#quick-start-guides" class="btn btn-primary">Choose Your Starting Point</a>
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