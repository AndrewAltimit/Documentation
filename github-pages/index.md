---
layout: docs
title: Andrew's Notebook
description: A reference wiki for software infrastructure, distributed systems, machine learning, physics, and theoretical computer science.
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #4facfe 100%);">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Andrew's Notebook</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.92;">A reference wiki for infrastructure, software, AI, physics, and theory &mdash; from first principles to research depth.</p>
</div>

Andrew's Notebook is a technical reference covering the software delivery stack, distributed systems, machine learning and generative AI, physics, and theoretical computer science. Pages are written as reference material rather than tutorials: each topic has an overview that states the core ideas, with focused sub-pages that go deeper, and the most advanced pages carry the formal treatment through to current research.

**New here?** [Getting Started](getting-started.html) explains how the site is organized and where to begin. To find a specific page, use the [documentation index](docs/index.html), which lists every page, or [search](search.html).

## Sections

<div class="category-grid">

  <div class="category-card tech">
    <a class="category-card__main" href="docs/technology/">
      <div class="category-icon"><i class="fas fa-server"></i></div>
      <h3>Technology</h3>
      <p>Containers, orchestration, cloud, infrastructure as code, version control, databases, networking, and security.</p>
    </a>
    <div class="category-links">
      <a href="docs/technology/docker/">Docker</a> &middot;
      <a href="docs/technology/kubernetes/">Kubernetes</a> &middot;
      <a href="docs/technology/terraform/">Terraform</a> &middot;
      <a href="docs/technology/aws/">AWS</a> &middot;
      <a href="docs/technology/git/">Git</a> &middot;
      <a href="docs/technology/database-design/">Databases</a>
    </div>
  </div>

  <div class="category-card tech">
    <a class="category-card__main" href="docs/distributed-systems/">
      <div class="category-icon"><i class="fas fa-network-wired"></i></div>
      <h3>Architecture &amp; Operations</h3>
      <p>How services agree, communicate, fail gracefully, and stay observable and fast in production.</p>
    </a>
    <div class="category-links">
      <a href="docs/distributed-systems/">Distributed Systems</a> &middot;
      <a href="docs/api-design/">API Design</a> &middot;
      <a href="docs/event-driven/">Event-Driven</a> &middot;
      <a href="docs/observability/">Observability</a> &middot;
      <a href="docs/testing/">Testing</a> &middot;
      <a href="docs/optimization/">Performance</a>
    </div>
  </div>

  <div class="category-card aiml">
    <a class="category-card__main" href="docs/artificial-intelligence/">
      <div class="category-icon"><i class="fas fa-brain"></i></div>
      <h3>AI &amp; Machine Learning</h3>
      <p>Machine-learning concepts and theory, plus hands-on diffusion image models, their tooling, and production use.</p>
    </a>
    <div class="category-links">
      <a href="docs/technology/ai/">ML Concepts</a> &middot;
      <a href="docs/ai-ml/">Generative AI</a> &middot;
      <a href="docs/ai-ml/base-models-comparison.html">Base Models</a> &middot;
      <a href="docs/ai-ml/comfyui-guide.html">ComfyUI</a> &middot;
      <a href="docs/ai-ml/lora-training.html">LoRA</a>
    </div>
  </div>

  <div class="category-card aiml">
    <a class="category-card__main" href="docs/gamedev/">
      <div class="category-icon"><i class="fas fa-gamepad"></i></div>
      <h3>Game Development &amp; Graphics</h3>
      <p>Game architecture, the rendering pipeline, shaders, multiplayer networking, and XR.</p>
    </a>
    <div class="category-links">
      <a href="docs/gamedev/">Game Dev</a> &middot;
      <a href="docs/graphics/3d-rendering.html">Rendering</a> &middot;
      <a href="docs/graphics/shaders.html">Shaders</a> &middot;
      <a href="docs/technology/unreal.html">Unreal</a> &middot;
      <a href="docs/vr-ar/">VR/AR</a>
    </div>
  </div>

  <div class="category-card physics">
    <a class="category-card__main" href="docs/physics/">
      <div class="category-icon"><i class="fas fa-atom"></i></div>
      <h3>Physics</h3>
      <p>Classical mechanics and thermodynamics through relativity, quantum field theory, condensed matter, and computation.</p>
    </a>
    <div class="category-links">
      <a href="docs/physics/classical-mechanics/">Mechanics</a> &middot;
      <a href="docs/physics/relativity/">Relativity</a> &middot;
      <a href="docs/physics/quantum-mechanics/">Quantum</a> &middot;
      <a href="docs/physics/quantum-field-theory.html">QFT</a> &middot;
      <a href="docs/physics/condensed-matter/">Condensed Matter</a>
    </div>
  </div>

  <div class="category-card advanced">
    <a class="category-card__main" href="docs/advanced/">
      <div class="category-icon"><i class="fas fa-flask"></i></div>
      <h3>Theory &amp; Research</h3>
      <p>Graduate-level, proof-oriented computer science and mathematics.</p>
    </a>
    <div class="category-links">
      <a href="docs/advanced/complexity-theory/">Complexity</a> &middot;
      <a href="docs/advanced/cryptography/">Cryptography</a> &middot;
      <a href="docs/advanced/ai-mathematics/">Learning Theory</a> &middot;
      <a href="docs/advanced/distributed-systems-theory/">Consensus</a> &middot;
      <a href="docs/advanced/quantum-algorithms-research/">Quantum Algorithms</a>
    </div>
  </div>

</div>

## Ways into the site

| Page | Use it for |
|------|------------|
| [Getting Started](getting-started.html) | How the site is organized, the kinds of pages, and a first page for common roles |
| [Documentation index](docs/index.html) | The complete list of pages, one row per topic |
| [Topic Map](docs/topic-map.html) | Prerequisites between topics and ordered reading paths by role |
| [Quick Reference](docs/reference/) | Command cheat sheets (Git, Docker, kubectl, AWS CLI), physics constants and equations, complexity tables |
| [Search](search.html) | Keyword lookup across the main hub and overview pages |

## First pages for newcomers

| Area | Start with |
|------|------------|
| Software tooling | [Git Crash Course](docs/technology/git-crash-course.html) · [Database Crash Course](docs/technology/database-crash-course.html) · [Docker Essentials](docs/technology/docker-essentials.html) |
| Machine learning | [AI Fundamentals](docs/technology/ai-fundamentals-simple.html) (no math) · [Stable Diffusion Fundamentals](docs/ai-ml/stable-diffusion-fundamentals.html) |
| Physics | [Classical Mechanics](docs/physics/classical-mechanics/) · [Thermodynamics](docs/physics/thermodynamics.html) · [Special Relativity](docs/physics/relativity/special-relativity.html) |

---

*The notebook is maintained in the [AndrewAltimit/Documentation](https://github.com/AndrewAltimit/Documentation) repository; corrections and suggestions are welcome as issues or pull requests.*
