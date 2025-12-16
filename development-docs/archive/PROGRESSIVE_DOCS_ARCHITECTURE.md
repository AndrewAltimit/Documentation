# Progressive Documentation Architecture

## Overview

This document outlines a comprehensive progressive documentation system that implements:
- **Progressive Disclosure**: Content revealed based on user expertise level
- **Prerequisite Tracking**: Clear learning paths with dependency management
- **Difficulty Tiering**: Beginner, Intermediate, and Advanced content separation
- **Multiple Content Types**: Tutorials, How-To Guides, Deep Dives, and Reference
- **Visual Learning Paths**: Interactive roadmaps for learning journeys

## 1. Site Structure

### Directory Organization

```
github-pages/
├── docs/
│   ├── tutorials/           # Tier 1: Get Started
│   │   ├── index.md
│   │   ├── quickstart/
│   │   └── first-steps/
│   ├── guides/             # Tier 2: How-To Guides
│   │   ├── index.md
│   │   ├── beginner/
│   │   ├── intermediate/
│   │   └── advanced/
│   ├── concepts/           # Tier 3: Deep Dives
│   │   ├── index.md
│   │   ├── fundamentals/
│   │   ├── architecture/
│   │   └── theory/
│   ├── reference/          # Tier 4: API Reference
│   │   ├── index.md
│   │   ├── api/
│   │   ├── cli/
│   │   └── config/
│   └── roadmaps/           # Learning Paths
│       ├── index.md
│       ├── beginner.md
│       ├── intermediate.md
│       └── advanced.md
├── _data/
│   ├── navigation.yml      # Navigation structure
│   ├── metadata.yml        # Global metadata definitions
│   ├── prerequisites.yml   # Prerequisite relationships
│   └── roadmaps.yml       # Learning path definitions
├── _includes/
│   ├── prerequisite-check.html
│   ├── difficulty-badge.html
│   ├── learning-objectives.html
│   ├── expandable-section.html
│   ├── code-tabs.html
│   └── roadmap-visual.html
└── _layouts/
    ├── tutorial.html       # Tutorial-specific layout
    ├── guide.html         # How-to guide layout
    ├── concept.html       # Deep dive layout
    ├── reference.html     # Reference layout
    └── roadmap.html       # Learning path layout
```

## 2. Metadata System

### YAML Front Matter Schema

```yaml
---
# Required fields
title: "Understanding Git Internals"
description: "Deep dive into Git's object model and internal architecture"
type: "concept"  # tutorial | guide | concept | reference
difficulty: "advanced"  # beginner | intermediate | advanced
estimated_time: "45 minutes"

# Learning metadata
prerequisites:
  - id: "git-basics"
    title: "Git Basics Tutorial"
    required: true
  - id: "data-structures"
    title: "Understanding Data Structures"
    required: false

learning_objectives:
  - "Understand Git's object model (blobs, trees, commits)"
  - "Explore the .git directory structure"
  - "Learn how Git stores and retrieves data"

skills_gained:
  - "Git internals"
  - "Content-addressable storage"
  - "Merkle trees"

# Organization
tags:
  - "version-control"
  - "git"
  - "distributed-systems"

category: "technology/git"
order: 3  # Position in learning path

# Content variations
has_interactive: true
has_exercises: true
has_video: false

# Related content
related:
  - id: "git-branching"
    type: "guide"
    relationship: "next"
  - id: "git-performance"
    type: "concept"
    relationship: "advanced"

# Tracking
last_updated: "2025-07-18"
version: "2.0"
contributors:
  - "AndrewAltimit"
---
```

### Metadata Definitions (_data/metadata.yml)

```yaml
difficulties:
  beginner:
    label: "Beginner"
    color: "#28a745"
    icon: "seedling"
    description: "No prior knowledge required"
  intermediate:
    label: "Intermediate"
    color: "#ffc107"
    icon: "plant-wilt"
    description: "Basic understanding needed"
  advanced:
    label: "Advanced"
    color: "#dc3545"
    icon: "tree"
    description: "Solid foundation required"

content_types:
  tutorial:
    label: "Tutorial"
    icon: "graduation-cap"
    description: "Step-by-step learning"
    template: "tutorial"
  guide:
    label: "How-To Guide"
    icon: "tools"
    description: "Solve specific problems"
    template: "guide"
  concept:
    label: "Deep Dive"
    icon: "microscope"
    description: "Understand the theory"
    template: "concept"
  reference:
    label: "Reference"
    icon: "book"
    description: "Complete documentation"
    template: "reference"

skill_levels:
  1: "Awareness"
  2: "Basic Understanding"
  3: "Working Knowledge"
  4: "Proficiency"
  5: "Expert"
```

## 3. Content Templates

### Tutorial Template (_layouts/tutorial.html)

```html
---
layout: default
---

<div class="tutorial-container">
  <!-- Progress indicator -->
  <div class="progress-bar">
    <div class="progress-fill" style="width: 0%"></div>
  </div>

  <!-- Header with metadata -->
  <header class="tutorial-header">
    <div class="metadata-badges">
      {% include difficulty-badge.html difficulty=page.difficulty %}
      <span class="time-estimate">
        <i class="far fa-clock"></i> {{ page.estimated_time }}
      </span>
      <span class="content-type">
        <i class="fas fa-{{ site.data.metadata.content_types[page.type].icon }}"></i>
        {{ site.data.metadata.content_types[page.type].label }}
      </span>
    </div>
    
    <h1>{{ page.title }}</h1>
    <p class="description">{{ page.description }}</p>
  </header>

  <!-- Prerequisites check -->
  {% if page.prerequisites %}
    {% include prerequisite-check.html prerequisites=page.prerequisites %}
  {% endif %}

  <!-- Learning objectives -->
  {% if page.learning_objectives %}
    {% include learning-objectives.html objectives=page.learning_objectives %}
  {% endif %}

  <!-- Main content with automatic section tracking -->
  <main class="tutorial-content" data-track-progress="true">
    {{ content }}
  </main>

  <!-- Interactive elements -->
  {% if page.has_interactive %}
    <div class="interactive-section">
      <h2>Try It Yourself</h2>
      <div id="interactive-sandbox"></div>
    </div>
  {% endif %}

  <!-- Exercises -->
  {% if page.has_exercises %}
    <section class="exercises">
      <h2>Practice Exercises</h2>
      {% include exercises.html page_id=page.id %}
    </section>
  {% endif %}

  <!-- Next steps -->
  <footer class="tutorial-footer">
    <div class="skills-gained">
      <h3>Skills Gained</h3>
      <ul class="skill-tags">
        {% for skill in page.skills_gained %}
          <li class="skill-tag">{{ skill }}</li>
        {% endfor %}
      </ul>
    </div>
    
    <nav class="next-steps">
      <h3>What's Next?</h3>
      {% for item in page.related %}
        {% if item.relationship == "next" %}
          <a href="/docs/{{ item.type }}s/{{ item.id }}" class="next-link">
            <i class="fas fa-arrow-right"></i> {{ item.title }}
          </a>
        {% endif %}
      {% endfor %}
    </nav>
  </footer>
</div>

<script src="{{ '/assets/js/tutorial-progress.js' | relative_url }}"></script>
```

### Guide Template (_layouts/guide.html)

```html
---
layout: default
---

<div class="guide-container">
  <header class="guide-header">
    {% include difficulty-badge.html difficulty=page.difficulty %}
    <h1>{{ page.title }}</h1>
    <p class="description">{{ page.description }}</p>
    
    <!-- Quick navigation for long guides -->
    {% if page.toc %}
      <nav class="guide-toc">
        <h3>In This Guide</h3>
        {{ content | toc_only }}
      </nav>
    {% endif %}
  </header>

  <!-- Prerequisites -->
  {% if page.prerequisites %}
    <aside class="prerequisites-sidebar">
      <h3>Before You Begin</h3>
      {% include prerequisite-check.html prerequisites=page.prerequisites compact=true %}
    </aside>
  {% endif %}

  <!-- Main content with collapsible advanced sections -->
  <main class="guide-content">
    {{ content | wrap_advanced_sections }}
  </main>

  <!-- Code examples with difficulty tabs -->
  <section class="code-examples">
    {% include code-tabs.html examples=page.code_examples %}
  </section>

  <!-- Related guides -->
  <aside class="related-guides">
    <h3>Related Guides</h3>
    {% include related-content.html related=page.related type="guide" %}
  </aside>
</div>
```

### Deep Dive Template (_layouts/concept.html)

```html
---
layout: default
---

<div class="concept-container">
  <header class="concept-header">
    <div class="concept-meta">
      {% include difficulty-badge.html difficulty=page.difficulty %}
      <span class="reading-time">{{ content | reading_time }}</span>
    </div>
    
    <h1>{{ page.title }}</h1>
    <p class="abstract">{{ page.description }}</p>
    
    <!-- Table of contents for deep dives -->
    <details class="concept-toc">
      <summary>Table of Contents</summary>
      {{ content | toc_tree }}
    </details>
  </header>

  <!-- Key concepts callout -->
  {% if page.key_concepts %}
    <aside class="key-concepts">
      <h3>Key Concepts</h3>
      <dl>
        {% for concept in page.key_concepts %}
          <dt>{{ concept.term }}</dt>
          <dd>{{ concept.definition }}</dd>
        {% endfor %}
      </dl>
    </aside>
  {% endif %}

  <!-- Main content with progressive disclosure -->
  <main class="concept-content">
    {{ content | add_complexity_toggles }}
  </main>

  <!-- Mathematical notation support -->
  {% if page.has_math %}
    <script>
      MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']],
          displayMath: [['$$', '$$'], ['\\[', '\\]']]
        }
      };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  {% endif %}

  <!-- Further reading -->
  <footer class="concept-footer">
    <h3>Further Reading</h3>
    {% include further-reading.html page_id=page.id %}
  </footer>
</div>
```

### Reference Template (_layouts/reference.html)

```html
---
layout: default
---

<div class="reference-container">
  <header class="reference-header">
    <h1>{{ page.title }}</h1>
    <div class="reference-meta">
      <span class="version">v{{ page.version }}</span>
      <span class="last-updated">Updated: {{ page.last_updated | date: "%B %d, %Y" }}</span>
    </div>
  </header>

  <!-- Quick search -->
  <div class="reference-search">
    <input type="text" id="reference-search" placeholder="Search this reference...">
  </div>

  <!-- Sidebar navigation -->
  <aside class="reference-sidebar">
    <nav class="reference-nav">
      {{ content | extract_headings | generate_nav }}
    </nav>
  </aside>

  <!-- Main content -->
  <main class="reference-content">
    {{ content | add_anchor_links | add_copy_buttons }}
  </main>

  <!-- API signature highlighting -->
  {% if page.api_reference %}
    <script src="{{ '/assets/js/syntax-highlight.js' | relative_url }}"></script>
  {% endif %}
</div>
```

## 4. Visual Roadmap System

### Roadmap Data Structure (_data/roadmaps.yml)

```yaml
beginner:
  title: "Beginner Learning Path"
  description: "Start your journey from zero"
  estimated_duration: "2-3 months"
  
  milestones:
    - id: "foundations"
      title: "Build Your Foundation"
      description: "Essential concepts and tools"
      items:
        - type: "tutorial"
          id: "git-quickstart"
          title: "Git Quick Start"
          required: true
        - type: "tutorial"
          id: "docker-basics"
          title: "Docker Basics"
          required: true
        - type: "guide"
          id: "terminal-essentials"
          title: "Terminal Essentials"
          required: false
    
    - id: "first-project"
      title: "Your First Project"
      description: "Apply what you've learned"
      items:
        - type: "guide"
          id: "create-repo"
          title: "Create Your First Repository"
        - type: "guide"
          id: "docker-compose"
          title: "Multi-Container Apps"

intermediate:
  title: "Intermediate Learning Path"
  description: "Level up your skills"
  estimated_duration: "3-4 months"
  
  prerequisites:
    - "beginner"
  
  milestones:
    - id: "advanced-workflows"
      title: "Advanced Workflows"
      description: "Professional development practices"
      items:
        - type: "concept"
          id: "git-internals"
          title: "Git Under the Hood"
        - type: "guide"
          id: "ci-cd-pipelines"
          title: "CI/CD Pipelines"

advanced:
  title: "Advanced Learning Path"
  description: "Master complex systems"
  estimated_duration: "6+ months"
  
  prerequisites:
    - "intermediate"
```

### Visual Roadmap Component (_includes/roadmap-visual.html)

```html
<div class="roadmap-container" data-level="{{ include.level }}">
  <svg class="roadmap-svg" viewBox="0 0 1200 800">
    <!-- Dynamic SVG generation from roadmap data -->
    {% assign roadmap = site.data.roadmaps[include.level] %}
    
    <!-- Milestone nodes -->
    {% for milestone in roadmap.milestones %}
      <g class="milestone-group" data-milestone-id="{{ milestone.id }}">
        <circle cx="{{ forloop.index | times: 200 }}" cy="100" r="50" 
                class="milestone-node {% if milestone.completed %}completed{% endif %}" />
        <text x="{{ forloop.index | times: 200 }}" y="105" text-anchor="middle">
          {{ milestone.title }}
        </text>
        
        <!-- Connection lines -->
        {% unless forloop.last %}
          <line x1="{{ forloop.index | times: 200 | plus: 50 }}" y1="100"
                x2="{{ forloop.index | times: 200 | plus: 150 }}" y2="100"
                class="connection-line" />
        {% endunless %}
        
        <!-- Learning items -->
        {% for item in milestone.items %}
          <g class="item-node" data-item-id="{{ item.id }}">
            <rect x="{{ forloop.parent.index | times: 200 | minus: 30 }}" 
                  y="{{ forloop.index | times: 60 | plus: 150 }}"
                  width="60" height="40" rx="5"
                  class="item-rect {{ item.type }}" />
            <text x="{{ forloop.parent.index | times: 200 }}" 
                  y="{{ forloop.index | times: 60 | plus: 175 }}"
                  text-anchor="middle" class="item-text">
              {{ item.title | truncate: 10 }}
            </text>
          </g>
        {% endfor %}
      </g>
    {% endfor %}
  </svg>
  
  <!-- Interactive tooltip -->
  <div class="roadmap-tooltip" style="display: none;">
    <h4 class="tooltip-title"></h4>
    <p class="tooltip-description"></p>
    <a href="#" class="tooltip-link">Start Learning →</a>
  </div>
</div>

<style>
.roadmap-container {
  position: relative;
  max-width: 100%;
  margin: 2rem 0;
}

.milestone-node {
  fill: #e0e0e0;
  stroke: #333;
  stroke-width: 2;
  cursor: pointer;
  transition: all 0.3s ease;
}

.milestone-node:hover {
  fill: #4CAF50;
  transform: scale(1.1);
}

.milestone-node.completed {
  fill: #4CAF50;
}

.connection-line {
  stroke: #999;
  stroke-width: 2;
  stroke-dasharray: 5,5;
}

.item-rect {
  fill: #f0f0f0;
  stroke: #666;
  cursor: pointer;
}

.item-rect.tutorial { fill: #e3f2fd; }
.item-rect.guide { fill: #fff3e0; }
.item-rect.concept { fill: #f3e5f5; }
.item-rect.reference { fill: #e8f5e9; }

.roadmap-tooltip {
  position: absolute;
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  z-index: 1000;
}
</style>

<script>
// Interactive roadmap functionality
document.addEventListener('DOMContentLoaded', function() {
  const roadmap = document.querySelector('.roadmap-container');
  const tooltip = roadmap.querySelector('.roadmap-tooltip');
  
  // Add interactivity to nodes
  roadmap.querySelectorAll('.milestone-node, .item-rect').forEach(node => {
    node.addEventListener('mouseenter', function(e) {
      // Show tooltip with relevant information
      const rect = e.target.getBoundingClientRect();
      const containerRect = roadmap.getBoundingClientRect();
      
      tooltip.style.left = (rect.left - containerRect.left + rect.width / 2) + 'px';
      tooltip.style.top = (rect.bottom - containerRect.top + 10) + 'px';
      tooltip.style.display = 'block';
      
      // Populate tooltip content based on node data
      // ... tooltip population logic
    });
    
    node.addEventListener('mouseleave', function() {
      tooltip.style.display = 'none';
    });
    
    node.addEventListener('click', function(e) {
      // Navigate to the relevant content
      const itemId = e.target.closest('[data-item-id]')?.dataset.itemId;
      if (itemId) {
        window.location.href = `/docs/${itemId}`;
      }
    });
  });
});
</script>
```

## 5. Implementation Plan

### Phase 1: Foundation (Week 1-2)
1. **Set up directory structure**
   - Create new directories for tutorials, guides, concepts, reference
   - Move existing content to appropriate tiers
   
2. **Implement metadata system**
   - Add front matter to all existing documents
   - Create metadata.yml with definitions
   - Build prerequisite tracking system

3. **Create base templates**
   - Develop layouts for each content type
   - Implement reusable includes

### Phase 2: Content Migration (Week 3-4)
1. **Audit existing content**
   - Categorize by type and difficulty
   - Identify prerequisites
   - Add learning objectives

2. **Restructure content**
   - Split complex pages into tiered versions
   - Create beginner-friendly tutorials
   - Extract reference documentation

3. **Add progressive disclosure**
   - Implement collapsible sections
   - Add code complexity tabs
   - Create simplified examples

### Phase 3: Enhanced Features (Week 5-6)
1. **Build interactive components**
   - Prerequisite checker
   - Progress tracking
   - Interactive code sandboxes

2. **Implement visual roadmaps**
   - Create SVG-based roadmap component
   - Add progress tracking
   - Build milestone system

3. **Add search and filtering**
   - Enhance search with metadata
   - Create difficulty filters
   - Implement tag-based navigation

### Phase 4: Polish and Launch (Week 7-8)
1. **User experience improvements**
   - Add animations and transitions
   - Implement responsive design
   - Optimize performance

2. **Documentation and guides**
   - Create contributor guidelines
   - Document the new system
   - Build content creation templates

3. **Testing and refinement**
   - User testing with different skill levels
   - Performance optimization
   - Accessibility audit

## 6. Success Metrics

### Quantitative Metrics
- **Time to First Success**: Measure how quickly beginners complete their first tutorial
- **Content Completion Rate**: Track percentage of users who finish each content type
- **Navigation Efficiency**: Monitor how users move through prerequisites
- **Search Effectiveness**: Measure search-to-success rates

### Qualitative Metrics
- **User Feedback**: Collect satisfaction ratings for each difficulty level
- **Learning Effectiveness**: Survey users on skill acquisition
- **Content Clarity**: Track questions and confusion points
- **Navigation Intuitiveness**: Observe user pathways through content

## 7. Maintenance Guidelines

### Content Review Cycle
- **Monthly**: Review and update prerequisites
- **Quarterly**: Audit difficulty ratings
- **Bi-annually**: Major content restructuring
- **Annually**: Complete system evaluation

### Adding New Content
1. Determine appropriate tier and difficulty
2. Identify prerequisites
3. Create learning objectives
4. Use appropriate template
5. Add to relevant roadmaps
6. Update navigation

### Quality Standards
- All content must have complete metadata
- Examples must work at specified difficulty level
- Prerequisites must be accurate
- Code examples must be tested
- Progressive disclosure must be consistent