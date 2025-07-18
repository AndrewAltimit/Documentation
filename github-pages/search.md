---
layout: docs
title: Search Documentation
---

# Search Documentation

<p class="lead">Find what you need quickly across all documentation sections. Start typing to search through topics, commands, and concepts.</p>

<div class="search-container">
  <input type="text" id="search-input" placeholder="Search documentation..." autofocus>
  <div id="search-results"></div>
</div>

<script>
// Simple client-side search implementation
const searchIndex = [];

// Function to load search index
async function loadSearchIndex() {
  try {
    // In production, you would generate this index during build time
    // For now, we'll create a basic search functionality
    const pages = [
      // Technology
      { title: "Terraform", url: "/Documentation/docs/technology/terraform.html", category: "Infrastructure", content: "Infrastructure as Code, AWS, Azure, GCP, providers, modules" },
      { title: "Docker", url: "/Documentation/docs/technology/docker.html", category: "Containerization", content: "Containers, images, Dockerfile, docker-compose, volumes" },
      { title: "Kubernetes", url: "/Documentation/docs/technology/kubernetes.html", category: "Orchestration", content: "Pods, services, deployments, ingress, helm" },
      { title: "AWS", url: "/Documentation/docs/technology/aws.html", category: "Cloud", content: "EC2, S3, Lambda, CloudFormation, IAM" },
      { title: "Git", url: "/Documentation/docs/technology/git.html", category: "Version Control", content: "Commits, branches, merge, rebase, workflows" },
      { title: "Database Design", url: "/Documentation/docs/technology/database-design.html", category: "Databases", content: "SQL, NoSQL, normalization, indexing, transactions" },
      
      // Physics
      { title: "Classical Mechanics", url: "/Documentation/docs/physics/classical-mechanics.html", category: "Physics", content: "Newton's laws, forces, energy, momentum" },
      { title: "Quantum Mechanics", url: "/Documentation/docs/physics/quantum-mechanics.html", category: "Physics", content: "Wave functions, uncertainty, Schrödinger equation" },
      { title: "Thermodynamics", url: "/Documentation/docs/physics/thermodynamics.html", category: "Physics", content: "Heat, entropy, laws of thermodynamics" },
      { title: "Relativity", url: "/Documentation/docs/physics/relativity.html", category: "Physics", content: "Special relativity, general relativity, spacetime" },
      
      // AI/ML
      { title: "AI Fundamentals - Simplified", url: "/Documentation/docs/technology/ai-fundamentals-simple.html", category: "AI/ML", content: "Machine learning, neural networks, deep learning, no math required" },
      { title: "AI Fundamentals", url: "/Documentation/docs/technology/ai.html", category: "AI/ML", content: "Comprehensive AI overview, neural networks, deep learning" },
      { title: "AI Deep Dive", url: "/Documentation/docs/technology/ai-lecture-2023.html", category: "AI/ML", content: "Advanced AI concepts, research, mathematical foundations" },
      { title: "Stable Diffusion", url: "/Documentation/docs/ai-ml/stable-diffusion-fundamentals.html", category: "AI/ML", content: "Text-to-image, diffusion models, prompting" },
      { title: "ComfyUI Guide", url: "/Documentation/docs/ai-ml/comfyui-guide.html", category: "AI/ML", content: "Nodes, workflows, custom nodes, API" },
      { title: "LoRA Training", url: "/Documentation/docs/ai-ml/lora-training.html", category: "AI/ML", content: "Fine-tuning, datasets, training parameters" },
      
      // Reference Pages
      { title: "Git Command Reference", url: "/Documentation/docs/technology/git-reference.html", category: "Reference", content: "Comprehensive Git commands, workflows, branching, rebasing" },
      { title: "Docker Essentials", url: "/Documentation/docs/technology/docker-essentials.html", category: "Reference", content: "Docker commands, Dockerfile reference, compose, networking" },
      { title: "Database Design", url: "/Documentation/docs/technology/database-design.html", category: "Technology", content: "Database architecture, normalization, SQL, NoSQL" },
      
      // Additional Pages (2025 updates)
      { title: "Distributed Systems", url: "/Documentation/docs/distributed-systems/index.html", category: "Technology", content: "Microservices, consensus algorithms, fault tolerance" },
      { title: "Distributed Systems Theory", url: "/Documentation/docs/advanced/distributed-systems-theory.html", category: "Research", content: "Formal foundations, impossibility results, CAP theorem" },
      { title: "Advanced Topics", url: "/Documentation/docs/advanced/index.html", category: "Research", content: "Graduate-level mathematics, formal methods, research papers" },
      { title: "AI Mathematics", url: "/Documentation/docs/advanced/ai-mathematics.html", category: "Research", content: "Statistical learning theory, optimization, neural network mathematics" },
      { title: "Quantum Algorithms Research", url: "/Documentation/docs/advanced/quantum-algorithms-research.html", category: "Research", content: "Quantum algorithms, complexity theory, quantum advantage" },
      { title: "Networking", url: "/Documentation/docs/technology/networking.html", category: "Technology", content: "TCP/IP, protocols, SDN, network security" },
      { title: "Cybersecurity", url: "/Documentation/docs/technology/cybersecurity.html", category: "Technology", content: "Security principles, zero trust, encryption, threat mitigation" },
      { title: "CI/CD", url: "/Documentation/docs/technology/ci-cd.html", category: "DevOps", content: "Continuous integration, deployment pipelines, automation" },
      { title: "Quantum Field Theory", url: "/Documentation/docs/physics/quantum-field-theory.html", category: "Physics", content: "QFT, gauge theory, renormalization, standard model" },
      { title: "String Theory", url: "/Documentation/docs/physics/string-theory.html", category: "Physics", content: "String theory, M-theory, dualities, extra dimensions" },
      { title: "Condensed Matter", url: "/Documentation/docs/physics/condensed-matter.html", category: "Physics", content: "Solid state physics, superconductivity, topological materials" },
      { title: "Statistical Mechanics", url: "/Documentation/docs/physics/statistical-mechanics.html", category: "Physics", content: "Ensemble theory, phase transitions, critical phenomena" },
      { title: "Quick Reference", url: "/Documentation/docs/reference/index.html", category: "Reference", content: "Cheat sheets, formulas, algorithms, API patterns" },
      { title: "Getting Started", url: "/Documentation/getting-started.html", category: "Guide", content: "Documentation overview, navigation tips, prerequisites" },
      { title: "Topic Map", url: "/Documentation/docs/topic-map.html", category: "Guide", content: "Visual navigation, learning paths, interactive map" },
      { title: "AI/ML Hub", url: "/Documentation/docs/ai-ml/index.html", category: "AI/ML", content: "Generative AI, model training, workflows, tutorials" },
      { title: "Artificial Intelligence Hub", url: "/Documentation/docs/artificial-intelligence/index.html", category: "AI/ML", content: "AI fundamentals to advanced research, learning paths" },
      { title: "Quantum Computing Hub", url: "/Documentation/docs/quantum-computing/index.html", category: "Technology", content: "Quantum basics, programming, algorithms, hardware" },
      { title: "FLUX Models", url: "/Documentation/docs/ai-ml/base-models-comparison.html#flux", category: "AI/ML", content: "FLUX flow matching, schnell variant, guidance distillation" },
      { title: "Stable Diffusion 3", url: "/Documentation/docs/ai-ml/base-models-comparison.html#stable-diffusion-3", category: "AI/ML", content: "SD3, multimodal diffusion transformer, triple text encoding" },
      { title: "ControlNet", url: "/Documentation/docs/ai-ml/controlnet.html", category: "AI/ML", content: "Pose control, depth maps, edge detection, segmentation" },
      { title: "Advanced Techniques", url: "/Documentation/docs/ai-ml/advanced-techniques.html", category: "AI/ML", content: "Inpainting, outpainting, prompt weighting, batch processing" },
      { title: "Model Types", url: "/Documentation/docs/ai-ml/model-types.html", category: "AI/ML", content: "LoRAs, embeddings, VAEs, checkpoints explained" },
      { title: "Output Formats", url: "/Documentation/docs/ai-ml/output-formats.html", category: "AI/ML", content: "Image formats, video generation, audio output" }
    ];
    
    searchIndex.push(...pages);
  } catch (error) {
    console.error('Error loading search index:', error);
  }
}

// Function to perform search
function performSearch(query) {
  const results = [];
  const searchTerms = query.toLowerCase().split(' ').filter(term => term.length > 0);
  
  if (searchTerms.length === 0) {
    return results;
  }
  
  searchIndex.forEach(page => {
    let score = 0;
    const titleLower = page.title.toLowerCase();
    const contentLower = page.content.toLowerCase();
    const categoryLower = page.category.toLowerCase();
    
    searchTerms.forEach(term => {
      // Title matches (highest priority)
      if (titleLower.includes(term)) {
        score += 10;
      }
      // Category matches
      if (categoryLower.includes(term)) {
        score += 5;
      }
      // Content matches
      if (contentLower.includes(term)) {
        score += 1;
      }
    });
    
    if (score > 0) {
      results.push({ ...page, score });
    }
  });
  
  // Sort by score (highest first)
  return results.sort((a, b) => b.score - a.score);
}

// Function to display results
function displayResults(results) {
  const resultsContainer = document.getElementById('search-results');
  
  if (results.length === 0) {
    resultsContainer.innerHTML = '<p class="no-results">No results found.</p>';
    return;
  }
  
  const html = results.map(result => `
    <div class="search-result">
      <h3><a href="${result.url}">${result.title}</a></h3>
      <p class="category">${result.category}</p>
      <p class="snippet">${result.content.substring(0, 150)}...</p>
    </div>
  `).join('');
  
  resultsContainer.innerHTML = html;
}

// Initialize search
document.addEventListener('DOMContentLoaded', async () => {
  await loadSearchIndex();
  
  const searchInput = document.getElementById('search-input');
  let searchTimeout;
  
  searchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    const query = e.target.value;
    
    if (query.length < 2) {
      document.getElementById('search-results').innerHTML = '';
      return;
    }
    
    // Debounce search
    searchTimeout = setTimeout(() => {
      const results = performSearch(query);
      displayResults(results);
    }, 300);
  });
  
  // Handle search from URL parameter
  const urlParams = new URLSearchParams(window.location.search);
  const queryParam = urlParams.get('q');
  if (queryParam) {
    searchInput.value = queryParam;
    const results = performSearch(queryParam);
    displayResults(results);
  }
});
</script>

<style>
.search-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem 0;
}

#search-input {
  width: 100%;
  padding: 1rem;
  font-size: 1.1rem;
  border: 2px solid #ddd;
  border-radius: 8px;
  outline: none;
  transition: border-color 0.3s;
}

#search-input:focus {
  border-color: #007bff;
}

#search-results {
  margin-top: 2rem;
}

.search-result {
  padding: 1.5rem;
  border: 1px solid #eee;
  border-radius: 8px;
  margin-bottom: 1rem;
  transition: all 0.3s;
}

.search-result:hover {
  border-color: #007bff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.search-result h3 {
  margin: 0 0 0.5rem 0;
}

.search-result h3 a {
  color: #007bff;
  text-decoration: none;
}

.search-result h3 a:hover {
  text-decoration: underline;
}

.search-result .category {
  color: #666;
  font-size: 0.9rem;
  margin: 0 0 0.5rem 0;
}

.search-result .snippet {
  color: #444;
  margin: 0;
  line-height: 1.5;
}

.no-results {
  text-align: center;
  color: #666;
  font-size: 1.1rem;
  padding: 3rem;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  #search-input {
    background: #2a2a2a;
    border-color: #444;
    color: #fff;
  }
  
  #search-input:focus {
    border-color: #4a9eff;
  }
  
  .search-result {
    background: #2a2a2a;
    border-color: #444;
  }
  
  .search-result:hover {
    border-color: #4a9eff;
  }
  
  .search-result h3 a {
    color: #4a9eff;
  }
  
  .search-result .category {
    color: #aaa;
  }
  
  .search-result .snippet {
    color: #ccc;
  }
}
</style>

## Quick Links

### Popular Pages
- [Git Command Reference](/Documentation/docs/technology/git-reference.html) - Most comprehensive Git guide
- [Docker Essentials](/Documentation/docs/technology/docker-essentials.html) - Container commands and operations
- [ComfyUI Guide](/Documentation/docs/ai-ml/comfyui-guide.html) - Visual AI workflow creation
- [Kubernetes](/Documentation/docs/technology/kubernetes.html) - Container orchestration

### Browse by Category
- **[All Documentation](/Documentation/docs/index.html)** - Complete documentation index
- **[Technology](/Documentation/docs/index.html#technology)** - Infrastructure, development, and tools
- **[Physics](/Documentation/docs/index.html#physics)** - Classical and modern physics
- **[AI/ML](/Documentation/docs/ai-ml/index.html)** - Machine learning and generative AI
- **[Advanced Topics](/Documentation/docs/advanced/index.html)** - Research-level content
- **[Reference](/Documentation/docs/reference/index.html)** - Quick reference guides

### Can't find what you're looking for?
Try browsing our [complete documentation index](/Documentation/docs/index.html) or check the [topic map](/Documentation/docs/topic-map.html) for a visual overview of all available content.