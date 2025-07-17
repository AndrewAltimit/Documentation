---
layout: docs
title: Search Documentation
---

# Search Documentation

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
      { title: "AI Fundamentals", url: "/Documentation/docs/technology/ai-fundamentals-simple.html", category: "AI/ML", content: "Machine learning, neural networks, deep learning" },
      { title: "Stable Diffusion", url: "/Documentation/docs/ai-ml/stable-diffusion-fundamentals.html", category: "AI/ML", content: "Text-to-image, diffusion models, prompting" },
      { title: "ComfyUI Guide", url: "/Documentation/docs/ai-ml/comfyui-guide.html", category: "AI/ML", content: "Nodes, workflows, custom nodes, API" },
      { title: "LoRA Training", url: "/Documentation/docs/ai-ml/lora-training.html", category: "AI/ML", content: "Fine-tuning, datasets, training parameters" },
      
      // Tutorials
      { title: "Git Crash Course", url: "/Documentation/docs/technology/git-crash-course.html", category: "Tutorial", content: "Quick start guide for Git" },
      { title: "Docker Crash Course", url: "/Documentation/docs/technology/docker-crash-course.html", category: "Tutorial", content: "Learn Docker from scratch" },
      { title: "Database Crash Course", url: "/Documentation/docs/technology/database-crash-course.html", category: "Tutorial", content: "SQL and database fundamentals" }
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