// Search functionality for Andrew's Notebook
(function() {
    'use strict';
    
    // List of all documentation pages to index
    const PAGES = [
        // Getting Started / Quick Start pages
        { url: 'docs/technology/git-crash-course.html', title: 'Git in 5 Minutes', category: 'tutorials' },
        { url: 'docs/technology/docker-crash-course.html', title: 'Docker in 5 Minutes', category: 'tutorials' },
        { url: 'docs/technology/database-crash-course.html', title: 'Databases in 5 Minutes', category: 'tutorials' },
        { url: 'docs/technology/ai-fundamentals-simple.html', title: 'AI Fundamentals Simple', category: 'tutorials' },
        { url: 'docs/tutorials/git-quickstart.html', title: 'Git Quick Start Tutorial', category: 'tutorials' },
        
        // Physics pages
        { url: 'docs/physics/index.html', title: 'Physics Overview', category: 'physics' },
        { url: 'docs/physics/classical-mechanics.html', title: 'Classical Mechanics', category: 'physics' },
        { url: 'docs/physics/thermodynamics.html', title: 'Thermodynamics', category: 'physics' },
        { url: 'docs/physics/statistical-mechanics.html', title: 'Statistical Mechanics', category: 'physics' },
        { url: 'docs/physics/relativity.html', title: 'Relativity', category: 'physics' },
        { url: 'docs/physics/quantum-mechanics.html', title: 'Quantum Mechanics', category: 'physics' },
        { url: 'docs/physics/condensed-matter.html', title: 'Condensed Matter Physics', category: 'physics' },
        { url: 'docs/physics/quantum-field-theory.html', title: 'Quantum Field Theory', category: 'physics' },
        { url: 'docs/physics/string-theory.html', title: 'String Theory', category: 'physics' },
        
        // Technology pages
        { url: 'docs/technology/index.html', title: 'Technology Overview', category: 'technology' },
        { url: 'docs/technology/terraform.html', title: 'Terraform', category: 'technology' },
        { url: 'docs/technology/docker.html', title: 'Docker Containers', category: 'technology' },
        { url: 'docs/technology/aws.html', title: 'AWS', category: 'technology' },
        { url: 'docs/technology/kubernetes.html', title: 'Kubernetes', category: 'technology' },
        { url: 'docs/technology/database-design.html', title: 'Database Design', category: 'technology' },
        { url: 'docs/technology/networking.html', title: 'Networking', category: 'technology' },
        { url: 'docs/technology/cybersecurity.html', title: 'Cybersecurity', category: 'technology' },
        { url: 'docs/technology/git.html', title: 'Git Version Control', category: 'technology' },
        { url: 'docs/technology/branching.html', title: 'Branching Strategies', category: 'technology' },
        { url: 'docs/technology/unreal.html', title: 'Unreal Engine', category: 'technology' },
        { url: 'docs/technology/quantumcomputing.html', title: 'Quantum Computing', category: 'technology' },
        { url: 'docs/technology/ai.html', title: 'Artificial Intelligence', category: 'technology' },
        { url: 'docs/technology/ai-lecture-2023.html', title: 'AI Lecture 2023', category: 'technology' },
        { url: 'docs/technology/please-build.html', title: 'Please Build', category: 'technology' },
        
        // AI/ML - Generative AI pages
        { url: 'docs/ai-ml/index.html', title: 'AI/ML Overview', category: 'ai-ml' },
        { url: 'docs/ai-ml/stable-diffusion-fundamentals.html', title: 'Stable Diffusion Fundamentals', category: 'ai-ml' },
        { url: 'docs/ai-ml/base-models-comparison.html', title: 'Base Models Comparison', category: 'ai-ml' },
        { url: 'docs/ai-ml/model-types.html', title: 'Model Types', category: 'ai-ml' },
        { url: 'docs/ai-ml/lora-training.html', title: 'LoRA Training', category: 'ai-ml' },
        { url: 'docs/ai-ml/controlnet.html', title: 'ControlNet Guide', category: 'ai-ml' },
        { url: 'docs/ai-ml/comfyui-guide.html', title: 'ComfyUI Guide', category: 'ai-ml' },
        { url: 'docs/ai-ml/output-formats.html', title: 'Output Formats', category: 'ai-ml' },
        { url: 'docs/ai-ml/advanced-techniques.html', title: 'Advanced Techniques', category: 'ai-ml' },
        
        // Advanced Topics
        { url: 'docs/advanced/index.html', title: 'Research Hub', category: 'advanced' },
        { url: 'docs/advanced/ai-mathematics/index.html', title: 'AI Mathematics', category: 'advanced' },
        { url: 'docs/advanced/distributed-systems-theory/index.html', title: 'Distributed Systems Theory', category: 'advanced' },
        { url: 'docs/advanced/quantum-algorithms-research/index.html', title: 'Quantum Algorithms Research', category: 'advanced' },
        
        // Other pages
        { url: 'docs/topic-map.html', title: 'Interactive Topic Map', category: 'navigation' },
        { url: 'getting-started.html', title: 'Getting Started', category: 'navigation' },
        { url: 'index.html', title: 'Home', category: 'navigation' }
    ];
    
    let searchIndex = [];
    let indexedPages = 0;
    let selectedResultIndex = -1;
    
    // Initialize search when page loads
    document.addEventListener('DOMContentLoaded', function() {
        const searchInput = document.getElementById('search-input');
        const searchStatus = document.getElementById('search-status');
        const searchResults = document.getElementById('search-results');
        
        // Show loading status
        searchStatus.innerHTML = '<div class="loading">Indexing documentation...</div>';
        
        // Index all pages
        indexPages().then(() => {
            searchStatus.innerHTML = '';
            
            // Handle search input
            searchInput.addEventListener('input', debounce(function(e) {
                const query = e.target.value.trim();
                if (query.length >= 2) {
                    performSearch(query);
                } else {
                    searchResults.innerHTML = '';
                }
            }, 300));
            
            // Handle URL parameters
            const urlParams = new URLSearchParams(window.location.search);
            const query = urlParams.get('q');
            if (query) {
                searchInput.value = query;
                performSearch(query);
            }
            
            // Add keyboard navigation
            searchInput.addEventListener('keydown', function(e) {
                const results = document.querySelectorAll('.search-result');
                
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    selectedResultIndex = Math.min(selectedResultIndex + 1, results.length - 1);
                    updateSelectedResult(results, selectedResultIndex);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    selectedResultIndex = Math.max(selectedResultIndex - 1, -1);
                    updateSelectedResult(results, selectedResultIndex);
                } else if (e.key === 'Enter' && selectedResultIndex >= 0) {
                    e.preventDefault();
                    const link = results[selectedResultIndex].querySelector('h3 a');
                    if (link) {
                        window.location.href = link.href;
                    }
                } else if (e.key === 'Escape') {
                    searchInput.value = '';
                    searchResults.innerHTML = '';
                    selectedResultIndex = -1;
                }
            });
        });
    });
    
    // Update selected search result
    function updateSelectedResult(results, index) {
        results.forEach((result, i) => {
            if (i === index) {
                result.style.backgroundColor = '#e5e7eb';
                result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                result.style.backgroundColor = '';
            }
        });
    }
    
    // Index all documentation pages
    async function indexPages() {
        const promises = PAGES.map(page => indexPage(page));
        await Promise.all(promises);
    }
    
    // Index a single page
    async function indexPage(page) {
        try {
            const response = await fetch(page.url);
            const html = await response.text();
            
            // Parse HTML and extract text content
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            
            // Remove script and style elements
            const scripts = doc.querySelectorAll('script, style');
            scripts.forEach(el => el.remove());
            
            // Get text content
            const content = doc.body ? doc.body.textContent : '';
            
            // Clean up text
            const cleanContent = content
                .replace(/\s+/g, ' ')
                .trim()
                .toLowerCase();
            
            // Extract headings for better context
            const headings = [];
            doc.querySelectorAll('h1, h2, h3').forEach(heading => {
                headings.push(heading.textContent.trim());
            });
            
            searchIndex.push({
                ...page,
                content: cleanContent,
                headings: headings,
                originalContent: content
            });
            
            indexedPages++;
        } catch (error) {
            console.error(`Failed to index ${page.url}:`, error);
        }
    }
    
    // Perform search
    function performSearch(query) {
        const searchResults = document.getElementById('search-results');
        const lowerQuery = query.toLowerCase();
        const terms = lowerQuery.split(/\s+/).filter(term => term.length > 1);
        
        // Score and filter results
        const results = searchIndex
            .map(page => {
                let score = 0;
                let excerpts = [];
                
                // Score based on title match
                const titleLower = page.title.toLowerCase();
                if (titleLower.includes(lowerQuery)) {
                    score += 10;
                }
                terms.forEach(term => {
                    if (titleLower.includes(term)) {
                        score += 5;
                    }
                });
                
                // Score based on heading matches
                page.headings.forEach(heading => {
                    const headingLower = heading.toLowerCase();
                    if (headingLower.includes(lowerQuery)) {
                        score += 5;
                    }
                    terms.forEach(term => {
                        if (headingLower.includes(term)) {
                            score += 2;
                        }
                    });
                });
                
                // Score based on content matches
                const contentMatches = [];
                const uniqueExcerpts = new Set();
                
                terms.forEach(term => {
                    const regex = new RegExp(term, 'gi');
                    const matches = page.content.match(regex);
                    if (matches) {
                        score += matches.length;
                        
                        // Find excerpts around matches (up to 3 unique excerpts per term)
                        let excerptCount = 0;
                        let lastIndex = 0;
                        while (excerptCount < 3) {
                            const index = page.content.indexOf(term.toLowerCase(), lastIndex);
                            if (index === -1) break;
                            
                            // Get corresponding position in original content
                            const originalIndex = findOriginalIndex(page.originalContent, page.content, index);
                            if (originalIndex !== -1) {
                                const start = Math.max(0, originalIndex - 80);
                                const end = Math.min(page.originalContent.length, originalIndex + 80);
                                const excerpt = page.originalContent.substring(start, end).trim();
                                
                                // Only add if it's sufficiently different from existing excerpts
                                if (excerpt.length > 20 && !isDuplicateExcerpt(excerpt, uniqueExcerpts)) {
                                    contentMatches.push({ term, excerpt });
                                    uniqueExcerpts.add(excerpt.toLowerCase());
                                    excerptCount++;
                                }
                            }
                            
                            lastIndex = index + term.length;
                        }
                    }
                });
                
                // Create excerpts with highlighted terms
                if (contentMatches.length > 0) {
                    // Use the first few matches
                    contentMatches.slice(0, 3).forEach(match => {
                        excerpts.push(highlightTerms(match.excerpt, terms));
                    });
                }
                
                return {
                    ...page,
                    score,
                    excerpts
                };
            })
            .filter(result => result.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, 20); // Limit to top 20 results
        
        // Display results
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="no-results">No results found. Try different keywords.</div>';
        } else {
            const resultsHtml = `
                <div class="search-info">Found ${results.length} result${results.length === 1 ? '' : 's'}</div>
                ${results.map(result => `
                    <div class="search-result">
                        <div class="search-category ${result.category}">${getCategoryDisplayName(result.category)}</div>
                        <h3><a href="${result.url}">${highlightTerms(result.title, terms)}</a></h3>
                        ${result.excerpts.length > 0 ? 
                            '<div class="search-excerpt">...' + result.excerpts.join('...') + '...</div>' : 
                            ''}
                    </div>
                `).join('')}
            `;
            searchResults.innerHTML = resultsHtml;
        }
        
        // Reset selected index when results change
        selectedResultIndex = -1;
    }
    
    // Highlight search terms in text
    function highlightTerms(text, terms) {
        let highlighted = text;
        terms.forEach(term => {
            const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
            highlighted = highlighted.replace(regex, '<span class="search-highlight">$1</span>');
        });
        return highlighted;
    }
    
    // Escape special regex characters
    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    // Get display name for category
    function getCategoryDisplayName(category) {
        const categoryNames = {
            'physics': 'Physics',
            'technology': 'Technology',
            'ai-ml': 'AI/ML',
            'advanced': 'Advanced Topics',
            'tutorials': 'Tutorials',
            'navigation': 'Navigation'
        };
        return categoryNames[category] || category;
    }
    
    // Find original index from normalized content
    function findOriginalIndex(originalContent, normalizedContent, normalizedIndex) {
        // Simple approximation - could be improved for better accuracy
        const ratio = originalContent.length / normalizedContent.length;
        return Math.floor(normalizedIndex * ratio);
    }
    
    // Check if excerpt is duplicate
    function isDuplicateExcerpt(excerpt, existingExcerpts) {
        const normalizedExcerpt = excerpt.toLowerCase().substring(0, 50);
        for (const existing of existingExcerpts) {
            if (existing.includes(normalizedExcerpt) || normalizedExcerpt.includes(existing.substring(0, 50))) {
                return true;
            }
        }
        return false;
    }
    
    // Debounce function to limit search frequency
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
})();