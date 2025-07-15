// Search functionality for Andrew's Notebook
(function() {
    'use strict';
    
    // List of all documentation pages to index
    const PAGES = [
        // Physics pages
        { url: 'docs/physics/classical-mechanics.html', title: 'Classical Mechanics', category: 'Physics' },
        { url: 'docs/physics/thermodynamics.html', title: 'Thermodynamics', category: 'Physics' },
        { url: 'docs/physics/statistical-mechanics.html', title: 'Statistical Mechanics', category: 'Physics' },
        { url: 'docs/physics/relativity.html', title: 'Relativity', category: 'Physics' },
        { url: 'docs/physics/quantum-mechanics.html', title: 'Quantum Mechanics', category: 'Physics' },
        { url: 'docs/physics/condensed-matter.html', title: 'Condensed Matter Physics', category: 'Physics' },
        { url: 'docs/physics/quantum-field-theory.html', title: 'Quantum Field Theory', category: 'Physics' },
        { url: 'docs/physics/string-theory.html', title: 'String Theory', category: 'Physics' },
        
        // Technology pages
        { url: 'docs/technology/terraform.html', title: 'Terraform', category: 'Technology' },
        { url: 'docs/technology/docker.html', title: 'Docker', category: 'Technology' },
        { url: 'docs/technology/aws.html', title: 'AWS', category: 'Technology' },
        { url: 'docs/technology/kubernetes.html', title: 'Kubernetes', category: 'Technology' },
        { url: 'docs/technology/database-design.html', title: 'Database Design', category: 'Technology' },
        { url: 'docs/technology/networking.html', title: 'Networking', category: 'Technology' },
        { url: 'docs/technology/cybersecurity.html', title: 'Cybersecurity', category: 'Technology' },
        { url: 'docs/technology/git.html', title: 'Git Version Control', category: 'Technology' },
        { url: 'docs/technology/branching.html', title: 'Branching Strategies', category: 'Technology' },
        { url: 'docs/technology/unreal.html', title: 'Unreal Engine', category: 'Technology' },
        { url: 'docs/technology/quantumcomputing.html', title: 'Quantum Computing', category: 'Technology' },
        { url: 'docs/technology/ai.html', title: 'Artificial Intelligence', category: 'Technology' },
        { url: 'docs/technology/ai-lecture-2023.html', title: 'AI Deep Dive', category: 'Technology' },
        { url: 'docs/technology/please-build.html', title: 'Please Build', category: 'Technology' }
    ];
    
    let searchIndex = [];
    let indexedPages = 0;
    
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
        });
    });
    
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
                terms.forEach(term => {
                    const regex = new RegExp(term, 'gi');
                    const matches = page.content.match(regex);
                    if (matches) {
                        score += matches.length;
                        
                        // Find excerpts around matches
                        const index = page.content.indexOf(term);
                        if (index !== -1) {
                            const start = Math.max(0, index - 100);
                            const end = Math.min(page.content.length, index + 100);
                            const excerpt = page.originalContent.substring(start, end);
                            contentMatches.push({ term, excerpt });
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
            searchResults.innerHTML = results.map(result => `
                <div class="search-result">
                    <div class="search-category">${result.category}</div>
                    <h3><a href="${result.url}">${highlightTerms(result.title, terms)}</a></h3>
                    ${result.excerpts.length > 0 ? 
                        '<div class="search-excerpt">...' + result.excerpts.join('...') + '...</div>' : 
                        ''}
                </div>
            `).join('');
        }
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