/**
 * Progress Tracker for Documentation
 * Tracks user progress through documentation pages using localStorage
 */

(function() {
  'use strict';

  // Constants
  const STORAGE_KEY = 'doc_progress';
  const VISITED_CLASS = 'visited';
  const PROGRESS_BAR_ID = 'progress-indicator';

  // Progress Tracker Object
  const ProgressTracker = {
    // Get all visited pages
    getVisited: function() {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : {};
    },

    // Mark page as visited
    markVisited: function(path) {
      const visited = this.getVisited();
      visited[path] = {
        timestamp: new Date().toISOString(),
        count: (visited[path]?.count || 0) + 1
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(visited));
      this.updateUI();
    },

    // Get progress statistics
    getStats: function() {
      const visited = this.getVisited();
      const pages = Object.keys(visited);
      
      // Count by difficulty
      const byDifficulty = {
        beginner: 0,
        intermediate: 0,
        advanced: 0
      };

      pages.forEach(page => {
        if (page.includes('crash-course') || page.includes('5-minutes')) {
          byDifficulty.beginner++;
        } else if (page.includes('advanced') || page.includes('internals')) {
          byDifficulty.advanced++;
        } else {
          byDifficulty.intermediate++;
        }
      });

      return {
        total: pages.length,
        byDifficulty: byDifficulty,
        lastVisited: pages.sort((a, b) => 
          new Date(visited[b].timestamp) - new Date(visited[a].timestamp)
        ).slice(0, 5)
      };
    },

    // Update UI elements
    updateUI: function() {
      const visited = this.getVisited();
      
      // Mark visited links
      document.querySelectorAll('a[href*="/docs/"]').forEach(link => {
        const path = new URL(link.href).pathname;
        if (visited[path]) {
          link.classList.add(VISITED_CLASS);
          
          // Add visit count badge
          const count = visited[path].count;
          if (count > 1 && !link.querySelector('.visit-count')) {
            const badge = document.createElement('span');
            badge.className = 'visit-count';
            badge.textContent = count;
            badge.title = `Visited ${count} times`;
            link.appendChild(badge);
          }
        }
      });

      // Update progress bar if exists
      this.updateProgressBar();
    },

    // Create and update progress bar
    updateProgressBar: function() {
      let progressBar = document.getElementById(PROGRESS_BAR_ID);
      
      if (!progressBar) {
        // Create progress bar if it doesn't exist
        progressBar = this.createProgressBar();
        if (!progressBar) return;
      }

      const stats = this.getStats();
      const totalTopics = 50; // Approximate total number of documentation pages
      const percentage = Math.min(100, (stats.total / totalTopics) * 100);

      progressBar.querySelector('.progress-fill').style.width = `${percentage}%`;
      progressBar.querySelector('.progress-text').textContent = 
        `${stats.total} pages explored (${Math.round(percentage)}% complete)`;
      
      // Update difficulty breakdown
      const breakdown = progressBar.querySelector('.difficulty-breakdown');
      if (breakdown) {
        breakdown.innerHTML = `
          <span class="diff-beginner">🌱 ${stats.byDifficulty.beginner}</span>
          <span class="diff-intermediate">🌿 ${stats.byDifficulty.intermediate}</span>
          <span class="diff-advanced">🌳 ${stats.byDifficulty.advanced}</span>
        `;
      }
    },

    // Create progress bar element
    createProgressBar: function() {
      const targetElement = document.querySelector('.page__inner-wrap') || 
                           document.querySelector('main');
      
      if (!targetElement) return null;

      const progressHTML = `
        <div id="${PROGRESS_BAR_ID}" class="progress-tracker">
          <div class="progress-header">
            <h3>Your Learning Progress</h3>
            <button class="toggle-progress" title="Toggle progress details">
              <span class="icon">📊</span>
            </button>
          </div>
          <div class="progress-bar">
            <div class="progress-fill"></div>
          </div>
          <div class="progress-text"></div>
          <div class="difficulty-breakdown"></div>
          <div class="progress-details" style="display: none;">
            <h4>Recently Visited</h4>
            <ul class="recent-pages"></ul>
            <button class="clear-progress">Clear Progress</button>
          </div>
        </div>
      `;

      targetElement.insertAdjacentHTML('afterbegin', progressHTML);
      
      const progressBar = document.getElementById(PROGRESS_BAR_ID);
      
      // Add event listeners
      progressBar.querySelector('.toggle-progress').addEventListener('click', () => {
        const details = progressBar.querySelector('.progress-details');
        details.style.display = details.style.display === 'none' ? 'block' : 'none';
        this.updateRecentPages();
      });

      progressBar.querySelector('.clear-progress').addEventListener('click', () => {
        if (confirm('Clear all progress tracking? This cannot be undone.')) {
          localStorage.removeItem(STORAGE_KEY);
          location.reload();
        }
      });

      return progressBar;
    },

    // Update recent pages list
    updateRecentPages: function() {
      const stats = this.getStats();
      const recentList = document.querySelector('.recent-pages');
      
      if (!recentList) return;

      recentList.innerHTML = stats.lastVisited.map(path => {
        const title = this.getPageTitle(path);
        return `<li><a href="${path}">${title}</a></li>`;
      }).join('');
    },

    // Get page title from path
    getPageTitle: function(path) {
      // Extract title from path
      const parts = path.split('/');
      const lastPart = parts[parts.length - 1] || parts[parts.length - 2];
      
      return lastPart
        .replace(/-/g, ' ')
        .replace(/\.html?$/, '')
        .replace(/\b\w/g, l => l.toUpperCase());
    },

    // Initialize tracker
    init: function() {
      // Mark current page as visited
      const currentPath = window.location.pathname;
      if (currentPath.includes('/docs/')) {
        this.markVisited(currentPath);
      }

      // Update UI on page load
      this.updateUI();

      // Listen for navigation (for SPAs)
      window.addEventListener('popstate', () => this.updateUI());
    }
  };

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ProgressTracker.init());
  } else {
    ProgressTracker.init();
  }

  // Expose to global scope for debugging
  window.ProgressTracker = ProgressTracker;
})();

// Styles for progress tracker
const style = document.createElement('style');
style.textContent = `
  .progress-tracker {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 2rem;
  }

  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .progress-header h3 {
    margin: 0;
    font-size: 1.1rem;
  }

  .toggle-progress {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1.2rem;
    padding: 0.25rem;
  }

  .progress-bar {
    width: 100%;
    height: 20px;
    background: #e9ecef;
    border-radius: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 0.9rem;
    color: #6c757d;
    margin-bottom: 0.5rem;
  }

  .difficulty-breakdown {
    display: flex;
    gap: 1rem;
    font-size: 0.9rem;
  }

  .difficulty-breakdown span {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .progress-details {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #dee2e6;
  }

  .recent-pages {
    list-style: none;
    padding: 0;
    margin: 0.5rem 0 1rem 0;
  }

  .recent-pages li {
    margin: 0.25rem 0;
  }

  .clear-progress {
    background: #dc3545;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .clear-progress:hover {
    background: #c82333;
  }

  a.visited {
    position: relative;
  }

  a.visited::after {
    content: '✓';
    position: absolute;
    right: -1.2em;
    color: #28a745;
    font-size: 0.8em;
  }

  .visit-count {
    display: inline-block;
    background: #6c757d;
    color: white;
    font-size: 0.7em;
    padding: 0.1rem 0.3rem;
    border-radius: 10px;
    margin-left: 0.5rem;
    vertical-align: super;
  }
`;
document.head.appendChild(style);