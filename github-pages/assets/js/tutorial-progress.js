// Tutorial Progress Tracking System

(function() {
  'use strict';

  // Configuration
  const STORAGE_KEY_PREFIX = 'tutorial_progress_';
  const SCROLL_OFFSET = 100; // Pixels from top to consider section "read"
  const UPDATE_DELAY = 500; // Debounce delay for scroll events

  // State
  let currentProgress = 0;
  let sections = [];
  let completedSections = new Set();
  let scrollTimeout;

  // Initialize on page load
  document.addEventListener('DOMContentLoaded', function() {
    if (!document.querySelector('[data-track-progress="true"]')) {
      return; // Not a tutorial page
    }

    initializeSections();
    loadProgress();
    setupEventListeners();
    updateProgressDisplay();
  });

  // Initialize sections for tracking
  function initializeSections() {
    const content = document.querySelector('.tutorial-sections');
    if (!content) return;

    // Track all h2 elements as major sections
    const headings = content.querySelectorAll('h2');
    sections = Array.from(headings).map((heading, index) => {
      const id = heading.id || `section-${index}`;
      heading.id = id;
      
      return {
        id: id,
        element: heading,
        offset: heading.offsetTop,
        completed: false
      };
    });
  }

  // Load saved progress from localStorage
  function loadProgress() {
    const pageId = getPageId();
    const savedData = localStorage.getItem(STORAGE_KEY_PREFIX + pageId);
    
    if (savedData) {
      try {
        const data = JSON.parse(savedData);
        completedSections = new Set(data.completedSections || []);
        currentProgress = data.progress || 0;
        
        // Mark completed sections visually
        completedSections.forEach(sectionId => {
          const section = sections.find(s => s.id === sectionId);
          if (section) {
            section.element.classList.add('completed');
            section.completed = true;
          }
        });
      } catch (e) {
        console.error('Failed to load progress:', e);
      }
    }
  }

  // Save progress to localStorage
  function saveProgress() {
    const pageId = getPageId();
    const data = {
      completedSections: Array.from(completedSections),
      progress: currentProgress,
      lastUpdated: new Date().toISOString()
    };
    
    localStorage.setItem(STORAGE_KEY_PREFIX + pageId, JSON.stringify(data));
  }

  // Setup event listeners
  function setupEventListeners() {
    // Track scroll progress
    window.addEventListener('scroll', debounce(trackScrollProgress, UPDATE_DELAY));
    
    // Track when user reaches end of tutorial
    const observer = new IntersectionObserver(handleIntersection, {
      threshold: 0.9
    });
    
    const footer = document.querySelector('.tutorial-footer');
    if (footer) {
      observer.observe(footer);
    }
    
    // Track manual section completion (if checkboxes exist)
    document.querySelectorAll('.section-complete-checkbox').forEach(checkbox => {
      checkbox.addEventListener('change', handleManualCompletion);
    });
  }

  // Track scroll progress through sections
  function trackScrollProgress() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    
    // Update overall progress
    currentProgress = Math.min(
      Math.round((scrollTop / (documentHeight - windowHeight)) * 100),
      100
    );
    
    // Check which sections have been viewed
    sections.forEach(section => {
      if (!section.completed && scrollTop > section.offset - SCROLL_OFFSET) {
        markSectionComplete(section);
      }
    });
    
    updateProgressDisplay();
  }

  // Mark a section as complete
  function markSectionComplete(section) {
    section.completed = true;
    section.element.classList.add('completed');
    completedSections.add(section.id);
    
    // Animate completion
    section.element.classList.add('just-completed');
    setTimeout(() => {
      section.element.classList.remove('just-completed');
    }, 1000);
    
    saveProgress();
  }

  // Update progress display
  function updateProgressDisplay() {
    // Update progress bar
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    if (progressFill) {
      progressFill.style.width = currentProgress + '%';
    }
    
    if (progressText) {
      progressText.textContent = currentProgress + '% Complete';
    }
    
    // Update section counter
    const completedCount = completedSections.size;
    const totalCount = sections.length;
    
    if (totalCount > 0) {
      const sectionProgress = document.querySelector('.section-progress');
      if (sectionProgress) {
        sectionProgress.textContent = `${completedCount} of ${totalCount} sections completed`;
      }
    }
    
    // Show completion message if 100%
    if (currentProgress === 100) {
      showCompletionMessage();
    }
  }

  // Show completion message
  function showCompletionMessage() {
    const message = document.querySelector('.completion-message');
    if (message && message.style.display === 'none') {
      message.style.display = 'block';
      message.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
      // Mark tutorial as completed in global progress
      markTutorialComplete();
    }
  }

  // Mark tutorial as completed in global progress tracking
  function markTutorialComplete() {
    const pageId = getPageId();
    const completedTutorials = JSON.parse(
      localStorage.getItem('completed_tutorials') || '[]'
    );
    
    if (!completedTutorials.includes(pageId)) {
      completedTutorials.push(pageId);
      localStorage.setItem('completed_tutorials', JSON.stringify(completedTutorials));
      
      // Update any prerequisite checks on other pages
      updatePrerequisiteStatus(pageId);
    }
  }

  // Update prerequisite status
  function updatePrerequisiteStatus(tutorialId) {
    const prerequisites = JSON.parse(
      localStorage.getItem('completedPrerequisites') || '{}'
    );
    prerequisites[tutorialId] = true;
    localStorage.setItem('completedPrerequisites', JSON.stringify(prerequisites));
  }

  // Handle intersection observer callbacks
  function handleIntersection(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting && currentProgress < 100) {
        currentProgress = 100;
        updateProgressDisplay();
        saveProgress();
      }
    });
  }

  // Handle manual section completion
  function handleManualCompletion(event) {
    const sectionId = event.target.dataset.sectionId;
    const section = sections.find(s => s.id === sectionId);
    
    if (section) {
      if (event.target.checked) {
        markSectionComplete(section);
      } else {
        section.completed = false;
        section.element.classList.remove('completed');
        completedSections.delete(section.id);
        saveProgress();
      }
    }
  }

  // Utility: Get page identifier
  function getPageId() {
    // Try to get from page front matter first
    const pageIdMeta = document.querySelector('meta[name="page-id"]');
    if (pageIdMeta) {
      return pageIdMeta.content;
    }
    
    // Fallback to URL path
    return window.location.pathname.replace(/\//g, '_');
  }

  // Utility: Debounce function
  function debounce(func, wait) {
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(scrollTimeout);
        func(...args);
      };
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(later, wait);
    };
  }

  // Add completion animations
  const style = document.createElement('style');
  style.textContent = `
    .tutorial-sections h2.just-completed::before {
      animation: completeSection 0.5s ease;
    }
    
    @keyframes completeSection {
      0% {
        transform: translateY(-50%) scale(1);
      }
      50% {
        transform: translateY(-50%) scale(1.2);
      }
      100% {
        transform: translateY(-50%) scale(1);
      }
    }
    
    .completion-message {
      animation: fadeInScale 0.5s ease;
    }
    
    @keyframes fadeInScale {
      from {
        opacity: 0;
        transform: scale(0.9);
      }
      to {
        opacity: 1;
        transform: scale(1);
      }
    }
  `;
  document.head.appendChild(style);

  // Export for use in other scripts
  window.TutorialProgress = {
    getProgress: () => currentProgress,
    getCompletedSections: () => Array.from(completedSections),
    resetProgress: () => {
      completedSections.clear();
      currentProgress = 0;
      sections.forEach(s => {
        s.completed = false;
        s.element.classList.remove('completed');
      });
      saveProgress();
      updateProgressDisplay();
    }
  };

})();