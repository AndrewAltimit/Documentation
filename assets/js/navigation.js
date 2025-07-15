// Active navigation highlighting
document.addEventListener('DOMContentLoaded', function() {
  // Get current page URL
  const currentPath = window.location.pathname;
  
  // Find all navigation links
  const navLinks = document.querySelectorAll('.nav__list a');
  
  // Add active class to current page link
  navLinks.forEach(link => {
    const linkPath = new URL(link.href).pathname;
    if (linkPath === currentPath) {
      link.classList.add('active');
      
      // Also expand parent sections if nested
      let parent = link.closest('.nav__items');
      while (parent) {
        const parentToggle = parent.previousElementSibling;
        if (parentToggle && parentToggle.classList.contains('nav__toggle')) {
          parentToggle.checked = true;
        }
        parent = parent.parentElement.closest('.nav__items');
      }
    }
  });
  
  // Smooth scroll for TOC links
  const tocLinks = document.querySelectorAll('.toc__menu a');
  tocLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href').substring(1);
      const targetElement = document.getElementById(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
});