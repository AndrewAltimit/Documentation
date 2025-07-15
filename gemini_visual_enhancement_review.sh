#!/bin/bash

echo "Running Gemini visual enhancement review..."

# Navigate to the github-pages directory
cd /home/miku/Documents/repos/Documentation/github-pages

# Create a focused review request
gemini -p "I need your help updating Andrew's Notebook documentation to match the visual richness and thoroughness of docs/technology/ai-lecture-2023.md. This file serves as our gold standard with:

VISUAL ELEMENTS IN AI-LECTURE-2023.MD:
- Embedded images with clickable links (neural networks, transformer architecture)
- Animated GIFs (self-attention.gif)
- Custom icons for reference types (file-text-fill.svg, file-pdf-fill.svg, play-btn-fill.svg, git.svg)
- Centered images with captions using referenceBoxes
- Floating images with text wrapping
- Visual hierarchy with clear sections

CONTENT FEATURES:
- Comprehensive table of contents with anchor links
- Reference boxes with different styling (type2, type3)
- In-depth explanations with theory AND practical examples
- Code snippets and real-world applications
- Quote boxes for important information
- External links to papers, articles, videos, GitHub repos

CURRENT DOCUMENTATION THAT NEEDS ENHANCEMENT:
Physics: classical-mechanics.md, thermodynamics.md, statistical-mechanics.md, relativity.md, quantum-mechanics.md, quantum-field-theory.md, condensed-matter.md, string-theory.md, quantum-computing.md
Technology: terraform.md, docker.md, aws.md, kubernetes.md, git.md, ai.md, cybersecurity.md, networking.md, database-design.md

Please provide SPECIFIC recommendations for:
1. What diagrams/visualizations to add to each physics doc (e.g., Feynman diagrams, phase diagrams, crystal lattices)
2. What architecture diagrams to add to each technology doc (e.g., network topologies, security models, container orchestration)
3. Which concepts need animated visualizations (like the self-attention.gif)
4. What reference papers/articles/videos to include with proper styling
5. Where to add code examples and practical demonstrations
6. How to enhance mathematical content with visual representations

Focus on making each document as visually rich and comprehensive as ai-lecture-2023.md. Be specific about what images/diagrams would best illustrate each concept." 2>&1 | tee ../gemini_visual_enhancement_output.md

echo "Review complete!"