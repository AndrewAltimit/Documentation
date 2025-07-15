#!/bin/bash

echo "Running focused Gemini review..."

# Navigate to the github-pages directory
cd /home/miku/Documents/repos/Documentation/github-pages

# Create a focused review request
gemini -p "I've implemented your recommendations for Andrew's Notebook documentation. Please review these specific improvements:

NEW PHYSICS SECTIONS:
- docs/physics/thermodynamics.md - Laws of thermodynamics, processes, state functions
- docs/physics/statistical-mechanics.md - Microscopic to macroscopic connections  
- docs/physics/condensed-matter.md - Crystal structures, superconductivity, topological phases

NEW TECHNOLOGY SECTIONS:
- docs/technology/cybersecurity.md - Security principles, cryptography, best practices
- docs/technology/networking.md - OSI model, TCP/IP, routing, troubleshooting
- docs/technology/database-design.md - Relational/NoSQL, normalization, optimization

SEARCH FEATURE:
- search.html and search.js - Client-side full-text search with highlighting

AI ENHANCEMENTS in docs/technology/ai.md:
- Comprehensive Diffusion Models section
- Extensive AI Ethics coverage

MATHEMATICAL RIGOR:
- docs/physics/relativity.md - Enhanced Lorentz transformations, Einstein equations
- docs/physics/quantum-field-theory.md - Detailed renormalization, path integrals

Please assess:
1. Quality and accuracy of new content
2. Completeness and remaining gaps
3. Overall improvement vs previous version
4. Specific strengths and weaknesses
5. Priority next steps

Focus on the github-pages directory content." 2>&1 | tee ../gemini_focused_review_output.md

echo "Review complete!"