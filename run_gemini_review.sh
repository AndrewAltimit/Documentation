#!/bin/bash

echo "Starting Gemini review of updated documentation..."
echo "====================================================="

# Change to the documentation directory
cd /home/miku/Documents/repos/Documentation

# Create the prompt with file context
cat << 'EOF' | gemini -p "$(cat -)" > gemini_final_review.md 2>&1
I need you to conduct a thorough follow-up review of the Andrew's Notebook GitHub Pages documentation after implementing your previous recommendations.

Please examine the github-pages directory, particularly focusing on:

NEW PHYSICS CONTENT:
- github-pages/docs/physics/thermodynamics.md
- github-pages/docs/physics/statistical-mechanics.md  
- github-pages/docs/physics/condensed-matter.md

NEW TECHNOLOGY CONTENT:
- github-pages/docs/technology/cybersecurity.md
- github-pages/docs/technology/networking.md
- github-pages/docs/technology/database-design.md

SEARCH FUNCTIONALITY:
- github-pages/search.html
- github-pages/search.js

AI SECTION UPDATES in github-pages/docs/technology/ai.md:
- New Diffusion Models section
- New AI Ethics section

ENHANCED MATHEMATICAL RIGOR in:
- github-pages/docs/physics/relativity.md
- github-pages/docs/physics/quantum-field-theory.md

Please provide:
1. Assessment of implementation quality
2. Remaining gaps or issues
3. Overall improvement compared to before
4. Specific feedback on each new section
5. Priority recommendations for next steps

Be thorough and specific in your review.
EOF

echo "Review complete. Output saved to gemini_final_review.md"