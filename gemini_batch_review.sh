#!/bin/bash

echo "Running Gemini review in batch mode..."

# Create a comprehensive prompt file
cat > /tmp/gemini_doc_review.txt << 'EOF'
Review the updated Andrew's Notebook documentation in the github-pages directory.

FOCUS AREAS:
1. New physics docs: thermodynamics.md, statistical-mechanics.md, condensed-matter.md
2. New tech docs: cybersecurity.md, networking.md, database-design.md  
3. Search functionality: search.html and search.js
4. AI updates: Diffusion Models and Ethics sections in ai.md
5. Enhanced math in relativity.md and quantum-field-theory.md

Assess quality, completeness, and improvements. Provide specific feedback.
EOF

# Run Gemini with the prompt
cd /home/miku/Documents/repos/Documentation
cat /tmp/gemini_doc_review.txt | gemini -p "$(cat /tmp/gemini_doc_review.txt)" --all-files 2>&1 | head -2000 > gemini_batch_review_output.md

echo "Review saved to gemini_batch_review_output.md"