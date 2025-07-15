#!/bin/bash

echo "Getting Gemini recommendations for remaining technology documents..."

cd /home/miku/Documents/repos/Documentation/github-pages

gemini -p "Following the same format, provide specific visual enhancement recommendations for these remaining technology documents:

1. ai.md
2. cybersecurity.md
3. networking.md
4. database-design.md

For each, specify:
- Architecture diagrams needed
- Workflow/process visualizations
- Key references and documentation
- Code examples with outputs
- Interactive elements

Keep the same level of detail as previous recommendations." 2>&1 | tee ../gemini_technology_visual_2.md

echo "Technology recommendations part 2 complete!"