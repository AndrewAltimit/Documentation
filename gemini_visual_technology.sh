#!/bin/bash

echo "Getting Gemini recommendations for technology documents..."

cd /home/miku/Documents/repos/Documentation/github-pages

gemini -p "Following the same format, provide specific visual enhancement recommendations for these technology documents:

1. terraform.md
2. docker.md
3. aws.md
4. kubernetes.md
5. git.md

For each, specify:
- Architecture diagrams needed
- Workflow visualizations
- Key references and documentation
- Code examples with outputs
- Interactive elements

Focus on visual elements that would help readers understand complex concepts." 2>&1 | tee ../gemini_technology_visual_1.md

echo "Technology recommendations part 1 complete!"