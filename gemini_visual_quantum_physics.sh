#!/bin/bash

echo "Getting Gemini recommendations for quantum physics documents..."

cd /home/miku/Documents/repos/Documentation/github-pages

gemini -p "Following the same format as before, provide specific visual enhancement recommendations for these quantum physics documents:

1. quantum-mechanics.md
2. quantum-field-theory.md
3. string-theory.md
4. quantum-computing.md

For each, specify:
- Diagrams needed (wave functions, Feynman diagrams, string vibrations, quantum circuits)
- Animated visualizations (particle-wave duality, quantum tunneling, superposition)
- Key papers and references
- Code examples (quantum simulations)
- Mathematical visualizations

Keep responses focused and specific." 2>&1 | tee ../gemini_quantum_physics_visual.md

echo "Quantum physics recommendations complete!"