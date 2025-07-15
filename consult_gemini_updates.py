#!/usr/bin/env python3
import subprocess
import json
import time

# Create the consultation request
consultation_request = """
Gemini, I've implemented your recommendations for improving Andrew's Notebook GitHub Pages documentation. Please conduct another thorough review of the updated documentation to assess:

1. How well the implementations address your original recommendations
2. The quality of the new content added
3. Any remaining gaps or areas for improvement
4. Overall assessment of the documentation's current state

Here's what I've implemented based on your review:

## Added Missing Topics:
- Physics: Thermodynamics, Statistical Mechanics, Condensed Matter Physics
- Technology: Cybersecurity, Networking, Database Design

## Implemented Search Functionality:
- Created search.html and search.js for client-side search
- Indexes all documentation with relevance scoring

## Updated AI Section:
- Added comprehensive Diffusion Models section
- Added extensive AI Ethics section

## Enhanced Cross-References:
- Added "See Also" sections to multiple documents
- Created interconnections between related topics

## Increased Mathematical Rigor:
- Enhanced Relativity with Lorentz transformation derivations
- Expanded QFT with detailed renormalization and path integrals

Please review the github-pages directory thoroughly and provide your assessment.
"""

# Save the request to a file
with open('/tmp/gemini_request.txt', 'w') as f:
    f.write(consultation_request)

# Use the Gemini CLI script
gemini_command = [
    "bash", 
    "/home/miku/Documents/repos/Documentation/tools/mcp/gemini_cli.sh",
    "/tmp/gemini_request.txt"
]

print("Consulting Gemini for documentation review...")
print("This may take a few moments...")

try:
    result = subprocess.run(gemini_command, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        print("\nGemini's Response:")
        print("=" * 80)
        print(result.stdout)
    else:
        print(f"Error running Gemini consultation: {result.stderr}")
        
except subprocess.TimeoutExpired:
    print("Gemini consultation timed out after 5 minutes")
except Exception as e:
    print(f"Error: {e}")