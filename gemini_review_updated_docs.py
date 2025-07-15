#!/usr/bin/env python3
import requests
import json
import os

def consult_gemini_for_review():
    """Consult Gemini AI to review the updated documentation"""
    
    # MCP server endpoint
    url = "http://localhost:8000/tools/execute"
    
    # Prepare the review request
    question = """Please conduct another thorough review of the updated Andrew's Notebook GitHub Pages documentation. 

I've implemented your previous recommendations:

1. ADDED MISSING TOPICS:
   - Physics: Thermodynamics, Statistical Mechanics, Condensed Matter Physics
   - Technology: Cybersecurity, Networking, Database Design

2. IMPLEMENTED SEARCH FUNCTIONALITY:
   - Created search.html and search.js for client-side search
   - Full-text search with relevance scoring and highlighting

3. UPDATED AI SECTION:
   - Added comprehensive Diffusion Models section
   - Added extensive AI Ethics section covering principles, challenges, frameworks, and best practices

4. ENHANCED CROSS-REFERENCES:
   - Added "See Also" sections to Quantum Mechanics, Classical Mechanics, Kubernetes, Quantum Computing
   - Created interconnections between related topics

5. INCREASED MATHEMATICAL RIGOR:
   - Enhanced Relativity with detailed Lorentz transformation derivations and Einstein Field Equations
   - Expanded QFT with comprehensive renormalization treatment and path integral formulation

Please assess:
- How well these implementations address your original recommendations
- The quality and depth of the new content
- Any remaining gaps or areas for improvement
- Overall documentation quality compared to before
- Specific strengths and weaknesses of the new sections

Please be thorough and specific in your review."""

    context = """This is a personal knowledge base website hosted on GitHub Pages. The documentation covers both physics topics (classical mechanics, quantum mechanics, relativity, etc.) and technology topics (AWS, Docker, Kubernetes, AI, etc.). The target audience includes students, professionals, and anyone interested in these topics. The documentation should be comprehensive, accurate, and accessible."""
    
    # Prepare request payload
    payload = {
        "tool": "consult_gemini",
        "arguments": {
            "question": question,
            "context": context
        }
    }
    
    print("Consulting Gemini for comprehensive documentation review...")
    print("This may take a few moments as Gemini analyzes the changes...")
    print("=" * 80)
    
    try:
        # Make request with longer timeout for thorough review
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                # Handle different response formats
                if isinstance(result.get("result"), dict):
                    gemini_response = result["result"].get("response", str(result["result"]))
                else:
                    gemini_response = str(result.get("result", "No response received"))
                
                # Print to console
                print("\nGemini's Review of Updated Documentation:")
                print("=" * 80)
                print(gemini_response)
                
                # Save to file
                output_file = "GEMINI_UPDATED_DOCS_REVIEW.md"
                with open(output_file, 'w') as f:
                    f.write("# Gemini's Review of Updated Documentation\n\n")
                    f.write(f"Date: {result['result'].get('timestamp', 'N/A')}\n\n")
                    f.write("## Review Summary\n\n")
                    f.write(gemini_response)
                
                print(f"\n\nReview saved to: {output_file}")
                return gemini_response
            else:
                print(f"Error from MCP server: {result.get('error')}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Request timed out after 5 minutes - Gemini may need more time for thorough review")
        return None
    except Exception as e:
        print(f"Error consulting Gemini: {e}")
        return None

if __name__ == "__main__":
    # Change to documentation directory
    os.chdir("/home/miku/Documents/repos/Documentation")
    
    # Run the consultation
    review = consult_gemini_for_review()
    
    if not review:
        print("\nFailed to get review from Gemini. Please check that the MCP server is running.")