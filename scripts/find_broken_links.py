#!/usr/bin/env python3
import os
import re
from pathlib import Path
import json

def find_markdown_links(content):
    """Extract all markdown links from content"""
    # Match [text](url) pattern
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.findall(pattern, content)

def is_internal_link(url):
    """Check if a link is internal (relative or local)"""
    if url.startswith(('http://', 'https://', 'mailto:', '#')):
        return False
    return True

def normalize_path(base_path, link_path):
    """Normalize a relative path to absolute"""
    if link_path.startswith('/'):
        # Absolute path from site root
        return Path('github-pages') / link_path.lstrip('/')
    else:
        # Relative path
        return (base_path.parent / link_path).resolve()

def check_link_exists(base_file, link):
    """Check if a linked file exists"""
    # Remove anchor if present
    if '#' in link:
        link = link.split('#')[0]
    
    if not link:  # Pure anchor link
        return True
        
    base_path = Path(base_file)
    
    # Handle different link formats
    if link.startswith('/Documentation/'):
        # GitHub Pages URL format
        link = link.replace('/Documentation/', '')
        target = Path('github-pages') / link
    else:
        target = normalize_path(base_path, link)
    
    # Check with and without .md extension
    if target.exists():
        return True
    if target.with_suffix('.md').exists():
        return True
    if (target / 'index.md').exists():
        return True
        
    return False

def analyze_broken_links():
    """Find all broken internal links in markdown files"""
    broken_links = {}
    all_links = {}
    
    # Focus on github-pages directory
    md_files = list(Path('github-pages').rglob('*.md'))
    
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        links = find_markdown_links(content)
        file_broken_links = []
        file_all_links = []
        
        for link_text, url in links:
            if is_internal_link(url):
                file_all_links.append({
                    'text': link_text,
                    'url': url,
                    'exists': check_link_exists(md_file, url)
                })
                
                if not check_link_exists(md_file, url):
                    file_broken_links.append({
                        'text': link_text,
                        'url': url
                    })
        
        if file_broken_links:
            broken_links[str(md_file)] = file_broken_links
        if file_all_links:
            all_links[str(md_file)] = file_all_links
    
    return broken_links, all_links

if __name__ == '__main__':
    print("Analyzing markdown files for broken links...")
    broken_links, all_links = analyze_broken_links()
    
    # Save results
    with open('broken_links_analysis.json', 'w') as f:
        json.dump({
            'broken_links': broken_links,
            'summary': {
                'total_files_with_broken_links': len(broken_links),
                'total_broken_links': sum(len(links) for links in broken_links.values())
            }
        }, f, indent=2)
    
    print(f"\nFound {len(broken_links)} files with broken links")
    print(f"Total broken links: {sum(len(links) for links in broken_links.values())}")
    
    # Print summary
    for file, links in broken_links.items():
        print(f"\n{file}:")
        for link in links:
            print(f"  - [{link['text']}]({link['url']})")