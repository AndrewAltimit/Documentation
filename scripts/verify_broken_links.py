#!/usr/bin/env python3
import os
import re
from pathlib import Path
import json

def find_all_links_in_file(file_path):
    """Find all links in a markdown file with their line numbers"""
    links = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        # Skip code blocks
        if line.strip().startswith('```'):
            continue
            
        # Find markdown links [text](url)
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.finditer(pattern, line)
        
        for match in matches:
            text = match.group(1)
            url = match.group(2)
            
            # Skip external links and anchors
            if url.startswith(('http://', 'https://', 'mailto:', '#', 'javascript:')):
                continue
            if not url or url == '':
                continue
                
            links.append({
                'text': text,
                'url': url,
                'line': line_num,
                'line_content': line.strip()
            })
    
    return links

def resolve_link_to_file(base_file, link_url):
    """Resolve a link URL to the actual file path it should point to"""
    # Remove anchor if present
    clean_url = link_url.split('#')[0] if '#' in link_url else link_url
    
    if not clean_url:
        return None, "anchor-only"
    
    base_path = Path(base_file).parent
    
    # Determine the expected file path
    if clean_url.startswith('/'):
        # Absolute path
        if clean_url.startswith('/Documentation/'):
            # GitHub Pages URL - remove the prefix
            clean_url = clean_url.replace('/Documentation/', '')
        # Remove leading slash and add to github-pages
        expected = Path('github-pages') / clean_url.lstrip('/')
    else:
        # Relative path
        expected = (base_path / clean_url).resolve()
    
    # Handle .html extension
    if str(expected).endswith('.html'):
        # Convert to .md
        expected = Path(str(expected)[:-5] + '.md')
    elif not str(expected).endswith('.md') and not expected.exists():
        # Might be a directory, check for index.md
        if (expected / 'index.md').exists():
            expected = expected / 'index.md'
        else:
            # Assume it should be a .md file
            expected = expected.with_suffix('.md')
    
    return expected, None

def check_all_links():
    """Check all internal links in the github-pages directory"""
    results = {
        'working_links': {},
        'broken_links': {},
        'external_links': {},
        'summary': {
            'total_files': 0,
            'total_links': 0,
            'working_links': 0,
            'broken_links': 0,
            'external_links': 0
        }
    }
    
    # Get all markdown files
    md_files = list(Path('github-pages').rglob('*.md'))
    results['summary']['total_files'] = len(md_files)
    
    for md_file in md_files:
        file_key = str(md_file)
        links = find_all_links_in_file(md_file)
        
        if not links:
            continue
            
        working = []
        broken = []
        external = []
        
        for link in links:
            results['summary']['total_links'] += 1
            
            # Check if external
            if link['url'].startswith(('http://', 'https://')):
                external.append(link)
                results['summary']['external_links'] += 1
                continue
            
            # Resolve the link
            expected_file, reason = resolve_link_to_file(md_file, link['url'])
            
            if reason == "anchor-only":
                working.append(link)
                results['summary']['working_links'] += 1
            elif expected_file and expected_file.exists():
                link['resolved_to'] = str(expected_file)
                working.append(link)
                results['summary']['working_links'] += 1
            else:
                link['expected_file'] = str(expected_file) if expected_file else "unknown"
                broken.append(link)
                results['summary']['broken_links'] += 1
        
        if working:
            results['working_links'][file_key] = working
        if broken:
            results['broken_links'][file_key] = broken
        if external:
            results['external_links'][file_key] = external
    
    return results

def main():
    print("Verifying all links in GitHub Pages documentation...")
    results = check_all_links()
    
    # Save full results
    with open('link_verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total files checked: {results['summary']['total_files']}")
    print(f"Total links found: {results['summary']['total_links']}")
    print(f"Working links: {results['summary']['working_links']}")
    print(f"Broken links: {results['summary']['broken_links']}")
    print(f"External links: {results['summary']['external_links']}")
    
    if results['broken_links']:
        print("\n=== BROKEN LINKS FOUND ===")
        for file_path, links in results['broken_links'].items():
            print(f"\n{file_path}:")
            for link in links:
                print(f"  Line {link['line']}: [{link['text']}]({link['url']})")
                print(f"    Expected file: {link['expected_file']}")
                print(f"    Context: {link['line_content']}")

if __name__ == '__main__':
    main()