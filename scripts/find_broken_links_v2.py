#!/usr/bin/env python3
import os
import re
from pathlib import Path
import json

def find_markdown_links(content):
    """Extract all markdown links from content, excluding math expressions"""
    # First, remove code blocks and inline code to avoid false positives
    # Remove fenced code blocks
    content = re.sub(r'```[\s\S]*?```', '', content)
    # Remove inline code
    content = re.sub(r'`[^`]+`', '', content)
    
    # Remove LaTeX math blocks
    content = re.sub(r'\$\$[\s\S]*?\$\$', '', content)
    # Remove inline LaTeX
    content = re.sub(r'\$[^\$]+\$', '', content)
    
    # Match [text](url) pattern
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(pattern, content)
    
    # Filter out likely mathematical expressions
    filtered_matches = []
    for text, url in matches:
        # Skip if URL is very short (likely part of math expression)
        if len(url) <= 2 and not url.startswith('/'):
            continue
        # Skip if text contains LaTeX commands
        if '\\' in text or '^' in text or '_' in text:
            continue
        # Skip if URL contains math symbols
        if any(char in url for char in ['\\', '^', '_', '{', '}']):
            continue
        filtered_matches.append((text, url))
    
    return filtered_matches

def is_internal_link(url):
    """Check if a link is internal (relative or local)"""
    if url.startswith(('http://', 'https://', 'mailto:', '#', 'javascript:')):
        return False
    # Skip anchors and queries
    if url == '' or url.startswith('?'):
        return False
    return True

def normalize_and_check_path(base_file, link):
    """Normalize a relative path and check if target exists"""
    # Remove anchor if present
    clean_link = link.split('#')[0] if '#' in link else link
    
    if not clean_link:  # Pure anchor link
        return True
        
    base_path = Path(base_file)
    
    # Handle different link formats
    if clean_link.startswith('/Documentation/'):
        # GitHub Pages URL format - strip the /Documentation/ prefix
        clean_link = clean_link.replace('/Documentation/', '')
        if clean_link.startswith('docs/'):
            target = Path('github-pages') / clean_link
        else:
            target = Path('github-pages/docs') / clean_link
    elif clean_link.startswith('/'):
        # Absolute path from site root
        target = Path('github-pages') / clean_link.lstrip('/')
    else:
        # Relative path
        target = (base_path.parent / clean_link).resolve()
    
    # Check various possibilities
    checks = [
        target,
        target.with_suffix('.md'),
        target / 'index.md',
        target.with_suffix('.html'),
        target / 'index.html'
    ]
    
    for check_path in checks:
        if check_path.exists():
            return True
    
    # Also check if it might be in docs subdirectory
    if not str(target).startswith('github-pages/docs/'):
        docs_target = Path('github-pages/docs') / target.name
        if docs_target.exists() or docs_target.with_suffix('.md').exists():
            return True
    
    return False

def get_expected_path(base_file, link):
    """Get the expected path for a broken link"""
    clean_link = link.split('#')[0] if '#' in link else link
    
    if clean_link.startswith('/Documentation/'):
        clean_link = clean_link.replace('/Documentation/', '')
        if clean_link.startswith('docs/'):
            return str(Path('github-pages') / clean_link)
        else:
            return str(Path('github-pages/docs') / clean_link)
    elif clean_link.startswith('/'):
        return str(Path('github-pages') / clean_link.lstrip('/'))
    else:
        base_path = Path(base_file)
        return str((base_path.parent / clean_link).resolve())

def analyze_broken_links():
    """Find all broken internal links in markdown files"""
    broken_links = {}
    
    # Focus on github-pages directory
    md_files = list(Path('github-pages').rglob('*.md'))
    
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        links = find_markdown_links(content)
        file_broken_links = []
        
        for link_text, url in links:
            if is_internal_link(url):
                if not normalize_and_check_path(md_file, url):
                    expected_path = get_expected_path(md_file, url)
                    file_broken_links.append({
                        'text': link_text,
                        'url': url,
                        'expected_path': expected_path,
                        'line': find_line_number(md_file, link_text, url)
                    })
        
        if file_broken_links:
            broken_links[str(md_file)] = file_broken_links
    
    return broken_links

def find_line_number(file_path, link_text, url):
    """Find the line number where a link appears"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    search_pattern = f"[{link_text}]({url})"
    for i, line in enumerate(lines, 1):
        if search_pattern in line:
            return i
    return None

def create_missing_pages_list(broken_links):
    """Create a deduplicated list of missing pages to create"""
    missing_pages = {}
    
    for file, links in broken_links.items():
        for link in links:
            expected_path = link['expected_path']
            # Normalize to .md extension
            if not expected_path.endswith('.md'):
                if expected_path.endswith('.html'):
                    expected_path = expected_path[:-5] + '.md'
                else:
                    expected_path = expected_path + '.md'
            
            # Skip if outside github-pages
            if not expected_path.startswith('github-pages/'):
                continue
                
            if expected_path not in missing_pages:
                missing_pages[expected_path] = {
                    'references': [],
                    'link_texts': set()
                }
            
            missing_pages[expected_path]['references'].append({
                'file': file,
                'line': link['line'],
                'url': link['url']
            })
            missing_pages[expected_path]['link_texts'].add(link['text'])
    
    # Convert sets to lists for JSON serialization
    for path in missing_pages:
        missing_pages[path]['link_texts'] = list(missing_pages[path]['link_texts'])
    
    return missing_pages

if __name__ == '__main__':
    print("Analyzing markdown files for broken internal links...")
    broken_links = analyze_broken_links()
    
    if not broken_links:
        print("No broken internal links found!")
    else:
        missing_pages = create_missing_pages_list(broken_links)
        
        # Save detailed results
        with open('broken_links_analysis.json', 'w') as f:
            json.dump({
                'broken_links': broken_links,
                'missing_pages': missing_pages,
                'summary': {
                    'total_files_with_broken_links': len(broken_links),
                    'total_broken_links': sum(len(links) for links in broken_links.values()),
                    'total_missing_pages': len(missing_pages)
                }
            }, f, indent=2)
        
        print(f"\nFound {len(broken_links)} files with broken links")
        print(f"Total broken links: {sum(len(links) for links in broken_links.values())}")
        print(f"Total missing pages to create: {len(missing_pages)}")
        
        # Print summary
        print("\nMissing pages that need to be created:")
        for page_path, info in missing_pages.items():
            print(f"\n{page_path}")
            print(f"  Referenced by: {len(info['references'])} links")
            print(f"  Link texts: {', '.join(info['link_texts'][:3])}{'...' if len(info['link_texts']) > 3 else ''}")