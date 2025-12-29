#!/usr/bin/env python3
"""
Enhanced Internal Link Validator for Jekyll Sites with GitHub Pages

This script validates internal markdown links by understanding Jekyll permalinks
and how relative links resolve based on source page URL structure.

Key features:
- Understands trailing-slash permalinks vs .html pages
- Correctly resolves relative links (../, ./, etc.)
- Validates that target pages exist
- Reports mismatches between link format and target permalink style
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from urllib.parse import urljoin
from collections import defaultdict


def extract_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    return yaml.safe_load(parts[1])
                except yaml.YAMLError:
                    return {}
    except Exception:
        pass
    return {}


def get_page_url(filepath, docs_root, url_prefix='/docs'):
    """Determine the URL for a page based on its permalink or file path."""
    frontmatter = extract_frontmatter(filepath)

    # If explicit permalink exists, use it
    if 'permalink' in frontmatter:
        return frontmatter['permalink']

    # Otherwise, derive from file path (Jekyll default behavior)
    rel_path = os.path.relpath(filepath, docs_root)

    # Convert path to URL with prefix
    url = url_prefix + '/' + rel_path.replace('\\', '/')

    # Replace .md with .html
    if url.endswith('.md'):
        url = url[:-3] + '.html'

    # index.html -> directory with trailing slash
    if url.endswith('/index.html'):
        url = url[:-10] + '/'

    return url


def extract_markdown_links(filepath):
    """Extract all markdown links from a file."""
    links = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Match markdown links: [text](url)
        pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(pattern, content):
            text, url = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            links.append({
                'text': text,
                'url': url,
                'line': line_num
            })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return links


def resolve_relative_url(base_url, relative_url):
    """Resolve a relative URL against a base URL."""
    # Handle absolute URLs (starting with /)
    if relative_url.startswith('/'):
        return relative_url

    # Handle external URLs
    if relative_url.startswith(('http://', 'https://', 'mailto:', '#')):
        return None  # Skip external links

    # For relative URLs, we need to resolve them based on the base URL
    # If base_url ends with /, it's a "directory"
    # If it ends with .html, it's a "file"

    if base_url.endswith('/'):
        # Base is a directory, resolve relative to it
        base = base_url
    else:
        # Base is a file, resolve relative to its directory
        base = base_url.rsplit('/', 1)[0] + '/'

    # Use urljoin for proper resolution
    resolved = urljoin(base, relative_url)

    return resolved


def build_page_index(docs_root, url_prefix='/docs'):
    """Build an index of all pages and their URLs."""
    pages = {}

    for root, dirs, files in os.walk(docs_root):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                url = get_page_url(filepath, docs_root, url_prefix)
                pages[url] = filepath

                # Also index common variations
                if url.endswith('/'):
                    # Trailing slash URLs might be accessed as .html
                    html_url = url.rstrip('/') + '.html'
                    pages[html_url] = filepath
                    # Or as index.html
                    pages[url + 'index.html'] = filepath
                elif url.endswith('.html'):
                    # .html URLs might be accessed without extension
                    no_ext = url[:-5]
                    pages[no_ext] = filepath
                    pages[no_ext + '/'] = filepath

    return pages


def validate_links(docs_root, url_prefix='/docs', verbose=False):
    """Validate all internal links in the documentation."""
    pages = build_page_index(docs_root, url_prefix)
    errors = []
    warnings = []

    for root, dirs, files in os.walk(docs_root):
        for file in files:
            if not file.endswith('.md'):
                continue

            filepath = os.path.join(root, file)
            source_url = get_page_url(filepath, docs_root, url_prefix)
            links = extract_markdown_links(filepath)

            for link in links:
                url = link['url']

                # Skip external links and anchors
                if url.startswith(('http://', 'https://', 'mailto:', '#')):
                    continue

                # Skip anchor-only links within same page
                if url.startswith('#'):
                    continue

                # Resolve the relative URL
                resolved = resolve_relative_url(source_url, url)

                if resolved is None:
                    continue

                # Strip anchor for validation
                resolved_path = resolved.split('#')[0]

                # Check if target exists
                if resolved_path not in pages:
                    # Try common variations
                    found = False
                    variations = [
                        resolved_path,
                        resolved_path.rstrip('/') + '/',
                        resolved_path.rstrip('/') + '.html',
                        resolved_path.rstrip('/'),
                    ]

                    for var in variations:
                        if var in pages:
                            found = True
                            # Check if link format matches target
                            if var != resolved_path:
                                warnings.append({
                                    'file': filepath,
                                    'line': link['line'],
                                    'link': url,
                                    'resolved': resolved_path,
                                    'suggestion': var,
                                    'message': f"Link works but format mismatch. Consider using '{var}'"
                                })
                            break

                    if not found:
                        errors.append({
                            'file': filepath,
                            'line': link['line'],
                            'link': url,
                            'resolved': resolved_path,
                            'source_url': source_url,
                            'message': f"Target not found: {resolved_path}"
                        })

    return errors, warnings, pages


def main():
    parser = argparse.ArgumentParser(
        description='Validate internal links in Jekyll markdown documentation'
    )
    parser.add_argument(
        'docs_root',
        nargs='?',
        default='github-pages/docs',
        help='Root directory of documentation (default: github-pages/docs)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    parser.add_argument(
        '--show-index',
        action='store_true',
        help='Show the page index (all known URLs)'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.docs_root):
        print(f"Error: {args.docs_root} is not a directory")
        return 1

    print(f"Validating links in: {args.docs_root}")
    print()

    # Use /docs prefix for GitHub Pages site structure
    errors, warnings, pages = validate_links(args.docs_root, '/docs', args.verbose)

    if args.show_index:
        print("=== Page Index ===")
        for url in sorted(pages.keys()):
            print(f"  {url}")
        print()

    if warnings:
        print(f"⚠️  {len(warnings)} warnings (links work but format could be improved):")
        print()
        for w in warnings:
            rel_file = os.path.relpath(w['file'], args.docs_root)
            print(f"  {rel_file}:{w['line']}")
            print(f"    Link: {w['link']}")
            print(f"    {w['message']}")
            print()

    if errors:
        print(f"❌ {len(errors)} broken internal links found:")
        print()
        for e in errors:
            rel_file = os.path.relpath(e['file'], args.docs_root)
            print(f"  {rel_file}:{e['line']}")
            print(f"    Link: {e['link']}")
            print(f"    Source URL: {e['source_url']}")
            print(f"    Resolved to: {e['resolved']}")
            print(f"    {e['message']}")
            print()
        return 1
    else:
        print(f"✅ All internal links are valid! ({len(pages)} pages indexed)")
        return 0


if __name__ == '__main__':
    exit(main())
