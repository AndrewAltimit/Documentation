#!/usr/bin/env python3
"""Shared markdown link checker module for reuse across the codebase"""

import asyncio
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import mistune
from mistune.renderers.rst import RSTRenderer


class LinkExtractorRenderer(RSTRenderer):
    """Custom mistune renderer to extract links from markdown"""

    def __init__(self):
        super().__init__()
        self.links: Set[str] = set()

    def link(self, token, state) -> str:
        """Extract link URLs during rendering"""
        attrs = token.get("attrs", {})
        url = attrs.get("url", "")
        if url:
            self.links.add(url)
        return ""

    def image(self, token, state) -> str:
        """Extract image URLs during rendering"""
        attrs = token.get("attrs", {})
        url = attrs.get("url", "")
        if url:
            self.links.add(url)
        return ""

    def autolink(self, token, state) -> str:
        """Extract autolink URLs"""
        children = token.get("children", [])
        if children and isinstance(children, list) and len(children) > 0:
            url = children[0].get("raw", "")
            if url and not url.startswith("mailto:"):
                self.links.add(url)
        return ""


class MarkdownLinkChecker:
    """Check links in markdown files"""

    def __init__(self):
        self.default_ignore_patterns = [
            r"^http://localhost",
            r"^http://127\.0\.0\.1",
            r"^http://192\.168\.",
            r"^http://0\.0\.0\.0",
            r"^#",
            r"^mailto:",
            r"^chrome://",
            r"^file://",
            r"^ftp://",
        ]
        # Create markdown parser with link extractor
        self.link_renderer = LinkExtractorRenderer()
        self.markdown = mistune.create_markdown(renderer=self.link_renderer)

    async def check_markdown_links(
        self,
        path: str,
        check_external: bool = True,
        ignore_patterns: Optional[List[str]] = None,
        timeout: int = 10,
        concurrent_checks: int = 10,
    ) -> Dict[str, Any]:
        """Check links in markdown files for validity"""

        if ignore_patterns is None:
            ignore_patterns = self.default_ignore_patterns

        # Compile ignore patterns
        compiled_patterns = [re.compile(pattern) for pattern in ignore_patterns]

        # Find all markdown files
        markdown_files = []
        path_obj = Path(path)

        if path_obj.is_file() and path_obj.suffix in [".md", ".markdown"]:
            markdown_files = [path_obj]
        elif path_obj.is_dir():
            markdown_files = list(path_obj.rglob("*.md")) + list(path_obj.rglob("*.markdown"))
        else:
            return {
                "success": False,
                "error": f"Path {path} is not a valid markdown file or directory",
            }

        if not markdown_files:
            return {
                "success": True,
                "files_checked": 0,
                "message": "No markdown files found",
            }

        # Process all files
        all_results = []
        total_links = 0
        broken_links = 0

        for md_file in markdown_files:
            try:
                links = await self._extract_links_from_markdown(md_file)
                file_results: Dict[str, Any] = {
                    "file": str(md_file),
                    "links": [],
                    "broken_count": 0,
                    "total_count": len(links),
                }

                # Filter out ignored patterns
                links_to_check = []
                for link in links:
                    # Skip single-character "links" (false positives from LaTeX)
                    if len(link) <= 1:
                        continue
                    should_ignore = any(pattern.match(link) for pattern in compiled_patterns)
                    if not should_ignore:
                        if not check_external and link.startswith(("http://", "https://")):
                            continue
                        links_to_check.append(link)

                # Check links concurrently
                if links_to_check:
                    link_results = await self._check_links_batch(links_to_check, md_file.parent, timeout, concurrent_checks)

                    for link, is_valid, error in link_results:
                        file_results["links"].append({"url": link, "valid": is_valid, "error": error})
                        total_links += 1
                        if not is_valid:
                            broken_links += 1
                            file_results["broken_count"] += 1

                all_results.append(file_results)

            except Exception as e:
                print(f"Error processing {md_file}: {str(e)}", file=sys.stderr)
                all_results.append({"file": str(md_file), "error": str(e)})

        return {
            "success": True,
            "files_checked": len(markdown_files),
            "total_links": total_links,
            "broken_links": broken_links,
            "all_valid": broken_links == 0,
            "results": all_results,
        }

    async def _extract_links_from_markdown(self, file_path: Path) -> List[str]:
        """Extract all links from a markdown file using mistune parser"""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Remove code blocks before parsing to prevent false positives
            # While mistune should handle this, it sometimes extracts text from code blocks
            # Remove fenced code blocks (```...```)
            content = re.sub(r"```[\s\S]*?```", "", content)
            # Remove inline code (`...`)
            content = re.sub(r"`[^`]+`", "", content)

            # Clear previous links and parse with mistune
            self.link_renderer.links.clear()
            self.markdown(content)

            # Get extracted links
            links = list(self.link_renderer.links)

            # Also check for reference-style links (mistune handles these but just to be sure)
            ref_pattern = r"^\[([^\]]+)\]:\s*(.+)$"
            for line in content.split("\n"):
                ref_match = re.match(ref_pattern, line.strip())
                if ref_match is not None:
                    url = ref_match.group(2).strip()
                    if url not in links:
                        links.append(url)

            return links

        except Exception as e:
            print(f"Error extracting links from {file_path}: {str(e)}", file=sys.stderr)
            return []

    async def _check_links_batch(
        self, links: List[str], base_dir: Path, timeout: int, max_concurrent: int
    ) -> List[Tuple[str, bool, Optional[str]]]:
        """Check multiple links concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_single_link(link: str) -> Tuple[str, bool, Optional[str]]:
            async with semaphore:
                return await self._check_single_link(link, base_dir, timeout)

        tasks = [check_single_link(link) for link in links]
        return await asyncio.gather(*tasks)

    def _resolve_jekyll_path(self, file_path: Path, base_dir: Path) -> bool:
        """
        Resolve a path using Jekyll conventions.

        Jekyll converts .md files to .html and supports permalinks.
        This method checks multiple possible source file locations.
        """
        path_str = str(file_path)

        # Direct file exists check
        if file_path.exists():
            return True

        # For relative paths, also check from base_dir
        full_path = base_dir / file_path
        if full_path.exists():
            return True

        # Try to find the Jekyll source root (look for github-pages or _config.yml)
        jekyll_root = self._find_jekyll_root(base_dir)

        # List of path variations to try
        paths_to_try = []

        # Handle .html -> .md conversion
        if path_str.endswith(".html"):
            md_path = path_str[:-5] + ".md"
            paths_to_try.append(Path(md_path))

        # Handle permalink-style paths (e.g., /docs/foo/bar/)
        if path_str.endswith("/"):
            # Could be docs/foo/bar.md or docs/foo/bar/index.md
            base_path = path_str.rstrip("/")
            paths_to_try.append(Path(base_path + ".md"))
            paths_to_try.append(Path(base_path) / "index.md")

        # Handle paths without extension (could be permalink)
        if not path_str.endswith((".md", ".html", "/")):
            paths_to_try.append(Path(path_str + ".md"))
            paths_to_try.append(Path(path_str) / "index.md")
            paths_to_try.append(Path(path_str + ".html"))

        # Check all variations
        for try_path in paths_to_try:
            # Check relative to base_dir
            if (base_dir / try_path).exists():
                return True
            # Check as absolute from current dir
            if try_path.exists():
                return True
            # Check relative to Jekyll root if found
            if jekyll_root and (jekyll_root / try_path).exists():
                return True

        # For absolute paths starting with /docs/, try github-pages prefix
        if path_str.startswith("docs/") or path_str.startswith("/docs/"):
            clean_path = path_str.lstrip("/")
            gh_pages_paths = [
                Path("github-pages") / clean_path,
                Path("github-pages") / (clean_path.rstrip("/") + ".md"),
                Path("github-pages") / clean_path.rstrip("/") / "index.md",
            ]
            # Handle .html -> .md
            if clean_path.endswith(".html"):
                gh_pages_paths.append(Path("github-pages") / (clean_path[:-5] + ".md"))

            for gh_path in gh_pages_paths:
                if gh_path.exists():
                    return True
                if jekyll_root and (jekyll_root.parent / gh_path).exists():
                    return True

        return False

    def _find_jekyll_root(self, start_dir: Path) -> Optional[Path]:
        """Find the Jekyll site root by looking for _config.yml"""
        current = start_dir
        for _ in range(10):  # Limit search depth
            if (current / "_config.yml").exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        return None

    async def _check_single_link(self, link: str, base_dir: Path, timeout: int) -> Tuple[str, bool, Optional[str]]:
        """Check if a single link is valid"""
        try:
            # Check if it's a relative file link
            if not link.startswith(("http://", "https://", "ftp://", "//")):
                # Handle relative paths
                if link.startswith("/"):
                    # Absolute path - likely a Jekyll permalink
                    file_path = Path(link[1:])
                else:
                    # Relative to current file
                    file_path = base_dir / link

                # Remove anchor if present
                if "#" in str(file_path):
                    file_path = Path(str(file_path).split("#", maxsplit=1)[0])

                # Check if file exists using Jekyll-aware resolution
                if self._resolve_jekyll_path(file_path, base_dir):
                    return (link, True, None)
                else:
                    return (link, False, "File not found")

            # Check external links
            if link.startswith("//"):
                link = "https:" + link

            # Use aiohttp to check HTTP/HTTPS links
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                try:
                    async with session.head(link, allow_redirects=True) as response:
                        if response.status < 400:
                            return (link, True, None)
                        else:
                            return (link, False, f"HTTP {response.status}")
                except aiohttp.ClientError as e:
                    # Try GET if HEAD fails
                    try:
                        async with session.get(link, allow_redirects=True) as response:
                            if response.status < 400:
                                return (link, True, None)
                            else:
                                return (link, False, f"HTTP {response.status}")
                    except Exception:
                        return (link, False, str(e))

        except Exception as e:
            return (link, False, str(e))
