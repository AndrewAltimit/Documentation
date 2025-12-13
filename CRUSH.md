# CRUSH.md - Agentic Coding Agent Configuration

This file contains essential commands and guidelines for agentic coding agents working in this repository.

## Project Context

This is **Andrew's Notebook** - a technical documentation site hosted on GitHub Pages at https://andrewaltimit.github.io/Documentation/. The site covers Physics and Technology topics with professional wiki-style documentation.

## Core Commands

### Build and Code Quality

```bash
# Full CI pipeline (formatting, linting)
./automation/ci-cd/run-ci.sh full

# Individual stages
./automation/ci-cd/run-ci.sh format      # Check formatting
./automation/ci-cd/run-ci.sh lint-basic  # Basic linting
./automation/ci-cd/run-ci.sh lint-full   # Full linting suite
./automation/ci-cd/run-ci.sh autoformat  # Auto-format code
```

### Jekyll Site

```bash
# Build site locally
docker-compose run --rm jekyll

# Check markdown links
python automation/analysis/check-markdown-links.py
```

### Container Operations

```bash
# Build and start all services
docker-compose up -d

# Rebuild after changes
docker-compose build python-ci

# Run any Python command in CI container
docker-compose run --rm python-ci python --version
```

## Code Style Guidelines

### Python Standards

- Line length: 88 characters (Black), 127 for docstrings/comments (Flake8)
- Import sorting: isort with Black profile
- Type hints: Use where possible
- Docstrings: Google style preferred

### Naming Conventions

- Variables: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private members: prefixed with underscore

### Documentation Style

- Markdown: GitHub-flavored
- Wiki format: Professional reference, not tutorials
- Diagrams: Mermaid, TikZ, or images

## Project-Specific Patterns

### Content Creation

- Manim animations for physics visualizations
- LaTeX for equations and technical diagrams
- TikZ for standalone diagrams

### Security

- Never commit secrets or API keys
- Use environment variables for configuration
- All containers run as non-root user
