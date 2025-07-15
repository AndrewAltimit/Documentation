# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Documentation Repository** with three components:
- **GitHub Pages site** (`github-pages/`): Public technical documentation
- **Development docs** (`development-docs/`): Internal AI agent and process documentation  
- **MCP Server**: FastAPI-based AI and development tools server (port 8005)

Single-maintainer project by @AndrewAltimit with **container-first philosophy** - all Python operations run in Docker.

## Essential Commands

### Running Tests
```bash
# Run all tests with coverage
./scripts/run-ci.sh test

# Run specific test file
docker-compose run --rm python-ci pytest tests/test_mcp_tools.py -v

# Run tests matching pattern
docker-compose run --rm python-ci pytest -k "test_format" -v
```

### Code Quality Checks
```bash
# Check formatting
./scripts/run-ci.sh format

# Basic linting (black, isort, flake8)
./scripts/run-ci.sh lint-basic

# Full linting (includes pylint, mypy)
./scripts/run-ci.sh lint-full

# Auto-format code
./scripts/run-ci.sh autoformat

# Run all CI checks
./scripts/run-ci.sh format && ./scripts/run-ci.sh lint-basic && ./scripts/run-ci.sh lint-full && ./scripts/run-ci.sh test
```

### MCP Server Operations
```bash
# Start MCP server
docker-compose up -d mcp-server

# View logs
docker-compose logs -f mcp-server

# Test server health
curl http://localhost:8005/health

# Stop services
docker-compose down
```

### Additional CI Stages
```bash
./scripts/run-ci.sh security    # Security scans (bandit, safety)
./scripts/run-ci.sh yaml-lint   # Validate YAML files
./scripts/run-ci.sh json-lint   # Validate JSON files
```

### Building GitHub Pages Site Locally
```bash
# Build the Jekyll site using Docker (creates _site folder)
cd github-pages
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  jekyll/jekyll:4.2.2 \
  jekyll build

# Alternative: Build using docker-compose from root directory
docker-compose run --rm jekyll

# Generate Gemfile.lock if needed (for dependency management)
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  jekyll/jekyll:4.2.2 \
  bundle install

# Serve the site locally for testing (available at http://localhost:4000)
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  -p 4000:4000 \
  jekyll/jekyll:4.2.2 \
  jekyll serve --host 0.0.0.0

# Clean build artifacts
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  jekyll/jekyll:4.2.2 \
  jekyll clean
```

## Architecture

### MCP Server (`tools/mcp/mcp_server.py`)
FastAPI server providing tools via Model Context Protocol:
- **Code Quality**: format_check, lint, analyze
- **AI Integration**: consult_gemini, clear_gemini_history
- **Content Creation**: create_manim_animation, compile_latex
- **Remote Services**: ComfyUI (image generation), AI Toolkit (LoRA training)

Configuration in `.mcp.json` defines tools, security, and rate limits.

### Container Architecture
- **python-ci**: All Python CI/CD tools (Python 3.11)
- **mcp-server**: FastAPI MCP server
- **manim-latex**: Animation and LaTeX compilation
- **mcp-http-bridge**: Remote service integration

All containers run with user permissions (non-root) via USER_ID/GROUP_ID.

### Testing Strategy
- Tests run in containers with pytest
- Mock external dependencies (subprocess, HTTP)
- Async support with pytest-asyncio
- Coverage reporting with pytest-cov
- No pytest cache (PYTHONDONTWRITEBYTECODE=1)

### GitHub Actions
All workflows run on self-hosted runners:
- **main-ci.yml**: Comprehensive CI pipeline
- **pr-validation.yml**: PR validation with Gemini review  
- **lint-stages.yml**: Multi-stage linting
- **mcp-tools.yml**: MCP server validation

## Development Guidelines

1. **Container-First**: Use Docker for all Python operations
2. **Testing**: Mock external dependencies, use async fixtures
3. **Paths**: Always use absolute paths in configurations
4. **Security**: No hardcoded credentials, use environment variables
5. **AI Tools**: Clear Gemini history before PR reviews

## Working with AI Services

### Remote MCP Integration
- **Dataset Paths**: Use absolute paths (`/ai-toolkit/datasets/name`)
- **LoRA Transfer**: For >100MB files, use chunked upload
- **FLUX Workflows**: Different from SD - use FluxGuidance, cfg=1.0

### AI Agent Collaboration
You work alongside:
- **Gemini CLI**: Automated PR reviews (runs on host)
- **GitHub Copilot**: Code review suggestions

Your role: Primary development (architecture, implementation, debugging, docs).

## Critical Reminders

- **After completing tasks**: ALWAYS run quality checks before marking complete
- **Commits**: ONLY commit when explicitly asked by the user
- **No Cache**: Python cache disabled in all containers
- **Permissions**: No `chmod 777` or overly permissive operations