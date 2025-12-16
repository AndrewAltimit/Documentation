# Andrew's Notebook

Technical documentation repository hosting a GitHub Pages site covering physics, technology, and AI/ML topics.

**Live Site:** https://andrewaltimit.github.io/Documentation/

## Overview

This repository contains:

- **GitHub Pages Site** - Public-facing technical wiki covering quantum computing, AI/ML, containerization, cloud infrastructure, and more
- **MCP Tools** - Model Context Protocol servers for code quality, content creation, and AI integration
- **CI/CD Infrastructure** - Container-first automation with self-hosted GitHub Actions runners

## Repository Structure

```
Documentation/
├── github-pages/          # Jekyll site source (docs, layouts, assets)
├── tools/mcp/             # MCP server implementations
├── automation/            # CI/CD scripts and analysis tools
├── docker/                # Dockerfiles for all services
├── development-docs/      # Internal docs for AI agents and processes
├── examples/              # Example scripts and demos
└── config/                # Configuration files
```

## Quick Start

### Build the Site Locally

```bash
# Using docker-compose (recommended)
docker-compose run --rm jekyll

# Serve locally at http://localhost:4000
cd github-pages
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  -p 4000:4000 \
  jekyll/jekyll:4.2.2 \
  jekyll serve --host 0.0.0.0
```

### Run Code Quality Checks

```bash
./automation/ci-cd/run-ci.sh full
```

## Documentation Topics

### Physics
- Quantum Mechanics and Quantum Field Theory
- Special and General Relativity
- Statistical Mechanics and Thermodynamics
- Condensed Matter Physics

### Technology
- Docker and Kubernetes
- Terraform and AWS
- Git and Version Control
- Database Design
- Cybersecurity

### AI/ML
- Stable Diffusion and Diffusion Models
- ComfyUI Workflows
- LoRA Training
- Transformer Architectures

## MCP Tools

FastAPI-based Model Context Protocol servers providing:

| Tool | Description |
|------|-------------|
| mcp_core | Base MCP server framework |
| mcp_content_creation | Manim animations and LaTeX compilation |
| mcp_gemini | Gemini AI integration |
| mcp_code_quality | Linting and formatting tools |

## Development

This project follows a container-first philosophy. All Python operations run in Docker containers to ensure consistency and reproducibility.

Key files for contributors:
- `CLAUDE.md` - Instructions for Claude Code AI assistant
- `development-docs/` - Internal documentation and guides
- `pyproject.toml` - Python tool configurations (black, ruff, mypy, etc.)

## License

See [LICENSE](LICENSE) for details.

---

Maintained by [@AndrewAltimit](https://github.com/AndrewAltimit)
