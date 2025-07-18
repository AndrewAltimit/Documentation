# Documentation Repository

This repository contains comprehensive technical documentation with three main components:

## 📁 github-pages/
Contains the GitHub Pages website documentation for "Andrew's Notebook":
- Public-facing technical wiki/knowledgebase covering physics and technology
- Professional reference documentation with Jekyll-based web presentation
- Topics include: Quantum Computing, AI/ML, Docker, Kubernetes, Terraform, AWS, Git, and more
- Available at: https://andrewaltimit.github.io/Documentation/

### Building the Site Locally
```bash
# Using docker-compose (from repository root)
docker-compose run --rm jekyll

# Or using Docker directly (from github-pages directory)
cd github-pages
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  jekyll/jekyll:4.2.2 \
  jekyll build

# Serve locally for testing (http://localhost:4000)
docker run --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --volume="$PWD/vendor/bundle:/usr/local/bundle:Z" \
  -p 4000:4000 \
  jekyll/jekyll:4.2.2 \
  jekyll serve --host 0.0.0.0
```

## 📁 development-docs/
Contains internal documentation for AI agents and development processes:
- `CLAUDE.md` - Instructions for Claude Code AI assistant
- `PROJECT_CONTEXT.md` - Project context for AI code review
- `AI_AGENTS.md` - Documentation about the three AI agents (Claude, Gemini, Copilot)
- `AI_TOOLKIT_COMFYUI_INTEGRATION_GUIDE.md` - AI toolkit and ComfyUI integration guide
- `CONTAINERIZED_CI.md` - Container-first CI/CD documentation
- `GEMINI_SETUP.md` - Gemini CLI setup and configuration
- `LORA_TRANSFER_DOCUMENTATION.md` - LoRA model transfer documentation
- `MCP_TOOLS.md` - Model Context Protocol tools documentation
- `SELF_HOSTED_RUNNER_SETUP.md` - GitHub Actions self-hosted runner setup

## 🔧 MCP Server
FastAPI-based Model Context Protocol server (port 8005) providing:
- Code quality tools (format checking, linting, analysis)
- AI integration tools (Gemini consultation, history management)
- Content creation tools (Manim animations, LaTeX compilation)
- Remote service integration (ComfyUI, AI Toolkit)

## Other Key Directories
- `docker/` - Dockerfiles for containerized services (python-ci, mcp-server, manim-latex, mcp-http-bridge)
- `scripts/` - Utility scripts for CI/CD, development, and maintenance
- `tools/` - Python tools including MCP server implementation and Gemini integration
- `tests/` - Test suite for MCP tools and server functionality
- `examples/` - Example scripts demonstrating tool usage and visualizations
- `.github/workflows/` - GitHub Actions workflows for CI/CD

## Project Philosophy
- **Container-First**: All Python operations run in Docker containers
- **Professional Documentation**: Wiki-style technical reference (not tutorials)
- **AI-Assisted Development**: Integrated AI agents for code review and documentation
- **Automated Quality**: Comprehensive CI/CD with linting, testing, and security checks

## Recent Updates (2024-2025)
- Transformed from tutorial platform to professional technical wiki
- Enhanced visual documentation with diagrams, animations, and interactive elements
- Implemented progressive documentation architecture
- Fixed broken links and updated all content for accuracy
- Added comprehensive physics visualizations and code examples

## Important Notes
- The GitHub Pages documentation is meant for public technical reference
- The development documentation guides AI agents and internal processes
- All Python operations should use the provided Docker containers
- The project follows a single-maintainer model by @AndrewAltimit