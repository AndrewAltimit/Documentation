# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is **Andrew's Notebook** - a technical documentation site hosted on GitHub Pages covering science and technology topics. It's a **single-maintainer project** by @AndrewAltimit with a **container-first philosophy**:

- **GitHub Pages Site**: https://andrewaltimit.github.io/Documentation/
- **Content Focus**: Physics (quantum mechanics, relativity, thermodynamics) and Technology (Docker, Kubernetes, AWS, Git)
- **Jekyll-based**: Uses minimal-mistakes theme with professional wiki-style presentation
- **Self-hosted CI/CD**: All workflows run on self-hosted GitHub Actions runners
- **Container-first**: All Python operations run in Docker containers

## Site Structure

```
Documentation/
├── github-pages/           # Jekyll site source
│   ├── docs/              # Main documentation pages
│   │   ├── physics/       # Physics documentation
│   │   ├── technology/    # Technology documentation
│   │   └── ai-ml/         # AI/ML documentation
│   ├── _data/             # Navigation and site data
│   ├── _layouts/          # Custom Jekyll layouts
│   └── assets/            # CSS, JS, images
├── automation/            # CI/CD and automation scripts
│   ├── ci-cd/            # CI/CD helper scripts
│   ├── analysis/         # Link checking, analysis tools
│   └── setup/            # Runner and environment setup
├── docker/               # Dockerfiles
├── tools/                # MCP servers and utilities
│   └── mcp/             # Content creation tools (Manim, LaTeX)
└── outputs/              # Generated content output
```

## AI Agent Collaboration

You are working alongside these AI agents:

1. **Gemini CLI** - Handles automated PR code reviews
2. **GitHub Copilot** - Provides code review suggestions in PRs

Your role as Claude Code is the primary development assistant, handling:

- Documentation content creation and editing
- CI/CD pipeline development
- Automation script development
- Jekyll site configuration

## Commands

### Building the Site

```bash
# Build Jekyll site locally (from repository root)
docker-compose run --rm jekyll

# Or from github-pages directory
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

### Code Quality

```bash
# Using containerized CI scripts (recommended)
./automation/ci-cd/run-ci.sh format      # Check formatting
./automation/ci-cd/run-ci.sh lint-basic   # Basic linting
./automation/ci-cd/run-ci.sh lint-full    # Full linting suite
./automation/ci-cd/run-ci.sh autoformat   # Auto-format code

# Direct Docker Compose commands
docker-compose run --rm python-ci black --check .
docker-compose run --rm python-ci flake8 .

# Run all checks at once
./automation/ci-cd/run-ci.sh full
```

### Link Checking

```bash
# Check all markdown links
python automation/analysis/check-markdown-links.py

# Check only internal links
python automation/analysis/check-markdown-links.py --internal-only

# Check specific directory
python automation/analysis/check-markdown-links.py --file github-pages/docs/
```

### Docker Operations

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f python-ci

# Stop services
docker-compose down

# Rebuild after changes
docker-compose build python-ci
```

### Content Creation (MCP Servers)

```bash
# Start content creation server for Manim animations and LaTeX
docker-compose up -d mcp-content-creation

# Or run locally
python -m mcp_content_creation.server
```

## Available MCP Tools

### Content Creation (Port 8011)

For creating educational content in documentation:

- **create_manim_animation** - Create mathematical/technical animations for physics visualizations
- **compile_latex** - Generate PDF documents from LaTeX (equations, diagrams)
- **render_tikz** - Render TikZ diagrams as standalone images

### Gemini Integration (Port 8006)

For AI-assisted review and consultation:

- **consult_gemini** - Get AI assistance for technical questions
- **clear_gemini_history** - Clear conversation history

## GitHub Actions Integration

The repository includes CI/CD workflows:

- **PR Validation**: Automatic Gemini AI code review
- **Code Quality**: Multi-stage linting in Docker containers
- **Link Checking**: Automated markdown link validation
- **Jekyll Build**: Site builds on GitHub Pages
- **Self-hosted Runners**: All workflows run on self-hosted infrastructure

## Development Reminders

- **Content Focus**: This is a documentation site - focus on clear, accurate technical writing
- **Wiki Style**: Professional reference documentation, not tutorials
- **Visual Elements**: Use diagrams, animations, and code examples where helpful
- **Link Integrity**: Always verify internal links when moving/renaming pages
- IMPORTANT: When you have completed a task, you MUST run the lint and quality checks:
  ```bash
  ./automation/ci-cd/run-ci.sh full
  ```
- NEVER commit changes unless the user explicitly asks you to
- Always follow the container-first philosophy - use Docker for all Python operations

## GitHub Etiquette

**IMPORTANT**: When working with GitHub issues, PRs, and comments:

- **NEVER use @ mentions** unless referring to actual repository maintainers
- Do NOT use @Gemini, @Claude, @OpenAI, etc. - these may ping unrelated GitHub users
- Only @ mention the repository owner (@AndrewAltimit)
- When referencing AI reviews, use phrases like "As noted in Gemini's review..."

### PR Comments and Reactions

**Use Custom Reaction Images**: When commenting on PRs and issues, use our custom reaction images.

- **Available reactions**: https://raw.githubusercontent.com/AndrewAltimit/Media/refs/heads/main/reaction/config.yaml
- **Format**: `![Reaction](https://raw.githubusercontent.com/AndrewAltimit/Media/refs/heads/main/reaction/[filename])`

**CRITICAL: Proper Method for GitHub Comments with Reaction Images**

When posting PR/issue comments with reaction images:

1. **Use the Write tool** to create a temporary markdown file (e.g., `/tmp/comment.md`)
2. Use `gh pr comment --body-file /tmp/filename.md` to post the comment

**DO NOT USE** direct `--body` flag, heredocs, or echo commands - they will escape the `!` in `![Reaction]`.

## Documentation Topics

### Physics
- Classical Mechanics
- Quantum Mechanics
- Quantum Field Theory
- Relativity (Special and General)
- Statistical Mechanics
- Thermodynamics
- Condensed Matter Physics
- String Theory

### Technology
- Docker containerization
- Kubernetes orchestration
- Terraform infrastructure as code
- AWS cloud services
- Git version control
- Branching strategies
- Networking fundamentals
- Database design
- Cybersecurity basics

### AI/ML
- Stable Diffusion fundamentals
- ComfyUI workflows
- LoRA training concepts
- Base model comparisons
- ControlNet techniques

## Container Architecture

1. **Jekyll Container**: Site building and local preview
2. **Python CI Container**: Linting, formatting, testing
3. **Content Creation Container**: Manim animations, LaTeX compilation

All containers run with user permissions (non-root) for security.

## Security Considerations

- API key management via environment variables
- Docker network isolation for services
- No hardcoded credentials in codebase
- Self-hosted runners for full control
