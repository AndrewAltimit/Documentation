# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Last updated: 2025.

## Project Context

This is a **single-maintainer project** by @AndrewAltimit with a **container-first philosophy**:

- All Python operations run in Docker containers
- Self-hosted infrastructure for zero-cost operation
- Designed for maximum portability - works on any Linux system with Docker
- No contributors model - optimized for individual developer efficiency

## AI Agent Collaboration

You are working alongside two other AI agents:

1. **Gemini CLI** - Handles automated PR code reviews (using gemini-2.5-pro model)
2. **GitHub Copilot** - Provides code review suggestions in PRs

Your role as Claude Code is the primary development assistant, handling:

- Architecture decisions and implementation
- Complex refactoring and debugging  
- Documentation and test writing
- CI/CD pipeline development
- MCP tool integration and usage

## Commands

### Running Tests

```bash
# Run all tests with coverage (containerized with Python 3.11+)
docker-compose run --rm python-ci pytest tests/ -v --cov=. --cov-report=xml --cov-report=term

# Run a specific test file
docker-compose run --rm python-ci pytest tests/test_mcp_tools.py -v

# Run tests with specific test name pattern
docker-compose run --rm python-ci pytest -k "test_format" -v

# Quick test run using helper script (recommended)
./scripts/run-ci.sh test

# Run tests with debug output
docker-compose run --rm python-ci pytest tests/ -v -s --log-cli-level=DEBUG
```

### Code Quality

```bash
# Using containerized CI scripts (recommended)
./scripts/run-ci.sh format      # Check formatting
./scripts/run-ci.sh lint-basic   # Basic linting
./scripts/run-ci.sh lint-full    # Full linting suite
./scripts/run-ci.sh autoformat   # Auto-format code

# Direct Docker Compose commands
docker-compose run --rm python-ci black --check .
docker-compose run --rm python-ci flake8 .
docker-compose run --rm python-ci pylint tools/ scripts/
docker-compose run --rm python-ci mypy . --ignore-missing-imports

# Note: All Python CI/CD tools run in containers to ensure consistency

# Run all checks at once (recommended before commits)
./scripts/run-ci.sh full

# Run specific linting stages
./scripts/run-lint-stage.sh format
./scripts/run-lint-stage.sh basic
./scripts/run-lint-stage.sh full
```

### Development

```bash
# Start MCP server via Docker (recommended)
docker-compose up -d mcp-server

# View MCP server logs
docker-compose logs -f mcp-server

# Test MCP server health (port 8005)
curl http://localhost:8005/health

# List available MCP tools
curl http://localhost:8005/tools

# Test with Python script
python scripts/test-mcp-server.py

# Run the main application
python main.py

# For local development without Docker (not recommended)
# Requires Python 3.11+
pip install -r requirements.txt
python tools/mcp/mcp_server.py
```

### Docker Operations

```bash
# Build and start all services
docker-compose up -d

# View specific container logs
docker-compose logs -f python-ci

# Stop all services
docker-compose down

# Rebuild after changes
docker-compose build mcp-server
docker-compose build python-ci
```

### Helper Scripts

```bash
# CI/CD operations script
./scripts/run-ci.sh [stage]
# Stages: format, lint-basic, lint-full, security, test, yaml-lint, json-lint, autoformat

# Lint stage helper (used in workflows)
./scripts/run-lint-stage.sh [stage]
# Stages: format, basic, full

# Fix runner permission issues
./scripts/fix-runner-permissions.sh
```

## Architecture

### MCP Server Architecture

The project centers around a Model Context Protocol (MCP) server that provides various AI and development tools:

1. **FastAPI Server** (`tools/mcp/mcp_server.py`): Main HTTP API on port 8005
2. **Tool Categories**:
   - **Code Quality**:
     - `format_check` - Check code formatting (Python, JS, TS, Go, Rust)
     - `lint` - Run static analysis with optional config
     - `analyze` - Deep code analysis with security checks
   - **AI Integration**:
     - `consult_gemini` - Get AI assistance for technical questions
     - `clear_gemini_history` - Clear conversation history for fresh responses
     - `create_manim_animation` - Create mathematical/technical animations
     - `compile_latex` - Generate PDF/DVI/PS documents from LaTeX
   - **Remote Services**: ComfyUI (image generation), AI Toolkit (LoRA training)

3. **Containerized CI/CD**:
   - **Python CI Container** (`docker/python-ci.Dockerfile`): All Python tools (Black, isort, flake8, pylint, mypy, pytest) with Python 3.11+
   - **Helper Scripts**: Centralized CI operations to reduce workflow complexity
   - **Cache Prevention**: PYTHONDONTWRITEBYTECODE=1, pytest cache disabled via pytest.ini
   - **User Permissions**: Containers run with USER_ID:GROUP_ID to avoid permission issues

4. **Configuration** (`.mcp.json`): Defines available tools, security settings, and rate limits

### GitHub Actions Integration

The repository includes comprehensive CI/CD workflows:

- **PR Validation**: Automatic Gemini AI code review with history clearing
- **Testing Pipeline**: Containerized pytest with coverage reporting
- **Code Quality**: Multi-stage linting in Docker containers
- **Self-hosted Runners**: All workflows run on self-hosted infrastructure
- **Runner Maintenance**: Automated cleanup and health checks

### Container Architecture Philosophy

1. **Everything Containerized**:
   - Python CI/CD tools run in `python-ci` container (Python 3.11)
   - MCP server runs in its own container
   - Only exception: Gemini CLI (would require Docker-in-Docker)
   - All containers run with user permissions (non-root)

2. **Zero Local Dependencies**:
   - No need to install Python, Node.js, or any tools locally
   - All operations available through Docker Compose
   - Portable across any Linux system

3. **Self-Hosted Infrastructure**:
   - All GitHub Actions run on self-hosted runners (v2.326.0+)
   - No cloud costs or external dependencies
   - Full control over build environment
   - Docker images cached locally for fast builds (BuildKit enabled)

### Key Integration Points

1. **AI Services**:
   - Gemini API for code review (runs on host due to Docker requirements)
   - Support for Claude and OpenAI integrations
   - Remote ComfyUI workflows for image generation

2. **Testing Strategy**:
   - All tests run in containers with Python 3.11+
   - Mock external dependencies (subprocess, HTTP calls) using pytest-mock
   - Async test support with pytest-asyncio
   - Coverage reporting with pytest-cov (XML and terminal output)
   - No pytest cache to avoid permission issues (-p no:cacheprovider in pytest.ini)

3. **Client Pattern** (`main.py`):
   - MCPClient class for interacting with MCP server
   - Example workflow demonstrating tool usage
   - Environment-based configuration

### Security Considerations

- API key management via environment variables
- Rate limiting configured in .mcp.json
- Docker network isolation for services
- No hardcoded credentials in codebase
- Containers run as non-root user

## Development Reminders

- IMPORTANT: When you have completed a task, you MUST run the lint and quality checks:
  ```bash
  # Run full CI checks (recommended - includes all stages)
  ./scripts/run-ci.sh full

  # Or individual checks in order
  ./scripts/run-ci.sh format      # Check code formatting
  ./scripts/run-ci.sh lint-basic   # Basic linting (black, isort, flake8)
  ./scripts/run-ci.sh lint-full    # Full linting (includes pylint, mypy)
  ./scripts/run-ci.sh test         # Run all tests with coverage
  ```
- NEVER commit changes unless the user explicitly asks you to
- Always follow the container-first philosophy - use Docker for all Python operations
- Remember that Gemini CLI cannot be containerized (needs Docker access)
- Use pytest fixtures and mocks for testing external dependencies
- Ensure all file operations preserve user permissions (use USER_ID/GROUP_ID)
- Run containers as non-root user for security

## AI Toolkit & ComfyUI Integration Notes

When working with the remote MCP servers (AI Toolkit and ComfyUI):

1. **Dataset Paths**: Always use absolute paths in AI Toolkit configs:
   - ✅ `/ai-toolkit/datasets/dataset_name`
   - ❌ `dataset_name` (will fail with "No such file or directory")

2. **LoRA Transfer**: For files >100MB, use chunked upload:
   - See `transfer_lora_between_services.py` for working implementation
   - Parameters: `upload_id` (provide UUID), `total_size` (bytes), `chunk` (not `chunk_data`)

3. **FLUX Workflows**: Different from SD workflows:
   - Use FluxGuidance node (guidance ~3.5)
   - KSampler: cfg=1.0, sampler="heunpp2", scheduler="simple"
   - Negative prompt cannot be null (use empty string)

4. **MCP Tool Discovery**: The `/mcp/tools` endpoint may not list all tools
   - Check the gist source directly for complete tool list
   - Chunked upload tools exist even if not shown

See `AI_TOOLKIT_COMFYUI_INTEGRATION_GUIDE.md` for comprehensive details.

## Related Documentation

- [AI_AGENTS.md](development-docs/AI_AGENTS.md) - Overview of the three-agent system
- [CONTAINERIZED_CI.md](development-docs/CONTAINERIZED_CI.md) - Container-first CI/CD details
- [MCP_TOOLS.md](development-docs/MCP_TOOLS.md) - Complete MCP tool reference
- [GEMINI_SETUP.md](development-docs/GEMINI_SETUP.md) - Gemini CLI configuration

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