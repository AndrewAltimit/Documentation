# Python CI/CD Image with all testing and linting tools
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    ffmpeg \
    shellcheck \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Update pip and setuptools to secure versions
RUN pip install --no-cache-dir --upgrade pip>=23.3 setuptools>=78.1.1

# Copy requirements first to leverage Docker layer caching
COPY config/python/requirements.txt ./

# Install all dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Install workspace packages in editable mode
# Note: This happens before copying the full codebase to leverage caching
# The actual code will be mounted at runtime via docker-compose volumes
COPY pyproject.toml /app/pyproject.toml
COPY tools /app/tools
COPY automation /app/automation

# Install MCP packages that exist in this repo
RUN pip install --no-cache-dir /app/tools/mcp/mcp_core && \
    pip install --no-cache-dir /app/tools/mcp/mcp_codex && \
    pip install --no-cache-dir /app/tools/mcp/mcp_content_creation && \
    pip install --no-cache-dir /app/tools/mcp/mcp_crush && \
    pip install --no-cache-dir /app/tools/mcp/mcp_gemini && \
    pip install --no-cache-dir /app/tools/mcp/mcp_github_board && \
    pip install --no-cache-dir /app/tools/mcp/mcp_opencode && \
    pip install --no-cache-dir /app/tools/mcp/mcp_agentcore_memory

# Copy linting configuration files to both /workspace and /app
# Note: Files are copied to both locations to support different tool contexts:
# - /workspace is the primary working directory for most CI operations
# - /app is used by some tools that expect absolute paths or when running
#   with read-only mounts where the code is mounted at /app
COPY .flake8 .pylintrc ./
COPY .flake8 .pylintrc /app/
# Copy pyproject.toml files for proper isort and black configuration
# Duplicated to ensure tools can find configs regardless of working directory
COPY pyproject.toml ./pyproject.toml
COPY pyproject.toml /app/pyproject.toml

# Python environment configuration to prevent cache issues
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    PYTHONUTF8=1

# Create a non-root user that will be overridden by docker-compose
RUN useradd -m -u 1000 ciuser

# Copy security hooks and set up universal gh alias
COPY automation/security /app/security
RUN chmod +x /app/security/*.sh && \
    echo 'alias gh="/app/security/gh-wrapper.sh"' >> /etc/bash.bashrc

# Default command
CMD ["bash"]
