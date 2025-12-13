# Self-Hosted Runner Setup Guide

This guide documents the setup process for self-hosted GitHub Actions runners used in this container-first project. Updated for 2024 with the latest runner version and best practices.

## Philosophy

This project uses **self-hosted runners exclusively** to:

- Maintain zero infrastructure costs
- Have full control over the build environment
- Enable caching of Docker images for faster builds
- Support the container-first approach without cloud limitations

## System Requirements

### Operating System

- **Tested on**: Zorin OS 17, Ubuntu 22.04 LTS, Ubuntu 24.04 LTS
- **Supported**: Ubuntu 20.04+, Debian 11+, RHEL 8+, or compatible Linux distributions
- **Architecture**: x86_64 (AMD64) or ARM64

### Required Software

1. **Docker** (v24.0+ recommended, minimum v20.10)

   ```bash
   # Install Docker (latest stable)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add user to docker group
   sudo usermod -aG docker $USER
   newgrp docker  # Apply group changes without logout
   
   # Verify installation
   docker --version
   docker run hello-world
   ```

2. **Docker Compose** (v2.20+ recommended)

   ```bash
   # Docker Compose v2 is included with Docker Desktop
   # For standalone installation:
   sudo apt-get update
   sudo apt-get install docker-compose-plugin
   
   # Verify installation
   docker compose version
   
   # Enable BuildKit by default
   echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
   echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Git** (v2.25+)

   ```bash
   sudo apt-get update
   sudo apt-get install git
   ```

4. **Node.js** (v22.16.0 or latest LTS) via nvm

   ```bash
   # Install nvm (check for latest version)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
   source ~/.bashrc

   # Install Node.js
   nvm install --lts  # Install latest LTS
   nvm install 22.16.0  # Or specific version
   nvm use 22.16.0
   nvm alias default 22.16.0
   ```

5. **Gemini CLI** (pre-authenticated)

   ```bash
   # Install Gemini CLI
   npm install -g @google/gemini-cli

   # Authenticate (happens automatically on first use)
   gemini
   ```

   Note: Gemini CLI cannot be containerized as it may need to invoke Docker commands.

6. **Python** (v3.11+ recommended)

   ```bash
   # Only needed for running helper scripts
   # All Python CI/CD operations run in containers
   python3 --version
   
   # Install Python 3.11+ if needed
   sudo apt-get update
   sudo apt-get install python3.11 python3.11-venv python3-pip
   
   # Set as default (optional)
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
   ```

## GitHub Actions Runner Installation

1. **Download and Configure Runner**

   ```bash
   # Create a directory for the runner
   mkdir ~/actions-runner && cd ~/actions-runner

   # Download the latest runner package (v2.327.0 as of 2024)
   # IMPORTANT: Always check https://github.com/actions/runner/releases for the latest version
   RUNNER_VERSION="2.327.0"
   curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L \
     https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

   # Verify checksum (recommended)
   curl -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz.sha256 \
     -o checksum.sha256
   sha256sum -c checksum.sha256

   # Extract the installer
   tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
   rm ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
   ```

2. **Configure the Runner**

   ```bash
   # Run the configuration script
   # Note: Get your token from GitHub: Settings > Actions > Runners > New self-hosted runner
   ./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO \
     --token YOUR_TOKEN \
     --name "$(hostname)-runner" \
     --labels "self-hosted,Linux,X64,docker" \
     --work "_work"
   ```

3. **Install as a Service**

   ```bash
   # Install the runner as a service (runs as current user)
   sudo ./svc.sh install
   
   # Configure service to start on boot
   sudo systemctl enable actions.runner.YOUR_ORG-YOUR_REPO.$(hostname)-runner.service
   
   # Start the service
   sudo ./svc.sh start

   # Check service status
   sudo ./svc.sh status
   
   # View logs
   sudo journalctl -u actions.runner.YOUR_ORG-YOUR_REPO.$(hostname)-runner.service -f
   ```

## Environment Configuration

### User Permissions

The CI/CD pipeline runs Docker containers with the current user's UID/GID to avoid permission issues:

```bash
# These are automatically set by the scripts
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
```

Note: We use `USER_ID` and `GROUP_ID` instead of `UID` and `GID` because `UID` is a readonly variable in some shells.

### Docker Configuration

Ensure Docker can be run without sudo:

```bash
# Test Docker access
docker run hello-world
```

### Disk Space Requirements

- **Minimum**: 20GB free space
- **Recommended**: 50GB+ free space
- **For CI/CD**: 100GB+ if building large images
- Docker images and build cache can consume significant space

```bash
# Check disk space
df -h

# Monitor Docker space usage
docker system df
```

### Network Requirements

- Access to Docker Hub for pulling base images
- Access to PyPI for Python packages
- Access to GitHub for cloning repositories

## Maintenance

### Regular Cleanup

To prevent disk space issues, regularly clean Docker resources:

```bash
# Automated cleanup script (save as ~/cleanup-runner.sh)
#!/bin/bash
set -euo pipefail

echo "Starting cleanup at $(date)"

# Remove unused containers, networks, images (keep last 7 days)
docker system prune -a --filter "until=168h" -f

# Remove unused volumes (interactive)
docker volume prune

# Clean builder cache
docker builder prune --keep-storage 10GB -f

# Check disk usage
echo "\nDisk usage after cleanup:"
docker system df
df -h /var/lib/docker

# Clean runner work directory (if safe)
find ~/actions-runner/_work -type d -name "_temp" -mtime +7 -exec rm -rf {} + 2>/dev/null || true

echo "Cleanup completed at $(date)"
```

```bash
# Make executable and add to cron
chmod +x ~/cleanup-runner.sh

# Add to crontab (runs daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /home/$USER/cleanup-runner.sh >> /home/$USER/cleanup-runner.log 2>&1") | crontab -
```

### Runner Updates

GitHub Actions runners can auto-update, but manual updates may be needed:

```bash
# Check current version
cd ~/actions-runner
./run.sh --version

# Update runner (automated script)
cat > ~/update-runner.sh << 'EOF'
#!/bin/bash
set -euo pipefail

RUNNER_DIR="$HOME/actions-runner"
cd "$RUNNER_DIR"

# Get latest version
LATEST=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep tag_name | cut -d '"' -f 4 | sed 's/v//')
CURRENT=$(./run.sh --version | grep -oP '\d+\.\d+\.\d+')

if [ "$LATEST" = "$CURRENT" ]; then
    echo "Already on latest version: $CURRENT"
    exit 0
fi

echo "Updating from $CURRENT to $LATEST"

# Stop service
sudo ./svc.sh stop

# Backup config
cp .runner .runner.backup
cp .credentials .credentials.backup
cp .credentials_rsaparams .credentials_rsaparams.backup 2>/dev/null || true

# Download new version
curl -o actions-runner-linux-x64-${LATEST}.tar.gz -L \
  https://github.com/actions/runner/releases/download/v${LATEST}/actions-runner-linux-x64-${LATEST}.tar.gz

# Extract (overwrites files)
tar xzf ./actions-runner-linux-x64-${LATEST}.tar.gz
rm ./actions-runner-linux-x64-${LATEST}.tar.gz

# Restore config
cp .runner.backup .runner
cp .credentials.backup .credentials
cp .credentials_rsaparams.backup .credentials_rsaparams 2>/dev/null || true

# Restart service
sudo ./svc.sh install
sudo ./svc.sh start

echo "Update completed successfully"
EOF

chmod +x ~/update-runner.sh

# Run update
~/update-runner.sh
```

## Troubleshooting

### Permission Issues

If you encounter permission errors:

1. Ensure your user is in the `docker` group
   ```bash
   groups | grep docker || sudo usermod -aG docker $USER
   ```
2. Verify the UID/GID environment variables are set correctly
   ```bash
   echo "USER_ID=$(id -u) GROUP_ID=$(id -g)"
   ```
3. Check that the workspace directory has proper permissions
   ```bash
   ls -la ~/actions-runner/_work
   ```
4. Use the fix script: `./scripts/fix-runner-permissions.sh`

**Python Cache Prevention:**
The CI/CD system prevents Python cache issues by:

- Setting `PYTHONDONTWRITEBYTECODE=1` in all containers
- Setting `PYTHONPYCACHEPREFIX=/tmp/pycache` to redirect cache
- Disabling pytest cache via `pytest.ini` configuration (`-p no:cacheprovider`)
- Running containers with proper user permissions (USER_ID:GROUP_ID)
- Using Python 3.11+ slim base image for consistency
- Adding `__pycache__` to `.dockerignore`

### Docker Build Failures

1. Check available disk space: `df -h`
2. Clean Docker cache: `docker builder prune`
3. Verify network connectivity to Docker Hub

### Runner Connection Issues

1. Check runner service logs:
   ```bash
   sudo journalctl -u actions.runner.* -f
   # Or find exact service name
   systemctl list-units | grep actions.runner
   ```
2. Verify GitHub token hasn't expired (tokens expire after 1 hour)
3. Ensure firewall allows outbound HTTPS connections
   ```bash
   sudo ufw status
   # If needed: sudo ufw allow out 443/tcp
   ```
4. Test GitHub connectivity:
   ```bash
   curl -I https://api.github.com
   ```

### Container-Specific Issues

1. **"Cannot connect to Docker daemon"**:
   ```bash
   # Ensure user is in docker group
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Permission denied on files created by containers**:
   ```bash
   # Use the fix script
   ./scripts/fix-runner-permissions.sh
   ```

3. **Out of space errors**:
   ```bash
   # Check Docker space usage
   docker system df
   
   # Check disk space
   df -h /var/lib/docker
   
   # Aggressive cleanup (warning: removes all unused images)
   docker system prune -a --volumes -f
   
   # Move Docker root (if needed)
   # Edit /etc/docker/daemon.json:
   # { "data-root": "/path/to/new/docker/root" }
   # Then: sudo systemctl restart docker
   ```

## Security Considerations

1. **Never run runners with root privileges**
2. **Use dedicated runner machines** when possible
3. **Regularly update all software** components
4. **Monitor runner logs** for suspicious activity
5. **Restrict repository access** to trusted contributors only

## Performance Optimization

1. **Enable Docker BuildKit**:
   ```bash
   echo '{"features": {"buildkit": true}}' | sudo tee /etc/docker/daemon.json
   sudo systemctl restart docker
   ```

2. **Configure runner for performance**:
   ```bash
   # Increase runner work folder cleanup threshold
   echo "RUNNER_WORK_FOLDER_CLEANUP_THRESHOLD=50" >> ~/actions-runner/.env
   ```

3. **Use local registry mirror** (optional):
   ```bash
   # For faster image pulls in corporate environments
   sudo mkdir -p /etc/docker
   echo '{
     "registry-mirrors": ["https://mirror.gcr.io"]
   }' | sudo tee /etc/docker/daemon.json
   ```

## Additional Resources

- [GitHub Actions Self-Hosted Runners Documentation](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Runner Releases](https://github.com/actions/runner/releases)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Linux Security Hardening Guide](https://www.debian.org/doc/manuals/securing-debian-manual/)

## Related Documentation

- [CONTAINERIZED_CI.md](CONTAINERIZED_CI.md) - Container-first CI/CD approach
- [AI_AGENTS.md](AI_AGENTS.md) - AI agent collaboration
- [GEMINI_SETUP.md](GEMINI_SETUP.md) - Gemini CLI setup for PR reviews
