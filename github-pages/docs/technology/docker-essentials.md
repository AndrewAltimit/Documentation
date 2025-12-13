---
layout: docs
title: "Docker Essentials"
---

# Docker Essentials

Docker is a platform for developing, shipping, and running applications using containerization technology. It packages applications and their dependencies into portable containers that can run consistently across different computing environments.

## Architecture Overview

Docker uses a client-server architecture:
- **Docker Client**: CLI tool that sends commands to the daemon
- **Docker Daemon**: Background service managing containers, images, networks, and volumes
- **Docker Registry**: Stores Docker images (Docker Hub is the default public registry)

## Core Concepts

### Images
Immutable templates containing application code, runtime, libraries, and dependencies. Images are built from Dockerfiles and stored in registries.

### Containers
Running instances of Docker images. Containers are isolated processes with their own filesystem, networking, and process tree.

### Dockerfile
Text document containing instructions to build a Docker image. Each instruction creates a layer in the image.

### Registry
Repository for storing and distributing Docker images. Docker Hub is the default public registry, but private registries can be configured.

## Installation

### Linux
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

### macOS and Windows
Download Docker Desktop from the official Docker website. Docker Desktop includes Docker Engine, CLI, and additional tools.

## Image Management

### Pulling Images
```bash
docker pull <image>                  # Pull latest version
docker pull <image>:<tag>            # Pull specific version
docker pull <registry>/<image>:<tag> # Pull from custom registry
```

### Listing Images
{% raw %}
```bash
docker images                        # List all images
docker images -a                     # Include intermediate images
docker image ls --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```
{% endraw %}

### Removing Images
```bash
docker rmi <image>                   # Remove image
docker rmi $(docker images -q)       # Remove all images
docker image prune                   # Remove unused images
docker image prune -a                # Remove all unused images
```

## Container Operations

### Running Containers
```bash
docker run <image>                   # Run container
docker run -d <image>                # Run in detached mode
docker run -it <image> /bin/bash     # Interactive mode with bash
docker run --name <name> <image>     # Named container
docker run -p 8080:80 <image>        # Port mapping
docker run -v /host:/container <image> # Volume mounting
docker run --rm <image>              # Remove after exit
```

### Container Management
```bash
docker ps                            # List running containers
docker ps -a                         # List all containers
docker start <container>             # Start stopped container
docker stop <container>              # Stop running container
docker restart <container>           # Restart container
docker rm <container>                # Remove stopped container
docker rm -f <container>             # Force remove running container
```

### Container Inspection
```bash
docker logs <container>              # View container logs
docker logs -f <container>           # Follow log output
docker exec -it <container> /bin/bash # Execute command in container
docker inspect <container>           # Detailed container information
docker stats                         # Real-time resource usage
docker top <container>               # Running processes in container
```

## Dockerfile Reference

### Common Instructions
```dockerfile
# Base image
FROM ubuntu:24.04

# Metadata
LABEL maintainer="email@example.com"
LABEL version="1.0"

# Environment variables
ENV NODE_ENV=production
ENV APP_PORT=3000

# Working directory
WORKDIR /app

# Copy files
COPY package*.json ./
COPY . .

# Add files (supports URLs and tar extraction)
ADD https://example.com/file.tar.gz /tmp/

# Run commands during build
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# Expose ports
EXPOSE 3000

# Default command
CMD ["node", "app.js"]

# Entrypoint (not overridable)
ENTRYPOINT ["python"]

# User context
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:3000/health || exit 1
```

### Multi-stage Builds
```dockerfile
# Build stage
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

## Networking

### Network Types
- **bridge**: Default network for containers
- **host**: Container uses host networking
- **none**: No networking
- **overlay**: Multi-host networking for Swarm

### Network Commands
```bash
docker network ls                    # List networks
docker network create <network>      # Create network
docker network inspect <network>     # Inspect network
docker network rm <network>          # Remove network
docker run --network <network> <image> # Connect container to network
docker network connect <network> <container> # Connect running container
```

## Volumes and Storage

### Volume Management
```bash
docker volume create <volume>        # Create named volume
docker volume ls                     # List volumes
docker volume inspect <volume>       # Inspect volume
docker volume rm <volume>            # Remove volume
docker volume prune                  # Remove unused volumes
```

### Mount Types
```bash
# Named volume
docker run -v myvolume:/data <image>

# Bind mount
docker run -v /host/path:/container/path <image>

# tmpfs mount
docker run --tmpfs /tmp <image>
```

## Docker Compose

### Basic docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - db
    volumes:
      - ./app:/app
      - node_modules:/app/node_modules

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
  node_modules:
```

### Compose Commands
```bash
docker-compose up                    # Start services
docker-compose up -d                 # Start in detached mode
docker-compose down                  # Stop and remove
docker-compose ps                    # List services
docker-compose logs                  # View logs
docker-compose exec <service> bash   # Execute command
docker-compose build                 # Build/rebuild services
```

## Best Practices

### Image Optimization
1. Use specific base image tags, not `latest`
2. Minimize layers by combining RUN commands
3. Order Dockerfile instructions from least to most frequently changing
4. Use multi-stage builds to reduce final image size
5. Clean up package manager caches in the same RUN instruction

### Security
1. Run containers as non-root user
2. Scan images for vulnerabilities
3. Use official base images when possible
4. Don't store secrets in images
5. Keep images up to date

### Performance
1. Use .dockerignore to exclude unnecessary files
2. Leverage build cache effectively
3. Use appropriate base images (alpine for smaller size)
4. Limit container resources when necessary

## Troubleshooting

### Common Issues

**Container exits immediately**
```bash
docker logs <container>              # Check logs for errors
docker run -it <image> /bin/sh       # Debug with shell
```

**Permission denied errors**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in
```

**Disk space issues**
```bash
docker system df                     # Check Docker disk usage
docker system prune -a               # Clean up everything unused
```

**Network connectivity**
```bash
docker network inspect bridge        # Check network configuration
docker exec <container> ping google.com # Test from container
```

## Advanced Topics

### Resource Limits
```bash
docker run -m 512m --cpus="1.5" <image> # Memory and CPU limits
docker update --memory 1g <container>    # Update running container
```

### Docker in Docker
```bash
docker run -v /var/run/docker.sock:/var/run/docker.sock docker:dind
```

### Remote Docker Daemon
```bash
export DOCKER_HOST=tcp://remote-host:2375
docker ps  # Commands now execute on remote host
```

## Docker Updates

### New Features
- **Docker Scout**: Integrated vulnerability scanning and SBOM analysis
- **Docker Build Cloud**: Remote builders for faster CI/CD pipelines
- **Docker Debug**: Interactive debugging for containers
- **Compose Watch**: Auto-sync for development workflows

### Security Enhancements
- **Attestations**: Supply chain security with provenance
- **Rootless mode**: Run Docker daemon without root privileges
- **Enhanced build secrets**: Improved secret handling during builds

### Alternative Tools
- **Podman**: Daemonless, rootless container engine
- **Colima**: Lightweight Docker Desktop alternative for macOS
- **Rancher Desktop**: Kubernetes-focused container management
- **OrbStack**: Fast, efficient Docker Desktop alternative

## Related Docker Documentation

- [Containers - Complete Guide](docker.html) - In-depth container concepts, networking, and orchestration
- [Kubernetes](kubernetes.html) - Container orchestration at scale
- [CI/CD Pipelines](ci-cd.html) - Docker in continuous integration workflows

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Docker Scout](https://docs.docker.com/scout/)
- [Docker Build Cloud](https://docs.docker.com/build/cloud/)