---
layout: docs
title: Docker Essentials
description: Quick reference guide for essential Docker commands and operations
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #00aaff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Docker Essentials</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Quick reference for container commands and operations</p>
</div>

<div class="intro-card">
  <p class="lead-text">This quick reference covers the most commonly used Docker commands for daily development work. For comprehensive explanations and deeper concepts, see our <a href="docker/">full Docker documentation</a>.</p>
</div>

## Container Lifecycle

### Running Containers

```bash
# Run a container from an image
docker run <image>

# Run in detached mode (background)
docker run -d <image>

# Run with interactive terminal
docker run -it <image> /bin/bash

# Run with port mapping (host:container)
docker run -p 8080:80 <image>

# Run with volume mount
docker run -v /host/path:/container/path <image>

# Run with environment variables
docker run -e "ENV_VAR=value" <image>

# Run with automatic removal when stopped
docker run --rm <image>

# Run with custom name
docker run --name my-container <image>
```

### Managing Containers

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a running container
docker stop <container>

# Start a stopped container
docker start <container>

# Restart a container
docker restart <container>

# Remove a container
docker rm <container>

# Remove all stopped containers
docker container prune

# Force remove running container
docker rm -f <container>
```

### Interacting with Containers

```bash
# Execute command in running container
docker exec <container> <command>

# Open interactive shell in container
docker exec -it <container> /bin/bash

# View container logs
docker logs <container>

# Follow logs in real-time
docker logs -f <container>

# Show last N lines of logs
docker logs --tail 100 <container>

# Copy files to/from container
docker cp <container>:/path/to/file /local/path
docker cp /local/file <container>:/path/to/file
```

## Image Management

### Working with Images

```bash
# List local images
docker images

# Pull image from registry
docker pull <image>:<tag>

# Build image from Dockerfile
docker build -t <name>:<tag> .

# Build with no cache
docker build --no-cache -t <name>:<tag> .

# Tag an image
docker tag <image> <new-name>:<tag>

# Push image to registry
docker push <image>:<tag>

# Remove an image
docker rmi <image>

# Remove unused images
docker image prune

# Remove all unused images
docker image prune -a
```

### Inspecting Images

```bash
# Show image details
docker inspect <image>

# Show image history/layers
docker history <image>

# Search Docker Hub
docker search <term>
```

## Docker Compose

### Basic Operations

```bash
# Start services defined in docker-compose.yml
docker-compose up

# Start in detached mode
docker-compose up -d

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View service logs
docker-compose logs

# Follow logs
docker-compose logs -f

# List running services
docker-compose ps
```

### Service Management

```bash
# Build or rebuild services
docker-compose build

# Force rebuild without cache
docker-compose build --no-cache

# Scale a service
docker-compose up -d --scale web=3

# Execute command in service
docker-compose exec <service> <command>

# Run one-off command
docker-compose run <service> <command>
```

## Networking

```bash
# List networks
docker network ls

# Create a network
docker network create <name>

# Connect container to network
docker network connect <network> <container>

# Disconnect from network
docker network disconnect <network> <container>

# Inspect network
docker network inspect <network>

# Remove network
docker network rm <network>
```

## Volumes

```bash
# List volumes
docker volume ls

# Create a volume
docker volume create <name>

# Inspect volume
docker volume inspect <name>

# Remove volume
docker volume rm <name>

# Remove unused volumes
docker volume prune
```

## System Maintenance

```bash
# Show disk usage
docker system df

# Show detailed disk usage
docker system df -v

# Remove all unused resources
docker system prune

# Remove everything including volumes
docker system prune -a --volumes

# Show system-wide information
docker info

# Show Docker version
docker version
```

## Debugging & Troubleshooting

```bash
# View container resource usage
docker stats

# View resource usage for specific containers
docker stats <container1> <container2>

# Inspect container details
docker inspect <container>

# View container processes
docker top <container>

# Show container port mappings
docker port <container>

# View container changes (filesystem diff)
docker diff <container>
```

## Common Patterns

### Development Environment

```bash
# Run with live code reload (mount source directory)
docker run -v $(pwd):/app -w /app node:18 npm run dev

# Run database for development
docker run -d \
  --name postgres-dev \
  -e POSTGRES_PASSWORD=devpass \
  -p 5432:5432 \
  postgres:15
```

### Quick Testing

```bash
# Run temporary container for testing
docker run --rm -it alpine sh

# Test network connectivity from container
docker run --rm alpine ping -c 4 google.com

# Quick Python environment
docker run --rm -it -v $(pwd):/work -w /work python:3.11 python
```

### Cleanup Commands

```bash
# Remove all stopped containers, unused networks, and dangling images
docker system prune

# Full cleanup (includes unused images and volumes)
docker system prune -a --volumes

# Remove containers older than 24h
docker container prune --filter "until=24h"
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Run container | `docker run <image>` |
| Run interactive | `docker run -it <image> bash` |
| List containers | `docker ps -a` |
| Stop container | `docker stop <container>` |
| Remove container | `docker rm <container>` |
| View logs | `docker logs <container>` |
| Execute command | `docker exec -it <container> bash` |
| List images | `docker images` |
| Build image | `docker build -t <name> .` |
| Pull image | `docker pull <image>` |
| Compose up | `docker-compose up -d` |
| Compose down | `docker-compose down` |
| System cleanup | `docker system prune -a` |

---

## Related Documentation

- **[Docker Fundamentals](docker/fundamentals.html)** - Core concepts and architecture explained
- **[Docker Storage & Security](docker/storage-security.html)** - Volumes, networking, and security best practices
- **[Dockerfiles Guide](docker/dockerfiles.html)** - Building custom images
- **[Advanced Docker](docker/advanced.html)** - Multi-stage builds, optimization, and orchestration
- **[Kubernetes](kubernetes/)** - Container orchestration at scale
- **[CI/CD](ci-cd.html)** - Automating Docker workflows in pipelines
