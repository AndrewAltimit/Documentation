---
layout: docs
title: Docker Fundamentals
section: technology
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Docker Fundamentals</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Containerization essentials for application development and deployment</p>
</div>

## Overview

Docker is a platform for developing, shipping, and running applications using containerization technology. It packages applications and their dependencies into portable containers that can run consistently across different computing environments.

## Core Concepts

### Containers
A container is a lightweight, standalone, executable package that includes everything needed to run an application: code, runtime, system tools, libraries, and settings. Containers are isolated from each other and the host system.

### Images
Docker images are read-only templates used to create containers. An image includes the application code, libraries, dependencies, tools, and other files needed for an application to run. Images are built from a set of instructions called a Dockerfile.

### Docker Engine
The Docker Engine is the core runtime that creates and manages containers. It consists of:
- **Docker Daemon**: Background service managing Docker objects
- **Docker Client**: Command-line interface for interacting with the daemon
- **REST API**: Interface for programmatic access

### Registry
A Docker registry stores Docker images. Docker Hub is the default public registry. Private registries can be hosted for proprietary images.

## Architecture

Docker uses a client-server architecture:

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Client    │────▶│  Docker Daemon  │────▶│   Registry   │
│  (docker)   │     │    (dockerd)    │     │ (Docker Hub) │
└─────────────┘     └─────────────────┘     └──────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Containers   │
                    │    Images     │
                    │   Networks    │
                    │   Volumes     │
                    └───────────────┘
```

## Basic Commands

### Container Management
```bash
docker run <image>              # Create and start container
docker ps                       # List running containers
docker ps -a                    # List all containers
docker stop <container>         # Stop running container
docker start <container>        # Start stopped container
docker rm <container>           # Remove container
docker logs <container>         # View container logs
docker exec -it <container> sh  # Execute command in container
```

### Image Management
```bash
docker images                   # List local images
docker pull <image>             # Download image from registry
docker build -t <tag> .         # Build image from Dockerfile
docker push <image>             # Upload image to registry
docker rmi <image>              # Remove local image
docker tag <source> <target>    # Create image tag
```

### Network and Volume Commands
```bash
docker network ls               # List networks
docker network create <name>    # Create network
docker volume ls                # List volumes
docker volume create <name>     # Create volume
```

## Dockerfile

A Dockerfile is a text file containing instructions for building a Docker image:

```dockerfile
# Base image
FROM node:14-alpine

# Set working directory
WORKDIR /app

# Copy dependency files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Define startup command
CMD ["node", "server.js"]
```

### Common Dockerfile Instructions
- `FROM`: Specify base image
- `WORKDIR`: Set working directory
- `COPY`/`ADD`: Copy files into image
- `RUN`: Execute commands during build
- `ENV`: Set environment variables
- `EXPOSE`: Document exposed ports
- `CMD`: Default command for container
- `ENTRYPOINT`: Configure container executable

## Docker Compose

Docker Compose is a tool for defining and running multi-container applications:

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
      
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=secret

volumes:
  postgres_data:
```

### Compose Commands
```bash
docker-compose up              # Start services
docker-compose down            # Stop and remove services
docker-compose ps              # List services
docker-compose logs            # View service logs
docker-compose build           # Build service images
```

## Networking

Docker provides several network drivers:

### Bridge (default)
- Containers connected to the same bridge network can communicate
- Provides isolation between different bridge networks

### Host
- Container uses host's network directly
- No network isolation between container and host

### None
- Container has no network access
- Complete network isolation

### Custom Networks
```bash
docker network create myapp-network
docker run --network myapp-network myapp
```

## Storage

### Volumes
Managed by Docker, stored in Docker's storage directory:
```bash
docker run -v myvolume:/data myapp
```

### Bind Mounts
Map host directory to container:
```bash
docker run -v /host/path:/container/path myapp
```

### tmpfs Mounts
Store data in host memory:
```bash
docker run --tmpfs /tmp myapp
```

## Best Practices

### Image Building
- Use specific base image tags
- Minimize layers by combining RUN commands
- Order Dockerfile instructions from least to most frequently changing
- Use .dockerignore to exclude unnecessary files
- Don't run containers as root when possible

### Security
- Scan images for vulnerabilities
- Use minimal base images (alpine, distroless)
- Don't store secrets in images
- Keep Docker and base images updated
- Use read-only containers when possible

### Resource Management
```bash
docker run --memory="256m" --cpus="1.0" myapp
```

## Container Orchestration

For production deployments, container orchestration platforms manage multiple containers:

- **Kubernetes**: Industry standard for container orchestration
- **Docker Swarm**: Docker's native clustering solution
- **Amazon ECS**: AWS container management service
- **Google GKE**: Google's managed Kubernetes service

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)