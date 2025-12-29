---
layout: docs
title: "Docker: Dockerfiles & CI/CD"
permalink: /docs/technology/docker/dockerfiles.html
toc: true
toc_sticky: true
---

## Writing Dockerfiles

A Dockerfile is a recipe for building a Docker image. It defines exactly what goes into your container: the operating system, libraries, configuration, and application code. When you share a Dockerfile, anyone can reproduce your exact environment.

### Why Dockerfiles Matter

Consider the following benefits:

- **Reproducibility**: The same Dockerfile always produces the same image (given the same build context)
- **Documentation**: The Dockerfile serves as documentation for your environment
- **Automation**: CI/CD pipelines can automatically build images from Dockerfiles
- **Version control**: Track environment changes alongside code changes

### Your First Dockerfile

A Dockerfile consists of instructions, each creating a layer in the final image. Here is a minimal but complete example:

```dockerfile
FROM python:3.12-slim    # Start from an official base image
WORKDIR /app             # Set working directory
COPY requirements.txt .  # Copy dependency file first (caching)
RUN pip install -r requirements.txt  # Install dependencies
COPY . .                 # Copy application code
CMD ["python", "app.py"] # Default command when container starts
```

Build and run this with:
```bash
docker build -t my-app .
docker run -p 8080:80 my-app
```

### Dockerfile Instructions Reference

Each instruction serves a specific purpose. Here is a quick reference:

| Instruction | Purpose | Example |
|-------------|---------|---------|
| FROM | Base image to start from | `FROM python:3.12-slim` |
| WORKDIR | Set working directory | `WORKDIR /app` |
| COPY | Copy files from host to image | `COPY . /app` |
| RUN | Execute command during build | `RUN pip install -r requirements.txt` |
| CMD | Default command when container starts | `CMD ["python", "app.py"]` |
| ENTRYPOINT | Configure container as executable | `ENTRYPOINT ["./start.sh"]` |
| EXPOSE | Document which port the app uses | `EXPOSE 8080` |
| ENV | Set environment variable | `ENV NODE_ENV=production` |
| ARG | Build-time variable | `ARG VERSION=1.0` |
| USER | Run as specific user | `USER appuser` |

#### CMD vs ENTRYPOINT

These two instructions are often confused. Here is when to use each:

- **CMD**: Use when you want to provide defaults that can be easily overridden
- **ENTRYPOINT**: Use when the container should always run a specific executable

```dockerfile
# CMD: user can override with "docker run my-app /bin/bash"
CMD ["python", "app.py"]

# ENTRYPOINT: container always runs this, CMD provides default arguments
ENTRYPOINT ["python"]
CMD ["app.py"]  # User can override: docker run my-app other.py
```

#### COPY vs ADD

Prefer **COPY** unless you specifically need ADD's features:

- **COPY**: Simple file copy (recommended)
- **ADD**: Also extracts tar files and downloads URLs (use sparingly)

### Multistage Builds

Multi-stage builds solve a common problem: build tools make images large. Your Go compiler, Node.js build tools, or Java SDK add hundreds of megabytes that are not needed at runtime.

**The solution**: Use one stage to build, another to run. The final image only contains what is needed to run your application.

```dockerfile
# Stage 1: Build (large image with all build tools)
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Run (small image with only runtime)
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
```

**Result**: Final image is ~25MB instead of ~1GB.

| Scenario | Without Multi-stage | With Multi-stage |
|----------|---------------------|------------------|
| Node.js app | ~1 GB | ~100 MB |
| Go binary | ~800 MB | ~10 MB |
| Java app | ~500 MB | ~200 MB |

## Best Practices

These practices will make your images smaller, more secure, and faster to build.

### Image Size and Security

| Practice | Why It Matters |
|----------|----------------|
| Use minimal base images (alpine, slim) | Smaller attack surface, faster pulls |
| Run as non-root user | Prevents container escape attacks |
| Use specific version tags | Avoids surprise breaking changes |
| Scan images for vulnerabilities | Catches known security issues |

### Build Speed

Order your Dockerfile to maximize cache hits. Put things that change rarely at the top:

```dockerfile
# Good: dependencies change less often than code
COPY package.json package-lock.json ./
RUN npm ci
COPY . .  # Code changes invalidate only this layer

# Bad: any code change reinstalls all dependencies
COPY . .
RUN npm ci
```

### Layer Optimization

Combine related commands to reduce layers and image size:

```dockerfile
# Good: single layer, cleanup in same layer
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Bad: cleanup in separate layer does not reduce size
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*
```

### Essential .dockerignore

Always create a `.dockerignore` file to exclude unnecessary files:

```
node_modules
.git
*.log
.env
```

## Performance Optimization

Container performance depends on resource allocation, storage configuration, and network setup. Here are the key levers you can adjust.

### Resource Limits

Always set resource limits in production to prevent runaway containers from affecting other services.

```bash
# Memory: set limit and reservation
docker run -d --memory="1g" --memory-reservation="750m" my-app

# CPU: limit to 2 cores
docker run -d --cpus="2" my-app
```

### Build Performance

Use BuildKit cache mounts to dramatically speed up builds:

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.20 AS builder
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o app .
```

### Network Performance

| Need | Solution |
|------|----------|
| Maximum performance | `--network host` (no isolation) |
| Good performance + isolation | Custom bridge network |
| Multi-host communication | Overlay network |

### Monitoring

```bash
# Real-time resource usage
docker stats

# Container-specific metrics
docker stats container-name --no-stream
```

For production monitoring, consider cAdvisor with Prometheus for detailed metrics and alerting.

## Docker Swarm: Native Orchestration

Docker Swarm turns multiple Docker hosts into a single cluster. It handles service deployment, scaling, and rolling updates with built-in load balancing.

**When to use Swarm vs Kubernetes:**

| Factor | Docker Swarm | Kubernetes |
|--------|--------------|------------|
| Complexity | Simple, quick to learn | Complex, steep learning curve |
| Setup time | Minutes | Hours to days |
| Scalability | Good for small/medium | Excellent for large scale |
| Feature set | Essential features | Comprehensive ecosystem |
| Best for | Small teams, simpler apps | Large teams, complex apps |

### Quick Start

```bash
# Initialize swarm on first manager
docker swarm init

# Join workers (run this on each worker node)
docker swarm join --token <token> <manager-ip>:2377

# Deploy a service with 3 replicas
docker service create --name web --replicas 3 -p 80:80 nginx

# Scale up
docker service scale web=5
```

### Stack Deployment

For multi-service applications, use stack files (compose format):

```yaml
# stack.yml
version: '3.8'
services:
  web:
    image: nginx:alpine
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
    ports:
      - "80:80"
  api:
    image: my-api:latest
    deploy:
      replicas: 2
```

```bash
# Deploy and manage stack
docker stack deploy -c stack.yml myapp
docker stack services myapp
docker stack rm myapp
```

### High Availability Tips

- Use an **odd number of managers** (3, 5, or 7) for quorum
- Distribute managers across availability zones
- Use node labels and constraints for placement control

## CI/CD Integration with Docker

Docker enables consistent builds across all CI/CD platforms. The pattern is always the same: build image, test, scan for vulnerabilities, push to registry, deploy.

### GitHub Actions

The most common approach for GitHub-hosted projects:

```yaml
# .github/workflows/docker.yml
name: Docker Build
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: docker/setup-buildx-action@v3
    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    - uses: docker/build-push-action@v5
      with:
        push: true
        tags: ghcr.io/${{ github.repository }}:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
build:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

### Key CI/CD Practices

| Practice | Why |
|----------|-----|
| Cache layers | Faster builds |
| Scan images | Catch vulnerabilities before deployment |
| Tag with commit SHA | Traceable deployments |
| Use multi-stage builds | Smaller production images |
| Avoid `latest` tag in production | Reproducible deployments |

