---
layout: docs
title: "Docker: Dockerfiles & CI/CD"
permalink: /docs/technology/docker/dockerfiles.html
toc: true
toc_sticky: true
---

## Writing Dockerfiles

A Dockerfile is a script containing instructions to build a Docker image. It automates the process of creating a container by specifying the base image, configuration, application code, and dependencies. This documentation will cover the basics of writing a Dockerfile, its syntax, and using multistage builds.

### Dockerfile Example

A Dockerfile consists of a series of instructions, each starting with an uppercase keyword followed by arguments. The instructions are executed in the order they appear, and each instruction creates a new layer in the Docker image. Comments can be added using the # symbol.

Here's a simple example:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

### Dockerfile Instructions

#### FROM

The **FROM** instruction sets the base image for your Dockerfile. It must be the first instruction in the file. You can use an official image from the Docker Hub or a custom image.

```dockerfile
FROM <image>[:<tag>] [AS <name>]
```

#### WORKDIR

The **WORKDIR** instruction sets the working directory for any subsequent **RUN**, **CMD**, **ENTRYPOINT**, **COPY**, and **ADD** instructions. If the directory does not exist, it will be created.

```dockerfile
WORKDIR <path>
```

#### COPY

The **COPY** instruction copies files or directories from the local filesystem to the container's filesystem.

```dockerfile
COPY <src> <dest>
```

#### ADD

The **ADD** instruction is similar to **COPY**, but it can also download remote files and extract compressed files.

```dockerfile
ADD <src> <dest>
```

#### RUN

The **RUN** instruction executes a command during the build process, creating a new layer.

```dockerfile
RUN <command>
```

#### CMD

The **CMD** instruction provides the default command that will be executed when running a container.

```dockerfile
CMD ["executable", "param1", "param2"]
```

#### ENTRYPOINT

The **ENTRYPOINT** instruction allows you to configure a container that will run as an executable.

```dockerfile
ENTRYPOINT ["executable", "param1", "param2"]
```

#### EXPOSE

The **EXPOSE** instruction informs Docker that the container listens on the specified network ports at runtime.

```dockerfile
EXPOSE <port> [<port>/<protocol>...]
```

#### ENV

The **ENV** instruction sets an environment variable.

```dockerfile
ENV <key>=<value> ...
```

#### ARG

The **ARG** instruction defines a variable that can be passed to the build process using the `--build-arg` flag.

```dockerfile
ARG <name>[=<default value>]
```

### Multistage Builds

Multistage builds allow you to optimize the Dockerfile by using multiple FROM instructions, each with a unique name. This is useful when you need to use multiple images or want to reduce the final image size.

Here's an example:

```dockerfile
# Stage 1: Build the application
FROM node:14 AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Create the final image
FROM nginx:1.19-alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

In this example, the first stage uses the node:14 image to build the application, and the second stage uses the nginx:1.19-alpine image to serve the application. The `COPY --from=build` command copies the built application files from the first stage to the final image. This results in a smaller final image without the build dependencies.

## Best Practices

- Use official base images: Official images are maintained and optimized by the creators of the respective software.
- Be specific with base image tags: Specify an exact version or use a specific tag to avoid breaking changes in the future.
- Keep layers to a minimum: Group related commands together and use a single RUN instruction whenever possible.
- Use .dockerignore file: Exclude unnecessary files from the build context to reduce build time and prevent sensitive data from being included in the image.
- Cache dependencies: Copy dependency files separately from the application code to take advantage of Docker's build cache and avoid unnecessary re-installations.
- Use multi-stage builds: Multi-stage builds can help reduce the final image size by only including the necessary files for the runtime environment.
- Run as non-root user: Always specify a non-root user in your Dockerfile for security.
- Order layers efficiently: Put commands that change less frequently (like package installation) before frequently changing items (like application code).
- Use COPY instead of ADD: Unless you need ADD's tar extraction or URL features, COPY is more predictable.
- Set WORKDIR early: Establish your working directory early to avoid using long paths.
- Combine RUN commands: Reduce layers by combining related commands with && operators.

## Performance Optimization

<div class="performance-section">
  <h3><i class="fas fa-tachometer-alt"></i> Docker Performance Tuning</h3>
  
  <p class="perf-intro">Optimizing Docker performance involves understanding resource allocation, storage drivers, networking overhead, and build optimization. These techniques can significantly improve your container performance.</p>
  
  <h3><i class="fas fa-memory"></i> Resource Management</h3>
  
  <div class="resource-optimization">
    <h4>Memory Optimization</h4>
    <pre><code class="language-bash"># Set memory limits and reservations
docker run -d \
  --name optimized-app \
  --memory="1g" \
  --memory-reservation="750m" \
  --memory-swap="2g" \
  --oom-kill-disable=false \
  my-app

# Java application optimization
docker run -d \
  --name java-app \
  --memory="2g" \
  -e JAVA_OPTS="-Xmx1800m -Xms1800m -XX:MaxMetaspaceSize=256m" \
  my-java-app</code></pre>
    
    <h4>CPU Optimization</h4>
    <pre><code class="language-bash"># CPU limits and shares
docker run -d \
  --name cpu-app \
  --cpus="2.5" \
  --cpu-shares=1024 \
  --cpuset-cpus="0-3" \
  my-app

# Real-time scheduling
docker run -d \
  --name realtime-app \
  --cpu-rt-runtime=950000 \
  --cpu-rt-period=1000000 \
  my-realtime-app</code></pre>
  </div>
  
  <h3><i class="fas fa-hdd"></i> Storage Driver Performance</h3>
  
  <div class="storage-performance">
    <h4>Choosing the Right Storage Driver</h4>
    <pre><code class="language-bash"># Check current storage driver
docker info | grep -i storage

# Configure storage driver (daemon.json)
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true",
    "overlay2.size=20G"
  ]
}

# Performance comparison
# overlay2: Best overall performance
# devicemapper: Good for high I/O workloads
# btrfs: Good for snapshots, CoW operations
# zfs: Best for data integrity, compression</code></pre>
    
    <h4>Build Cache Optimization</h4>
    <pre><code class="language-dockerfile"># Optimize Dockerfile for caching
FROM node:18-alpine

# Dependencies change less frequently
COPY package*.json ./
RUN npm ci --only=production

# Application code changes frequently
COPY . .

# BuildKit cache mounts
# syntax=docker/dockerfile:1
FROM golang:1.20-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go mod download</code></pre>
  </div>
  
  <h3><i class="fas fa-network-wired"></i> Network Performance</h3>
  
  <div class="network-performance">
    <h4>Network Driver Performance</h4>
    <pre><code class="language-bash"># Host networking for maximum performance
docker run -d --network host high-perf-app

# SR-IOV for near-native performance
docker run -d \
  --name sriov-app \
  --network sriov-net \
  --device=/dev/vfio/vfio \
  high-bandwidth-app

# Optimize bridge network
sudo sysctl -w net.bridge.bridge-nf-call-iptables=0
sudo sysctl -w net.bridge.bridge-nf-call-ip6tables=0</code></pre>
  </div>
  
  <h3><i class="fas fa-chart-line"></i> Monitoring and Profiling</h3>
  
  <div class="performance-monitoring">
    <h4>Container Profiling</h4>
    <pre><code class="language-bash"># CPU profiling with perf
docker run -d --name app --cap-add SYS_ADMIN my-app
docker exec app perf record -g -p 1 -- sleep 30
docker exec app perf report

# Memory profiling
docker stats --no-stream
docker exec app cat /proc/1/smaps | grep -E "^Rss|^Pss|^Shared"

# I/O profiling
docker exec app iotop -b -n 1</code></pre>
    
    <h4>Metrics Collection</h4>
    <pre><code class="language-yaml"># docker-compose.yml with Prometheus monitoring
version: '3.8'
services:
  app:
    image: my-app
    ports:
      - "8080:8080"
      
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8081:8080"
      
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"</code></pre>
  </div>
</div>

## Docker Swarm: Native Orchestration

<div class="swarm-section">
  <h3><i class="fas fa-th"></i> Introduction to Docker Swarm</h3>
  
  <p class="swarm-intro">Docker Swarm is Docker's native clustering and orchestration solution. It turns a pool of Docker hosts into a single, virtual Docker host, providing high availability and load balancing for your services.</p>
  
  <h3><i class="fas fa-server"></i> Setting Up a Swarm Cluster</h3>
  
  <div class="swarm-setup">
    <h4>Initialize Swarm</h4>
    <pre><code class="language-bash"># On manager node
docker swarm init --advertise-addr 192.168.1.100

# Output will include join token:
# docker swarm join --token SWMTKN-1-xxx 192.168.1.100:2377

# Get join tokens
docker swarm join-token worker
docker swarm join-token manager

# On worker nodes
docker swarm join --token SWMTKN-1-xxx 192.168.1.100:2377

# Verify cluster
docker node ls</code></pre>
    
    <h4>Deploy Services</h4>
    <pre><code class="language-bash"># Create a service
docker service create \
  --name web \
  --replicas 3 \
  --publish 80:80 \
  --mount type=volume,source=web-data,destination=/data \
  --constraint 'node.role==worker' \
  --update-delay 10s \
  --update-parallelism 2 \
  nginx:alpine

# Scale service
docker service scale web=5

# Rolling update
docker service update \
  --image nginx:latest \
  --update-failure-action rollback \
  web</code></pre>
    
    <h4>Stack Deployment</h4>
    <pre><code class="language-yaml"># stack.yml
version: '3.8'

services:
  web:
    image: nginx:alpine
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      placement:
        constraints:
          - node.role == worker
          - node.labels.zone == frontend
    ports:
      - "80:80"
    networks:
      - webnet
      
  api:
    image: my-api:latest
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.labels.zone == backend
    environment:
      - DB_HOST=db
    networks:
      - webnet
      - backend
      
  db:
    image: postgres:15
    deploy:
      placement:
        constraints:
          - node.labels.type == database
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - backend
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password

networks:
  webnet:
    driver: overlay
    encrypted: true
  backend:
    driver: overlay
    internal: true

volumes:
  db-data:
    driver: local

secrets:
  db_password:
    external: true</code></pre>
    <pre><code class="language-bash"># Deploy stack
docker stack deploy -c stack.yml myapp

# Monitor stack
docker stack services myapp
docker stack ps myapp

# Remove stack
docker stack rm myapp</code></pre>
  </div>
  
  <h3><i class="fas fa-balance-scale"></i> High Availability</h3>
  
  <div class="swarm-ha">
    <h4>Multi-Manager Setup</h4>
    <pre><code class="language-bash"># Best practice: odd number of managers (3, 5, 7)
# Promote workers to managers
docker node promote worker1 worker2

# Verify manager status
docker node ls

# Configure manager availability
docker node update --availability drain manager1</code></pre>
    
    <h4>Service Constraints and Preferences</h4>
    <pre><code class="language-bash"># Node labels
docker node update --label-add zone=frontend worker1
docker node update --label-add zone=backend worker2
docker node update --label-add type=database worker3

# Deploy with constraints
docker service create \
  --name frontend \
  --constraint 'node.labels.zone==frontend' \
  --placement-pref 'spread=node.labels.zone' \
  nginx</code></pre>
  </div>
</div>

## CI/CD Integration with Docker

<div class="cicd-section">
  <h3><i class="fas fa-infinity"></i> Docker in CI/CD Pipelines</h3>
  
  <p class="cicd-intro">Docker revolutionizes CI/CD by providing consistent environments across all pipeline stages. Here's how to integrate Docker into popular CI/CD platforms.</p>
  
  <h3><i class="fas fa-github"></i> GitHub Actions</h3>
  
  <div class="github-actions">
    <h4>Basic Docker Build and Push</h4>
    <pre><code class="language-yaml"># .github/workflows/docker.yml
name: Docker CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache
        cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache,mode=max</code></pre>
    
    <h4>Multi-Platform Builds</h4>
    <pre><code class="language-yaml"># Multi-architecture build
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v2
      
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64,linux/arm/v7
        push: true
        tags: ${{ steps.meta.outputs.tags }}</code></pre>
  </div>
  
  <h3><i class="fas fa-gitlab"></i> GitLab CI/CD</h3>
  
  <div class="gitlab-ci">
    <h4>GitLab Pipeline with Docker</h4>
    <pre><code class="language-yaml"># .gitlab-ci.yml
variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG

stages:
  - build
  - test
  - deploy

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build --pull -t $IMAGE_TAG .
    - docker push $IMAGE_TAG
    
test:
  stage: test
  image: $IMAGE_TAG
  script:
    - pytest tests/
    - flake8 .
    
scan:
  stage: test
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker pull $IMAGE_TAG
    - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        aquasec/trivy image --exit-code 1 --severity HIGH,CRITICAL $IMAGE_TAG
        
deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/app app=$IMAGE_TAG
  only:
    - main</code></pre>
  </div>
  
  <h3><i class="fas fa-jenkins"></i> Jenkins Pipeline</h3>
  
  <div class="jenkins-pipeline">
    <h4>Declarative Pipeline with Docker</h4>
    <pre><code class="language-groovy">// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.company.com'
        DOCKER_CREDENTIALS = credentials('docker-registry-creds')
        IMAGE_NAME = "${DOCKER_REGISTRY}/myapp"
        IMAGE_TAG = "${BUILD_NUMBER}-${GIT_COMMIT.take(7)}"
    }
    
    stages {
        stage('Build') {
            steps {
                script {
                    docker.build("${IMAGE_NAME}:${IMAGE_TAG}")
                }
            }
        }
        
        stage('Test') {
            steps {
                script {
                    docker.image("${IMAGE_NAME}:${IMAGE_TAG}").inside {
                        sh 'npm test'
                        sh 'npm run lint'
                    }
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                sh '''
                    docker run --rm \
                      -v /var/run/docker.sock:/var/run/docker.sock \
                      -v $HOME/.cache:/root/.cache \
                      aquasec/trivy image ${IMAGE_NAME}:${IMAGE_TAG}
                '''
            }
        }
        
        stage('Push') {
            when {
                branch 'main'
            }
            steps {
                script {
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-creds') {
                        docker.image("${IMAGE_NAME}:${IMAGE_TAG}").push()
                        docker.image("${IMAGE_NAME}:${IMAGE_TAG}").push('latest')
                    }
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    kubectl set image deployment/myapp \
                      myapp=${IMAGE_NAME}:${IMAGE_TAG} \
                      --record=true
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
            sh 'docker system prune -f'
        }
    }
}</code></pre>
  </div>
  
  <h3><i class="fas fa-cloud"></i> Cloud Native CI/CD</h3>
  
  <div class="cloud-native-cicd">
    <h4>Tekton Pipeline</h4>
    <pre><code class="language-yaml"># tekton-pipeline.yaml
apiVersion: tekton.dev/v1beta1
kind: Pipeline
metadata:
  name: docker-build-push
spec:
  params:
    - name: repo-url
      type: string
    - name: image-reference
      type: string
  workspaces:
    - name: shared-data
  tasks:
    - name: fetch-source
      taskRef:
        name: git-clone
      workspaces:
        - name: output
          workspace: shared-data
      params:
        - name: url
          value: $(params.repo-url)
          
    - name: build-push
      runAfter: ["fetch-source"]
      taskRef:
        name: kaniko
      workspaces:
        - name: source
          workspace: shared-data
      params:
        - name: IMAGE
          value: $(params.image-reference)
        - name: DOCKERFILE
          value: ./Dockerfile
        - name: CONTEXT
          value: ./
        - name: EXTRA_ARGS
          value:
            - --cache=true
            - --cache-ttl=24h</code></pre>
  </div>
</div>

