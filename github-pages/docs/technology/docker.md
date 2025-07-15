---
layout: single
title: Containers
---

# Containers

<html><header><link rel="stylesheet" href="https://andrewaltimit.github.io/Documentation/style.css"></header></html>

<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">Containers & Docker</h1>
    <p class="hero-subtitle">Build, Ship, and Run Anywhere</p>
  </div>
</div>

<div class="intro-card">
  <p class="lead-text">Containers provide a consistent environment for applications by packaging software, dependencies, and configurations into a single, portable unit. However, there are some cases where this consistency might be compromised, particularly when dealing with kernel differences.</p>
  
  <div class="key-insights">
    <div class="insight-card">
      <i class="fas fa-box"></i>
      <h4>Lightweight</h4>
      <p>Share host OS kernel</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-rocket"></i>
      <h4>Fast Startup</h4>
      <p>Seconds vs minutes</p>
    </div>
    <div class="insight-card">
      <i class="fas fa-globe"></i>
      <h4>Portable</h4>
      <p>Run anywhere consistently</p>
    </div>
  </div>
</div>

## Containers vs Virtual Machines

<div class="comparison-section">
  <p class="section-intro">Containers are lightweight, resource-efficient, and portable, making them suitable for modern, scalable applications. Virtual machines provide strong isolation, full OS support, and hardware emulation but can be resource-intensive and slower to start up.</p>
  
  <div class="architecture-comparison">
    <div class="architecture-diagram">
      <svg viewBox="0 0 600 350">
        <!-- Container Architecture -->
        <g id="container-arch">
          <text x="150" y="20" text-anchor="middle" font-size="14" font-weight="bold">Container Architecture</text>
          
          <!-- Hardware -->
          <rect x="50" y="300" width="200" height="40" fill="#34495e" />
          <text x="150" y="325" text-anchor="middle" font-size="11" fill="white">Hardware</text>
          
          <!-- Host OS -->
          <rect x="50" y="250" width="200" height="40" fill="#3498db" />
          <text x="150" y="275" text-anchor="middle" font-size="11" fill="white">Host OS + Kernel</text>
          
          <!-- Container Runtime -->
          <rect x="50" y="200" width="200" height="40" fill="#e74c3c" />
          <text x="150" y="225" text-anchor="middle" font-size="11" fill="white">Container Runtime</text>
          
          <!-- Containers -->
          <rect x="60" y="80" width="55" height="110" fill="#2ecc71" opacity="0.7" stroke="#27ae60" stroke-width="2" />
          <text x="87" y="100" text-anchor="middle" font-size="9" fill="white">App A</text>
          <text x="87" y="115" text-anchor="middle" font-size="8" fill="white">Bins/Libs</text>
          
          <rect x="125" y="80" width="55" height="110" fill="#f39c12" opacity="0.7" stroke="#d68910" stroke-width="2" />
          <text x="152" y="100" text-anchor="middle" font-size="9" fill="white">App B</text>
          <text x="152" y="115" text-anchor="middle" font-size="8" fill="white">Bins/Libs</text>
          
          <rect x="190" y="80" width="55" height="110" fill="#9b59b6" opacity="0.7" stroke="#8e44ad" stroke-width="2" />
          <text x="217" y="100" text-anchor="middle" font-size="9" fill="white">App C</text>
          <text x="217" y="115" text-anchor="middle" font-size="8" fill="white">Bins/Libs</text>
        </g>
        
        <!-- VM Architecture -->
        <g id="vm-arch" transform="translate(300,0)">
          <text x="150" y="20" text-anchor="middle" font-size="14" font-weight="bold">Virtual Machine Architecture</text>
          
          <!-- Hardware -->
          <rect x="50" y="300" width="200" height="40" fill="#34495e" />
          <text x="150" y="325" text-anchor="middle" font-size="11" fill="white">Hardware</text>
          
          <!-- Host OS -->
          <rect x="50" y="250" width="200" height="40" fill="#3498db" />
          <text x="150" y="275" text-anchor="middle" font-size="11" fill="white">Host OS</text>
          
          <!-- Hypervisor -->
          <rect x="50" y="200" width="200" height="40" fill="#e74c3c" />
          <text x="150" y="225" text-anchor="middle" font-size="11" fill="white">Hypervisor</text>
          
          <!-- VMs -->
          <rect x="60" y="50" width="55" height="140" fill="#2ecc71" opacity="0.7" stroke="#27ae60" stroke-width="2" />
          <text x="87" y="70" text-anchor="middle" font-size="9" fill="white">App A</text>
          <text x="87" y="85" text-anchor="middle" font-size="8" fill="white">Bins/Libs</text>
          <rect x="65" y="120" width="45" height="25" fill="#27ae60" opacity="0.5" />
          <text x="87" y="137" text-anchor="middle" font-size="8" fill="white">Guest OS</text>
          
          <rect x="125" y="50" width="55" height="140" fill="#f39c12" opacity="0.7" stroke="#d68910" stroke-width="2" />
          <text x="152" y="70" text-anchor="middle" font-size="9" fill="white">App B</text>
          <text x="152" y="85" text-anchor="middle" font-size="8" fill="white">Bins/Libs</text>
          <rect x="130" y="120" width="45" height="25" fill="#d68910" opacity="0.5" />
          <text x="152" y="137" text-anchor="middle" font-size="8" fill="white">Guest OS</text>
          
          <rect x="190" y="50" width="55" height="140" fill="#9b59b6" opacity="0.7" stroke="#8e44ad" stroke-width="2" />
          <text x="217" y="70" text-anchor="middle" font-size="9" fill="white">App C</text>
          <text x="217" y="85" text-anchor="middle" font-size="8" fill="white">Bins/Libs</text>
          <rect x="195" y="120" width="45" height="25" fill="#8e44ad" opacity="0.5" />
          <text x="217" y="137" text-anchor="middle" font-size="8" fill="white">Guest OS</text>
        </g>
      </svg>
    </div>
  </div>
</div>

<div class="pros-cons-comparison">
  <div class="container-pros-cons">
    <h3><i class="fas fa-box"></i> Container Pros/Cons</h3>
    
    <div class="pros-cons-grid">
      <div class="pros-section">
        <h4><i class="fas fa-check-circle"></i> Pros</h4>
        <div class="pro-item">
          <i class="fas fa-feather-alt"></i>
          <div>
            <strong>Lightweight</strong>
            <p>Share the host OS kernel, minimal overhead</p>
          </div>
        </div>
        <div class="pro-item">
          <i class="fas fa-rocket"></i>
          <div>
            <strong>Fast startup</strong>
            <p>Start in seconds for rapid deployment</p>
          </div>
        </div>
        <div class="pro-item">
          <i class="fas fa-chart-line"></i>
          <div>
            <strong>Resource efficiency</strong>
            <p>Higher density on single host</p>
          </div>
        </div>
        <div class="pro-item">
          <i class="fas fa-ship"></i>
          <div>
            <strong>Portability</strong>
            <p>Consistent deployment across environments</p>
          </div>
        </div>
        <div class="pro-item">
          <i class="fas fa-shield-alt"></i>
          <div>
            <strong>Process Isolation</strong>
            <p>Applications run without interference</p>
          </div>
        </div>
      </div>
      
      <div class="cons-section">
        <h4><i class="fas fa-times-circle"></i> Cons</h4>
        <div class="con-item">
          <i class="fas fa-link"></i>
          <div>
            <strong>Kernel dependency</strong>
            <p>Limited cross-platform compatibility</p>
          </div>
        </div>
        <div class="con-item">
          <i class="fas fa-lock-open"></i>
          <div>
            <strong>Security boundaries</strong>
            <p>Weaker isolation than VMs</p>
          </div>
        </div>
        <div class="con-item">
          <i class="fas fa-ban"></i>
          <div>
            <strong>Limited applications</strong>
            <p>Not suitable for kernel modifications</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="vm-pros-cons">
    <h3><i class="fas fa-desktop"></i> Virtual Machine Pros/Cons</h3>
    
    <div class="pros-cons-grid">
      <div class="pros-section">
        <h4><i class="fas fa-check-circle"></i> Pros</h4>
        <div class="pro-item">
          <i class="fas fa-lock"></i>
          <div>
            <strong>Strong isolation</strong>
            <p>Complete OS separation for security</p>
          </div>
        </div>
        <div class="pro-item">
          <i class="fas fa-layer-group"></i>
          <div>
            <strong>Full OS support</strong>
            <p>Run any OS version or distribution</p>
          </div>
        </div>
        <div class="pro-item">
          <i class="fas fa-microchip"></i>
          <div>
            <strong>Hardware emulation</strong>
            <p>Support legacy and platform-specific apps</p>
          </div>
        </div>
        <div class="pro-item">
          <i class="fas fa-history"></i>
          <div>
            <strong>Mature ecosystem</strong>
            <p>Extensive tooling and management</p>
          </div>
        </div>
      </div>
      
      <div class="cons-section">
        <h4><i class="fas fa-times-circle"></i> Cons</h4>
        <div class="con-item">
          <i class="fas fa-weight-hanging"></i>
          <div>
            <strong>Resource-intensive</strong>
            <p>Full OS stack overhead</p>
          </div>
        </div>
        <div class="con-item">
          <i class="fas fa-hourglass-half"></i>
          <div>
            <strong>Slow startup</strong>
            <p>Minutes to boot and initialize</p>
          </div>
        </div>
        <div class="con-item">
          <i class="fas fa-database"></i>
          <div>
            <strong>Storage overhead</strong>
            <p>Duplicated OS and libraries</p>
          </div>
        </div>
        <div class="con-item">
          <i class="fas fa-random"></i>
          <div>
            <strong>Deployment complexity</strong>
            <p>Manual dependency management</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

### Container Consistency

- **Application dependencies:** Containers bundle all required libraries and dependencies, ensuring that the application runs consistently across different environments.
- **Configuration:** Containers encapsulate the application's configuration, making it easy to reproduce and share across teams and environments.
- **Isolation:** Containers provide process isolation, so applications running in separate containers won't interfere with one another.
- **Portability:** Containers can run on any system with container runtime support, regardless of the host's underlying hardware or operating system.

### Container Inconsistency

- **Kernel differences:** Containers share the host's kernel, which means that they are susceptible to inconsistencies stemming from kernel differences. For example, a container running on a host with an older kernel version may not have access to newer kernel features. Additionally, certain system calls or kernel modules may not be available or compatible across different host systems.
- **Host-specific resources:** Containers can access host resources like filesystems, devices, and network interfaces. However, these resources may not be consistent across different host systems, leading to potential inconsistencies in container behavior.
- **Resource limits and constraints:** Containers can be limited in terms of resources, such as CPU, memory, or I/O. These limits may vary between host systems and can impact the consistency of container performance.
- **Platform-specific features:** Some features, such as hardware acceleration, are platform-specific and may not be consistently available across different host systems. As a result, containers relying on these features may experience inconsistent behavior.

While containers provide a high level of consistency for application dependencies, configuration, isolation, and portability, they can be susceptible to inconsistencies due to kernel differences, host-specific resources, resource limits, and platform-specific features. To minimize these inconsistencies, it is essential to understand the requirements of your application and ensure that the host systems are compatible with the desired container environment.

# Docker Architecture Deep Dive

## Container Runtime Architecture

Docker uses a layered architecture to manage containers:

### OCI (Open Container Initiative) Runtime

The low-level container runtime follows the OCI specification:

**Key Components**:
- **OCI Bundle**: Directory with `config.json` and `rootfs/`
- **Runtime Spec**: JSON configuration defining container properties
- **Lifecycle**: Create, start, kill, delete operations

**Container Configuration**:
- **Process**: Command, environment, working directory, capabilities
- **Root Filesystem**: Mount points and readonly settings  
- **Namespaces**: PID, Network, IPC, UTS, Mount, User, Cgroup isolation
- **Resources**: Memory, CPU, PID limits, block I/O controls
- **Security**: Seccomp profiles, masked paths, readonly paths

**Resource Management**:
- Memory limits and reservations
- CPU shares, quotas, and cpuset assignment
- Process ID limits
- Block I/O weight and throttling
- Network class IDs and priorities

### Containerd Integration

High-level container management daemon:

**Architecture**:
- **gRPC API**: Remote procedure calls for container operations
- **Namespaces**: Logical grouping of containers and images
- **Snapshots**: Filesystem snapshots for container rootfs
- **Content Store**: Storage for image layers and manifests

**Key Operations**:
- Pull images from registries
- Create containers from images
- Manage container lifecycle (start, stop, pause, resume)
- Execute commands in running containers
- Stream logs and attach to containers

> **Code Reference**: For complete implementation of OCI runtime and containerd integration, see:
> - [`container_runtime.py`](../../code-examples/technology/docker/container_runtime.py) - OCI runtime implementation
> - [`containerd_integration.py`](../../code-examples/technology/docker/containerd_integration.py) - Containerd client

## Docker Network Architecture

### Container Network Interface (CNI)

Standard for container networking plugins:

**CNI Specification**:
- **ADD**: Connect container to network
- **DEL**: Disconnect container from network
- **CHECK**: Verify container connectivity
- **VERSION**: Report plugin version

**Network Configuration**:
- Create veth (virtual ethernet) pairs
- Configure network namespaces
- Assign IP addresses and routes
- Setup DNS configuration

### Docker Bridge Network Driver

Default network driver for containers:

**Components**:
- **Linux Bridge**: Virtual switch for container communication
- **veth Pairs**: Virtual cable connecting container to bridge
- **iptables Rules**: NAT and packet filtering
- **IP Address Management**: Allocation and tracking

**Network Isolation**:
- Network namespaces per container
- Bridge isolation between networks
- ICC (Inter-Container Communication) control
- Port mapping and exposure

### IP Address Management (IPAM)

Manages IP allocation for containers:

**Features**:
- Subnet management
- Dynamic IP allocation
- Address release and reuse
- Conflict prevention
- Gateway reservation

> **Code Reference**: For complete networking implementation, see [`container_networking.py`](../../code-examples/technology/docker/container_networking.py)

# Docker

<div class="docker-section">
  <div class="docker-intro">
    <p>Docker is a platform for developing, shipping, and running applications via containerization technology which packages applications and their dependencies into lightweight and portable containers that can run consistently across different environments.</p>
    
    <div class="docker-workflow">
      <h3><i class="fas fa-cogs"></i> Docker Workflow</h3>
      <svg viewBox="0 0 700 200" class="workflow-diagram">
        <!-- Build Stage -->
        <rect x="50" y="50" width="120" height="100" fill="#3498db" opacity="0.3" stroke="#2980b9" stroke-width="2" />
        <text x="110" y="80" text-anchor="middle" font-size="14" font-weight="bold">BUILD</text>
        <rect x="70" y="95" width="80" height="40" fill="#2c3e50" />
        <text x="110" y="120" text-anchor="middle" font-size="10" fill="white">Dockerfile</text>
        
        <!-- Arrow -->
        <path d="M 175 100 L 225 100" stroke="#34495e" stroke-width="3" marker-end="url(#arrow)" />
        <text x="200" y="90" text-anchor="middle" font-size="10">docker build</text>
        
        <!-- Ship Stage -->
        <rect x="230" y="50" width="120" height="100" fill="#e74c3c" opacity="0.3" stroke="#c0392b" stroke-width="2" />
        <text x="290" y="80" text-anchor="middle" font-size="14" font-weight="bold">SHIP</text>
        <circle cx="290" cy="110" r="25" fill="#c0392b" />
        <text x="290" y="115" text-anchor="middle" font-size="10" fill="white">Image</text>
        
        <!-- Registry -->
        <rect x="380" y="30" width="140" height="140" fill="#f39c12" opacity="0.2" stroke="#d68910" stroke-width="2" />
        <text x="450" y="55" text-anchor="middle" font-size="12">Docker Registry</text>
        <rect x="400" y="70" width="40" height="30" fill="#d68910" opacity="0.5" />
        <rect x="450" y="70" width="40" height="30" fill="#d68910" opacity="0.5" />
        <rect x="400" y="110" width="40" height="30" fill="#d68910" opacity="0.5" />
        <rect x="450" y="110" width="40" height="30" fill="#d68910" opacity="0.5" />
        
        <!-- Push arrow -->
        <path d="M 315 100 Q 350 80, 375 100" stroke="#34495e" stroke-width="2" marker-end="url(#arrow)" />
        <text x="345" y="75" text-anchor="middle" font-size="9">push</text>
        
        <!-- Pull arrow -->
        <path d="M 520 100 Q 545 120, 545 150" stroke="#34495e" stroke-width="2" marker-end="url(#arrow)" />
        <text x="535" y="125" text-anchor="middle" font-size="9">pull</text>
        
        <!-- Run Stage -->
        <rect x="480" y="155" width="120" height="40" fill="#27ae60" opacity="0.3" stroke="#229954" stroke-width="2" />
        <text x="540" y="180" text-anchor="middle" font-size="14" font-weight="bold">RUN</text>
        <text x="620" y="175" font-size="10">Containers</text>
      </svg>
    </div>
  </div>
</div>

## Installing Docker

- [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Install Docker on Debian](https://docs.docker.com/engine/install/debian/)
- [Install Docker on Fedora](https://docs.docker.com/engine/install/fedora/)
- [Install Docker on CentOS](https://docs.docker.com/engine/install/centos/)
- [Install Docker on Windows](https://docs.docker.com/docker-for-windows/install/)
- [Install Docker on macOS](https://docs.docker.com/docker-for-mac/install/)

## Common Docker CLI Commands

<div class="docker-commands">
  <div class="command-category">
    <h3><i class="fas fa-images"></i> Images</h3>
    <div class="command-grid">
      <div class="command-item">
        <code>docker images</code>
        <span>List all local images</span>
      </div>
      <div class="command-item">
        <code>docker pull &lt;image&gt;:&lt;tag&gt;</code>
        <span>Download image from registry</span>
      </div>
      <div class="command-item">
        <code>docker rmi &lt;image&gt;:&lt;tag&gt;</code>
        <span>Remove an image</span>
      </div>
    </div>
  </div>
  
  <div class="command-category">
    <h3><i class="fas fa-cube"></i> Containers</h3>
    <div class="command-grid">
      <div class="command-item">
        <code>docker ps</code>
        <span>List running containers</span>
      </div>
      <div class="command-item">
        <code>docker ps -a</code>
        <span>List all containers</span>
      </div>
      <div class="command-item">
        <code>docker run -it --rm --name &lt;name&gt; &lt;image&gt;:&lt;tag&gt;</code>
        <span>Run interactive container</span>
      </div>
      <div class="command-item">
        <code>docker stop &lt;container&gt;</code>
        <span>Stop a running container</span>
      </div>
      <div class="command-item">
        <code>docker rm &lt;container&gt;</code>
        <span>Remove a container</span>
      </div>
    </div>
  </div>
  
  <div class="command-category">
    <h3><i class="fas fa-file-alt"></i> Container Operations</h3>
    <div class="command-grid">
      <div class="command-item">
        <code>docker logs &lt;container&gt;</code>
        <span>View container logs</span>
      </div>
      <div class="command-item">
        <code>docker exec -it &lt;container&gt; &lt;command&gt;</code>
        <span>Execute command in container</span>
      </div>
    </div>
  </div>
  
  <div class="command-category">
    <h3><i class="fas fa-hammer"></i> Building & Publishing</h3>
    <div class="command-grid">
      <div class="command-item">
        <code>docker build -t &lt;image&gt;:&lt;tag&gt; .</code>
        <span>Build image from Dockerfile</span>
      </div>
      <div class="command-item">
        <code>docker push &lt;image&gt;:&lt;tag&gt;</code>
        <span>Push image to registry</span>
      </div>
    </div>
  </div>
  
  <div class="command-flow">
    <h3><i class="fas fa-stream"></i> Common Workflow</h3>
    <div class="workflow-steps">
      <div class="workflow-step">
        <div class="step-number">1</div>
        <code>docker build -t myapp:1.0 .</code>
        <span>Build your application image</span>
      </div>
      <div class="workflow-step">
        <div class="step-number">2</div>
        <code>docker run -d -p 8080:80 myapp:1.0</code>
        <span>Run container in detached mode</span>
      </div>
      <div class="workflow-step">
        <div class="step-number">3</div>
        <code>docker logs -f &lt;container_id&gt;</code>
        <span>Monitor application logs</span>
      </div>
      <div class="workflow-step">
        <div class="step-number">4</div>
        <code>docker push myapp:1.0</code>
        <span>Share image via registry</span>
      </div>
    </div>
  </div>
</div>

### Docker Compose

- Start a multi-container application: `docker-compose up -d`
- Stop a multi-container application: `docker-compose down`

### Docker Volumes

- Create a volume: `docker volume create <volume_name>`
- List volumes: `docker volume ls`
- Remove a volume: `docker volume rm <volume_name>`

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
RUN pip install --trusted-host pypi.python.org -r requirements.txt

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
