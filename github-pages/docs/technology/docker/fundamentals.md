---
layout: docs
title: "Docker: Fundamentals"
permalink: /docs/technology/docker/fundamentals.html
toc: true
toc_sticky: true
hide_title: true
---

[Docker](./) &raquo; Fundamentals

The core concepts of containerization: images, containers, networking, and the Docker architecture.

## Why Docker? The Problems It Solves

An application that works perfectly on your machine often fails the moment a colleague runs it — a different Python version, a missing library, a conflicting config. Docker eliminates this entire class of problems. The recurring pain points it addresses:

- **Dependency hell** — different projects need different versions of libraries, languages, and tools, forcing complex virtual-environment juggling.
- **Environment parity** — code that runs on a laptop fails in production over OS differences, missing dependencies, or config mismatches.
- **Onboarding time** — new contributors lose days installing tools and troubleshooting their environment.
- **Resource efficiency** — traditional VMs consume enough resources to limit how many applications fit on one server.

Docker answers each: **isolated environments** (each container has its own filesystem, network, and process space), **reproducible builds** (a Dockerfile defines the exact build steps), **instant setup** (a new developer runs one `docker run`), and **efficient layering** (shared read-only layers cut disk and memory overhead).

## Essential Docker Commands

Before running any Docker commands, it helps to understand the mental model: Docker images are like recipes (blueprints), while containers are the actual dishes you create from those recipes. You can make many containers from the same image, and each one runs independently.

### Core Operations

These examples build from a throwaway container up to a multi-service stack.

**Running containers** — start here when learning Docker or testing something in a clean environment:

```bash
# Run an interactive Ubuntu container
docker run -it ubuntu:22.04 bash

# You are now inside a minimal Linux system
cat /etc/os-release && exit
```

The `-it` flags create an interactive terminal session; typing `exit` stops the container.

**Web server deployment** — run a service in the background (web server, database, API):

```bash
# Run Nginx web server in the background
docker run -d -p 8080:80 --name my-web nginx

# Visit http://localhost:8080, then clean up
docker stop my-web && docker rm my-web
```

The `-d` flag runs the container in the background; `-p 8080:80` maps host port 8080 to the container's port 80.

**Building custom images** — package your own application with its dependencies:

```dockerfile
# Dockerfile - save this file, then build with: docker build -t my-app .
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```bash
# Build and run your custom image
docker build -t my-app .
docker run -d -p 5000:5000 my-app
```

A Dockerfile defines your environment step by step. Docker caches each step, so rebuilds are fast when only your code changes.

**Using Docker Compose** — when your application needs multiple services (web server + database) working together:

```yaml
# compose.yaml
services:
  web:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - redis
  redis:
    image: redis:alpine
```

```bash
# Start all services with one command
docker compose up -d

# Stop everything
docker compose down
```

Compose creates a network where services find each other by name — your web service connects to `redis` without knowing its IP.

The essentials to carry forward: **images** are blueprints and **containers** are running instances; **Dockerfiles** build images reproducibly; **port mapping** connects container services to the host; **Compose** manages multi-container apps; and **volumes** persist data beyond a container's lifecycle.

## Understanding Container Technology

Before diving into Docker's specific implementation, it helps to understand how containers differ from traditional virtualization. This context will help you make informed decisions about when to use each approach.

### Containers vs Virtual Machines

Consider the following when choosing between containers and VMs:

| Factor | Choose Containers | Choose VMs |
|--------|-------------------|------------|
| Startup time matters | Yes - seconds vs minutes | Boot time acceptable |
| Running many instances | Yes - minimal overhead | Fewer, larger workloads |
| Need different OS | No - share host kernel | Yes - run Windows on Linux |
| Security isolation | Process-level sufficient | Need hardware-level isolation |
| Legacy applications | May need refactoring | Run as-is |

Containers are lightweight, resource-efficient, and portable, suited to modern scalable applications; virtual machines provide strong isolation, full OS support, and hardware emulation at the cost of higher resource use and slower startup. The two architectures side by side:

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

Containers and VMs make opposite trade-offs:

| | Containers | Virtual machines |
|---|------------|------------------|
| **Strengths** | Lightweight (share the host kernel), start in seconds, high density on one host, portable across environments, process-level isolation | Strong OS-level isolation, run any OS/distribution, hardware emulation for legacy/platform-specific apps, mature tooling |
| **Weaknesses** | Tied to the host kernel (limited cross-platform), weaker isolation than VMs, unsuitable for kernel modifications | Resource-intensive (full OS per VM), minutes to boot, duplicated OS/library storage, heavier dependency management |

### What Containers Guarantee

Containers provide strong guarantees in several areas:

- **Application dependencies**: All libraries and tools are bundled together, eliminating "missing dependency" errors
- **Configuration**: Environment settings travel with the container, making deployments reproducible
- **Isolation**: Each container has its own filesystem and process space, preventing conflicts between applications
- **Portability**: The same container runs on any machine with Docker installed

### What Containers Cannot Guarantee

However, some factors remain outside container control:

- **Kernel features**: Containers share the host kernel, so a container expecting Linux 5.x features will not work on a host running Linux 4.x
- **Hardware access**: GPU acceleration, specialized devices, and host-specific resources may behave differently across machines
- **Resource limits**: CPU and memory constraints vary by host, affecting performance consistency
- **Platform differences**: A Linux container cannot run natively on Windows without a Linux VM layer

**Practical guidance**: For maximum portability, avoid dependencies on specific kernel versions or hardware features. When these are unavoidable, document the requirements clearly.

Now that we understand what containers do, let's briefly look at how Docker implements them. You do not need to memorize these details to use Docker effectively, but understanding the architecture helps when troubleshooting or optimizing performance.

### Docker Architecture Overview

Docker uses a layered architecture where each component has a specific responsibility:

| Component | Role | When You Interact With It |
|-----------|------|---------------------------|
| Docker CLI | User interface | Every docker command you run |
| Docker Daemon | Manages containers, images, networks | Runs in background |
| containerd | Container lifecycle management | Rarely directly |
| runc | Actually runs containers | Never directly |

### Container Runtime Basics

At its core, a container is defined by:

- **Filesystem**: What files the container can see (its "root filesystem")
- **Namespaces**: Isolation boundaries for processes, network, users, and more
- **Cgroups**: Resource limits for CPU, memory, and I/O
- **Security**: Capabilities, seccomp profiles, and mandatory access controls

The Open Container Initiative (OCI) standardizes these specifications, allowing containers to run on any compliant runtime.

> **Note**: Most users never interact with these low-level components directly. Docker's CLI abstracts away this complexity while giving you control when needed.

### How Images Are Layered

A Docker image is not a single blob; it is a stack of read-only layers, one per build instruction. When you run a container, Docker adds a thin writable layer on top. Multiple containers from the same image share the underlying read-only layers, which is why containers are so lightweight.

```mermaid
flowchart TB
    subgraph Image["Image (read-only, shared)"]
        L1["FROM python:3.12-slim<br/>(base OS layer)"]
        L2["RUN pip install -r requirements.txt<br/>(dependencies)"]
        L3["COPY . .<br/>(application code)"]
        L1 --> L2 --> L3
    end
    L3 --> C1["Container A<br/>writable layer"]
    L3 --> C2["Container B<br/>writable layer"]
```

**Why this matters for builds**: Docker caches each layer. If a layer's inputs are unchanged, Docker reuses the cached layer instead of rebuilding it. Ordering instructions so that rarely-changing steps (installing dependencies) come before frequently-changing steps (copying source code) maximizes cache hits and dramatically speeds up rebuilds.

### The Container Lifecycle

A container is not simply "on" or "off." It moves through a small set of well-defined states, and knowing them explains why a stopped container still occupies disk, and why `docker run` and `docker start` are different operations.

```mermaid
stateDiagram-v2
    [*] --> Created: docker create
    Created --> Running: docker start
    [*] --> Running: docker run
    Running --> Paused: docker pause
    Paused --> Running: docker unpause
    Running --> Stopped: docker stop / process exits
    Stopped --> Running: docker start
    Stopped --> [*]: docker rm
```

| State | Meaning | Writable layer kept? |
|-------|---------|----------------------|
| Created | Filesystem prepared, process not started | Yes |
| Running | Main process executing | Yes |
| Paused | Processes frozen via cgroup freezer | Yes |
| Stopped (Exited) | Process ended; container retained | Yes (until `docker rm`) |
| Removed | Container and its writable layer deleted | No |

**Key insight**: `docker run` is shorthand for `docker create` + `docker start`. A stopped container keeps its writable layer and configuration, so `docker start` resumes it with state intact. Only `docker rm` reclaims that space — which is why `docker ps -a` often reveals a long tail of forgotten exited containers.

### Operating a Running Container

A handful of commands cover almost all day-to-day inspection and debugging:

```bash
docker ps                       # List running containers
docker ps -a                    # Include stopped containers
docker logs -f my-web           # Stream a container's stdout/stderr
docker exec -it my-web bash     # Open a shell inside a running container
docker inspect my-web           # Full JSON: networks, mounts, env, state
docker stats                    # Live CPU/memory/IO per container
docker cp my-web:/app/log.txt . # Copy a file out of a container
```

`docker exec` is the workhorse for debugging: it starts a *new* process inside an already-running container, unlike `docker run` which creates a fresh container. If `exec` fails with "executable not found," the image is likely a minimal one (e.g. `distroless` or `scratch`) that ships no shell.

### The Build Context and .dockerignore

When you run `docker build .`, the trailing `.` is the **build context** — the entire directory tree is sent to the Docker daemon before the build begins. A bloated context (a stray `node_modules/` or `.git/`) slows every build and can leak secrets into images via a careless `COPY . .`.

A `.dockerignore` file, with the same syntax as `.gitignore`, excludes paths from the context:

```dockerignore
.git
node_modules
*.log
.env
Dockerfile
```

This both speeds uploads and prevents accidental inclusion of credentials or local artifacts in the final image.

## Docker Network Architecture

Networking is often the trickiest part of containerization. Containers need to communicate with each other, with the host, and with external services, all while maintaining isolation. Docker provides several network drivers for different scenarios.

### Network Driver Quick Reference

| Driver | Use Case | Isolation | Performance |
|--------|----------|-----------|-------------|
| bridge | Default for standalone containers | Good | Good |
| host | Maximum performance needed | None | Best |
| overlay | Multi-host communication (Swarm) | Good | Good |
| macvlan | Container needs real IP on network | Varies | Good |
| none | Complete network isolation | Maximum | N/A |

### How Bridge Networking Works

When you run a container without specifying a network, Docker uses the default bridge network. Here is what happens:

1. Docker creates a virtual network interface pair (veth)
2. One end attaches to the container, the other to a bridge on the host
3. The container gets an IP address from Docker's internal DHCP
4. Containers on the same bridge can communicate by IP
5. For external access, Docker uses NAT and port mapping

**Key insight**: Containers on the same user-defined bridge network can find each other by container name (automatic DNS). The default bridge does not have this feature, which is why creating custom networks is recommended.

The diagram below shows how an external request reaches a container on a bridge network, and how two containers on the same network reach each other directly by name:

```mermaid
flowchart LR
    Client([External Client]) -->|"host:8080"| Host[Docker Host]
    Host -->|"NAT / port map"| Bridge(["docker0 bridge"])
    Bridge -->|"172.17.0.2:80"| Web[web container]
    Web -->|"DNS: db"| DB[db container]
    Bridge --- DB
```

### When to Use Each Network Type

- **bridge**: Development, single-host deployments, isolated applications
- **host**: Performance-critical applications, when container port must match host port
- **overlay**: Docker Swarm services spanning multiple hosts
- **macvlan**: When container must appear as physical device on network (legacy integration)
- **none**: Security-sensitive workloads that should have no network access

<div class="notice--warning">
  <h4>Common Pitfalls</h4>
  <ul>
    <li><strong>Relying on the default bridge for DNS:</strong> Containers on the default <code>bridge</code> cannot resolve each other by name. Always create a user-defined network for multi-container apps.</li>
    <li><strong>Expecting data to persist:</strong> A container's writable layer is destroyed with the container. Use volumes for anything you need to keep (see Storage &amp; Security).</li>
    <li><strong>Using <code>latest</code> tags:</strong> <code>latest</code> is not a fixed version; it moves. Pin explicit tags for reproducible builds.</li>
    <li><strong>Running as root:</strong> Containers default to root inside the container. Add a non-root <code>USER</code> for anything beyond local experiments.</li>
  </ul>
</div>

## Key Takeaways

- **Images vs containers.** An image is an immutable, layered blueprint; a container is a running instance with a thin writable layer on top.
- **Containers share the kernel.** Unlike VMs, containers share the host kernel — sub-second startup and minimal overhead, at the cost of weaker isolation.
- **Layers drive caching.** Each Dockerfile instruction is a cached layer. Order from least- to most-frequently-changing to speed up rebuilds.
- **Custom networks for DNS.** User-defined bridge networks give containers automatic name resolution; the default bridge does not.

---

## See Also

- [Storage &amp; Security](storage-security.html) - Persist data with volumes and harden containers
- [Dockerfiles &amp; CI/CD](dockerfiles.html) - Build optimized images and automate pipelines
- [Advanced Patterns](advanced.html) - Production architectures and WebAssembly runtimes
- [Docker Essentials](../docker-essentials.html) - Quick command reference
- [Kubernetes](../kubernetes/) - Orchestrate containers at scale
