---
layout: docs
title: "Docker: Advanced Patterns"
permalink: /docs/technology/docker/advanced.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #00aaff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Docker: Advanced Patterns</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Production-ready architectures, real-world case studies, and cutting-edge container technologies including WebAssembly runtimes.</p>
</div>

<div class="intro-card">
  <p class="lead-text">This page assumes you are comfortable with images, containers, and Dockerfiles. It looks at how Docker is used <strong>in production at scale</strong>: real migration case studies, recurring design patterns, performance and security hardening, and where containers are heading next (including WebAssembly). Treat it as a tour of "what good looks like" once the basics are second nature.</p>
</div>

## Real-World Examples and Case Studies

<div class="case-studies-section">
  <h3><i class="fas fa-building"></i> Enterprise Microservices Architecture</h3>
  
  <div class="case-study">
    <h4>E-Commerce Platform Migration</h4>
    <p class="case-intro">A major e-commerce company migrated from monolithic architecture to Docker-based microservices, achieving 70% reduction in deployment time and 50% infrastructure cost savings.</p>
    
    <pre><code class="language-yaml"># docker-compose.production.yml
version: '3.8'

services:
  # API Gateway
  gateway:
    image: company/api-gateway:${VERSION}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    ports:
      - "443:443"
    environment:
      - RATE_LIMIT=1000
      - JWT_SECRET_FILE=/run/secrets/jwt_key
    secrets:
      - jwt_key
    networks:
      - frontend
      - backend

  # Product Service
  product-service:
    image: company/product-service:${VERSION}
    deploy:
      replicas: 5
      update_config:
        parallelism: 2
        delay: 10s
        failure_action: rollback
    environment:
      - DB_HOST=product-db
      - CACHE_HOST=redis-product
    depends_on:
      - product-db
      - redis-product
    networks:
      - backend

  # Order Service
  order-service:
    image: company/order-service:${VERSION}
    deploy:
      replicas: 3
    environment:
      - DB_HOST=order-db
      - KAFKA_BROKERS=kafka:9092
    depends_on:
      - order-db
      - kafka
    networks:
      - backend

  # Databases
  product-db:
    image: postgres:15-alpine
    volumes:
      - product-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    networks:
      - backend

  order-db:
    image: postgres:15-alpine
    volumes:
      - order-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    networks:
      - backend

  # Caching
  redis-product:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    deploy:
      replicas: 2
    networks:
      - backend

  # Message Queue
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
    depends_on:
      - zookeeper
    networks:
      - backend

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    networks:
      - backend

networks:
  frontend:
    driver: overlay
    driver_opts:
      encrypted: "true"
  backend:
    driver: overlay
    driver_opts:
      encrypted: "true"
    internal: true

volumes:
  product-data:
    driver: local
  order-data:
    driver: local

secrets:
  db_password:
    external: true
  jwt_key:
    external: true</code></pre>
    
    <h4>Implementation Highlights</h4>
    <ul>
      <li><strong>Service Mesh:</strong> Implemented Istio for advanced traffic management and observability</li>
      <li><strong>Auto-scaling:</strong> Used Kubernetes HPA with custom metrics for demand-based scaling</li>
      <li><strong>Zero-downtime:</strong> Achieved through rolling updates and health checks</li>
      <li><strong>Security:</strong> Implemented mutual TLS between services and secret rotation</li>
      <li><strong>Monitoring:</strong> Full observability with Prometheus, Grafana, and distributed tracing</li>
    </ul>
  </div>
  
  <h3><i class="fas fa-robot"></i> ML Pipeline Architecture</h3>
  
  <div class="ml-case-study">
    <h4>Containerized ML Model Serving</h4>
    <pre><code class="language-dockerfile"># Dockerfile for ML model serving
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mluser
USER mluser

# Copy model and application
WORKDIR /app
COPY --chown=mluser:mluser model/ ./model/
COPY --chown=mluser:mluser src/ ./src/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health').raise_for_status()"

# Serve model
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "120", "src.app:app"]</code></pre>
    
    <h4>Training Pipeline</h4>
    <pre><code class="language-yaml"># kubernetes-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-job
spec:
  template:
    spec:
      containers:
      - name: training
        image: company/ml-training:latest
        resources:
          limits:
            nvidia.com/gpu: 2
            memory: 32Gi
            cpu: 8
          requests:
            nvidia.com/gpu: 2
            memory: 16Gi
            cpu: 4
        volumeMounts:
        - name: dataset
          mountPath: /data
        - name: model-output
          mountPath: /output
        env:
        - name: EPOCHS
          value: "100"
        - name: BATCH_SIZE
          value: "64"
        - name: LEARNING_RATE
          value: "0.001"
      volumes:
      - name: dataset
        persistentVolumeClaim:
          claimName: training-dataset
      - name: model-output
        persistentVolumeClaim:
          claimName: model-storage
      restartPolicy: OnFailure
      nodeSelector:
        gpu-type: nvidia-v100</code></pre>
  </div>
</div>

## Advanced Docker Patterns and Techniques

<div class="advanced-patterns-section">
  <h3><i class="fas fa-puzzle-piece"></i> Design Patterns for Production</h3>
  
  <div class="pattern-grid">
    <div class="pattern-card">
      <h4><i class="fas fa-sync-alt"></i> Sidecar Pattern</h4>
      <p>Deploy helper containers alongside your main application container</p>
      <pre><code class="language-yaml"># Logging sidecar example (compose.yaml)
services:
  app:
    image: my-app:latest
    volumes:
      - logs:/var/log/app
      
  log-forwarder:
    image: fluent/fluent-bit:latest
    volumes:
      - logs:/var/log/app:ro
      - ./fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf
    environment:
      - ELASTICSEARCH_HOST=elasticsearch
      
volumes:
  logs:</code></pre>
    </div>
    
    <div class="pattern-card">
      <h4><i class="fas fa-shield-alt"></i> Ambassador Pattern</h4>
      <p>Proxy container that handles external communication</p>
      <pre><code class="language-yaml"># Service mesh ambassador
services:
  app:
    image: my-app:latest
    network_mode: "service:envoy"
    
  envoy:
    image: envoyproxy/envoy:v1.31-latest
    ports:
      - "8080:8080"
    volumes:
      - ./envoy.yaml:/etc/envoy/envoy.yaml</code></pre>
    </div>
    
    <div class="pattern-card">
      <h4><i class="fas fa-code-branch"></i> Adapter Pattern</h4>
      <p>Standardize output from different containers</p>
      <pre><code class="language-yaml"># Metrics adapter example
services:
  legacy-app:
    image: legacy-app:latest
    
  metrics-adapter:
    image: prom-exporter:latest
    environment:
      - LEGACY_APP_URL=http://legacy-app:8080
      - METRICS_PATH=/legacy/stats
    ports:
      - "9090:9090"</code></pre>
    </div>
  </div>
  
  <h3><i class="fas fa-lock"></i> Advanced Security Patterns</h3>
  
  <div class="security-patterns">
    <h4>Distroless Images</h4>
    <pre><code class="language-dockerfile"># Multi-stage build with distroless
FROM golang:1.23 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o myapp .

# Distroless image - no shell, package manager, or utilities
FROM gcr.io/distroless/static:nonroot
COPY --from=builder /app/myapp /
USER nonroot:nonroot
ENTRYPOINT ["/myapp"]</code></pre>
    
    <h4>Runtime Security with Falco</h4>
    <pre><code class="language-yaml"># falco-rules.yaml
- rule: Unauthorized Process in Container
  desc: Detect unauthorized process execution
  condition: >
    container and
    not proc.name in (allowed_processes) and
    not container.image.repository in (trusted_images)
  output: >
    Unauthorized process in container 
    (user=%user.name command=%proc.cmdline container=%container.name)
  priority: WARNING</code></pre>
  </div>
  
  <h3><i class="fas fa-compress-alt"></i> Image Optimization Techniques</h3>
  
  <div class="optimization-techniques">
    <h4>Advanced Multi-Stage Patterns</h4>
    <pre><code class="language-dockerfile"># Parallel multi-stage builds
# syntax=docker/dockerfile:1
FROM node:22 AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM golang:1.23 AS backend-builder
WORKDIR /app
COPY backend/go.* ./
RUN go mod download
COPY backend/ ./
RUN go build -ldflags="-s -w" -o server .

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=backend-builder /app/server /
COPY --from=frontend-builder /app/dist /static
EXPOSE 8080
CMD ["/server"]</code></pre>
    
    <h4>Layer Caching Strategies</h4>
    <pre><code class="language-dockerfile"># Dependency caching with BuildKit
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Cache mount for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy pandas scikit-learn

# Bind mount for development
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt</code></pre>
  </div>
</div>

While Docker and traditional container runtimes have revolutionized application deployment, the technology continues to evolve. One of the most promising developments is the emergence of WebAssembly as a potential container runtime alternative. This represents a significant shift in how we think about application isolation and portability.

## Future of Container Runtimes: WebAssembly

### WASM/WASI as Container Runtime Alternative

WebAssembly (WASM) and the WebAssembly System Interface (WASI) represent a potential paradigm shift in container technology, offering a lightweight, secure, and portable alternative to traditional container runtimes. Unlike traditional containers that share the host kernel, WebAssembly provides a completely sandboxed execution environment that can run anywhere - from browsers to servers to edge devices.

#### Understanding WebAssembly

To appreciate why WebAssembly is relevant to containerization, let's examine its core characteristics that make it suitable as a container runtime alternative:

**Core Characteristics:**
- **Binary Instruction Format**: Designed for stack-based virtual machines
- **Near-Native Performance**: Compiles to machine code with minimal overhead
- **Language Agnostic**: Supports C/C++, Rust, Go, and many other languages
- **Sandboxed Execution**: Strong security guarantees through capability-based security
- **Platform Independent**: True write-once, run-anywhere portability

#### WASI (WebAssembly System Interface)

WASI provides a standardized system interface for WebAssembly modules:

```rust
// Example WASI application in Rust
use std::env;
use std::fs;

fn main() {
    // WASI provides standard file system access
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 {
        match fs::read_to_string(&args[1]) {
            Ok(contents) => println!("File contents: {}", contents),
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
}
```

**WASI Capabilities:**
- **File System Access**: Sandboxed file operations
- **Network Access**: Controlled socket operations
- **Environment Variables**: Secure environment access
- **Random Number Generation**: Cryptographically secure randomness
- **Clock Access**: Time and timer functionality

While WASI provides essential system interfaces, some applications require more extensive POSIX compatibility. This is where WASIX comes in.

#### WASIX: Extended WASI

WASIX extends WASI with additional POSIX compatibility:

- **Threading Support**: Full POSIX threads
- **Process Forking**: Fork/exec capabilities
- **Signals**: POSIX signal handling
- **Sockets**: Extended networking support
- **Shared Memory**: Inter-process communication

```c
// WASIX example with threading
#include <pthread.h>
#include <stdio.h>

void* worker(void* arg) {
    printf("Worker thread: %ld\n", (long)arg);
    return NULL;
}

int main() {
    pthread_t thread;
    pthread_create(&thread, NULL, worker, (void*)42);
    pthread_join(thread, NULL);
    return 0;
}
```

With these extended capabilities, WebAssembly becomes viable for a broader range of applications. But how do we actually run WebAssembly modules as containers? This is where specialized runtimes like crun come into play.

#### crun: WebAssembly Container Runtime

crun is an OCI-compliant container runtime that supports WebAssembly:

```bash
# Running WASM containers with crun
sudo crun --runtime=/usr/bin/crun-wasm run wasm-container

# Container configuration for WASM
{
  "ociVersion": "1.0.2",
  "process": {
    "args": ["app.wasm"],
    "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
    "cwd": "/"
  },
  "root": {
    "path": "rootfs"
  },
  "annotations": {
    "module.wasm.image/variant": "compat"
  }
}
```

Now that we've seen how WebAssembly can function as a container runtime, let's examine the compelling advantages it offers over traditional container technologies.

### Advantages of WASM Containers

WebAssembly's appeal as a runtime comes down to three properties: it starts almost instantly, it isolates by *capability* rather than by kernel namespace, and its modules are tiny. The figures below are representative order-of-magnitude comparisons (exact numbers vary by workload and host); treat them as relative, not absolute.

#### Startup time and footprint

| Metric | Docker container | Firecracker microVM | WASM module |
|--------|------------------|---------------------|-------------|
| Cold start | ~1 s | ~125 ms | ~1 ms |
| Memory overhead | ~50 MB | ~150 MB | ~1-5 MB |
| Disk footprint | 100 MB - 1 GB | 100 MB - 1 GB | 1-10 MB |
| CPU overhead | 5-10% | 5-15% | under 1% |

The sub-millisecond cold start is the headline number: it makes WASM attractive for serverless and edge workloads where a traditional container's ~1 second startup dominates request latency.

#### Capability-based security

A container restricts a process *after* it has full access to a shared kernel — you drop capabilities and add seccomp profiles to claw privileges back. WebAssembly inverts this: a module starts with **no** access to the host and can only touch resources its host explicitly hands it (a pre-opened directory, a socket, a clock). There is no ambient authority to escape from.

```rust
// A WASI module can only open files under a directory the host pre-opened.
// Without that grant, path_open simply fails — the module never had access.
use wasi::{Errno, Fd};

fn open_under_preopened(dir_fd: Fd, path: &str) -> Result<Fd, Errno> {
    unsafe {
        wasi::path_open(
            dir_fd, // a directory FD the host chose to expose
            0,      // dirflags
            path,
            0,      // open flags
            0,      // rights base
            0,      // rights inheriting
            0,      // fd flags
        )
    }
}
```

These advantages make WebAssembly particularly attractive for modern cloud-native applications. But how do we manage WebAssembly containers at scale? The answer lies in integrating with existing orchestration platforms.

### WASM Container Orchestration

#### Kubernetes Integration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: wasm-app
  annotations:
    module.wasm.image/variant: "compat-smart"
spec:
  runtimeClassName: wasmtime
  containers:
  - name: app
    image: myregistry/wasm-app:latest
    resources:
      limits:
        memory: "10Mi"
        cpu: "100m"
```

#### Krustlet: Kubernetes Kubelet for WASM

```rust
// Krustlet provider implementation
use kubelet::Provider;

struct WasmProvider {
    runtime: wasmtime::Engine,
}

impl Provider for WasmProvider {
    async fn add(&self, pod: Pod) -> Result<()> {
        let module = self.fetch_wasm_module(&pod)?;
        let instance = self.runtime.instantiate(&module)?;
        instance.run().await
    }
}
```

While the integration with Kubernetes and other orchestration platforms is promising, it's important to understand where WebAssembly containers excel and where traditional containers might still be the better choice.

### Use Cases and Limitations

**Ideal Use Cases:**
- **Edge Computing**: Ultra-low latency requirements
- **Serverless Functions**: Fast cold starts
- **Plugin Systems**: Secure, sandboxed extensions
- **IoT Devices**: Minimal resource footprint
- **Multi-tenant Platforms**: Strong isolation guarantees

**Current Limitations:**
- **Ecosystem Maturity**: Tooling still evolving
- **Language Support**: Not all languages compile efficiently to WASM
- **System Calls**: Limited compared to native containers
- **Debugging**: More challenging than traditional containers

Despite these limitations, many organizations are exploring WebAssembly for specific workloads. If you're considering this transition, here's a practical approach to migration.

### Migration Path

```python
# Gradual migration strategy
class ContainerMigrationStrategy:
    def assess_workload(self, app):
        """Determine if app is suitable for WASM"""
        criteria = {
            "stateless": app.is_stateless(),
            "cpu_bound": app.is_cpu_intensive(),
            "small_footprint": app.size < 50 * 1024 * 1024,  # 50MB
            "supported_language": app.language in ["rust", "c", "go"],
        }
        
        score = sum(criteria.values()) / len(criteria)
        return score > 0.7  # 70% criteria met
    
    def migrate_to_wasm(self, app):
        """Step-by-step migration"""
        steps = [
            self.compile_to_wasm,
            self.add_wasi_bindings,
            self.test_functionality,
            self.optimize_performance,
            self.deploy_hybrid,
            self.monitor_and_validate,
            self.complete_migration
        ]
        
        for step in steps:
            if not step(app):
                return self.rollback(app)
```

To make informed decisions about migration, it's essential to understand the real-world performance characteristics of WebAssembly containers compared to traditional Docker containers.

### Performance Characteristics

The table below sketches how the two runtimes compare across a few common workload shapes. Startup and memory are reported in the same units across rows so the trend is visible; per-request latency is roughly comparable once warm, which is the key takeaway — **WASM's win is in cold start and footprint, not steady-state throughput.**

| Workload | Runtime | Cold start | Per-request latency | Memory |
|----------|---------|-----------|---------------------|--------|
| HTTP request handler | Docker | ~1200 ms | ~0.5 ms | ~50 MB |
| HTTP request handler | WASM | ~1 ms | ~0.6 ms | ~2 MB |
| Image processing | Docker | ~1500 ms | ~10 ms | ~200 MB |
| Image processing | WASM | ~2 ms | ~12 ms | ~20 MB |
| API gateway | Docker | ~1000 ms | ~0.2 ms | ~100 MB |
| API gateway | WASM | ~0.5 ms | ~0.25 ms | ~5 MB |

Two patterns stand out. First, cold start collapses from roughly a second to single-digit milliseconds — decisive for serverless and scale-to-zero. Second, steady-state per-request latency is essentially a wash; WASM does not make a warm handler meaningfully faster, so it is not a drop-in throughput upgrade for long-running services. As the technology matures, we can expect even more improvements. Let's look at what's on the horizon.

### Future Developments

**Component Model:**
```wit
// WebAssembly Interface Types (WIT)
interface http-handler {
  use types.{request, response}
  
  handle: func(req: request) -> response
}

world service {
  import wasi:filesystem/types
  import wasi:sockets/tcp
  
  export http-handler
}
```

**WASM-native Development:**
```rust
// Future: Direct WASM targeting without WASI
#[no_std]
#[wasm_module]
pub mod app {
    #[wasm_export]
    pub fn handle_request(ptr: *const u8, len: usize) -> Vec<u8> {
        // Direct memory manipulation
        // No system calls needed
    }
}
```

## Putting It Into Practice

The patterns on this page — sidecars, distroless images, multi-stage builds, and WASM runtimes — all serve the same goals: smaller attack surface, faster delivery, and predictable behavior at scale. Which ones matter most depends on your role.

### Where to Focus by Role

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4><i class="fas fa-book"></i> Newcomers</h4>
    <ul>
      <li>Solidify the basics in <a href="fundamentals.html">Fundamentals</a> before adopting these patterns</li>
      <li>Reach for Docker Compose before any orchestrator</li>
      <li>Adopt the sidecar pattern only once a single container is too limiting</li>
      <li>Follow security best practices from day one, not as a retrofit</li>
    </ul>
  </div>
  
  <div class="takeaway-card">
    <h4><i class="fas fa-code"></i> For Developers</h4>
    <ul>
      <li>Optimize Dockerfiles for build caching</li>
      <li>Use multi-stage builds to reduce image size</li>
      <li>Implement proper health checks</li>
      <li>Integrate Docker into your CI/CD pipeline</li>
    </ul>
  </div>
  
  <div class="takeaway-card">
    <h4><i class="fas fa-server"></i> For DevOps/SRE</h4>
    <ul>
      <li>Master networking for service mesh architectures</li>
      <li>Implement comprehensive monitoring and logging</li>
      <li>Use orchestration for high availability</li>
      <li>Plan for security at every layer</li>
    </ul>
  </div>
  
  <div class="takeaway-card">
    <h4><i class="fas fa-building"></i> For Architects</h4>
    <ul>
      <li>Design with microservices patterns in mind</li>
      <li>Consider WebAssembly for edge computing</li>
      <li>Plan for scalability and resilience</li>
      <li>Balance complexity with operational overhead</li>
    </ul>
  </div>
</div>

### The Modern Toolchain

The ecosystem around Docker has matured well beyond the original CLI and daemon. The tools below are worth knowing because they change how you build, scan, and run images day to day.

| Area | Tool | What it gives you |
|------|------|-------------------|
| Supply chain | Docker Scout | Vulnerability scanning and SBOM generation |
| Supply chain | Build attestations | SLSA provenance baked into the image |
| Build | BuildKit | The default builder: parallel stages, cache mounts, secrets |
| Build | Docker Build Cloud | Remote, shared builders for faster CI |
| Runtime | containerd | The OCI runtime Docker and Kubernetes share |
| Dev loop | Compose Watch | Auto-sync source into running containers |

**Drop-in alternatives** are also worth tracking. **Podman** runs daemonless, rootless containers with a Docker-compatible CLI; **Colima**, **Rancher Desktop**, and **OrbStack** are lighter Docker Desktop replacements for macOS. WASM runtimes (via `runwasi`/`crun`) sit at the other end of the spectrum for ultra-light, sandboxed workloads.

The durable principles do not change with the tooling: build for **consistency, isolation, and portability**, and choose the lightest runtime that meets your isolation needs.

## See Also

- [Docker Essentials](../docker-essentials.html) - Quick reference and command cheat sheet
- [Kubernetes](../kubernetes/) - Container orchestration at scale
- [CI/CD](../ci-cd/) - Docker in continuous integration workflows
- [AWS](../aws/) - ECS, EKS, and cloud container services
- [Terraform](../terraform/) - Infrastructure as Code for container deployments
- [Networking](../networking/) - Network concepts and container networking
- [Distributed Systems](../../distributed-systems/) - Distributed computing principles
