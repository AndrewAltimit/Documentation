---
layout: docs
title: "Docker: Storage & Security"
permalink: /docs/technology/docker/storage-security.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #0066cc 0%, #00aaff 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Docker: Storage & Security</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Learn to persist data with volumes, configure networks, and implement security best practices for production containers.</p>
</div>

## Docker in Practice

This page covers the practical aspects of working with Docker: persisting data, configuring networks, and securing your containers. These topics become essential as you move beyond simple experiments into real-world deployments.

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

### Installing Docker

Before you can use Docker, you need to install it on your system. The installation process varies by operating system, but the result is the same: a working Docker daemon that can build and run containers.

<div class="installation-guide">
  <p class="install-intro">Choose your platform below. Most users will want Docker Desktop (Windows/Mac) or Docker Engine (Linux).</p>
  
  <div class="install-tabs">
    <h4><i class="fas fa-linux"></i> Linux Installation</h4>
    
    <div class="install-method">
      <h5>Ubuntu/Debian Quick Install</h5>
      <pre><code class="language-bash"># Install Docker using the convenience script
curl -fsSL https://get.docker.com | sudo sh

# Add your user to the docker group (logout required)
sudo usermod -aG docker $USER

# Verify installation after logging back in
docker --version</code></pre>
      <p class="explanation">The convenience script handles repository setup automatically. For production systems, see the <a href="https://docs.docker.com/engine/install/ubuntu/">official installation guide</a> for manual setup.</p>
    </div>

    <div class="install-method">
      <h5>Post-Installation Setup</h5>
      <pre><code class="language-bash"># Enable Docker to start on boot
sudo systemctl enable docker

# Verify everything works
docker run hello-world</code></pre>
    </div>
  </div>
</div>

The first step in your Docker journey is installation. Docker provides packages for all major operating systems:

- [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Install Docker on Debian](https://docs.docker.com/engine/install/debian/)
- [Install Docker on Fedora](https://docs.docker.com/engine/install/fedora/)
- [Install Docker on CentOS](https://docs.docker.com/engine/install/centos/)
- [Install Docker on Windows](https://docs.docker.com/docker-for-windows/install/)
- [Install Docker on macOS](https://docs.docker.com/docker-for-mac/install/)

Once Docker is installed, you'll interact with it primarily through the command-line interface. Let's explore the essential commands that will become part of your daily workflow.

### Common Docker CLI Commands

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
    <div class="workflow-commands">
      <div class="workflow-item">
        <code>docker build -t myapp:1.0 .</code>
        <span>Build your application image</span>
      </div>
      <div class="workflow-item">
        <code>docker run -d -p 8080:80 myapp:1.0</code>
        <span>Run container in detached mode</span>
      </div>
      <div class="workflow-item">
        <code>docker logs -f &lt;container_id&gt;</code>
        <span>Monitor application logs</span>
      </div>
      <div class="workflow-item">
        <code>docker push myapp:1.0</code>
        <span>Share image via registry</span>
      </div>
    </div>
  </div>
</div>

### Docker Compose

- Start a multi-container application: `docker-compose up -d`
- Stop a multi-container application: `docker-compose down`

## Docker Storage: Volumes, Bind Mounts, and tmpfs

By default, data inside a container disappears when the container stops. This is actually a feature, not a bug: it keeps containers lightweight and reproducible. However, most real applications need to persist data somewhere.

Consider the following scenarios and which storage type fits each:

| Scenario | Best Storage Type | Why |
|----------|-------------------|-----|
| Database files | Volume | Docker manages it, easy backups, best performance |
| Source code during development | Bind mount | See changes instantly without rebuilding |
| Configuration files | Bind mount | Edit on host, container reads immediately |
| Sensitive data (secrets, tokens) | tmpfs | Never written to disk, cleared when container stops |
| Build cache | Volume | Persists between builds, improves speed |

<div class="storage-section">
  <h3><i class="fas fa-database"></i> Understanding Docker Storage</h3>

  <p class="storage-intro">Docker provides three ways to persist data beyond the container lifecycle. The right choice depends on your use case.</p>
  
  <div class="storage-comparison">
    <table class="storage-table">
      <thead>
        <tr>
          <th>Storage Type</th>
          <th>Use Case</th>
          <th>Performance</th>
          <th>Portability</th>
          <th>Management</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Volumes</strong></td>
          <td>Production data, databases, shared data between containers</td>
          <td>Best (native to Docker)</td>
          <td>High (managed by Docker)</td>
          <td>Easy (Docker commands)</td>
        </tr>
        <tr>
          <td><strong>Bind Mounts</strong></td>
          <td>Development, config files, source code</td>
          <td>Good (direct filesystem)</td>
          <td>Low (host-dependent)</td>
          <td>Manual (filesystem)</td>
        </tr>
        <tr>
          <td><strong>tmpfs</strong></td>
          <td>Temporary data, secrets, caches</td>
          <td>Excellent (memory)</td>
          <td>None (memory only)</td>
          <td>Automatic (cleared on stop)</td>
        </tr>
      </tbody>
    </table>
  </div>
  
  <h3><i class="fas fa-hdd"></i> Docker Volumes</h3>

  <p><strong>When to use:</strong> Production databases, application state, any data that must survive container restarts.</p>

  <div class="volume-examples">
    <h4>Essential Volume Commands</h4>
    <pre><code class="language-bash"># Create and use a named volume
docker volume create app-data
docker run -d -v app-data:/var/lib/postgresql/data postgres:15

# List and clean up volumes
docker volume ls
docker volume prune  # Remove unused volumes</code></pre>

    <h4>Backup and Restore</h4>
    <pre><code class="language-bash"># Backup: mount volume read-only, tar to host
docker run --rm -v app-data:/source:ro -v $(pwd):/backup \
  alpine tar czf /backup/backup.tar.gz -C /source .

# Restore: extract tar into volume
docker run --rm -v app-data:/target -v $(pwd):/backup:ro \
  alpine tar xzf /backup/backup.tar.gz -C /target</code></pre>
  </div>
  
  <h3><i class="fas fa-link"></i> Bind Mounts</h3>

  <p><strong>When to use:</strong> Development workflows where you want to edit files on your host and see changes immediately in the container.</p>

  <div class="bind-mount-examples">
    <h4>Development Workflow</h4>
    <pre><code class="language-bash"># Mount source code for live development
docker run -d -v $(pwd)/src:/app/src -p 3000:3000 node:18 npm run dev

# Mount config file read-only (container cannot modify)
docker run -d -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro nginx</code></pre>
    <p class="explanation">The <code>:ro</code> suffix makes the mount read-only, preventing the container from modifying your host files.</p>
  </div>
  
  <h3><i class="fas fa-memory"></i> tmpfs Mounts</h3>

  <p><strong>When to use:</strong> Sensitive data like secrets or tokens that should never be written to disk, or temporary caches that can be discarded.</p>

  <div class="tmpfs-examples">
    <h4>Secure Temporary Storage</h4>
    <pre><code class="language-bash"># Store secrets in memory only (never touches disk)
docker run -d --tmpfs /run/secrets:size=10m,mode=0700 my-app

# Fast temporary cache
docker run -d --tmpfs /app/cache:size=100m my-app</code></pre>
    <p class="explanation">tmpfs mounts exist only in memory. When the container stops, the data is gone. This is ideal for sensitive information.</p>
  </div>
  
  <h3><i class="fas fa-share-alt"></i> Sharing Data Between Containers</h3>

  <p><strong>When to use:</strong> When multiple containers need to read or write the same data, such as a web server and a log processor.</p>

  <div class="data-sharing-examples">
    <h4>Volume Sharing Pattern</h4>
    <pre><code class="language-bash"># Both containers access the same volume
docker volume create shared-data
docker run -d -v shared-data:/data --name writer my-app
docker run -d -v shared-data:/data:ro --name reader log-processor</code></pre>
    <p class="explanation">The writer container can modify data; the reader has read-only access. Both see the same files.</p>
  </div>
</div>

## Docker Networking In-Depth

Networking determines how containers communicate with each other, with the host, and with external services. Getting this right is essential for both functionality and security.

<div class="networking-section">
  <h3><i class="fas fa-network-wired"></i> Docker Network Architecture</h3>

  <p class="network-intro">Docker provides several network drivers for different scenarios. The default (bridge) works for most cases, but understanding the alternatives helps you make better architectural decisions.</p>
  
  <div class="network-drivers">
    <h4>Network Driver Overview</h4>
    
    <div class="driver-grid">
      <div class="driver-card">
        <h5><i class="fas fa-bridge"></i> Bridge (default)</h5>
        <p>Default network for standalone containers. Provides network isolation and automatic DNS resolution between containers.</p>
        <code>docker network create --driver bridge my-bridge</code>
      </div>
      
      <div class="driver-card">
        <h5><i class="fas fa-server"></i> Host</h5>
        <p>Removes network isolation between container and host. Container uses host's network directly.</p>
        <code>docker run --network host nginx</code>
      </div>
      
      <div class="driver-card">
        <h5><i class="fas fa-layer-group"></i> Overlay</h5>
        <p>Creates distributed networks among multiple Docker hosts. Used in Swarm mode for multi-host communication.</p>
        <code>docker network create --driver overlay --attachable my-overlay</code>
      </div>
      
      <div class="driver-card">
        <h5><i class="fas fa-ethernet"></i> Macvlan</h5>
        <p>Assigns MAC address to containers, making them appear as physical devices on the network.</p>
        <code>docker network create -d macvlan --subnet=192.168.1.0/24 my-macvlan</code>
      </div>
      
      <div class="driver-card">
        <h5><i class="fas fa-ban"></i> None</h5>
        <p>Disables all networking for the container. Used for maximum isolation.</p>
        <code>docker run --network none alpine</code>
      </div>
    </div>
  </div>
  
  <h3><i class="fas fa-project-diagram"></i> Bridge Networking Deep Dive</h3>

  <p><strong>Key concept:</strong> Always create custom bridge networks for your applications. Unlike the default bridge, custom networks provide automatic DNS resolution between containers.</p>

  <div class="bridge-networking">
    <h4>Creating Custom Networks</h4>
    <pre><code class="language-bash"># Create a network and run containers on it
docker network create my-app-network
docker run -d --name web --network my-app-network nginx
docker run -d --name db --network my-app-network postgres

# Containers can find each other by name
docker exec web ping db  # Works!</code></pre>

    <h4>Network Isolation Pattern</h4>
    <pre><code class="language-bash"># Isolate frontend from database
docker network create frontend
docker network create backend
docker run -d --name webapp --network frontend nginx
docker run -d --name api --network backend my-api

# Connect API to both networks (acts as bridge)
docker network connect frontend api</code></pre>
    <p class="explanation">The webapp can reach the api, but not the database directly. The api can reach both. This is a common security pattern.</p>
  </div>
  
  <h3><i class="fas fa-globe"></i> Overlay Networking for Swarm</h3>

  <p><strong>When to use:</strong> Docker Swarm deployments where services need to communicate across multiple hosts.</p>

  <div class="overlay-networking">
    <pre><code class="language-bash"># Create encrypted overlay network
docker network create --driver overlay --opt encrypted my-overlay

# Services on this network can find each other across hosts
docker service create --name api --network my-overlay my-api</code></pre>
  </div>

  <h3><i class="fas fa-ethernet"></i> Macvlan Networking</h3>

  <p><strong>When to use:</strong> When containers need to appear as physical devices on your network (legacy system integration, specific IP requirements).</p>

  <div class="macvlan-networking">
    <pre><code class="language-bash"># Container gets a real IP on your network
docker network create -d macvlan \
  --subnet=192.168.1.0/24 --gateway=192.168.1.1 \
  -o parent=eth0 my-macvlan

docker run -d --network my-macvlan --ip 192.168.1.100 nginx</code></pre>
  </div>
  
  <h3><i class="fas fa-chart-network"></i> Advanced Networking Patterns</h3>

  <p>For complex deployments, consider these patterns:</p>

  <div class="network-patterns">
    <ul>
      <li><strong>Service mesh</strong>: Use a proxy (Envoy, Traefik) to handle routing, load balancing, and observability</li>
      <li><strong>Network segmentation</strong>: Create separate networks for frontend, backend, and database tiers</li>
      <li><strong>Firewall rules</strong>: Use iptables DOCKER-USER chain to restrict container traffic</li>
    </ul>
    <p>These patterns are typically managed through orchestration tools like Kubernetes or Docker Swarm rather than manual configuration.</p>
  </div>
</div>

## Docker Security Best Practices

Container security is not about a single setting. It is about applying multiple layers of protection, from how you build images to how you run containers in production.

Consider the following security layers:

| Layer | What It Protects | Key Actions |
|-------|------------------|-------------|
| Image | What goes into containers | Use minimal base images, scan for vulnerabilities |
| Build | The build process | Use BuildKit secrets, multi-stage builds |
| Runtime | Running containers | Drop capabilities, run as non-root, limit resources |
| Network | Container communication | Use custom networks, encrypt overlay traffic |
| Host | The Docker host | Keep Docker updated, use user namespaces |

<div class="security-section">
  <h3><i class="fas fa-shield-alt"></i> Container Security Fundamentals</h3>

  <p class="security-intro">The following practices significantly reduce your attack surface. Start with the basics and add more controls as your security requirements grow.</p>
  
  <div class="security-principles">
    <h4>Security Principles</h4>
    
    <div class="principle-grid">
      <div class="principle-card">
        <i class="fas fa-user-lock"></i>
        <h5>Least Privilege</h5>
        <p>Run containers with minimal permissions required for operation</p>
      </div>
      
      <div class="principle-card">
        <i class="fas fa-layer-group"></i>
        <h5>Defense in Depth</h5>
        <p>Multiple security layers from host to application</p>
      </div>
      
      <div class="principle-card">
        <i class="fas fa-shield-virus"></i>
        <h5>Immutability</h5>
        <p>Containers should be stateless and read-only where possible</p>
      </div>
      
      <div class="principle-card">
        <i class="fas fa-search"></i>
        <h5>Vulnerability Scanning</h5>
        <p>Regular scanning of images for known vulnerabilities</p>
      </div>
    </div>
  </div>
  
  <h3><i class="fas fa-user-shield"></i> Running Containers Securely</h3>

  <p><strong>The most impactful change:</strong> Run containers as non-root users. This single practice prevents many container escape vulnerabilities.</p>

  <div class="secure-runtime">
    <h4>Non-Root User in Dockerfile</h4>
    <pre><code class="language-dockerfile">FROM alpine:3.18
RUN adduser -D appuser
COPY --chown=appuser . /app
USER appuser
CMD ["python3", "app.py"]</code></pre>

    <h4>Runtime Hardening</h4>
    <pre><code class="language-bash"># Read-only filesystem with necessary tmpfs
docker run -d --read-only --tmpfs /tmp my-app

# Drop all capabilities, add only what is needed
docker run -d --cap-drop ALL --cap-add NET_BIND_SERVICE nginx

# Limit resources to prevent DoS
docker run -d --memory 512m --cpus 0.5 --pids-limit 100 my-app</code></pre>
    <p class="explanation">Each flag adds a layer of protection. <code>--read-only</code> prevents filesystem modifications. <code>--cap-drop ALL</code> removes Linux capabilities. Resource limits prevent runaway processes.</p>
  </div>
  
  <h3><i class="fas fa-key"></i> Secrets Management</h3>

  <p><strong>Never put secrets in:</strong> Dockerfiles, environment variables in compose files committed to git, or image layers. These are all visible to anyone with access to the image or source code.</p>

  <div class="secrets-management">
    <h4>Safe Options for Secrets</h4>

| Method | Use Case | How It Works |
|--------|----------|--------------|
| Docker Secrets (Swarm) | Production services | Secrets stored encrypted, mounted as files at /run/secrets/ |
| BuildKit secrets | Build-time credentials | Secret available only during build, not in final image |
| External secrets manager | Enterprise deployments | Vault, AWS Secrets Manager inject at runtime |
| Environment file | Development only | .env file loaded at runtime (never commit to git) |

    <pre><code class="language-bash"># BuildKit: secret available only during build
DOCKER_BUILDKIT=1 docker build --secret id=token,src=./token.txt .

# Development: use .env file (add to .gitignore!)
docker run --env-file .env my-app</code></pre>
  </div>
  
  <h3><i class="fas fa-search-plus"></i> Image Security Scanning</h3>

  <p>Scan images for known vulnerabilities before deploying them. Integrate scanning into your CI/CD pipeline to catch issues early.</p>

  <div class="image-scanning">
    <pre><code class="language-bash"># Docker Scout (built into Docker Desktop)
docker scout cves my-app:latest

# Trivy (open source, widely used)
trivy image my-app:latest

# Enable image signing to verify provenance
export DOCKER_CONTENT_TRUST=1
docker pull my-registry/my-app:latest  # Fails if not signed</code></pre>
  </div>
  
  <h3><i class="fas fa-clipboard-check"></i> Security Compliance Checklist</h3>
  
  <div class="security-checklist">
    <h4>Image Security</h4>
    <ul>
      <li>✓ Use minimal base images (alpine, distroless)</li>
      <li>✓ Scan images for vulnerabilities regularly</li>
      <li>✓ Don't store secrets in images</li>
      <li>✓ Use specific version tags, not 'latest'</li>
      <li>✓ Sign images with Docker Content Trust</li>
      <li>✓ Remove unnecessary packages and files</li>
    </ul>
    
    <h4>Runtime Security</h4>
    <ul>
      <li>✓ Run containers as non-root user</li>
      <li>✓ Use read-only root filesystems</li>
      <li>✓ Drop unnecessary capabilities</li>
      <li>✓ Limit resources (memory, CPU, PIDs)</li>
      <li>✓ Use security profiles (AppArmor, SELinux, Seccomp)</li>
      <li>✓ Isolate containers with user namespaces</li>
    </ul>
    
    <h4>Network Security</h4>
    <ul>
      <li>✓ Use custom bridge networks, not default</li>
      <li>✓ Encrypt overlay network traffic</li>
      <li>✓ Implement network segmentation</li>
      <li>✓ Use TLS for container communication</li>
      <li>✓ Restrict container-to-container communication</li>
    </ul>
  </div>
</div>

## Troubleshooting Common Docker Issues

When something goes wrong, start with the simplest checks and work your way to more detailed investigation.

<div class="troubleshooting-section">
  <h3><i class="fas fa-tools"></i> Debugging Containers</h3>

  <div class="debug-techniques">
    <h4>Container Will Not Start</h4>
    <pre><code class="language-bash"># First, check the logs
docker logs container-name

# Get the exit code (non-zero means error)
docker inspect container-name --format='{% raw %}{{.State.ExitCode}}{% endraw %}'

# Start an interactive shell to investigate
docker run -it --entrypoint /bin/sh my-image</code></pre>

    <h4>Connectivity Issues</h4>
    <pre><code class="language-bash"># Use netshoot to debug networking
docker run --rm --network container:my-app nicolaka/netshoot

# Inside: test DNS and connectivity
nslookup service-name
curl -v http://service-name:port</code></pre>

    <h4>Performance Problems</h4>
    <pre><code class="language-bash"># Real-time stats for all containers
docker stats

# Check disk usage
docker system df</code></pre>
  </div>
  
  <h3><i class="fas fa-exclamation-circle"></i> Common Error Solutions</h3>

  <div class="error-solutions">
| Error | Quick Fix |
|-------|-----------|
| "Cannot connect to Docker daemon" | `sudo systemctl start docker` or add user to docker group |
| "No space left on device" | `docker system prune -a --volumes` |
| "Port already in use" | `sudo lsof -i :8080` to find the process, then kill it or use a different port |
| "Permission denied" | Run with sudo, or add user to docker group and log out/in |

    <h4>Cleaning Up Disk Space</h4>
    <pre><code class="language-bash"># See what is using space
docker system df

# Remove everything unused (images, containers, volumes)
docker system prune -a --volumes</code></pre>
  </div>

  <h3><i class="fas fa-heartbeat"></i> Health Checks</h3>

  <p>Health checks let Docker know if your application is actually working, not just running.</p>

  <div class="health-monitoring">
    <pre><code class="language-dockerfile"># Add to Dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1</code></pre>
    <pre><code class="language-bash"># Check health status
docker ps  # Shows health in STATUS column
docker inspect --format='{% raw %}{{.State.Health.Status}}{% endraw %}' container-name</code></pre>
  </div>
</div>

After mastering the basic Docker commands, the next crucial skill is creating your own Docker images. This is where Dockerfiles come in - they are the blueprint for building custom container images.

