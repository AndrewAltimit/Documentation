---
layout: docs
title: "Docker: Storage & Security"
permalink: /docs/technology/docker/storage-security.html
toc: true
toc_sticky: true
hide_title: true
---

[Docker](./) &raquo; Storage &amp; Security

## Two Sides of the Same Problem

Storage and security are the two concerns that turn a throwaway container into something you can run in production. Storage decides what survives when a container stops; security decides who and what can touch that data while the container runs. They are tightly coupled — a database volume that outlives its container is also the most valuable thing an attacker can reach, and a secret that gets baked into an image layer is a storage decision with a security consequence.

This page covers both: the three ways Docker persists data, how to back it up and restore it, and the layered defenses that protect images, builds, and the runtime.

> **Looking for networking?** Network drivers, DNS-based service discovery, port publishing, overlay/macvlan, and network segmentation now live on their own page: **[Docker: Networking](docker-networking.html)**. Network isolation is a security control too, so treat the two pages as companions.

## Docker Storage: Volumes, Bind Mounts, and tmpfs

By default, data inside a container disappears when the container stops. This is actually a feature, not a bug: it keeps containers lightweight and reproducible. Every write a container makes goes into a thin, copy-on-write **writable layer** stacked on top of the read-only image layers. Delete the container and that layer is gone with it.

Most real applications, however, need to persist *something* — a database, uploaded files, a cache — beyond the lifetime of any single container. Docker offers three mount types for that, each with a different tradeoff:

| Scenario | Best Storage Type | Why |
|----------|-------------------|-----|
| Database files | Volume | Docker manages it, easy backups, best performance |
| Source code during development | Bind mount | See changes instantly without rebuilding |
| Configuration files | Bind mount | Edit on host, container reads immediately |
| Sensitive data (secrets, tokens) | tmpfs | Never written to disk, cleared when container stops |
| Build cache | Volume | Persists between builds, improves speed |

The three mount types compared across the dimensions that matter:

| Storage type | Use case | Performance | Portability | Management |
|--------------|----------|-------------|-------------|------------|
| **Volumes** | Production data, databases, shared data between containers | Best (native to Docker) | High (managed by Docker) | Easy (Docker commands) |
| **Bind mounts** | Development, config files, source code | Good (direct filesystem) | Low (host-dependent) | Manual (filesystem) |
| **tmpfs** | Temporary data, secrets, caches | Excellent (memory) | None (memory only) | Automatic (cleared on stop) |

### Where Each Mount Type Lives

Knowing *where* the bytes actually sit explains every tradeoff above:

- **Volumes** live in a directory Docker controls — on Linux, typically `/var/lib/docker/volumes/<name>/_data`. You never address that path directly; you refer to the volume by name and Docker handles the rest. Because Docker owns the lifecycle, volumes survive `docker rm`, can be listed and pruned with first-class commands, and can be backed by plugins (NFS, cloud block storage) for networked or replicated storage.
- **Bind mounts** map a specific host path straight into the container. There is no indirection: the container sees exactly the host directory you point at, with the host's permissions and ownership. That directness is the point for development and the hazard for production — the container inherits whatever lives at that path and can modify it unless you mount read-only.
- **tmpfs** mounts are backed by RAM (and swap). Nothing is written to the host filesystem, so the data is fast, volatile, and never lingers on disk after the container stops.

### Docker Volumes

**When to use:** Production databases, application state, any data that must survive container restarts.

Prefer the explicit `--mount` syntax in scripts and Compose — it is self-documenting and fails loudly on typos, whereas the terser `-v` syntax silently creates a new anonymous volume if you misspell a name.

```bash
# Create and use a named volume
docker volume create app-data
docker run -d -v app-data:/var/lib/postgresql/data postgres:16

# Equivalent with the explicit, self-documenting --mount syntax
docker run -d \
  --mount type=volume,source=app-data,target=/var/lib/postgresql/data \
  postgres:16

# List and clean up volumes
docker volume ls
docker volume inspect app-data   # See driver, mountpoint, labels
docker volume prune              # Remove unused (dangling) volumes
```

Anonymous volumes (created when an image's `VOLUME` instruction has no host counterpart, or when you run `-v /data` with no name) accumulate quickly and are the usual culprit behind a Docker host that mysteriously fills up. Audit them with `docker volume ls -f dangling=true` and reclaim space with `docker volume prune`.

### Bind Mounts

**When to use:** Development workflows where you want to edit files on your host and see changes immediately in the container.

```bash
# Mount the project (with its package.json) and set it as the workdir
docker run -d -v $(pwd):/app -w /app -p 3000:3000 node:22 npm run dev

# Mount config file read-only (container cannot modify)
docker run -d -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro nginx
```

The `:ro` suffix makes the mount read-only, preventing the container from modifying your host files.

Two bind-mount footguns worth internalizing:

- **A bind mount shadows whatever is already at the target path** inside the image. Mount your host's empty-ish project directory over `/app` and you hide the image's `/app` — including any `node_modules` the image built. The common fix is an anonymous volume "above" the bind mount, e.g. `-v $(pwd):/app -v /app/node_modules`, so the dependency directory the image built is preserved.
- **Ownership is the host's, not the container's.** If the container process runs as UID 1000 but your host files are owned by a different UID, you get permission errors. Align the UID with the `user:` directive (Compose) or `--user` flag, or `chown` on the host.

### tmpfs Mounts

**When to use:** Sensitive data like secrets or tokens that should never be written to disk, or temporary caches that can be discarded.

```bash
# Store secrets in memory only (never touches disk)
docker run -d --tmpfs /run/secrets:size=10m,mode=0700 my-app

# Fast temporary cache
docker run -d --tmpfs /app/cache:size=100m my-app
```

tmpfs mounts exist only in memory. When the container stops, the data is gone — ideal for sensitive information.

Because tmpfs consumes RAM, always set a `size=` limit — an unbounded tmpfs that fills up will pressure the host's memory and can trigger the OOM killer. tmpfs mounts are also a natural partner for `--read-only` containers (covered below): the root filesystem is immutable, but the handful of paths that genuinely need to be writable (`/tmp`, a PID file directory) are backed by disposable memory.

### Sharing Data Between Containers

**When to use:** When multiple containers need to read or write the same data, such as a web server and a log processor.

```bash
# Both containers access the same volume
docker volume create shared-data
docker run -d -v shared-data:/data --name writer my-app
docker run -d -v shared-data:/data:ro --name reader log-processor
```

The writer container can modify data; the reader has read-only access. Both see the same files.

Grant the least access that works: if a sidecar only needs to *read* logs, mount the volume `:ro` so a compromise of that container cannot corrupt the shared data. Note that Docker does not coordinate concurrent writes for you — if two containers write the same files, the application is responsible for locking.

### Backup and Restore

Volumes are managed by Docker, but Docker does not back them up for you. The portable pattern is to run a short-lived helper container (Alpine is ideal — tiny and has `tar`) that mounts the volume and streams a tarball to a bind-mounted host directory. Mount the source volume read-only during backup so a stuck job can never mutate live data.

```bash
# Backup: mount volume read-only, tar to host
docker run --rm -v app-data:/source:ro -v $(pwd):/backup \
  alpine tar czf /backup/backup.tar.gz -C /source .

# Restore: extract tar into a (typically fresh) volume
docker run --rm -v app-data:/target -v $(pwd):/backup:ro \
  alpine tar xzf /backup/backup.tar.gz -C /target
```

A few production cautions:

- **Quiesce stateful services first.** Backing up a live database volume with `tar` can capture a torn, mid-write state. For databases, prefer the engine's own consistent dump (`pg_dump`, `mysqldump`, `mongodump`) and back up *that* file, or stop/pause the container during the snapshot.
- **Restore into a fresh volume** when you can, so a partial extraction never leaves a half-old, half-new dataset mixed together.
- **Test your restores.** A backup you have never restored is a hypothesis, not a backup. Periodically restore into a throwaway volume and start a container against it.

For volume drivers that support it (cloud block storage, LVM, ZFS), filesystem- or block-level snapshots are faster and more consistent than `tar`, but they are storage-backend specific rather than portable Docker commands.

## Docker Security Best Practices

Container security is not about a single setting. It is about applying multiple layers of protection, from how you build images to how you run containers in production. No single control is sufficient; the goal is **defense in depth**, so that a failure at one layer is contained by the next.

Consider the following security layers:

| Layer | What It Protects | Key Actions |
|-------|------------------|-------------|
| Image | What goes into containers | Use minimal base images, scan for vulnerabilities |
| Build | The build process | Use BuildKit secrets, multi-stage builds |
| Runtime | Running containers | Drop capabilities, run as non-root, limit resources |
| Network | Container communication | Use custom networks, encrypt overlay traffic |
| Host | The Docker host | Keep Docker updated, use user namespaces |

The **Network** layer — custom bridges, segmentation, encrypted overlays, and restricting container-to-container traffic — is covered in depth on the [Docker: Networking](docker-networking.html) page; the rest of this section drills into image, build, runtime, and secret controls.

These practices significantly reduce your attack surface; start with the basics and add controls as requirements grow. Four principles run through all of them: **least privilege** (run containers with the minimal permissions they need), **defense in depth** (multiple security layers from host to application), **immutability** (keep containers stateless and read-only where possible), and **vulnerability scanning** (scan images regularly for known CVEs).

### Image Security

Everything downstream depends on what you put *into* the image. A container can only be as trustworthy as the layers it is built from.

```dockerfile
# Pin a specific minimal base, not a moving 'latest' tag
FROM alpine:3.20

# Install only what the app needs, then remove the package cache
RUN apk add --no-cache ca-certificates

# For compiled/static binaries, go further: a distroless or scratch base
# has no shell and no package manager, shrinking the attack surface
```

Key image-hardening habits:

- **Use minimal base images** (Alpine, Google's distroless, or `scratch` for static binaries). Fewer packages means fewer CVEs and a smaller attack surface.
- **Pin specific version tags**, never `latest`. `latest` is a moving target that makes builds non-reproducible and can silently pull in a vulnerable or breaking update.
- **Remove build-time cruft** — package caches, compilers, and dev headers — from the final image. A multi-stage build (compile in a fat builder stage, copy only the artifact into a slim runtime stage) is the cleanest way to do this. See [Dockerfiles & CI/CD](dockerfiles.html) for the full multi-stage pattern.
- **Never store secrets in image layers.** Layers are cached and shareable; anyone who pulls the image can `docker history` or unpack it to recover a baked-in token. (See *Secrets Management* below.)

### Running Containers Securely

**The most impactful change:** Run containers as non-root users. This single practice prevents many container escape vulnerabilities — if a process is compromised but only runs as an unprivileged UID, the blast radius is dramatically smaller.

Add a non-root user in the Dockerfile:

```dockerfile
FROM alpine:3.20
RUN adduser -D appuser
COPY --chown=appuser . /app
USER appuser
CMD ["python3", "app.py"]
```

Then harden the runtime at launch:

```bash
# Read-only filesystem with necessary tmpfs
docker run -d --read-only --tmpfs /tmp my-app

# Drop all capabilities, add only what is needed
docker run -d --cap-drop ALL --cap-add NET_BIND_SERVICE nginx

# Prevent a process from gaining new privileges (defeats setuid escalation)
docker run -d --security-opt no-new-privileges my-app

# Limit resources to prevent DoS
docker run -d --memory 512m --cpus 0.5 --pids-limit 100 my-app
```

Each flag adds a layer: `--read-only` prevents filesystem modifications, `--cap-drop ALL` removes Linux capabilities, `--security-opt no-new-privileges` blocks setuid escalation, and the resource limits prevent runaway processes from starving the host.

Beyond these flags, three host-level controls deepen runtime isolation:

- **Linux capabilities** are fine-grained slices of root's power. Dropping all and re-adding only what you need (e.g. `NET_BIND_SERVICE` to bind ports below 1024) is far safer than running with the default capability set.
- **Seccomp** filters which syscalls a container may make. Docker's default profile already blocks dozens of dangerous syscalls; only loosen it deliberately.
- **AppArmor / SELinux** apply mandatory access control policies to the container process. Keep the default `docker-default` AppArmor profile (or the SELinux `container_t` type) enabled rather than running with `--security-opt apparmor=unconfined`.
- **User namespaces** (`userns-remap`) map container UID 0 to an unprivileged UID on the host, so even a "root" process inside the container is non-root outside it.

### Secrets Management

**Never put secrets in:** Dockerfiles, environment variables in compose files committed to git, or image layers. These are all visible to anyone with access to the image or source code — `docker history` and `docker inspect` will happily reveal them.

| Method | Use case | How it works |
|--------|----------|--------------|
| Docker Secrets (Swarm) | Production services | Stored encrypted, mounted as files at `/run/secrets/` |
| BuildKit secrets | Build-time credentials | Available only during build, not in the final image |
| External secrets manager | Enterprise deployments | Vault, AWS Secrets Manager inject at runtime |
| Environment file | Development only | `.env` file loaded at runtime (never commit to git) |

```bash
# BuildKit: secret available only during build, never in a layer
DOCKER_BUILDKIT=1 docker build --secret id=token,src=./token.txt .

# Development: use .env file (add to .gitignore!)
docker run --env-file .env my-app
```

The throughline is *don't bake, mount instead*: a good secret is delivered to the running container as a file (`/run/secrets/...`) or pulled at startup from a secrets manager, so it never lands in an image layer, in your shell history, or in `docker inspect` output. Prefer file-mounted secrets over environment variables even at runtime, since env vars are visible to any process that can read `/proc/<pid>/environ` and are frequently leaked into logs and crash dumps.

### Image Security Scanning

Scan images for known vulnerabilities before deploying them, and integrate scanning into your CI/CD pipeline to catch issues early — a vulnerable base image discovered in CI is cheap to fix; the same image discovered in production is an incident.

```bash
# Docker Scout (built into Docker Desktop)
docker scout cves my-app:latest

# Trivy (open source, widely used)
trivy image my-app:latest

# Enable image signing to verify provenance
export DOCKER_CONTENT_TRUST=1
docker pull my-registry/my-app:latest  # Fails if not signed
```

Scanning tells you *what is vulnerable*; signing (Docker Content Trust, or the newer Sigstore/cosign workflow) tells you *whether the image is the one you built*. Use both: scan on every build, and require signatures on pull so a tampered or unexpected image is rejected before it ever runs.

### Security Compliance Checklist

**Image security**

- Use minimal base images (alpine, distroless)
- Scan images for vulnerabilities regularly
- Don't store secrets in images
- Use specific version tags, not `latest`
- Sign images with Docker Content Trust
- Remove unnecessary packages and files

**Runtime security**

- Run containers as a non-root user
- Use read-only root filesystems
- Drop unnecessary capabilities
- Limit resources (memory, CPU, PIDs)
- Use security profiles (AppArmor, SELinux, Seccomp)
- Add `--security-opt no-new-privileges`
- Isolate containers with user namespaces

**Network security** (detailed on the [Docker: Networking](docker-networking.html) page)

- Use custom bridge networks, not the default
- Encrypt overlay network traffic
- Implement network segmentation
- Use TLS for container communication
- Restrict container-to-container communication

## Troubleshooting Common Docker Issues

When something goes wrong, start with the simplest checks and work your way to more detailed investigation. Many storage and security problems surface as cryptic permission errors, exhausted disk, or containers that exit immediately — the techniques below isolate the cause quickly.

### Debugging Containers

**Container will not start:**

```bash
# First, check the logs
docker logs container-name

# Get the exit code (non-zero means error)
docker inspect container-name --format='{% raw %}{{.State.ExitCode}}{% endraw %}'

# Start an interactive shell to investigate
docker run -it --entrypoint /bin/sh my-image
```

**Permission and mount problems:**

```bash
# Inspect exactly what is mounted and how
docker inspect container-name --format='{% raw %}{{json .Mounts}}{% endraw %}'

# Check the UID the process runs as vs. the file ownership
docker exec container-name id
docker exec container-name ls -ln /data
```

Most volume/bind-mount errors are UID mismatches: the in-container user does not own the mounted files. Align the user with `--user` or `chown` on the host.

**Performance problems:**

```bash
# Real-time stats for all containers
docker stats

# Check disk usage broken down by images, containers, and volumes
docker system df -v
```

### Common Error Solutions

| Error | Quick Fix |
|-------|-----------|
| "Cannot connect to Docker daemon" | `sudo systemctl start docker` or add user to docker group |
| "No space left on device" | `docker system prune -a --volumes` (note: also deletes unused volumes) |
| "Permission denied" on a mount | UID mismatch — align `--user` with host file ownership, or `chown` the host path |
| "Permission denied" on the socket | Run with sudo, or add user to docker group and log out/in |
| Read-only filesystem error | Container started `--read-only`; add a `--tmpfs` for the path that needs writes |

**Cleaning up disk space:**

```bash
# See what is using space
docker system df

# Remove everything unused (images, containers, AND volumes)
docker system prune -a --volumes

# Reclaim only dangling/anonymous volumes (safer)
docker volume prune
```

Be deliberate with `--volumes`: it deletes any volume not currently attached to a container, which can destroy data you meant to keep. Prefer `docker volume prune` when you only want to reclaim anonymous leftovers.

### Health Checks

Health checks let Docker know if your application is actually working, not just running.

```dockerfile
# Add to Dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

```bash
# Check health status
docker ps  # Shows health in STATUS column
docker inspect --format='{% raw %}{{.State.Health.Status}}{% endraw %}' container-name
```

## Key Takeaways

- **Volumes for persistence.** Container writable layers are ephemeral. Use named volumes for production data, bind mounts for local development, and tmpfs for secrets that must never touch disk.
- **Back up volumes yourself.** Docker does not back up volumes for you. Tar them from a read-only helper container, quiesce databases before snapshotting, and actually test your restores.
- **Run as non-root.** The single highest-impact hardening step. Add a `USER` directive, drop capabilities with `--cap-drop ALL`, and add `--security-opt no-new-privileges`.
- **Mount secrets, don't bake them.** Keep secrets out of image layers and env vars. Use BuildKit secrets at build time and file-mounted secrets or a secrets manager at runtime, and scan and sign images before deploy.

---

## See Also

- [Fundamentals](fundamentals.html) - Images, containers, layers, and CLI basics
- [Docker: Networking](docker-networking.html) - Network drivers, service discovery, and network-layer security
- [Dockerfiles &amp; CI/CD](dockerfiles.html) - Multi-stage builds, BuildKit secrets, and automated pipelines
- [Advanced Patterns](advanced.html) - Production architectures and runtime alternatives
- [Docker Essentials](../docker-essentials.html) - Quick command reference
- [Cybersecurity](../cybersecurity/) - Broader security fundamentals
