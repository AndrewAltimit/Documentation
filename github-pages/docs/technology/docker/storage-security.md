---
layout: docs
title: "Docker: Storage & Security"
description: "Persisting container data with volumes, bind mounts, tmpfs, and image mounts; backup and restore; and defense-in-depth hardening of Docker images, builds, secrets, the runtime, and the host."
permalink: /docs/technology/docker/storage-security.html
toc: true
toc_sticky: true
hide_title: true
---

[Docker](./) &raquo; Storage &amp; Security

Storage decides what survives when a container is removed; security decides who and what can reach that data, and the host, while the container runs. The two are coupled: a database volume that outlives its container is the most valuable thing an attacker can reach, and a secret copied into an image layer is a storage decision with a security consequence. This page covers Docker's mount types, backup and restore, and the layered controls that harden images, builds, secrets, the container runtime, and the host. Network isolation, also a security control, has its own page: [Docker: Networking](docker-networking.html).

## Docker Storage: Volumes, Bind Mounts, and tmpfs

Every write a container makes lands in its **writable layer**, a thin copy-on-write layer on top of the read-only image (see [Fundamentals](fundamentals.html#images-and-layers)). That layer is deleted with the container and is slow for write-heavy workloads. Anything that must persist, be shared, or be written heavily should live in a **mount** instead.

```mermaid
flowchart LR
    subgraph C["Container filesystem view"]
        direction TB
        root["/ (image layers + writable layer)"]
        m1["/var/lib/postgresql/data"]
        m2["/app/src"]
        m3["/tmp"]
        m4["/tools"]
    end
    V[("Volume<br/>/var/lib/docker/volumes/pgdata<br/>or a volume driver (NFS, cloud disk)")] --> m1
    B["Host directory<br/>~/project/src"] --> m2
    T["Host RAM (tmpfs)"] --> m3
    I["Another image<br/>(read-only)"] --> m4
```

| Mount type | Backed by | Lifecycle | Typical use | Caveats |
|------------|-----------|-----------|-------------|---------|
| **Volume** | Directory managed by Docker, or a volume driver | Independent of containers; removed only explicitly | Databases, uploads, any persistent state | Host path is an implementation detail; back it up yourself |
| **Bind mount** | Any host path | Tied to the host filesystem | Source code in development, host config files | Host-dependent, inherits host ownership, container can modify host files |
| **tmpfs** | Host memory (may be swapped) | Discarded when the container stops | Scratch space, runtime sockets, PID files, secrets in flight | Consumes RAM; Linux only |
| **Image mount** | Layers of another image, read-only | Lives with the container | Bringing debug tools or shared assets into a container | Requires the containerd image store |
| **Named pipe** | Windows named pipe | Host-defined | Windows containers talking to the Engine API | Windows only |

`--mount` is the explicit syntax and is preferred in scripts: `type=`, `source=`, `target=` and options are named, and a misspelled bind source is an error. The short `-v` form is terser but silently creates a new volume or host directory when a name or path is wrong.

### Volumes

Named volumes are the default choice for persistent data. Docker owns their lifecycle, they work the same on every platform (including Docker Desktop, where bind mounts cross a VM boundary and are slower), and they can be backed by drivers for NFS, CIFS, or cloud block storage.

```bash
docker volume create pgdata
docker run -d --name db \
  -e POSTGRES_PASSWORD=example \
  --mount type=volume,source=pgdata,target=/var/lib/postgresql/data \
  postgres:17

docker volume ls
docker volume inspect pgdata          # driver, mountpoint, labels
docker volume ls -f dangling=true     # volumes attached to no container
docker volume rm pgdata
```

Behaviors worth knowing:

- **Pre-population.** When an *empty* volume is mounted over a directory that already contains files in the image, Docker copies those files into the volume first. Add `volume-nocopy` to disable this. Bind mounts never pre-populate; they hide the image's content.
- **Subpaths.** One volume can serve several containers at different subdirectories: `--mount type=volume,src=logs,dst=/var/log/app1,volume-subpath=app1`. The subdirectory must already exist in the volume.
- **Anonymous volumes.** A `VOLUME` instruction in an image, or `-v /data` without a name, creates a volume with a random name. These accumulate and are the usual reason a Docker host fills up; `docker run --rm` removes a container's anonymous volumes with it, and `docker volume prune` removes unused anonymous volumes (add `-a` to include unused named volumes).

In Compose, declare volumes at the top level and reference them by name:

```yaml
services:
  db:
    image: postgres:17
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata: {}
```

`docker compose down` keeps named volumes; `docker compose down -v` deletes them.

### Bind Mounts

A bind mount exposes a host path directly. There is no copy and no indirection, which is what makes it useful for development and risky in production.

```bash
# Live-edit source in a dev container
docker run --rm -it -p 3000:3000 \
  --mount type=bind,source="$(pwd)",target=/app \
  -w /app node:24 npm run dev

# Read-only config file
docker run -d \
  --mount type=bind,source="$(pwd)/nginx.conf",target=/etc/nginx/nginx.conf,readonly \
  nginx:1.29
```

- **Shadowing.** A bind mount hides whatever the image had at the target path. Mounting a project directory over `/app` also hides the image's `/app/node_modules`; a common workaround is an extra anonymous volume at `/app/node_modules`. For Compose development loops, `docker compose watch` (file sync and rebuild on change) avoids bind mounts entirely.
- **Ownership.** Files keep their host UID/GID. If the container runs as UID 1000 and the host files belong to another UID, writes fail. Match the UID with `--user` or `user:`, or adjust ownership on the host. On SELinux hosts, add the `:z` (shared) or `:Z` (private) suffix with `-v` so Docker relabels the content.
- **Blast radius.** Never bind-mount sensitive host paths (`/`, `/etc`, `/var/run/docker.sock`) into untrusted containers. Mount read-only whenever the container only needs to read.

### tmpfs Mounts

A tmpfs mount lives in memory and disappears when the container stops.

```bash
docker run -d --read-only \
  --tmpfs /tmp:rw,size=64m,mode=1777 \
  --tmpfs /run:rw,size=8m \
  my-app:1.4.2
```

Always set `size=`: tmpfs consumes host RAM, counts against the container's memory limit, and an unbounded mount can drive the host into memory pressure. tmpfs pages can be written to swap, so "never touches disk" holds only on hosts without swap or with encrypted swap. The natural pairing is a `--read-only` root filesystem with small tmpfs mounts for the few paths that must be writable.

### Image Mounts

An image mount makes another image's filesystem available read-only at a chosen path, without adding it to the container's own image. It is useful for bringing a tool-rich image into a minimal (distroless) container for debugging, or for sharing read-only assets such as model weights or static files across containers built from different images. Image mounts require the containerd image store, the default on fresh Docker Engine 29 installs.

### Sharing Data Between Containers

Several containers can mount the same volume. Grant the least access that works, and remember that Docker does not coordinate concurrent writers: file locking is the application's job.

```bash
docker volume create shared-logs
docker run -d --name app      --mount type=volume,src=shared-logs,dst=/var/log/app        my-app
docker run -d --name shipper  --mount type=volume,src=shared-logs,dst=/logs,readonly       log-shipper
```

### Backup and Restore

Docker does not back up volumes. The portable approach runs a short-lived helper container that mounts the volume and streams a tarball to the host:

```bash
# Backup (source mounted read-only)
docker run --rm \
  --mount type=volume,src=pgdata,dst=/source,readonly \
  --mount type=bind,src="$(pwd)",dst=/backup \
  alpine:3.22 tar czf /backup/pgdata-$(date +%F).tar.gz -C /source .

# Restore into a fresh volume
docker volume create pgdata-restored
docker run --rm \
  --mount type=volume,src=pgdata-restored,dst=/target \
  --mount type=bind,src="$(pwd)",dst=/backup,readonly \
  alpine:3.22 tar xzf /backup/pgdata-2026-09-22.tar.gz -C /target
```

For production data:

- **Use the application's consistent backup tool for databases.** A `tar` of a live database directory can capture a torn, mid-write state. Prefer `pg_dump`/`pg_basebackup`, `mysqldump`, or `mongodump` (run with `docker exec`), or stop the container during a file-level copy.
- **Restore into a new volume**, verify it, then switch the service over; never extract over live data.
- **Test restores regularly.** An untested backup is a hypothesis.
- **Use storage snapshots where available.** Volume drivers backed by LVM, ZFS, or cloud disks can take block-level snapshots that are faster and crash-consistent, though not portable across backends.

## Docker Security Best Practices

Container isolation depends on a shared kernel, so no single setting makes a container safe. The goal is **defense in depth**: each layer assumes the one before it may fail.

```mermaid
flowchart TB
    subgraph Supply["Supply chain"]
        I["Image: minimal base, pinned digest,<br/>scanned, signed"]
        Bd["Build: BuildKit secrets, multi-stage,<br/>no credentials in layers"]
    end
    subgraph Run["Runtime"]
        U["Non-root user, read-only rootfs"]
        K["Capabilities dropped, no-new-privileges,<br/>seccomp, AppArmor/SELinux"]
        R["cgroup limits: memory, CPU, PIDs"]
        N["Network: user-defined networks,<br/>no unnecessary published ports"]
    end
    subgraph Host["Host"]
        H["Patched kernel and Engine, rootless or userns-remap,<br/>docker.sock protected, audit / CIS benchmark"]
    end
    Supply --> Run --> Host
```

| Layer | Threat it addresses | Key controls |
|-------|---------------------|--------------|
| Image | Known-vulnerable or malicious packages | Minimal or hardened base, pinned digests, scanning, signature verification |
| Build | Credentials leaking into layers or build logs | BuildKit secret and SSH mounts, multi-stage builds, `.dockerignore` |
| Runtime | A compromised process escalating or escaping | Non-root user, capability drop, `no-new-privileges`, seccomp, LSM profiles, read-only rootfs |
| Resources | Denial of service against the host or neighbors | `--memory`, `--cpus`, `--pids-limit`, ulimits, log rotation |
| Network | Lateral movement, accidental exposure | Segmented user-defined networks, loopback-bound ports ([Networking](docker-networking.html)) |
| Host | Daemon or kernel compromise | Updates, rootless mode, socket protection, CIS Docker Benchmark |

### Image Security

A container is only as trustworthy as the layers it is built from.

- **Start from a minimal, maintained base.** Fewer packages means fewer CVEs and fewer tools for an attacker. Options, roughly from largest to smallest: `-slim` Debian variants, Alpine, Wolfi/Chainguard images, Google distroless, and `scratch` for static binaries. [Docker Hardened Images](https://docs.docker.com/dhi/) provide non-root, distroless-style variants of common bases with signed SBOMs, VEX statements, and SLSA Build Level 3 provenance; their core catalog is free under Apache 2.0.
- **Pin versions, and pin digests for production.** `FROM python:3.13-slim@sha256:...` makes builds reproducible; let Dependabot or Renovate propose digest bumps so pins do not go stale.
- **Keep build tooling out of the runtime image.** Compile in a builder stage and copy only the artifact into the final stage; see [Dockerfiles &amp; CI/CD](dockerfiles.html).
- **Never put secrets in layers.** Anyone who can pull an image can unpack every layer, including files deleted in later layers, and `docker history` shows `ARG` and `ENV` values.
- **Scan, sign, and attest.** Vulnerability scanning, signing, SBOMs, and provenance are covered in depth on [Registries &amp; Supply Chain](registry.html#vulnerability-scanning).

### Running Containers Securely

**Run as a non-root user.** Without user namespaces, UID 0 in a container is UID 0 on the host; a kernel or runtime bug that lets a process escape then yields host root. Set the user in the image:

```dockerfile
FROM python:3.13-slim
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --chown=app:app . .
USER 10001
CMD ["python", "app.py"]
```

Use a numeric `USER` so that orchestrators (for example Kubernetes `runAsNonRoot`) can verify it without resolving `/etc/passwd`. Then tighten the runtime:

```bash
docker run -d --name api \
  --user 10001:10001 \
  --read-only --tmpfs /tmp:size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --memory 512m --cpus 1 --pids-limit 200 \
  -p 127.0.0.1:8080:8080 \
  my-api:1.4.2
```

| Control | What it does |
|---------|--------------|
| `--user` / `USER` | Runs PID 1 as an unprivileged UID. |
| `--read-only` | Makes the root filesystem immutable; add tmpfs mounts for paths that need writes. |
| `--cap-drop ALL` / `--cap-add` | Removes Linux capabilities; add back only what is needed (for example `NET_BIND_SERVICE` to bind ports below 1024). |
| `--security-opt no-new-privileges=true` | Prevents gaining privileges through setuid/setgid binaries or file capabilities. |
| seccomp | Filters syscalls. The default profile blocks dozens of dangerous calls (for example `kexec_load`, `mount`, `reboot`, and kernel-module loading); supply a tighter custom profile with `--security-opt seccomp=profile.json`. Never use `seccomp=unconfined` in production. |
| AppArmor / SELinux | Mandatory access control. Keep the `docker-default` AppArmor profile or the SELinux `container_t` type enabled. |
| `--memory`, `--cpus`, `--pids-limit` | Contain runaway or malicious resource use; `--pids-limit` stops fork bombs. |

**Avoid `--privileged`.** It disables seccomp, AppArmor/SELinux confinement, and the capability drop, and exposes all host devices; a privileged container is effectively root on the host. If a workload needs one device or one capability, grant exactly that (`--device`, `--cap-add`).

**Treat `docker.sock` as root.** Mounting `/var/run/docker.sock` into a container (common for CI agents, reverse proxies, and dashboards) lets that container start a privileged container and take over the host. If unavoidable, put a filtering socket proxy in front of it that allows only the API calls needed.

### Host Hardening

- **Rootless mode.** [Rootless Docker](https://docs.docker.com/engine/security/rootless/) runs the daemon and containers as an ordinary user, so a full daemon compromise yields only that user's privileges. Trade-offs include restricted networking performance, no binding to ports below 1024 without extra configuration, and limited cgroup support on older setups.
- **User-namespace remapping.** With `userns-remap` in `daemon.json`, container UID 0 maps to an unprivileged range on the host while the daemon still runs as root. It is incompatible with some features, and on Engine 29 it keeps the classic image store rather than the containerd store.
- **Keep the kernel, runc, and Engine patched.** Container escapes are usually kernel or runtime vulnerabilities (CVE-2019-5736 and CVE-2024-21626 in runc are well-known examples); updates are the only fix.
- **cgroup v2.** cgroup v1 is deprecated as of Engine 29; cgroup v2 provides better resource accounting and is required by newer features.
- **Audit.** The CIS Docker Benchmark, and the `docker-bench-security` script that implements it, check daemon, host, and container configuration against published recommendations.

### Secrets Management

The rule is *mount, do not bake*: deliver secrets as files at build time or run time so they never enter an image layer, the image config, or `docker inspect` output.

| Stage | Mechanism | Notes |
|-------|-----------|-------|
| Build | BuildKit secret mounts (`RUN --mount=type=secret`) | Available only to the `RUN` step that mounts it; never written to a layer or the cache key. |
| Build | BuildKit SSH mounts (`RUN --mount=type=ssh`) | Forwards the host SSH agent for cloning private repositories. |
| Run (Compose) | Compose `secrets:` | Mounted read-only at `/run/secrets/<name>` from a file or environment variable. |
| Run (Swarm) | Docker secrets | Encrypted in the Raft store, delivered to tasks as tmpfs files. |
| Run (production) | External secret manager | Vault, AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, fetched at startup or injected by the orchestrator. |
| Development | `--env-file .env` | Convenient; keep `.env` out of version control and out of the build context. |

Build-time secret, using the current Dockerfile syntax:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.13-slim
RUN --mount=type=secret,id=pip_token,env=PIP_TOKEN \
    pip install --index-url "https://__token__:${PIP_TOKEN}@pypi.example.com/simple" private-pkg
```

```bash
docker build --secret id=pip_token,src=./pip_token.txt -t my-app .
```

Runtime secrets with Compose:

```yaml
services:
  db:
    image: postgres:17
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
secrets:
  db_password:
    file: ./secrets/db_password.txt
```

Prefer file-mounted secrets over environment variables even at run time: environment variables appear in `docker inspect`, are inherited by every child process, are readable through `/proc/<pid>/environ`, and frequently end up in logs and crash reports. Many official images support a `*_FILE` variant of their password variables for this reason.

### Security Checklist

**Image and build**

- Minimal or hardened base image, pinned by digest
- Multi-stage build; no compilers or package managers in the runtime image unless required
- No secrets in `COPY`, `ARG`, `ENV`, or layers; `.dockerignore` excludes `.env` and `.git`
- Scanned in CI and on a schedule; signed, with SBOM and provenance attached

**Runtime**

- Numeric non-root `USER`
- `--read-only` with sized tmpfs mounts
- `--cap-drop ALL`, adding back only required capabilities
- `--security-opt no-new-privileges=true`; default or stricter seccomp and AppArmor/SELinux
- Memory, CPU, and PID limits; log rotation configured
- No `--privileged`, no `docker.sock` mount, no sensitive host bind mounts

**Host and network**

- Kernel, runc, containerd, and Engine patched; cgroup v2
- Rootless mode or `userns-remap` where compatible
- User-defined networks; ports published on `127.0.0.1` unless they must be public ([Networking](docker-networking.html))
- CIS Docker Benchmark reviewed

## Troubleshooting Common Docker Issues

Most storage and security problems surface as permission errors, a full disk, or a container that exits immediately.

### Debugging Containers

**Container exits immediately:**

```bash
docker ps -a --filter name=api                      # status and exit code
docker logs api                                     # the process's own error output
docker inspect api --format '{% raw %}{{.State.ExitCode}} {{.State.OOMKilled}} {{.State.Error}}{% endraw %}'
docker run --rm -it --entrypoint sh my-api:1.4.2    # explore the image by hand
```

Exit code 137 with `OOMKilled=true` means the memory limit was hit; 126 or 127 usually means the entrypoint is not executable or does not exist in the image.

**Permission and mount problems:**

```bash
docker inspect api --format '{% raw %}{{json .Mounts}}{% endraw %}'
docker exec api id                  # UID/GID the process runs as
docker exec api ls -ln /data        # numeric ownership of the mounted files
```

Most "permission denied" errors on mounts are UID mismatches between the container user and the files. On SELinux hosts, an unlabeled bind mount produces the same symptom; add `:z` or `:Z`.

**Disk usage:**

```bash
docker system df -v                 # images, containers, volumes, build cache
```

### Common Error Solutions

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `Cannot connect to the Docker daemon` | Daemon not running, or wrong context | `sudo systemctl start docker`; check `docker context ls` and `DOCKER_HOST` |
| `permission denied ... docker.sock` | User lacks socket access | Use `sudo`, rootless Docker, or add the user to the `docker` group (root-equivalent; log out and back in) |
| `no space left on device` | Images, build cache, logs, or volumes | `docker system df`, then `docker builder prune`, `docker image prune -a`; rotate logs |
| `permission denied` on a mount | UID mismatch or SELinux label | Align `--user` with file ownership; `:z`/`:Z` on SELinux |
| `Read-only file system` | Container started with `--read-only` | Add a sized `--tmpfs` for the path that needs writes |
| `exec format error` | Image built for another CPU architecture | Pull or build the right platform (`--platform linux/arm64`) |

**Reclaiming space safely:**

```bash
docker system prune                 # stopped containers, unused networks, dangling images, build cache
docker volume prune                 # unused anonymous volumes only
docker system prune -a --volumes    # everything unused, including named volumes: destructive
```

Be deliberate with `--volumes`: it removes every volume not attached to a container, including databases whose container happens to be stopped.

### Health Checks

A health check lets Docker, Compose, and Swarm distinguish "running" from "working".

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["/app/healthcheck"]
```

```bash
docker ps                                            # STATUS shows (healthy) / (unhealthy)
docker inspect --format '{% raw %}{{json .State.Health}}{% endraw %}' api
```

Slim and distroless images often lack `curl` or `wget`, so a common pattern is a small health-check binary or script shipped with the application. `--start-period` gives slow-starting services a grace period during which failures do not count. Docker itself does not restart unhealthy standalone containers; Compose `depends_on: condition: service_healthy`, Swarm, or an external supervisor acts on the status. Kubernetes ignores `HEALTHCHECK` and uses its own probes.

---

## See Also

- [Fundamentals](fundamentals.html) - Images, layers, the engine architecture, and isolation primitives
- [Docker: Networking](docker-networking.html) - Network drivers, service discovery, and network-layer security
- [Dockerfiles &amp; CI/CD](dockerfiles.html) - Multi-stage builds, BuildKit features, and pipelines
- [Registries &amp; Supply Chain](registry.html) - Scanning, signing, SBOMs, and provenance
- [Container Runtimes](../container-runtimes.html) - Sandboxed runtimes (gVisor, Kata) for stronger isolation
- [Docker Essentials](../docker-essentials.html) - Command cheat sheet
- [Cloud and Container Security](../cybersecurity/cloud-and-container-security.html) - Broader container security context
