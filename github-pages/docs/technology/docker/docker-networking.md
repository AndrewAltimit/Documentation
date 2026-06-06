---
layout: docs
title: "Docker: Networking"
permalink: /docs/technology/docker/docker-networking.html
toc: true
toc_sticky: true
hide_title: true
---

[Docker](./) &raquo; Networking

Docker's network drivers, DNS-based service discovery, port publishing, and multi-host networking — and how to lock the whole thing down.

## Why Container Networking Matters

Networking determines how containers communicate with each other, with the host, and with external services. Getting this right is essential for both functionality and security. A misconfigured network can leave a database reachable from the public internet, or it can silently prevent two services that *should* talk to each other from ever connecting.

Docker builds its networking on top of standard Linux primitives — network namespaces, virtual Ethernet (veth) pairs, Linux bridges, iptables, and an embedded DNS server. Each container normally gets its own network namespace, which gives it a private view of network interfaces, routing tables, and firewall rules. The driver you choose decides how that namespace is wired to the outside world.

Docker provides several network drivers; the default (bridge) suits most cases, but knowing the alternatives improves your architectural decisions. Use this table to choose a driver quickly:

| Driver | Isolation | Cross-host | Performance | Typical use |
|--------|-----------|-----------|-------------|-------------|
| **bridge** | Per-network namespace, NAT to host | No | Good (slight NAT overhead) | Standalone containers, Compose stacks on one host |
| **host** | None — shares host namespace | No | Best (no NAT, no veth) | Latency-sensitive workloads, network appliances |
| **overlay** | Per-network, VXLAN tunnel | Yes | Good (VXLAN encapsulation) | Swarm services spanning multiple hosts |
| **macvlan** | Per-container MAC/IP on physical LAN | Yes (same L2) | Best (bypasses bridge/NAT) | Legacy apps needing a real LAN IP |
| **ipvlan** | Per-container IP, shared MAC | Yes (same L2) | Best | Same as macvlan where the network restricts MACs |
| **none** | Total — loopback only | No | N/A | Batch jobs, maximum isolation |

## Bridge Networking Deep Dive

The bridge driver is the workhorse of single-host Docker. When the daemon starts it creates a Linux bridge named `docker0` and attaches every container on the default bridge to it through a veth pair: one end lives inside the container (as `eth0`), the other end is plugged into `docker0` on the host. Traffic leaving the container is NATed to the host's IP via iptables `MASQUERADE` rules.

<p><strong>Key concept:</strong> Always create custom bridge networks for your applications. Unlike the default bridge, custom (user-defined) networks provide automatic DNS resolution between containers, better isolation, and the ability to attach/detach containers on the fly.</p>

Create a custom network and containers find each other by name:

```bash
# Create a network and run containers on it
docker network create my-app-network
docker run -d --name web --network my-app-network nginx
docker run -d --name db --network my-app-network postgres

# Containers can find each other by name
docker exec web ping db  # Works!
```

A common isolation pattern spans two networks so the web tier cannot reach the database directly:

```bash
# Isolate frontend from database
docker network create frontend
docker network create backend
docker run -d --name webapp --network frontend nginx
docker run -d --name api --network backend my-api

# Connect API to both networks (acts as bridge)
docker network connect frontend api
```

The webapp can reach the api, but not the database directly; the api can reach both.

### Default bridge vs. user-defined bridge

The distinction trips up many newcomers, so it is worth stating explicitly:

| Behavior | Default bridge (`docker0`) | User-defined bridge |
|----------|----------------------------|---------------------|
| DNS by container name | No (legacy `--link` only) | Yes — embedded DNS resolves names automatically |
| Containers isolated by default | All default-bridge containers can reach each other | Only containers on the same network can communicate |
| Attach/detach at runtime | No | Yes (`docker network connect`/`disconnect`) |
| Configurable subnet/gateway | Daemon-wide only | Per network |

The practical rule: never rely on the default bridge for anything beyond a throwaway `docker run`. Create a named network.

### Inspecting and configuring a bridge

You can pin the subnet, gateway, and IP range, which matters when you need to avoid clashing with an existing corporate network:

```bash
# Create a bridge with an explicit subnet and gateway
docker network create \
  --driver bridge \
  --subnet 172.28.0.0/16 \
  --gateway 172.28.0.1 \
  --ip-range 172.28.5.0/24 \
  app-net

# Assign a container a static address from that range
docker run -d --name api --network app-net --ip 172.28.5.10 my-api

# Inspect the network: subnet, gateway, and attached containers
docker network inspect app-net
```

`docker network inspect` returns JSON describing the `IPAM` (IP Address Management) config and a `Containers` map showing each attached container's IPv4/IPv6 address and MAC. It is the first tool to reach for when "container A cannot reach container B."

## Host Networking

With `--network host`, the container skips its own network namespace entirely and shares the host's. There is no veth pair, no NAT, and no port-publishing step — a process binding `0.0.0.0:8080` inside the container is binding the host's port 8080 directly.

```bash
# The container's listening sockets are the host's sockets
docker run -d --network host nginx
# nginx now answers on the host's port 80 with no -p flag
```

Host networking shines when you need the lowest possible latency or the highest packet throughput (NAT and the bridge add per-packet cost), or when an application enumerates host interfaces (monitoring agents, some VPNs). The trade-offs are real, though:

- **No port remapping.** You cannot run two containers that both want port 80; they collide on the host.
- **Weaker isolation.** The container sees every host interface and can bind any host port, including privileged ones if it has the capability.
- **Linux-centric semantics.** On Docker Desktop for macOS/Windows the container actually runs inside a Linux VM, so `host` means "the VM's network," not your laptop's — behavior differs from native Linux.

## None (No Networking)

The `none` driver gives the container a network namespace with only a loopback interface — no external connectivity at all.

```bash
docker run --rm --network none alpine ip addr
# Only "lo" appears; no eth0
```

Use it for batch/compute jobs that read from a mounted volume and write back to one, where network access would only be an attack surface. You can also start a container with `none` and attach it to a network later with `docker network connect` once you are ready for it to talk.

## DNS and Service Discovery

On any user-defined network, Docker runs an **embedded DNS server** reachable from every attached container at `127.0.0.11`. The container's `/etc/resolv.conf` is rewritten to point at it. This server resolves:

- **Container names** — `db` resolves to the database container's IP on that network.
- **Network aliases** — additional names you assign so several containers, or a renamed container, answer to a stable hostname.
- **Service names** (in Swarm) — a service name resolves to a virtual IP (VIP) that load-balances across the service's tasks.

Anything the embedded server cannot answer is forwarded to the host's upstream resolvers, so public DNS still works.

```bash
# Give a container extra DNS names on the network
docker run -d --name db \
  --network app-net \
  --network-alias database \
  --network-alias primary-db \
  postgres:16

# Any of these now resolve from another container on app-net:
#   db, database, primary-db
docker run --rm --network app-net alpine nslookup database
```

### Round-robin DNS for simple load balancing

When several containers share the **same network alias**, the embedded DNS returns all of their addresses, and the client's resolver rotates through them. This gives you crude client-side load balancing without a proxy:

```bash
docker network create web-net
docker run -d --network web-net --network-alias web nginx
docker run -d --network web-net --network-alias web nginx
docker run -d --network web-net --network-alias web nginx

# Repeated lookups cycle through the three container IPs
docker run --rm --network web-net alpine nslookup web
```

### Overriding DNS

For containers that need a specific resolver (split-horizon DNS, an internal corporate server) you can override the defaults at run time:

```bash
docker run -d \
  --dns 10.0.0.53 \
  --dns-search corp.example.com \
  --add-host legacy-host:192.168.1.50 \
  my-app
```

`--add-host` writes a static `/etc/hosts` entry, which is handy for pinning a name the embedded DNS does not know about.

## Port Publishing

A container on a bridge network is, by default, unreachable from outside the host. **Publishing** a port wires a host port to a container port by inserting iptables DNAT rules and running a small userland `docker-proxy` helper.

```bash
# -p HOST:CONTAINER  — host 8080 forwards to container 80
docker run -d -p 8080:80 nginx

# Bind only to localhost so the port is not exposed on the LAN
docker run -d -p 127.0.0.1:8080:80 nginx

# Let Docker pick an ephemeral host port (see it with docker port)
docker run -d -p 80 nginx
docker port <container>   # e.g. 80/tcp -> 0.0.0.0:49153

# Publish a UDP port
docker run -d -p 53:53/udp my-dns

# Publish a range
docker run -d -p 7000-7010:7000-7010 my-app

# -P publishes every EXPOSEd port to random host ports
docker run -d -P my-app
```

A few things that bite people:

- **`EXPOSE` documents, it does not publish.** An `EXPOSE 80` line in a Dockerfile is metadata for humans and for `-P`; it does not open a host port by itself. Use `-p`/`-P` to actually publish.
- **Bind address matters.** `-p 8080:80` binds `0.0.0.0` — every host interface, including the public one. Use `-p 127.0.0.1:8080:80` to keep a service host-local.
- **Containers on the same user-defined network never need published ports to talk to each other.** Publishing is only for traffic entering from outside Docker. Inter-container traffic flows over the bridge directly.

### Port publishing vs. internal connectivity

```
                External client
                      │  curl host:8080
                      ▼
   ┌──────────────────────────────────────────┐
   │  Host  (iptables DNAT 8080 -> 172.x:80)   │
   │   ┌────────────┐        ┌────────────┐    │
   │   │  web :80   │◀──────▶│  db :5432  │    │  internal traffic
   │   │ (published)│  app-net (no publish)│    │  needs NO -p
   │   └────────────┘        └────────────┘    │
   └──────────────────────────────────────────┘
```

Only `web` needs `-p`; `db` stays unpublished and is reachable from `web` by name over `app-net`.

## Multi-Host Networking with Overlay

A single bridge cannot span machines. To let containers on **different Docker hosts** communicate as if on one LAN, Docker uses the **overlay** driver, which builds a VXLAN tunnel between hosts and encapsulates container traffic inside it. Overlay networks are the native multi-host mechanism in **Docker Swarm**.

<p><strong>When to use:</strong> Docker Swarm deployments where services need to communicate across multiple hosts.</p>

```bash
# Create encrypted overlay network
docker network create --driver overlay --opt encrypted my-overlay

# Services on this network can find each other across hosts
docker service create --name api --network my-overlay my-api
```

### How overlay works

Swarm maintains a distributed key-value store (the Raft log on the managers) that every node consults to learn which container lives on which host and behind which VXLAN tunnel endpoint. When a container sends a packet to a peer on another host:

1. The local overlay bridge captures the frame.
2. It is encapsulated in a VXLAN header (UDP, port 4789) tagged with the network's VXLAN Network Identifier (VNI).
3. The packet travels host-to-host over the underlay network.
4. The receiving host strips the VXLAN header and delivers the frame to the destination container.

```bash
# Bootstrap a swarm (run on the first manager)
docker swarm init --advertise-addr <MANAGER-IP>
# Join workers with the token printed above
docker swarm join --token <TOKEN> <MANAGER-IP>:2377

# Create an attachable, encrypted overlay
docker network create \
  --driver overlay \
  --attachable \
  --opt encrypted \
  --subnet 10.10.0.0/24 \
  prod-net

# Deploy a replicated service onto it
docker service create \
  --name api \
  --network prod-net \
  --replicas 3 \
  my-api:1.0
```

Two flags worth knowing:

- `--attachable` lets *standalone* `docker run` containers join the overlay, not just Swarm services — useful for debugging.
- `--opt encrypted` turns on IPsec (ESP) encryption of the VXLAN data plane between hosts. The Swarm control plane is always encrypted; the data plane is not unless you ask.

### Required firewall ports for Swarm overlay

For overlay networking to work across hosts, the underlay firewall must allow:

| Port | Protocol | Purpose |
|------|----------|---------|
| 2377 | TCP | Swarm cluster management (managers) |
| 7946 | TCP + UDP | Node-to-node control plane / gossip |
| 4789 | UDP | VXLAN data plane (overlay traffic) |

If services can be created but cross-host traffic silently fails, 4789/udp or 7946 being blocked is the usual cause.

## Macvlan and ipvlan Networking

Sometimes a container must look like a real machine on the physical LAN — a legacy monitoring tool that expects a routable IP, or an app whose licensing is pinned to a MAC address. The **macvlan** driver gives each container its own MAC and IP directly on the parent interface's network, bypassing the Docker bridge and NAT.

<p><strong>When to use:</strong> When containers need to appear as physical devices on your network (legacy system integration, specific IP requirements).</p>

```bash
# Container gets a real IP on your network
docker network create -d macvlan \
  --subnet=192.168.1.0/24 --gateway=192.168.1.1 \
  -o parent=eth0 my-macvlan

docker run -d --network my-macvlan --ip 192.168.1.100 nginx
```

### Caveats and the ipvlan alternative

Macvlan is fast but has sharp edges:

- **Host-to-container traffic is blocked by default.** Because the container's MAC is distinct and rides the same physical NIC, the host kernel will not route packets to it without a dedicated macvlan sub-interface on the host. Container-to-container and container-to-LAN work fine; the *host itself* talking to its own macvlan containers does not, out of the box.
- **Promiscuous mode required.** Many switches and most cloud/virtual NICs reject the multiple-MAC-per-port behavior macvlan needs, so it often only works on bare metal with a NIC you control.
- **IP management is yours.** Docker does not coordinate with your LAN's DHCP server; reserve an `--ip-range` to avoid collisions.

When the network forbids multiple MACs per port (common in clouds and on Wi-Fi), use **ipvlan** instead. It gives each container a unique IP but shares the parent interface's single MAC:

```bash
docker network create -d ipvlan \
  --subnet 192.168.1.0/24 \
  --gateway 192.168.1.1 \
  -o parent=eth0 \
  -o ipvlan_mode=l2 \
  ipvlan-net
```

## Network Security

Containers are only as safe as the networks you put them on. Apply the same defense-in-depth thinking here as anywhere else.

- Use custom bridge networks, not the default — containers on the default bridge can all reach each other
- Encrypt overlay network traffic with `--opt encrypted`
- Implement network segmentation: separate frontend, backend, and database tiers
- Use TLS for container-to-container communication carrying sensitive data
- Restrict which containers can talk to each other; never publish a port wider than needed
- Bind published ports to `127.0.0.1` unless the service truly must be reachable on the LAN

### Internal networks

Mark a network `--internal` to forbid any traffic in or out of the Docker host — containers on it can talk to each other but have no route to the outside world and cannot be published. This is the cleanest way to wall off a database tier:

```bash
# Backend network with no external connectivity
docker network create --internal backend
docker run -d --name db --network backend postgres:16

# The API spans both networks; it is the only path to the DB
docker network create frontend
docker run -d --name api --network frontend my-api
docker network connect backend api
```

The database has no route to the internet and no published port; the only way to reach it is through the API, which sits on both networks.

### The DOCKER-USER iptables chain

Docker writes its own iptables rules and will overwrite changes you make to the `DOCKER` chain. The supported escape hatch is the **`DOCKER-USER`** chain, which Docker evaluates *before* its own rules and never clobbers. Put your custom filtering there:

```bash
# Allow only one trusted subnet to reach published container ports;
# drop everyone else on the external interface
iptables -I DOCKER-USER -i eth0 ! -s 10.0.0.0/24 -j DROP
```

Remember a subtle gotcha: by publishing a port, Docker inserts DNAT rules that **bypass the host's INPUT chain**, so a `ufw`/firewalld rule on INPUT will *not* protect a published container port. Filtering must happen in `DOCKER-USER`.

### Encryption and orchestration patterns

For complex deployments, three patterns recur:

- **Service mesh** — use a proxy (Envoy, Traefik) to handle routing, load balancing, mutual TLS, and observability.
- **Network segmentation** — separate networks for frontend, backend, and database tiers.
- **Firewall rules** — use the iptables `DOCKER-USER` chain to restrict container traffic.

These are typically managed through orchestration tools like Kubernetes or Docker Swarm rather than by hand.

## Compose Networking

Docker Compose makes single-host networking nearly invisible: every Compose project gets its own user-defined bridge network automatically, and each service is reachable from the others by its **service name**. You rarely need to declare networks unless you want segmentation.

```yaml
services:
  web:
    image: nginx
    ports:
      - "8080:80"        # published to the host
    networks: [frontend]
  api:
    image: my-api
    networks: [frontend, backend]   # bridges the two tiers
  db:
    image: postgres:16
    networks: [backend]             # not reachable from web

networks:
  frontend:
  backend:
    internal: true       # db tier has no external route
```

Here `api` resolves `db` by name over the `backend` network, `web` resolves `api` over `frontend`, and `web` has no path to `db` at all. Only `web` publishes a host port.

## Troubleshooting Container Networking

When connectivity breaks, work from the inside out: confirm the container has an interface and an address, then DNS, then reachability, then the published-port path.

**Connectivity issues** — `netshoot` is a debug container with every network tool preinstalled:

```bash
# Share the target container's network namespace
docker run --rm --network container:my-app nicolaka/netshoot

# Inside: test DNS and connectivity
nslookup service-name
curl -v http://service-name:port
```

**What network is the container on?**

```bash
# See every network and which containers are attached
docker network ls
docker network inspect app-net

# See the networks a single container joined
docker inspect --format '{% raw %}{{json .NetworkSettings.Networks}}{% endraw %}' my-app
```

**Is the port actually published?**

```bash
# Show host:container port mappings for a container
docker port my-app

# Confirm the host is listening
ss -tlnp | grep 8080
```

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ping db` fails by name, works by IP | Containers on the *default* bridge | Put both on a user-defined network |
| Service unreachable from LAN | Port bound to `127.0.0.1` | Publish on `0.0.0.0` or the right interface |
| Cross-host Swarm traffic dies | 4789/udp or 7946 firewalled | Open the overlay ports on the underlay |
| Host cannot reach its macvlan container | macvlan host isolation | Add a macvlan sub-interface on the host |
| Published port not firewalled by ufw | DNAT bypasses INPUT chain | Filter in `DOCKER-USER` instead |

## Key Takeaways

- **Custom networks by default.** User-defined bridge networks provide automatic DNS resolution between containers (the default bridge does not) plus stronger isolation.
- **DNS is built in.** Containers on the same user-defined network resolve each other by name via the embedded server at 127.0.0.11. Use network aliases for stable, load-balanced names.
- **Publish narrowly.** Only publish ports that must be reachable from outside Docker, and bind them to 127.0.0.1 unless the LAN truly needs them. Inter-container traffic needs no `-p`.
- **Overlay for multiple hosts.** Use encrypted overlay networks in Swarm to span hosts over VXLAN, and open ports 2377, 7946, and 4789 on the underlay firewall.

---

## See Also

- [Fundamentals](fundamentals.html) - Images, containers, layers, and bridge networking basics
- [Storage &amp; Security](storage-security.html) - Persist data with volumes and harden containers
- [Dockerfiles &amp; CI/CD](dockerfiles.html) - Build optimized images and automate pipelines
- [Advanced Patterns](advanced.html) - Production architectures and runtime alternatives
- [Docker Essentials](../docker-essentials.html) - Quick command reference
- [Kubernetes](../kubernetes/) - Container orchestration and its own networking model
- [Networking](../networking/) - TCP/IP, DNS, and firewall fundamentals
- [Cybersecurity](../cybersecurity/) - Broader security fundamentals
