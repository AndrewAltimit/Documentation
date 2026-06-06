---
layout: docs
title: "Networking: Cloud Networking"
permalink: /docs/technology/networking/cloud-networking.html
toc: true
toc_sticky: true
hide_title: true
---

[Networking](./) &raquo; Cloud Networking

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Cloud Networking</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">VPCs, subnets, load balancers, CDNs, and the shared-responsibility model that make networks rentable</p>
</div>

The cloud took the software-defined ideas behind SDN, NFV, and P4 and turned them into rentable primitives. Instead of racking switches, firewalls, and load balancers, you describe an entire network in an API call: address ranges, routing, internet access, and traffic distribution all become declarative resources. This page works through those primitives bottom-up — the **VPC** that isolates your network, the **subnets and route tables** that segment and steer it, the **load balancers** that spread traffic across servers, the **CDNs and anycast** that bring content close to users, the **NAT and egress** paths that control outbound traffic, the **service networking** layer that connects everything privately, and finally the **shared-responsibility model** that defines who secures what.

## Virtual Private Cloud (VPC)

A **Virtual Private Cloud (VPC)** is an isolated, software-defined network inside a cloud provider's infrastructure. It looks and behaves like the private network you would build in your own data center — your own address space, routing, and gateways — except the "wires" are virtual overlays implemented by the provider's SDN fabric, and you provision them through an API rather than a loading dock.

**Core components**:
- **CIDR block** — the private IPv4 (and optionally IPv6) address range the VPC owns, e.g. `10.0.0.0/16`.
- **Subnets** — slices of the VPC CIDR pinned to an availability zone.
- **Route tables** — per-subnet rules that decide where each destination prefix goes.
- **Internet gateway (IGW)** — the door to the public internet for inbound and outbound traffic.
- **NAT gateway** — outbound-only internet access for private subnets.
- **Security groups** — stateful, instance-level virtual firewalls.
- **Network ACLs** — stateless, subnet-level packet filters.

### Choosing an Address Range

A VPC is defined first by its CIDR block, which fixes how many addresses you have and — crucially — whether it can peer with other networks. Two networks with overlapping ranges cannot route to each other, so address planning is the single most consequential cloud-networking decision. RFC 1918 reserves three private ranges to draw from:

| Range | CIDR | Addresses | Typical use |
|-------|------|-----------|-------------|
| `10.0.0.0` – `10.255.255.255` | `10.0.0.0/8` | ~16.7M | Large enterprises, room to subdivide per region/account |
| `172.16.0.0` – `172.31.255.255` | `172.16.0.0/12` | ~1M | Mid-size deployments |
| `192.168.0.0` – `192.168.255.255` | `192.168.0.0/16` | ~65K | Small networks, labs, home |

A `/16` VPC gives 65,536 addresses; a `/24` subnet carved from it gives 256 (minus a handful the provider reserves for the gateway, DNS, and broadcast). For the mechanics of splitting a block into subnets, see [Layers & Addressing](fundamentals.html#subnetting-organizing-ip-addresses); for how on-premises and cloud routes interact, see [Routing & Switching](routing.html).

```python
import ipaddress

def plan_vpc(vpc_cidr, subnet_prefix):
    """Carve a VPC CIDR into evenly sized subnets.

    Cloud providers reserve a few addresses per subnet (network,
    gateway, DNS, broadcast), so usable hosts are smaller than 2^h.
    """
    vpc = ipaddress.ip_network(vpc_cidr)
    reserved_per_subnet = 5  # typical cloud reservation

    subnets = []
    for subnet in vpc.subnets(new_prefix=subnet_prefix):
        total = subnet.num_addresses
        subnets.append({
            'cidr': str(subnet),
            'total_addresses': total,
            'usable_hosts': max(total - reserved_per_subnet, 0),
            'gateway': str(subnet.network_address + 1),
        })
    return subnets

# A /16 VPC split into /24 subnets => 256 subnets of 251 usable hosts each
for s in plan_vpc('10.0.0.0/16', 24)[:3]:
    print(s)
```

### Isolation and Multi-VPC Topologies

A VPC is closed by default: nothing routes in or out until you add a gateway and a route. Connecting VPCs together — across teams, accounts, or regions — uses one of three patterns:

- **VPC peering** — a direct, non-transitive link between two VPCs. Simple, but the number of connections grows as `n(n-1)/2`, so a full mesh of many VPCs becomes unmanageable.
- **Transit gateway / hub-and-spoke** — a central router that every VPC attaches to, giving transitive any-to-any routing with `O(n)` attachments instead of `O(n^2)` peerings.
- **Private connectivity to on-prem** — a VPN over the public internet (encrypted, cheap, variable latency) or a dedicated circuit such as AWS Direct Connect / Azure ExpressRoute (private, consistent latency, higher cost).

> For provider-specific detail — VPC design, Direct Connect, and managed gateways — see [AWS](../aws/).

## Subnets and Route Tables

Subnets divide a VPC into segments, and route tables decide where traffic from each segment goes. Together they implement the public/private split that nearly every cloud architecture relies on.

### Public vs. Private Subnets

The only thing that makes a subnet "public" is its route table: a subnet is public if its route table sends `0.0.0.0/0` (the default route) to an **internet gateway**. A "private" subnet has no such route — its default route points at a **NAT gateway** (outbound only) or nowhere at all.

```
                         ┌─────────────── VPC 10.0.0.0/16 ───────────────┐
                         │                                               │
  Internet  ◄───────►  [IGW]                                             │
                         │                                               │
                         │   Public subnet 10.0.1.0/24                    │
                         │   route: 0.0.0.0/0 → IGW                       │
                         │   ┌──────────┐    ┌───────────┐                │
                         │   │   ALB    │    │ NAT GW     │               │
                         │   └────┬─────┘    └─────┬──────┘               │
                         │        │                │                      │
                         │   Private subnet 10.0.2.0/24                   │
                         │   route: 0.0.0.0/0 → NAT GW                    │
                         │   ┌──────────┐    ┌───────────┐                │
                         │   │ app srv  │    │ database  │                │
                         │   └──────────┘    └───────────┘                │
                         └───────────────────────────────────────────────┘
```

The pattern above — internet-facing load balancers and NAT gateways in public subnets, application servers and databases in private subnets — is the canonical "three-tier" cloud layout. Inbound traffic terminates at the load balancer in the public tier; the private tier reaches out for updates through NAT but is unreachable from the internet directly.

### How a Route Table Decides

Route tables follow the same **longest-prefix-match** rule as any IP router: among all routes whose destination prefix contains the packet's destination address, the most specific (longest mask) wins. A local route for the VPC's own CIDR is always present and most specific, so intra-VPC traffic never leaves; everything else falls to progressively broader routes down to the default `0.0.0.0/0`.

```python
import ipaddress

def route_lookup(dest_ip, route_table):
    """Longest-prefix-match route selection, as a cloud route table does it.

    route_table: list of (cidr, target) tuples.
    """
    dest = ipaddress.ip_address(dest_ip)
    best = None
    best_len = -1
    for cidr, target in route_table:
        net = ipaddress.ip_network(cidr)
        if dest in net and net.prefixlen > best_len:
            best, best_len = target, net.prefixlen
    return best

routes = [
    ('10.0.0.0/16', 'local'),       # the VPC itself
    ('10.50.0.0/16', 'pcx-peer'),   # peered VPC
    ('0.0.0.0/0',   'nat-gateway'), # default: everything else via NAT
]

print(route_lookup('10.0.2.5',   routes))  # local   (longest match)
print(route_lookup('10.50.1.1',  routes))  # pcx-peer
print(route_lookup('93.184.216.34', routes))  # nat-gateway (default)
```

### Stateful vs. Stateless Filtering

Two filtering layers guard subnet and instance traffic, and the distinction matters:

- **Security groups** are *stateful* and attach to instances/interfaces. If you allow an inbound request, the corresponding reply is automatically allowed — you only write rules for the direction you initiate. They are allow-only (no explicit deny).
- **Network ACLs (NACLs)** are *stateless* and attach to subnets. Every packet is evaluated independently, so you must allow both the request and the reply direction explicitly (including ephemeral high ports for return traffic). They support both allow and deny, evaluated by rule number.

In practice security groups carry the bulk of the policy because their statefulness matches how applications actually communicate; NACLs are a coarse subnet-wide backstop. The deeper treatment of stateful inspection lives in [Performance, QoS & Security](performance-and-security.html).

## Load Balancing

A **load balancer** distributes incoming traffic across a pool of backend servers so that no single server is overwhelmed, failed servers are taken out of rotation, and the pool can scale horizontally behind one stable address. In the cloud it is itself a managed, horizontally scaled service — you never see the boxes doing the balancing.

### Layer 4 vs. Layer 7

Where a load balancer makes its decision determines what it can do:

| | **Layer 4 (transport)** | **Layer 7 (application)** |
|---|---|---|
| Operates on | TCP/UDP connections | HTTP(S), gRPC requests |
| Sees | IP addresses, ports | URLs, headers, cookies, methods |
| Routing by | 5-tuple hashing, connection count | host/path rules, header values |
| Features | very low latency, protocol-agnostic | TLS termination, content routing, redirects, WAF |
| Examples | AWS NLB, Azure LB | AWS ALB, NGINX, Envoy, Azure App Gateway |

A Layer 4 balancer forwards connections without understanding their contents — it is fast and works for any TCP/UDP protocol but cannot route on a URL. A Layer 7 balancer terminates the application protocol, so it can send `/api/*` to one fleet and `/static/*` to another, terminate TLS, inject headers, and apply a web-application firewall — at the cost of doing more work per request.

### Balancing Algorithms

The balancer must choose *which* backend gets each new request. Common algorithms:

- **Round robin** — hand requests to backends in rotation. Simple and fair when requests are uniform.
- **Weighted round robin** — bias the rotation toward more capable backends.
- **Least connections** — send each request to the backend with the fewest active connections; adapts when request durations vary widely.
- **IP hash / consistent hashing** — hash the client (or 5-tuple) to a backend so a given client sticks to the same server (session affinity), while minimizing reshuffling when the pool changes.

```python
import hashlib
from collections import defaultdict

class LoadBalancer:
    """Illustrative L4/L7 backend selection.

    Models the four classic algorithms over a pool of backends,
    each carrying a weight and a live connection count.
    """

    def __init__(self, backends):
        # backends: list of {'id', 'weight'}
        self.backends = backends
        self.rr_index = 0
        self.connections = defaultdict(int)

    def round_robin(self):
        b = self.backends[self.rr_index % len(self.backends)]
        self.rr_index += 1
        return b['id']

    def weighted_round_robin(self):
        # Expand the pool by weight, then rotate through it.
        expanded = [b['id'] for b in self.backends for _ in range(b['weight'])]
        choice = expanded[self.rr_index % len(expanded)]
        self.rr_index += 1
        return choice

    def least_connections(self):
        return min(self.backends, key=lambda b: self.connections[b['id']])['id']

    def ip_hash(self, client_ip):
        h = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return self.backends[h % len(self.backends)]['id']

lb = LoadBalancer([
    {'id': 'srv-a', 'weight': 3},
    {'id': 'srv-b', 'weight': 1},
])
print([lb.weighted_round_robin() for _ in range(4)])  # a,a,a,b
print(lb.ip_hash('203.0.113.7'))                       # sticky per client
```

### Health Checks and Global vs. Regional

A load balancer is only as good as its knowledge of which backends are alive. It continuously probes each backend (a TCP connect, an HTTP `GET /healthz`, or a gRPC health RPC) and removes any target that fails a threshold of consecutive checks, restoring it once it passes again. This is what turns a server crash into a few dropped requests rather than an outage.

Load balancers also operate at two scopes:

- **Regional** balancers spread traffic across servers within one region's availability zones — the standard front door for an application.
- **Global** balancers spread traffic across *regions*, usually via DNS or anycast, steering each user to the nearest healthy region and failing whole regions over. This naturally leads into anycast and CDNs below.

## Content Delivery Networks and Anycast

### Content Delivery Networks (CDN)

A **Content Delivery Network (CDN)** caches content at **edge locations** geographically close to users, so a request for an image, script, or video segment is served from a nearby point of presence (PoP) instead of crossing the planet to the origin server.

**Benefits**:
- **Reduced latency** — content is served from the nearest PoP, cutting round-trip time dramatically.
- **Decreased bandwidth costs** — cached responses never touch the origin, so origin egress drops.
- **Improved availability** — the origin can be offline and cached content still serves.
- **DDoS protection** — the distributed edge absorbs and disperses attack traffic before it reaches the origin.

A CDN is fundamentally a cache hierarchy. On a **cache miss** the edge fetches from the origin (or a regional mid-tier cache) and stores the result according to its TTL; subsequent requests are **cache hits** served locally. Tuning a CDN is largely about maximizing the **hit ratio** through sensible `Cache-Control` headers, cache-key design, and edge TTLs.

```python
class EdgeCache:
    """Illustrative CDN edge node with TTL-based caching.

    Hit ratio is the metric that matters: every hit is latency and
    origin bandwidth saved.
    """

    def __init__(self, origin_latency_ms=120, edge_latency_ms=10):
        self.store = {}              # key -> (content, expires_at)
        self.origin_latency = origin_latency_ms
        self.edge_latency = edge_latency_ms
        self.hits = 0
        self.misses = 0

    def get(self, key, now, fetch_origin, ttl=60):
        entry = self.store.get(key)
        if entry and entry[1] > now:
            self.hits += 1
            return entry[0], self.edge_latency          # cache hit
        # miss: fetch from origin and cache
        self.misses += 1
        content = fetch_origin(key)
        self.store[key] = (content, now + ttl)
        return content, self.origin_latency

    @property
    def hit_ratio(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
```

The latency win is direct. If a fraction $h$ of requests hit the edge, the expected latency is a weighted average of the edge and origin paths:

$$L_{avg} = h \cdot L_{edge} + (1 - h) \cdot L_{origin}$$

With an edge RTT of 10 ms, an origin RTT of 120 ms, and a 90% hit ratio, the average request sees $0.9 \times 10 + 0.1 \times 120 = 21$ ms — roughly a 6x improvement over always reaching the origin.

### Anycast: One Address, Many Locations

CDNs and global load balancers steer users to the nearest edge using **anycast**. With anycast, the *same* IP address is announced from many physical locations simultaneously via BGP. Each user's packets follow normal internet routing toward whichever announcement is closest in BGP terms — so the same destination address resolves to a different nearby PoP depending on where you are.

Contrast the three IP addressing modes:

- **Unicast** — one address, one destination. The default for ordinary traffic.
- **Anycast** — one address announced from many places; the network delivers to the *nearest* one. Used by CDNs, public DNS resolvers (`1.1.1.1`, `8.8.8.8`), and root DNS servers.
- **Multicast** — one address delivered to *all* subscribed members of a group; used for one-to-many streaming within controlled networks.

Anycast gives three things at once: low latency (nearest PoP wins), implicit load distribution (traffic spreads across announcements), and resilience (withdraw a PoP's BGP announcement and traffic instantly reroutes to the next-nearest). The routing mechanics behind these announcements are covered in [Routing & Switching](routing.html).

## NAT and Egress

Private subnets, by design, have no inbound route from the internet — but their workloads still need to reach out for OS updates, package registries, and third-party APIs. **NAT (Network Address Translation)** provides exactly that: outbound connectivity without inbound exposure.

### How Outbound NAT Works

A **NAT gateway** sits in a public subnet with its own public IP. When a private instance opens a connection to the internet, the NAT gateway rewrites the packet's source from the private address to its own public address, records the mapping in a translation table, and forwards it. Return traffic matches the recorded mapping and is rewritten back to the private address. Because the gateway only forwards replies to connections it initiated, nothing on the internet can open a new connection inward.

```python
class NATGateway:
    """Illustrative source NAT (PAT) translation table.

    Many private hosts share one public IP, disambiguated by the
    source port the gateway assigns — 'port address translation'.
    """

    def __init__(self, public_ip):
        self.public_ip = public_ip
        self.table = {}            # (pub_ip, pub_port) -> (priv_ip, priv_port)
        self.next_port = 1024

    def outbound(self, priv_ip, priv_port, dst_ip, dst_port):
        pub_port = self.next_port
        self.next_port += 1
        self.table[(self.public_ip, pub_port)] = (priv_ip, priv_port)
        # packet leaves with rewritten source
        return (self.public_ip, pub_port, dst_ip, dst_port)

    def inbound(self, pub_port):
        # reply is delivered only if it matches an existing mapping
        return self.table.get((self.public_ip, pub_port))
```

The general NAT and PAT mechanism — including its use in home routers and the broader IPv4-exhaustion story — is covered in [Routing & Switching](routing.html#nat-network-address-translation); here it is the controlled egress door for private cloud subnets.

### Egress Control and Cost

Egress is not just a connectivity concern but a security and cost concern:

- **Egress filtering** — many architectures restrict *where* private workloads may connect (allowlisting specific domains or prefixes) to limit data exfiltration and blast radius after a compromise. This is a core idea in [zero-trust](../cybersecurity/) designs.
- **Data transfer pricing** — cloud providers typically charge for traffic leaving the cloud (and across regions/AZs) but not for inbound traffic. Because NAT gateways meter every byte that passes through them, routing high-volume egress (backups, large pulls) through NAT can be a meaningful cost line. Routing such traffic over private service endpoints (below) often avoids both the NAT and the internet entirely.
- **IPv6 egress** — with IPv6 there is no address scarcity, so the NAT-for-sharing rationale disappears. Providers offer an **egress-only internet gateway** that preserves the "outbound allowed, inbound blocked" property without any address translation.

## Service Networking

The final layer connects the *services* running inside (and outside) your VPC to each other — ideally without that traffic ever traversing the public internet.

### Private Service Endpoints

Reaching a managed cloud service (object storage, a database, a message queue) over its public DNS name would send traffic out the internet gateway, costing egress and widening exposure. **Private endpoints** (AWS PrivateLink / VPC endpoints, Azure Private Link, GCP Private Service Connect) instead project the service into your VPC as a private IP. The service's DNS name resolves to an address inside your subnet, so traffic stays on the provider's backbone, never touches a NAT gateway, and can be locked down with security groups.

- **Gateway endpoints** add a route-table entry for a whole service's prefix (used for object storage), keeping that traffic on the internal fabric for free.
- **Interface endpoints** place an actual elastic network interface (with a private IP) in your subnet that fronts the service.

### Service Discovery and Service Mesh

Inside a dynamic environment — containers and instances coming and going — clients need to find healthy service instances by *name*, not by hard-coded IP:

- **DNS-based discovery** — the platform registers each healthy instance under a service name (e.g. `payments.internal`) and clients resolve it. Kubernetes does this with cluster DNS; cloud providers offer managed equivalents (Cloud Map, internal DNS zones).
- **Service mesh** — a sidecar proxy (Envoy, deployed by Istio, Linkerd, or AWS App Mesh) is injected alongside each service. The mesh handles discovery, load balancing, mutual-TLS encryption, retries, and traffic-shifting *transparently*, lifting that logic out of application code. This is the natural extension of the Layer-7 load-balancing ideas above into east-west (service-to-service) traffic.

For how this plays out inside an orchestrator, see [Kubernetes](../kubernetes/) cluster networking and CNI.

## The Shared-Responsibility Model

Moving to the cloud does not eliminate operational responsibility — it *splits* it. The **shared-responsibility model** is the contract that defines where the provider's duties end and yours begin, and misunderstanding it is the root cause of a large share of cloud breaches (typically a customer misconfiguration, not a provider failure).

The dividing line is usually summarized as **"of the cloud" vs. "in the cloud"**:

| Layer | Responsibility |
|-------|----------------|
| Physical facilities, hardware, power, cooling | **Provider** ("security *of* the cloud") |
| Hypervisor, host OS, the SDN fabric itself | **Provider** |
| Network controls: security groups, NACLs, route tables, public/private subnet design | **Customer** |
| Guest OS patching, application code, IAM policies | **Customer** |
| Data: classification, encryption choices, access policy | **Customer** ("security *in* the cloud") |

The exact line shifts with the service model:

- **IaaS** (raw VMs, VPCs) — you own the most: OS, network configuration, firewalls, and data. The provider secures only the underlying infrastructure.
- **PaaS** (managed databases, app platforms) — the provider also manages the OS and runtime; you still own network access policy, IAM, and your data.
- **SaaS** — the provider runs almost everything; your responsibility narrows to access management and your data.

```
        IaaS                PaaS                SaaS
   ┌────────────┐      ┌────────────┐      ┌────────────┐
   │   Data     │ C    │   Data     │ C    │   Data     │ C
   │   IAM      │ C    │   IAM      │ C    │   IAM      │ C
   │  App code  │ C    │  App code  │ C    │  App code  │ P
   │  Runtime   │ C    │  Runtime   │ P    │  Runtime   │ P
   │  Guest OS  │ C    │  Guest OS  │ P    │  Guest OS  │ P
   │ Networking │ C/P  │ Networking │ C/P  │ Networking │ P
   │ Hypervisor │ P    │ Hypervisor │ P    │ Hypervisor │ P
   │  Hardware  │ P    │  Hardware  │ P    │  Hardware  │ P
   └────────────┘      └────────────┘      └────────────┘
     C = Customer   P = Provider   C/P = shared
```

For networking specifically, almost everything on this page — VPC design, subnet segmentation, route tables, security groups, NACLs, and egress filtering — falls on the **customer** side of the line. The provider guarantees the fabric works and is isolated; *you* decide whether a database sits in a public subnet, whether a security group allows `0.0.0.0/0` on port 22, and whether egress is filtered. Treating those as your responsibility, and validating them with tooling, is the heart of cloud network security. The broader principles connect directly to [Cybersecurity](../cybersecurity/) and zero-trust architecture.

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>A VPC is your network in software</h4>
    <p>An isolated CIDR with subnets, route tables, and gateways — provisioned by API. Address planning is the decision you cannot easily undo.</p>
  </div>
  <div class="takeaway-card">
    <h4>The route table defines public vs. private</h4>
    <p>A subnet is public only because its default route points at an internet gateway; private subnets reach out via NAT and stay unreachable inbound.</p>
  </div>
  <div class="takeaway-card">
    <h4>L4 is fast, L7 is smart</h4>
    <p>Layer-4 balancers move connections protocol-agnostically; Layer-7 balancers route on URLs and headers, terminate TLS, and apply WAF rules.</p>
  </div>
  <div class="takeaway-card">
    <h4>CDNs and anycast bring content close</h4>
    <p>Edge caches plus one anycast address announced from many PoPs cut latency, spread load, and absorb DDoS — all via normal BGP routing.</p>
  </div>
  <div class="takeaway-card">
    <h4>NAT controls egress</h4>
    <p>Outbound-only access for private workloads, plus a natural point to filter and meter traffic leaving the cloud.</p>
  </div>
  <div class="takeaway-card">
    <h4>Shared responsibility is mostly yours</h4>
    <p>The provider secures the fabric; you own security groups, subnet design, IAM, and data. Most cloud breaches are customer misconfigurations.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Related pages</h4>
  <ul>
    <li><a href="fundamentals.html">Layers &amp; Addressing</a> — CIDR subnetting that VPC and subnet planning depend on.</li>
    <li><a href="routing.html">Routing &amp; Switching</a> — longest-prefix match, NAT, and the BGP behind anycast.</li>
    <li><a href="performance-and-security.html">Performance, QoS &amp; Security</a> — stateful inspection, firewalls, and the defenses cloud controls implement.</li>
    <li><a href="modern-architecture.html">Modern &amp; Future Networking</a> — the SDN, NFV, and P4 foundations the cloud is built on.</li>
    <li><a href="../aws/">AWS</a> — VPC, Direct Connect, ELB, and CloudFront in a real cloud.</li>
    <li><a href="../kubernetes/">Kubernetes</a> — cluster networking, CNI, and service meshes.</li>
    <li><a href="../cybersecurity/">Cybersecurity</a> — zero-trust and the network security side of shared responsibility.</li>
  </ul>
</div>
