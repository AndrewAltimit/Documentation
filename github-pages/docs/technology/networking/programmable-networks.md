---
layout: docs
title: "Networking: Programmable Networks (SDN/NFV/P4)"
permalink: /docs/technology/networking/programmable-networks.html
toc: true
toc_sticky: true
hide_title: true
---

[Networking](./) &raquo; Programmable Networks

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Programmable Networks</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">SDN, NFV, P4, MPLS, and segment routing — turning fixed silicon into software you can change</p>
</div>

For decades a network device bundled three things in one box: the **data plane** that forwards packets, the **control plane** that decides where they go, and fixed silicon that dictated which protocols it understood. The defining trend of modern networking is pulling those apart and turning each into software you can program. This page traces that shift — separating the control and data planes with SDN, replacing hardware appliances with software in NFV, defining packet processing itself with P4, and the label- and segment-based forwarding fabrics (MPLS, SR, SRv6) that carry this traffic at scale — and closes with the IPv6 transition that underpins much of it.

The Python listings throughout this page are *illustrative* — compact models that show the logic of each idea rather than production code.

## Networks Become Software

A traditional router or switch is a vertically integrated, closed appliance: the vendor ships the forwarding silicon, the routing software, and the management interface as one unit, and you take what you are given. Programmable networking decomposes that monolith along three axes:

| Layer | Traditional | Programmable | Technology |
|-------|-------------|--------------|------------|
| Control plane | Distributed, per-device, vendor firmware | Centralized, open, software | SDN (OpenFlow, P4Runtime) |
| Network functions | Dedicated hardware appliances | Software on commodity servers | NFV (VNFs, service chains) |
| Data plane | Fixed-function ASIC | Reconfigurable pipeline | P4 |
| Forwarding fabric | Per-hop IP lookup | Label / segment forwarding | MPLS, SR, SRv6 |

The recurring theme is **disaggregation**: separate the thing that decides from the thing that forwards, separate the function from the box, and expose each through an open, programmable interface.

## SDN: Programming the Control Plane

Traditional networks are like city streets with fixed traffic lights. **Software-Defined Networking (SDN)** makes networks programmable, like having smart traffic lights that adapt to real-time conditions. It does this by separating the control plane from the data plane: a centralized controller computes forwarding decisions and pushes them down to simple switches. SDN has evolved well beyond data centers to enable 5G network slicing, edge-computing orchestration, and AI-driven network optimization.

### The Three-Plane Model

```
         +-----------------------------------------+
         |          Application Plane              |
         |  (traffic engineering, firewall apps,   |
         |   load balancing, network monitoring)   |
         +-----------------------------------------+
                  ^                       |
   Northbound API |  (REST / gRPC)        v
         +-----------------------------------------+
         |            Control Plane                |
         |   SDN controller (ONOS, OpenDaylight)   |
         |   topology, path computation, policy    |
         +-----------------------------------------+
                  ^                       |
   Southbound API | (OpenFlow / P4Runtime / NETCONF)
         +-----------------------------------------+
         |             Data Plane                  |
         |   "dumb" switches that match-and-act    |
         |   on flow rules pushed from above       |
         +-----------------------------------------+
```

**Components**:
- **Controller**: Centralized management (ONOS, OpenDaylight, Ryu, Faucet). Maintains a global view of the topology and computes forwarding decisions.
- **Southbound API**: Controller to switches (OpenFlow 1.5+, P4Runtime, NETCONF/YANG, gNMI).
- **Northbound API**: Applications to controller (REST, gRPC). Lets a traffic-engineering or security app express intent without touching individual devices.
- **Intent-based networking**: Declarative network management — describe *what* you want ("tenant A must never reach tenant B"), not *how* to wire the flow rules.

### OpenFlow and the Match-Action Abstraction

OpenFlow was the protocol that launched SDN (McKeown et al., 2008). Its core abstraction is the **flow table**: an ordered list of entries, each consisting of

- a **match** over packet header fields (Ethernet src/dst, VLAN, IP src/dst, TCP/UDP ports, ingress port, ...),
- a set of **actions** (forward to a port, drop, rewrite a field, push/pop a tag, send to controller),
- a **priority** (highest-priority matching entry wins),
- **counters** (packets/bytes), and **timeouts** (idle and hard).

A packet entering a switch is matched against the table top-down. If it matches an entry, the actions execute. If it matches nothing, the default behavior (in modern OpenFlow, an explicit table-miss entry) usually sends it to the controller as a **packet-in** message. The controller decides what to do, optionally installs a new flow entry via a **flow-mod** message so future packets of that flow are handled in hardware at line rate, and emits a **packet-out** for the current packet. This "first packet to the controller, rest in the fast path" pattern is the heart of reactive SDN.

OpenFlow 1.1+ generalized the single table into a **multi-table pipeline** with a metadata register carried between tables and a **group table** for multipath, multicast, and fast failover. The cost of the centralized model is the controller's reachability and scale — production deployments run controller clusters and pre-install proactive flows to avoid a packet-in storm.

In practice, an SDN controller learns the topology, installs flow rules into switches, and handles the packets that don't yet match any rule. The following model implements that core loop — MAC learning, flow installation, and shortest-path computation with a backup path:

```python
class SDNController:
    """Software-Defined Network Controller"""
    
    def __init__(self):
        self.switches = {}
        self.topology = nx.Graph()
        self.flow_tables = defaultdict(list)
        self.packet_in_handlers = []
        self.statistics = defaultdict(lambda: {'packets': 0, 'bytes': 0})
        
    def handle_switch_connect(self, switch_id, features):
        """Handle switch connection"""
        self.switches[switch_id] = {
            'features': features,
            'ports': features['ports'],
            'flow_table_size': 0
        }
        
        # Install default flows
        self.install_default_flows(switch_id)
        
    def install_flow(self, switch_id, match, actions, priority=0, idle_timeout=0):
        """Install OpenFlow rule"""
        flow_mod = {
            'match': match,
            'actions': actions,
            'priority': priority,
            'idle_timeout': idle_timeout,
            'cookie': random.randint(1, 2**32)
        }
        
        self.flow_tables[switch_id].append(flow_mod)
        
        # Send to switch
        self.send_flow_mod(switch_id, flow_mod)
        
    def handle_packet_in(self, switch_id, port, packet_data):
        """Handle packet not matching any flow"""
        # Parse packet
        packet = self.parse_packet(packet_data)
        
        # Learn source MAC
        self.mac_learning(switch_id, packet['src_mac'], port)
        
        # Find destination
        out_port = self.find_destination(switch_id, packet['dst_mac'])
        
        if out_port:
            # Install flow for future packets
            match = {
                'eth_dst': packet['dst_mac'],
                'eth_src': packet['src_mac']
            }
            actions = [{'type': 'output', 'port': out_port}]
            
            self.install_flow(switch_id, match, actions, priority=1, 
                            idle_timeout=300)
            
            # Send current packet
            self.packet_out(switch_id, packet_data, out_port)
        else:
            # Flood
            self.packet_out(switch_id, packet_data, 'FLOOD')
            
    def calculate_paths(self):
        """Calculate all shortest paths in topology"""
        paths = {}
        
        for src in self.topology.nodes():
            for dst in self.topology.nodes():
                if src != dst:
                    try:
                        # Primary path
                        path = nx.shortest_path(self.topology, src, dst, 
                                              weight='weight')
                        
                        # Backup path (node-disjoint)
                        temp_graph = self.topology.copy()
                        # Remove intermediate nodes from primary path
                        for node in path[1:-1]:
                            temp_graph.remove_node(node)
                            
                        backup_path = None
                        try:
                            backup_path = nx.shortest_path(temp_graph, src, dst,
                                                         weight='weight')
                        except nx.NetworkXNoPath:
                            pass
                            
                        paths[(src, dst)] = {
                            'primary': path,
                            'backup': backup_path
                        }
                    except nx.NetworkXNoPath:
                        paths[(src, dst)] = {'primary': None, 'backup': None}
                        
        return paths
```

### Reactive vs. Proactive, Centralized vs. Distributed

Two design choices dominate real SDN deployments:

- **Reactive flow installation** sends the first packet of every new flow to the controller, which installs a rule on demand. It uses table space efficiently but adds latency to flow setup and makes the controller a bottleneck under churn.
- **Proactive flow installation** pre-computes and pushes rules before traffic arrives (e.g. full-mesh forwarding in a data-center fabric). It avoids per-flow controller round-trips at the cost of larger flow tables.

Likewise, the "centralized" controller is logically central but physically a **cluster** — controllers like ONOS use a distributed store (Raft/Atomix) so the global view survives a controller failure. This is the modern reconciliation between SDN's logically-centralized model and the fault tolerance that distributed routing protocols gave you for free.

> For the distributed routing protocols SDN replaces — OSPF, BGP, and per-hop shortest-path forwarding — see [Routing & Switching](routing.html).

## P4: Programming the Data Plane

SDN lets us program the *control* plane, but the switches themselves still only understand protocols baked into their silicon — an OpenFlow switch can only match the header fields its ASIC was built to parse. **P4** (Programming Protocol-independent Packet Processors) goes further: it lets us define how switches parse and process packets. Imagine customizing not just traffic rules, but how traffic is *understood*. This makes it possible to deploy new protocols without new hardware, do in-network computing (processing data as it flows), and build advanced telemetry.

### The PISA Pipeline

P4 targets the **Protocol-Independent Switch Architecture (PISA)**, a reconfigurable pipeline with four stages:

1. **Parser** — a programmable state machine that walks the packet header-by-header, extracting fields into typed headers. You declare the grammar; the parser is protocol-agnostic.
2. **Ingress match-action** — a series of tables, each matching extracted fields and invoking actions (modify headers, set metadata, choose an egress port, drop).
3. **Egress match-action** — a second set of tables applied after the switch's queueing/replication, useful for per-output-port rewrites and multicast.
4. **Deparser** — reassembles the (possibly modified) headers back onto the wire.

The compiler maps your P4 program onto the target (a Tofino ASIC, a SmartNIC, BMv2 software switch, or an FPGA). The control plane populates the tables at runtime over **P4Runtime** (a gRPC API), giving you the same controller/data-plane split as OpenFlow, but with a data plane whose very structure you defined.

```python
class P4DataPlane:
    """Define custom packet processing behavior in switches.
    
    Use cases:
    - New protocols without hardware changes
    - In-network computing (processing data as it flows)
    - Advanced telemetry and monitoring
    """
    
    def __init__(self):
        self.tables = {}
        self.actions = {}
        self.parsers = {}
        self.metadata = {}
        
    def define_parser(self):
        """Define packet parser in P4 style"""
        parser_def = '''
        parser MyParser(packet_in packet,
                       out headers hdr,
                       inout metadata meta,
                       inout standard_metadata_t standard_metadata) {
            
            state start {
                transition parse_ethernet;
            }
            
            state parse_ethernet {
                packet.extract(hdr.ethernet);
                transition select(hdr.ethernet.etherType) {
                    0x0800: parse_ipv4;
                    0x86DD: parse_ipv6;
                    default: accept;
                }
            }
            
            state parse_ipv4 {
                packet.extract(hdr.ipv4);
                transition select(hdr.ipv4.protocol) {
                    6: parse_tcp;
                    17: parse_udp;
                    default: accept;
                }
            }
            
            state parse_tcp {
                packet.extract(hdr.tcp);
                transition accept;
            }
        }
        '''
        return parser_def
        
    def define_match_action_table(self, name, match_fields, actions, size=1024):
        """Define match-action table"""
        self.tables[name] = {
            'match_fields': match_fields,
            'actions': actions,
            'entries': {},
            'default_action': None,
            'size': size
        }
        
    def add_table_entry(self, table_name, match_values, action_name, action_params):
        """Add entry to match-action table"""
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found")
            
        # Create match key
        match_key = tuple(match_values)
        
        # Add entry
        self.tables[table_name]['entries'][match_key] = {
            'action': action_name,
            'params': action_params
        }
        
    def process_packet(self, packet):
        """Process packet through P4 pipeline"""
        # Parse packet
        headers = self.parse_packet(packet)
        metadata = {'ingress_port': packet.ingress_port}
        
        # Ingress pipeline
        headers, metadata = self.ingress_pipeline(headers, metadata)
        
        # Egress decision
        if metadata.get('drop', False):
            return None
            
        # Egress pipeline
        headers, metadata = self.egress_pipeline(headers, metadata)
        
        # Deparse
        output_packet = self.deparse_packet(headers)
        
        return output_packet, metadata.get('egress_port')
```

### Match Kinds and What the Data Plane Can Do

P4 tables match on different **match kinds** that map to different silicon:

- `exact` — hash table lookup (e.g. exact destination MAC).
- `lpm` — longest-prefix match, the IP-routing primitive (a TCAM or algorithmic LPM).
- `ternary` — masked match for ACLs and firewall rules (TCAM).
- `range` — port-range matching.

Beyond simple forwarding, the programmable data plane enables **in-network computing**: stateful registers and counters let a switch maintain per-flow state, implement load-balancing or caching, run sketches for heavy-hitter detection, or aggregate data for distributed training — all at terabit line rate. A flagship application is **In-band Network Telemetry (INT)**, where each switch on the path appends its queue depth, timestamp, and hop latency into a metadata stack, giving per-packet, per-hop visibility impossible with traditional sampling.

The limits are real, though: a hardware P4 pipeline runs each table at most once per packet, has bounded stages and memory, and offers no loops — it trades generality for line-rate determinism. Software targets (BMv2) lift these limits but at orders-of-magnitude lower throughput.

## NFV: Network Functions Become Software

Why buy expensive hardware firewalls, routers, and load balancers when software can do the job? **Network Function Virtualization (NFV)** transforms these appliances into software (Virtual Network Functions, or VNFs) running on standard servers. The payoff is deploying new services in minutes instead of months, scaling them up and down on demand, cutting hardware costs, and chaining functions together (firewall → IDS → load balancer) as a **service function chain**.

### The NFV Architecture (ETSI)

The ETSI NFV reference architecture separates three concerns:

- **NFVI (Infrastructure)** — the compute, storage, and network (often a cloud/OpenStack/Kubernetes substrate) on which functions run.
- **VNFs** — the virtualized functions themselves (a virtual firewall, vRouter, vBNG, 5G UPF, ...), packaged as VMs or, increasingly, containers (then called CNFs — Cloud-native Network Functions).
- **MANO (Management and Orchestration)** — the orchestrator, VNF managers, and Virtualized Infrastructure Manager that instantiate, scale, heal, and chain VNFs according to descriptors.

NFV and SDN are complementary: NFV virtualizes *what* the function is, while SDN programs *how* traffic is steered between functions. Together they enable **Service Function Chaining (SFC)** — pushing a packet through an ordered list of VNFs (encoded with NSH, the Network Service Header, or with SRv6) regardless of where each VNF physically runs.

```python
class VirtualNetworkFunction:
    """Transform hardware network appliances into flexible software.
    
    Benefits:
    - Deploy new services in minutes, not months
    - Scale up/down based on demand
    - Reduce hardware costs
    - Enable service chaining (firewall → IDS → load balancer)
    """
    
    def __init__(self, cpu_cores=1, memory_mb=1024):
        self.cpu_cores = cpu_cores
        self.memory_mb = memory_mb
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()
        self.statistics = {
            'packets_processed': 0,
            'packets_dropped': 0,
            'processing_time_ms': []
        }
        
    def process_packet(self, packet):
        """Override in subclasses"""
        raise NotImplementedError
        
    def run(self):
        """Main processing loop"""
        while True:
            try:
                packet = self.rx_queue.get(timeout=0.001)
                start_time = time.time()
                
                # Process packet
                result = self.process_packet(packet)
                
                if result:
                    self.tx_queue.put(result)
                    self.statistics['packets_processed'] += 1
                else:
                    self.statistics['packets_dropped'] += 1
                    
                # Record processing time
                proc_time = (time.time() - start_time) * 1000
                self.statistics['processing_time_ms'].append(proc_time)
                
            except queue.Empty:
                continue

class VirtualFirewall(VirtualNetworkFunction):
    """Stateful firewall VNF"""
    
    def __init__(self, rules_file=None, **kwargs):
        super().__init__(**kwargs)
        self.rules = self.load_rules(rules_file)
        self.connection_table = {}
        self.connection_timeout = 300  # seconds
        
    def process_packet(self, packet):
        """Apply firewall rules"""
        # Check established connections
        conn_key = self.get_connection_key(packet)
        
        if conn_key in self.connection_table:
            # Update timestamp
            self.connection_table[conn_key]['last_seen'] = time.time()
            return packet
            
        # Check rules
        for rule in self.rules:
            if self.match_rule(packet, rule):
                if rule['action'] == 'allow':
                    # Add to connection table
                    self.connection_table[conn_key] = {
                        'created': time.time(),
                        'last_seen': time.time(),
                        'packets': 1
                    }
                    return packet
                else:
                    return None  # Drop
                    
        # Default deny
        return None

class ServiceFunctionChain:
    """Chain multiple VNFs"""
    
    def __init__(self):
        self.vnfs = []
        self.links = []
        
    def add_vnf(self, vnf):
        """Add VNF to chain"""
        self.vnfs.append(vnf)
        
        # Create link queues
        if len(self.vnfs) > 1:
            link_queue = queue.Queue()
            self.links.append(link_queue)
            
            # Connect previous VNF output to current input
            self.vnfs[-2].tx_queue = link_queue
            self.vnfs[-1].rx_queue = link_queue
            
    def deploy(self):
        """Deploy service chain"""
        threads = []
        
        for vnf in self.vnfs:
            thread = threading.Thread(target=vnf.run)
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        return threads
```

### Making Software Forwarding Fast

A naive VNF that copies every packet through the kernel networking stack cannot keep up with a 10/40/100 GbE NIC. NFV in production leans on **fast-path** techniques:

- **DPDK (Data Plane Development Kit)** — poll-mode drivers that bypass the kernel, pin packet processing to dedicated cores, and use hugepages and lockless ring buffers to push tens of millions of packets per second from userspace.
- **eBPF / XDP** — run a verified program in the kernel at the earliest receive hook (the driver), dropping or redirecting packets before the stack touches them. Projects like Cilium and Katran build L3/L4 load balancers and firewalls this way.
- **SR-IOV** — hardware partitions one physical NIC into many virtual functions, giving each VNF near-native I/O without a software switch in the path.

These are the same ideas that let a P4 SmartNIC or a kernel eBPF program do "in-network" work — the boundary between "the network" and "the server" has largely dissolved.

## Label and Segment Forwarding: MPLS, SR, and SRv6

Per-hop IP forwarding makes an independent longest-prefix-match decision at every router, which is flexible but offers little control over *which* path a flow takes. Label- and segment-based forwarding instead encode the path (or a service) into the packet, so the network's interior just follows instructions.

### MPLS (Multiprotocol Label Switching)

**MPLS** forwards packets based on short, fixed-length **labels** rather than full routing-table lookups. An ingress router (the **Label Edge Router**, LER) classifies a packet into a **Forwarding Equivalence Class** and pushes a 4-byte label header (the "shim") between L2 and L3. Interior routers (**Label Switch Routers**, LSRs) do a fast exact-match lookup on the label, **swap** it for the next-hop label, and forward — never re-examining the IP header. The egress LER **pops** the label and forwards normally.

The label header carries a 20-bit label, a 3-bit traffic-class (QoS) field, a bottom-of-stack bit (labels can be stacked), and a TTL. Stacking enables hierarchy: an outer label selects the tunnel, an inner label selects the VPN or service.

Benefits:
- **Traffic engineering** — explicit **Label-Switched Paths (LSPs)** can be pinned along non-shortest routes (via RSVP-TE) to balance load or honor latency constraints.
- **QoS** — the traffic-class field lets each hop apply per-class queueing without deep inspection.
- **VPN services** — **L3VPN** (BGP/MPLS, RFC 4364) and **L2VPN/VPLS** let a provider carry many customers' overlapping address spaces over one core using the label stack.
- **Reduced lookups** — exact-match label swap is simpler and faster than recursive IP LPM (less relevant on modern hardware, but historically the point).

### Segment Routing (SR)

**Segment Routing** keeps the label/forwarding idea but removes MPLS's signaling baggage (no RSVP-TE or LDP state in the core). The source encodes the path as an ordered list of **segments** — instructions like "go to node X" (a *node* segment) or "traverse link L" (an *adjacency* segment). Interior routers hold no per-flow tunnel state; they just execute the segment on top of the stack and pop it. This **source routing** model scales because the network state lives in the packet, not in the routers.

Segments are advertised as **SIDs (Segment IDs)** in the IGP (OSPF/IS-IS extensions), so a controller or the ingress node can compute and impose an explicit path with nothing more than the existing link-state database.

```python
def build_sr_label_stack(path, sid_map, node_sids, adj_sids):
    """Translate an explicit hop list into an SR-MPLS label stack.

    node_sids: {node -> prefix-SID label}
    adj_sids:  {(a, b) -> adjacency-SID label}  (link-local)
    A node-SID routes by shortest path to that node; an adjacency-SID
    pins a specific link. Mixing them gives loose or strict source routes.
    """
    stack = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if (a, b) in adj_sids:
            stack.append(adj_sids[(a, b)])   # strict: this exact link
        else:
            stack.append(node_sids[b])       # loose: shortest path to b
    # Stack is imposed top-to-bottom; top label is the first instruction.
    return stack
```

Segment Routing has two data planes:

- **SR-MPLS** — segments are MPLS labels, reusing existing MPLS hardware with a much simpler control plane.
- **SRv6** — segments are full IPv6 addresses carried in an IPv6 extension header (the **Segment Routing Header**, SRH). Because each SID is a 128-bit IPv6 address, a SID can encode not just "where" but "what to do" — a **network program** (RFC 8986 "SRv6 Network Programming"): forwarding, VPN decapsulation, service-chaining a packet through a VNF, and more. SRv6 unifies overlay (VPN), underlay (TE), and service chaining into one IPv6-native mechanism, which is why it has become a centerpiece of 5G transport and modern provider cores.

> SRv6 builds directly on IPv6 extension headers — see the IPv6 section below for the header machinery it depends on.

## IPv6 Transition

IPv6 is the substrate SRv6 and much of modern transport assume, but the world still runs enormous amounts of IPv4. **Transition mechanisms** let the two coexist during the long migration. They fall into three families:

- **Dual stack** — hosts and routers run IPv4 and IPv6 simultaneously, choosing per-destination (typically via *Happy Eyeballs*, RFC 8305, racing A and AAAA). Conceptually the cleanest approach, but it doubles the addressing/operational burden and does nothing to relieve IPv4 exhaustion.
- **Tunneling** — carry IPv6 over an IPv4 network (or vice versa) by encapsulation: **6to4** and **6rd** (auto-derive an IPv6 prefix from an IPv4 address), **Teredo** (IPv6 through IPv4 NAT via UDP), and **GRE / manual tunnels** in provider cores. Useful to bridge IPv6 islands across legacy IPv4 transit.
- **Translation** — rewrite headers between families: **NAT64** + **DNS64** let IPv6-only clients reach IPv4-only servers (DNS64 synthesizes a AAAA record inside a well-known prefix `64:ff9b::/96`; NAT64 translates the resulting IPv6 flow to IPv4). **464XLAT** combines a stateless client-side translator with a stateful provider NAT64 so IPv6-only mobile networks can still carry IPv4-only applications.

The IPv6 header itself simplifies the data plane in ways that complement programmable networking: a fixed 40-byte base header, no in-network fragmentation (PMTUD is mandatory), no header checksum, and a chain of **extension headers** for options. That extension-header chain is exactly what SRv6 reuses to carry its segment list — making IPv6 not just the next addressing scheme but the programmable transport layer beneath SR.

## SRv6, eBPF, and the Convergence

Several of the threads above are converging in production today:

- **SRv6 (Segment Routing over IPv6)** — simplified network programming, source-routed traffic engineering, and IPv6-native service chaining in one mechanism, increasingly the default for 5G transport.
- **eBPF** — programmable kernel networking without modules, powering load balancers, firewalls, and observability (Cilium, Katran, Pixie). The server's kernel has become a programmable data plane in its own right.
- **In-network computing** — P4-programmable switches and SmartNICs that offload caching, load balancing, telemetry, and aggregation into the fabric.
- **AI/ML in networking** — controllers that predict congestion, auto-tune paths, and detect anomalies on top of the global view SDN exposes.

The common thread is the one this page opened with: every layer of the network — control plane, data plane, network function, and forwarding fabric — is becoming software you can change.

## See Also

- [Routing & Switching](routing.html) — the classical distributed control plane SDN replaces, plus OSPF and BGP.
- [Transport & Protocols](transport-and-protocols.html) — QUIC, TCP, and the protocols riding over these fabrics.
- [Performance, QoS & Security](performance-and-security.html) — the metrics, queueing, and defenses these architectures build on.
- [Modern & Future Networking](modern-architecture.html) — cloud-networking primitives and the research frontier built on programmable networks.
- [AWS](../aws/) — VPC, Direct Connect, and managed load balancers in a real cloud.
- [Kubernetes](../kubernetes/) — cluster networking, CNI plugins, and eBPF-based dataplanes.
