---
layout: default
title: Networking
---

# Networking

<html><header><link rel="stylesheet" href="https://andrewaltimit.github.io/Documentation/style.css"></header></html>

Computer networking is the practice of connecting computers and devices to share resources and communicate. Understanding networking fundamentals is essential for system administration, security, and modern application development.

## Mathematical Foundations of Networking

### Queueing Theory

#### M/M/1 Queue Analysis
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class MM1Queue:
    """M/M/1 queueing model for network performance analysis"""
    def __init__(self, arrival_rate, service_rate):
        self.lambda_ = arrival_rate  # λ
        self.mu = service_rate       # μ
        self.rho = arrival_rate / service_rate  # ρ = λ/μ
        
        if self.rho >= 1:
            raise ValueError("System is unstable (ρ >= 1)")
    
    def average_customers(self):
        """L = ρ / (1 - ρ)"""
        return self.rho / (1 - self.rho)
    
    def average_queue_length(self):
        """Lq = ρ² / (1 - ρ)"""
        return self.rho**2 / (1 - self.rho)
    
    def average_waiting_time(self):
        """W = 1 / (μ - λ)"""
        return 1 / (self.mu - self.lambda_)
    
    def average_queue_time(self):
        """Wq = ρ / (μ - λ)"""
        return self.rho / (self.mu - self.lambda_)
    
    def response_time_distribution(self, t):
        """P(T ≤ t) = 1 - e^(-(μ-λ)t)"""
        return 1 - np.exp(-(self.mu - self.lambda_) * t)
    
    def simulate(self, duration=1000):
        """Discrete event simulation"""
        events = []
        current_time = 0
        customers_in_system = 0
        
        while current_time < duration:
            # Next arrival
            inter_arrival = np.random.exponential(1/self.lambda_)
            arrival_time = current_time + inter_arrival
            
            # Service time
            service_time = np.random.exponential(1/self.mu)
            
            events.append({
                'time': arrival_time,
                'type': 'arrival',
                'service_time': service_time
            })
            
            current_time = arrival_time
            
        return events

# Example: Router buffer analysis
router = MM1Queue(arrival_rate=800, service_rate=1000)  # packets/second
print(f"Average queue length: {router.average_queue_length():.2f} packets")
print(f"Average waiting time: {router.average_queue_time()*1000:.2f} ms")
```

#### Network of Queues (Jackson Network)
```python
class JacksonNetwork:
    """Open Jackson network for multi-hop analysis"""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.routing_matrix = np.zeros((num_nodes, num_nodes))
        self.external_arrivals = np.zeros(num_nodes)
        self.service_rates = np.zeros(num_nodes)
        
    def set_routing(self, from_node, to_node, probability):
        """Set routing probability between nodes"""
        self.routing_matrix[from_node, to_node] = probability
        
    def solve_traffic_equations(self):
        """Solve λᵢ = γᵢ + Σⱼ λⱼ * Pⱼᵢ"""
        # Form system: (I - P^T)λ = γ
        I = np.eye(self.num_nodes)
        A = I - self.routing_matrix.T
        
        # Solve for arrival rates at each node
        arrival_rates = np.linalg.solve(A, self.external_arrivals)
        
        return arrival_rates
    
    def analyze_performance(self):
        """Analyze performance metrics for each node"""
        arrival_rates = self.solve_traffic_equations()
        metrics = []
        
        for i in range(self.num_nodes):
            if arrival_rates[i] < self.service_rates[i]:
                queue = MM1Queue(arrival_rates[i], self.service_rates[i])
                metrics.append({
                    'node': i,
                    'utilization': queue.rho,
                    'avg_delay': queue.average_waiting_time(),
                    'avg_queue': queue.average_customers()
                })
            else:
                metrics.append({
                    'node': i,
                    'utilization': float('inf'),
                    'avg_delay': float('inf'),
                    'avg_queue': float('inf')
                })
                
        return metrics
```

### Graph Theory for Networks

#### Shortest Path Algorithms
```python
import heapq
from collections import defaultdict

class NetworkGraph:
    """Advanced graph algorithms for network routing"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
        
    def add_edge(self, u, v, weight, metrics=None):
        """Add edge with weight and optional metrics"""
        self.graph[u].append({
            'vertex': v,
            'weight': weight,
            'metrics': metrics or {}
        })
        self.vertices.add(u)
        self.vertices.add(v)
        
    def dijkstra_multi_metric(self, source, metric='weight'):
        """Dijkstra with configurable metric"""
        distances = {vertex: float('infinity') for vertex in self.vertices}
        distances[source] = 0
        predecessors = {vertex: None for vertex in self.vertices}
        
        pq = [(0, source)]
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_distance > distances[current_vertex]:
                continue
                
            for neighbor in self.graph[current_vertex]:
                if metric == 'weight':
                    edge_weight = neighbor['weight']
                else:
                    edge_weight = neighbor['metrics'].get(metric, float('inf'))
                    
                distance = current_distance + edge_weight
                
                if distance < distances[neighbor['vertex']]:
                    distances[neighbor['vertex']] = distance
                    predecessors[neighbor['vertex']] = current_vertex
                    heapq.heappush(pq, (distance, neighbor['vertex']))
                    
        return distances, predecessors
    
    def k_shortest_paths(self, source, target, k):
        """Yen's algorithm for k-shortest paths"""
        # First shortest path
        distances, predecessors = self.dijkstra_multi_metric(source)
        
        if distances[target] == float('infinity'):
            return []
            
        # Reconstruct first path
        A = [self._reconstruct_path(predecessors, source, target)]
        B = []
        
        for k_iter in range(1, k):
            for i in range(len(A[k_iter-1]) - 1):
                spur_node = A[k_iter-1][i]
                root_path = A[k_iter-1][:i+1]
                
                # Remove edges used in previous paths
                removed_edges = []
                for path in A:
                    if len(path) > i and path[:i+1] == root_path:
                        u, v = path[i], path[i+1]
                        # Temporarily remove edge
                        for j, edge in enumerate(self.graph[u]):
                            if edge['vertex'] == v:
                                removed_edges.append((u, j, edge))
                                
                # Find spur path
                spur_distances, spur_pred = self.dijkstra_multi_metric(spur_node)
                
                if spur_distances[target] < float('infinity'):
                    spur_path = self._reconstruct_path(spur_pred, spur_node, target)
                    total_path = root_path[:-1] + spur_path
                    
                    if total_path not in B:
                        B.append(total_path)
                        
                # Restore edges
                for u, j, edge in removed_edges:
                    self.graph[u].insert(j, edge)
                    
            if not B:
                break
                
            # Sort B by path cost
            B.sort(key=lambda p: self._path_cost(p))
            A.append(B.pop(0))
            
        return A
```

### Network Flow Algorithms

#### Maximum Flow Implementation
```python
class MaxFlow:
    """Ford-Fulkerson with Edmonds-Karp implementation"""
    
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph)
        
    def bfs(self, source, sink, parent):
        """BFS to find augmenting path"""
        visited = [False] * self.n
        queue = [source]
        visited[source] = True
        
        while queue:
            u = queue.pop(0)
            
            for v in range(self.n):
                if not visited[v] and self.graph[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    
                    if v == sink:
                        return True
                        
        return False
    
    def max_flow(self, source, sink):
        """Find maximum flow from source to sink"""
        parent = [-1] * self.n
        max_flow_value = 0
        
        # Create residual graph
        residual = [[self.graph[i][j] for j in range(self.n)] 
                    for i in range(self.n)]
        
        while self.bfs(source, sink, parent):
            # Find minimum residual capacity
            path_flow = float('inf')
            s = sink
            
            while s != source:
                path_flow = min(path_flow, residual[parent[s]][s])
                s = parent[s]
                
            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = parent[v]
                
            max_flow_value += path_flow
            
        return max_flow_value, residual
    
    def min_cut(self, source, residual):
        """Find minimum cut after max flow"""
        visited = [False] * self.n
        queue = [source]
        visited[source] = True
        
        while queue:
            u = queue.pop(0)
            for v in range(self.n):
                if not visited[v] and residual[u][v] > 0:
                    visited[v] = True
                    queue.append(v)
                    
        # Find edges in cut
        cut_edges = []
        for i in range(self.n):
            for j in range(self.n):
                if visited[i] and not visited[j] and self.graph[i][j] > 0:
                    cut_edges.append((i, j, self.graph[i][j]))
                    
        return cut_edges
```

## OSI Model

The Open Systems Interconnection model conceptualizes network communication in seven layers:

### Layer 7: Application
User-facing protocols and services.
- HTTP/HTTPS
- FTP
- SMTP
- DNS
- SSH

### Layer 6: Presentation
Data formatting, encryption, compression.
- SSL/TLS
- JPEG, GIF
- ASCII, Unicode

### Layer 5: Session
Establishes, manages, and terminates connections.
- SQL
- RPC
- NetBIOS

### Layer 4: Transport
Reliable data transfer and error recovery.
- TCP (Transmission Control Protocol)
- UDP (User Datagram Protocol)
- Port numbers

### Layer 3: Network
Routing and logical addressing.
- IP (Internet Protocol)
- ICMP
- Routing protocols (OSPF, BGP)

### Layer 2: Data Link
Node-to-node delivery, error detection.
- Ethernet
- MAC addresses
- Switches
- ARP

### Layer 1: Physical
Bits over physical medium.
- Cables, fiber optics
- Hubs
- Signal encoding

## TCP/IP Stack

Practical four-layer model used in modern networks:

1. **Application Layer**: HTTP, FTP, SSH, DNS
2. **Transport Layer**: TCP, UDP
3. **Internet Layer**: IP, ICMP, ARP
4. **Network Access Layer**: Ethernet, Wi-Fi

## IP Addressing

### IPv4
32-bit addresses, written as four octets (e.g., 192.168.1.1)

**Address Classes**:
- Class A: 1.0.0.0 to 126.255.255.255 (/8)
- Class B: 128.0.0.0 to 191.255.255.255 (/16)
- Class C: 192.0.0.0 to 223.255.255.255 (/24)

**Private Address Ranges**:
- 10.0.0.0/8
- 172.16.0.0/12
- 192.168.0.0/16

### IPv6
128-bit addresses, written in hexadecimal (e.g., 2001:db8::1)

**Address Types**:
- Unicast: Single interface
- Multicast: Group of interfaces
- Anycast: Nearest of group

**Special Addresses**:
- ::1 - Loopback
- fe80::/10 - Link-local
- 2000::/3 - Global unicast

### Subnetting
Dividing networks into smaller segments.

**CIDR Notation**: 192.168.1.0/24
- Network: 192.168.1.0
- Broadcast: 192.168.1.255
- Usable hosts: 254

**Subnet Calculation**:
```
Network bits: 24
Host bits: 32 - 24 = 8
Total addresses: 2^8 = 256
Usable addresses: 256 - 2 = 254
```

## Advanced Protocol Analysis

### TCP Congestion Control Algorithms

#### TCP Reno Implementation
```python
class TCPReno:
    """TCP Reno congestion control algorithm"""
    
    def __init__(self, mss=1460):
        self.mss = mss  # Maximum Segment Size
        self.cwnd = 1 * mss  # Congestion window
        self.ssthresh = 64 * 1024  # Slow start threshold
        self.state = 'slow_start'
        self.dup_ack_count = 0
        self.rtt_samples = []
        self.srtt = None  # Smoothed RTT
        self.rttvar = None  # RTT variance
        self.rto = 1.0  # Retransmission timeout
        
    def on_ack(self, ack_num, is_duplicate=False):
        """Handle ACK reception"""
        if is_duplicate:
            self.dup_ack_count += 1
            
            if self.dup_ack_count == 3:
                # Fast retransmit/recovery
                self.ssthresh = max(self.cwnd // 2, 2 * self.mss)
                self.cwnd = self.ssthresh + 3 * self.mss
                self.state = 'fast_recovery'
                return 'fast_retransmit'
                
            elif self.state == 'fast_recovery':
                # Inflate window
                self.cwnd += self.mss
                
        else:
            self.dup_ack_count = 0
            
            if self.state == 'slow_start':
                # Exponential increase
                self.cwnd += self.mss
                
                if self.cwnd >= self.ssthresh:
                    self.state = 'congestion_avoidance'
                    
            elif self.state == 'congestion_avoidance':
                # Additive increase
                self.cwnd += (self.mss * self.mss) / self.cwnd
                
            elif self.state == 'fast_recovery':
                # Exit fast recovery
                self.cwnd = self.ssthresh
                self.state = 'congestion_avoidance'
                
        return None
    
    def on_timeout(self):
        """Handle retransmission timeout"""
        self.ssthresh = max(self.cwnd // 2, 2 * self.mss)
        self.cwnd = 1 * self.mss
        self.state = 'slow_start'
        self.dup_ack_count = 0
        
        # Back off RTO
        self.rto = min(self.rto * 2, 60)
        
    def update_rtt(self, measured_rtt):
        """Update RTT estimates (RFC 6298)"""
        alpha = 0.125
        beta = 0.25
        K = 4
        
        if self.srtt is None:
            # First measurement
            self.srtt = measured_rtt
            self.rttvar = measured_rtt / 2
        else:
            # Update estimates
            self.rttvar = (1 - beta) * self.rttvar + beta * abs(self.srtt - measured_rtt)
            self.srtt = (1 - alpha) * self.srtt + alpha * measured_rtt
            
        # Calculate RTO
        self.rto = self.srtt + K * self.rttvar
        self.rto = max(self.rto, 1.0)  # Minimum 1 second
```

#### TCP BBR (Bottleneck Bandwidth and RTT)
```python
class TCPBBR:
    """Google's BBR congestion control"""
    
    def __init__(self):
        self.mode = 'startup'
        self.pacing_rate = 0
        self.cwnd = 0
        self.min_rtt = float('inf')
        self.min_rtt_stamp = 0
        self.btl_bw = 0  # Bottleneck bandwidth
        self.rtprop = 0  # Min RTT
        self.bandwidth_samples = []
        
    def update_model(self, delivered, interval, rtt):
        """Update bandwidth and RTT model"""
        # Update bandwidth estimate
        if interval > 0:
            bandwidth = delivered / interval
            self.bandwidth_samples.append(bandwidth)
            
            # Use windowed max filter
            if len(self.bandwidth_samples) > 10:
                self.bandwidth_samples.pop(0)
                
            self.btl_bw = max(self.bandwidth_samples)
            
        # Update RTT estimate
        self.rtprop = min(self.rtprop, rtt) if self.rtprop > 0 else rtt
        
    def calculate_pacing_rate(self):
        """Calculate pacing rate based on model"""
        if self.mode == 'startup':
            # High gain to quickly discover bandwidth
            pacing_gain = 2.89  # 2/ln(2)
        elif self.mode == 'drain':
            # Drain queue built during startup
            pacing_gain = 0.35  # 1/2.89
        elif self.mode == 'probe_bw':
            # Cycle through different gains
            gains = [1.25, 0.75, 1, 1, 1, 1, 1, 1]
            pacing_gain = gains[self.cycle_index % len(gains)]
        else:  # probe_rtt
            pacing_gain = 1
            
        self.pacing_rate = pacing_gain * self.btl_bw
        
    def update_cwnd(self):
        """Update congestion window"""
        if self.mode == 'probe_rtt':
            # Minimal cwnd to measure RTT
            self.cwnd = 4 * self.mss
        else:
            # BDP + headroom for probing
            bdp = self.btl_bw * self.rtprop
            self.cwnd = max(bdp * self.cwnd_gain, 4 * self.mss)
```

### Advanced Routing Protocols

#### BGP Implementation
```python
class BGPRouter:
    """Simplified BGP implementation"""
    
    def __init__(self, as_number, router_id):
        self.as_number = as_number
        self.router_id = router_id
        self.peers = {}
        self.rib_in = {}  # Received routes
        self.rib_loc = {}  # Local routes
        self.rib_out = {}  # Advertised routes
        self.best_paths = {}  # Best path selection
        
    def add_peer(self, peer_ip, peer_as, peer_type='ebgp'):
        """Add BGP peer"""
        self.peers[peer_ip] = {
            'as_number': peer_as,
            'type': peer_type,
            'state': 'idle',
            'hold_timer': 90,
            'keepalive_timer': 30
        }
        
    def process_update(self, peer_ip, nlri, attributes):
        """Process BGP UPDATE message"""
        for prefix in nlri:
            route_key = (peer_ip, prefix)
            
            # Store in RIB-In
            self.rib_in[route_key] = {
                'prefix': prefix,
                'peer': peer_ip,
                'attributes': attributes,
                'timestamp': time.time()
            }
            
        # Run best path selection
        self.best_path_selection()
        
    def best_path_selection(self):
        """BGP best path selection algorithm"""
        prefix_routes = defaultdict(list)
        
        # Group routes by prefix
        for (peer, prefix), route in self.rib_in.items():
            prefix_routes[prefix].append(route)
            
        for prefix, routes in prefix_routes.items():
            # Apply BGP decision process
            best_route = self.select_best_route(routes)
            
            if best_route:
                self.best_paths[prefix] = best_route
                
                # Install in RIB-Loc if best
                self.rib_loc[prefix] = best_route
                
                # Advertise to other peers
                self.advertise_route(prefix, best_route)
                
    def select_best_route(self, routes):
        """Apply BGP decision criteria"""
        if not routes:
            return None
            
        # Sort by BGP decision criteria
        def route_preference(route):
            attrs = route['attributes']
            return (
                -attrs.get('local_pref', 100),  # Higher is better
                len(attrs.get('as_path', [])),   # Shorter is better
                attrs.get('origin', 2),          # Lower is better (IGP < EGP < Incomplete)
                attrs.get('med', 0),             # Lower is better
                self.peers[route['peer']]['type'] == 'ebgp',  # Prefer eBGP
                attrs.get('next_hop', ''),       # Lower IP is better
                route['peer']                    # Lower peer IP is better
            )
            
        return min(routes, key=route_preference)
```

#### OSPF Link State Implementation
```python
class OSPFRouter:
    """OSPF link state routing protocol"""
    
    def __init__(self, router_id):
        self.router_id = router_id
        self.lsdb = {}  # Link State Database
        self.neighbors = {}
        self.interfaces = {}
        self.routing_table = {}
        
    def generate_lsa(self):
        """Generate Router LSA"""
        lsa = {
            'type': 1,  # Router LSA
            'router_id': self.router_id,
            'sequence': self.get_next_sequence(),
            'age': 0,
            'links': []
        }
        
        for intf_id, intf in self.interfaces.items():
            link = {
                'type': intf['type'],  # p2p, transit, stub
                'id': intf['neighbor_id'] if intf['type'] == 'p2p' else intf_id,
                'data': intf['ip_address'],
                'metric': intf['cost']
            }
            lsa['links'].append(link)
            
        return lsa
    
    def dijkstra_spf(self):
        """Calculate shortest paths using Dijkstra"""
        # Build graph from LSDB
        graph = self.build_topology_graph()
        
        # Initialize
        distances = {node: float('inf') for node in graph}
        distances[self.router_id] = 0
        predecessors = {}
        unvisited = set(graph.keys())
        
        while unvisited:
            # Find minimum distance node
            current = min(unvisited, key=lambda x: distances[x])
            unvisited.remove(current)
            
            if distances[current] == float('inf'):
                break
                
            # Update neighbors
            for neighbor, cost in graph[current].items():
                if neighbor in unvisited:
                    alt_distance = distances[current] + cost
                    
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        predecessors[neighbor] = current
                        
        # Build routing table
        self.build_routing_table(distances, predecessors)
```

## TCP vs UDP

### TCP (Transmission Control Protocol)
**Characteristics**:
- Connection-oriented
- Reliable delivery
- Ordered packets
- Flow control
- Congestion control

**Three-way Handshake**:
1. SYN →
2. ← SYN-ACK
3. ACK →

**Use Cases**:
- Web browsing (HTTP)
- Email (SMTP)
- File transfer (FTP)
- SSH

### UDP (User Datagram Protocol)
**Characteristics**:
- Connectionless
- No delivery guarantee
- No ordering
- Lower overhead
- Faster

**Use Cases**:
- DNS queries
- VoIP
- Video streaming
- Gaming
- DHCP

## Common Protocols

### HTTP/HTTPS
Web communication protocol.

**HTTP Methods**:
- GET: Retrieve resource
- POST: Submit data
- PUT: Update resource
- DELETE: Remove resource
- HEAD: Headers only
- OPTIONS: Available methods

**Status Codes**:
- 1xx: Informational
- 2xx: Success (200 OK)
- 3xx: Redirection (301 Moved)
- 4xx: Client error (404 Not Found)
- 5xx: Server error (500 Internal Error)

### DNS (Domain Name System)
Translates domain names to IP addresses.

**Record Types**:
- A: IPv4 address
- AAAA: IPv6 address
- CNAME: Canonical name (alias)
- MX: Mail exchange
- TXT: Text information
- NS: Name server
- SOA: Start of authority

**DNS Query Process**:
1. Check local cache
2. Query recursive resolver
3. Query root server
4. Query TLD server
5. Query authoritative server

### DHCP (Dynamic Host Configuration Protocol)
Automatically assigns IP addresses.

**DORA Process**:
1. **Discover**: Client broadcasts request
2. **Offer**: Server offers configuration
3. **Request**: Client requests offered config
4. **Acknowledge**: Server confirms

### SSH (Secure Shell)
Encrypted remote access protocol.

**Key-based Authentication**:
```bash
# Generate key pair
ssh-keygen -t rsa -b 4096

# Copy public key to server
ssh-copy-id user@server

# Connect using key
ssh -i ~/.ssh/id_rsa user@server
```

## Routing

### Static Routing
Manually configured routes.

```bash
# Add route
ip route add 10.0.0.0/8 via 192.168.1.1

# Delete route
ip route del 10.0.0.0/8

# Show routing table
ip route show
```

### Dynamic Routing Protocols

**Interior Gateway Protocols (IGP)**:
- **RIP**: Distance vector, hop count metric
- **OSPF**: Link state, cost metric
- **EIGRP**: Hybrid, Cisco proprietary

**Exterior Gateway Protocol (EGP)**:
- **BGP**: Path vector, used between ISPs

### NAT (Network Address Translation)
Translates private IPs to public IPs.

**Types**:
- Static NAT: One-to-one mapping
- Dynamic NAT: Pool of public IPs
- PAT/Overload: Many-to-one using ports

## VLANs

Virtual LANs segment networks logically.

**Benefits**:
- Security isolation
- Broadcast domain reduction
- Flexible network design
- QoS implementation

**Configuration Example**:
```
# Create VLAN
vlan 10
 name Sales

# Assign port to VLAN
interface GigabitEthernet0/1
 switchport mode access
 switchport access vlan 10

# Configure trunk
interface GigabitEthernet0/24
 switchport mode trunk
 switchport trunk allowed vlan 10,20,30
```

## Network Security

### Firewalls
Filter traffic based on rules.

**Types**:
- Packet filtering
- Stateful inspection
- Application layer
- Next-generation (NGFW)

### VPN (Virtual Private Network)
Creates encrypted tunnels.

**Types**:
- Site-to-site
- Remote access
- SSL/TLS VPN
- IPSec

### Access Control Lists (ACLs)
Define permitted/denied traffic.

```
# Cisco ACL example
access-list 100 permit tcp any host 192.168.1.10 eq 80
access-list 100 deny ip any any
```

## Quality of Service (QoS)

Prioritizes network traffic.

**Methods**:
- Classification and marking
- Queuing
- Policing and shaping
- Congestion avoidance

**Common QoS Models**:
- Best Effort
- IntServ (Integrated Services)
- DiffServ (Differentiated Services)

## Network Troubleshooting

### Common Tools

**ping**: Test connectivity
```bash
ping -c 4 google.com
```

**traceroute**: Show path to destination
```bash
traceroute google.com
```

**netstat**: Display connections
```bash
netstat -tulpn
```

**tcpdump**: Capture packets
```bash
tcpdump -i eth0 -w capture.pcap
```

**nmap**: Network discovery
```bash
nmap -sS -p 1-1000 192.168.1.0/24
```

**dig**: DNS lookup
```bash
dig @8.8.8.8 example.com
```

### Common Issues

**No Connectivity**:
1. Check physical connection
2. Verify IP configuration
3. Test default gateway
4. Check DNS resolution
5. Verify firewall rules

**Slow Performance**:
1. Check bandwidth utilization
2. Look for packet loss
3. Measure latency
4. Check for duplex mismatch
5. Verify MTU settings

## Advanced Topics

### SDN (Software-Defined Networking)
Separates control plane from data plane.

**Components**:
- Controller: Centralized management
- Southbound API: Controller to switches (OpenFlow)
- Northbound API: Applications to controller

### Network Function Virtualization (NFV)
Virtualizes network services.

**Examples**:
- Virtual routers
- Virtual firewalls
- Virtual load balancers

### MPLS (Multiprotocol Label Switching)
Forwards packets based on labels.

**Benefits**:
- Traffic engineering
- QoS support
- VPN services
- Reduced routing lookups

### IPv6 Transition
**Mechanisms**:
- Dual stack
- Tunneling (6to4, Teredo)
- Translation (NAT64)

## Cloud Networking

### Virtual Private Cloud (VPC)
Isolated network in cloud.

**Components**:
- Subnets
- Route tables
- Internet gateway
- NAT gateway
- Security groups

### Load Balancing
Distributes traffic across servers.

**Types**:
- Layer 4 (Transport)
- Layer 7 (Application)
- Global vs Regional

**Algorithms**:
- Round robin
- Least connections
- IP hash
- Weighted

### Content Delivery Networks (CDN)
Caches content at edge locations.

**Benefits**:
- Reduced latency
- Decreased bandwidth costs
- Improved availability
- DDoS protection

## Advanced SDN and Network Programmability

### OpenFlow Controller Implementation
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

### P4 Programmable Data Plane
```python
class P4DataPlane:
    """P4 programmable switch implementation"""
    
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

### Network Function Virtualization
```python
class VirtualNetworkFunction:
    """Base class for VNFs"""
    
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

## Network Performance Analysis

### Advanced Traffic Modeling
```python
class TrafficGenerator:
    """Generate realistic network traffic patterns"""
    
    def __init__(self):
        self.models = {
            'poisson': self.poisson_traffic,
            'self_similar': self.self_similar_traffic,
            'mmpp': self.mmpp_traffic,
            'heavy_tail': self.heavy_tail_traffic
        }
        
    def poisson_traffic(self, rate, duration):
        """Generate Poisson traffic"""
        timestamps = []
        current_time = 0
        
        while current_time < duration:
            interval = np.random.exponential(1/rate)
            current_time += interval
            if current_time < duration:
                timestamps.append(current_time)
                
        return timestamps
    
    def self_similar_traffic(self, H=0.8, duration=1000, method='FGN'):
        """Generate self-similar traffic using Fractional Gaussian Noise"""
        if method == 'FGN':
            # Generate FGN with Hurst parameter H
            n = int(duration)
            
            # Covariance matrix
            cov_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    cov_matrix[i, j] = 0.5 * (abs(i-j+1)**(2*H) + 
                                              abs(i-j-1)**(2*H) - 
                                              2*abs(i-j)**(2*H))
                    
            # Generate using Cholesky decomposition
            L = np.linalg.cholesky(cov_matrix)
            z = np.random.normal(0, 1, n)
            fgn = L @ z
            
            # Convert to arrival process
            arrivals = np.cumsum(np.exp(fgn))
            return arrivals
```

## Network Monitoring

### SNMP (Simple Network Management Protocol)
Monitor and manage network devices.

**Components**:
- Manager: Monitoring system
- Agent: Device software
- MIB: Management Information Base

### NetFlow/sFlow
Collect traffic flow data.

**Use Cases**:
- Traffic analysis
- Security monitoring
- Capacity planning
- Billing

### Network Performance Metrics
- **Bandwidth**: Maximum data rate
- **Throughput**: Actual data rate
- **Latency**: Delay in transmission
- **Jitter**: Variation in latency
- **Packet loss**: Dropped packets

## Best Practices

### Network Design
- Follow hierarchical model (core, distribution, access)
- Implement redundancy
- Use standard protocols
- Document everything
- Plan for growth

### Security
- Implement defense in depth
- Use strong encryption
- Regular security audits
- Keep firmware updated
- Monitor for anomalies

### Performance
- Optimize routing paths
- Implement QoS appropriately
- Monitor utilization
- Use caching where possible
- Regular capacity planning

## Research Frontiers in Networking

### Information-Centric Networking (ICN)
```python
class NamedDataNetworking:
    """NDN/CCN implementation"""
    
    def __init__(self):
        self.content_store = {}  # Cache
        self.pit = {}  # Pending Interest Table
        self.fib = {}  # Forwarding Information Base
        
    def handle_interest(self, name, incoming_face):
        """Process Interest packet"""
        # Check content store
        if name in self.content_store:
            # Cache hit - return data
            return self.content_store[name]
            
        # Check PIT
        if name in self.pit:
            # Add incoming face to existing entry
            self.pit[name]['faces'].add(incoming_face)
            return None
        else:
            # Create new PIT entry
            self.pit[name] = {
                'faces': {incoming_face},
                'timestamp': time.time()
            }
            
        # Forward based on FIB
        next_hops = self.fib_lookup(name)
        for next_hop in next_hops:
            if next_hop != incoming_face:
                self.forward_interest(name, next_hop)
                
        return None
```

### Network Coding
```python
class NetworkCoding:
    """Random Linear Network Coding"""
    
    def __init__(self, field_size=256):
        self.field_size = field_size
        
    def encode_generation(self, packets, num_coded):
        """Create coded packets from generation"""
        n = len(packets)
        coded_packets = []
        
        for _ in range(num_coded):
            # Random coefficients
            coeffs = np.random.randint(0, self.field_size, n)
            
            # Linear combination in finite field
            coded_data = np.zeros_like(packets[0])
            for i, packet in enumerate(packets):
                coded_data = (coded_data + coeffs[i] * packet) % self.field_size
                
            coded_packets.append({
                'coefficients': coeffs,
                'data': coded_data
            })
            
        return coded_packets
    
    def decode_generation(self, coded_packets):
        """Decode using Gaussian elimination"""
        # Build coefficient matrix
        n = len(coded_packets)
        A = np.array([pkt['coefficients'] for pkt in coded_packets])
        B = np.array([pkt['data'] for pkt in coded_packets])
        
        # Solve in finite field
        decoded = self.gaussian_elimination_gf(A, B)
        
        return decoded
```

### Quantum Networking
```python
class QuantumNetwork:
    """Quantum network protocols"""
    
    def quantum_teleportation(self, alice_qubit):
        """Teleport quantum state"""
        # Create entangled pair
        bell_pair = self.create_bell_pair()
        
        # Alice performs Bell measurement
        measurement = self.bell_measurement(alice_qubit, bell_pair[0])
        
        # Send classical bits to Bob
        classical_bits = measurement
        
        # Bob applies corrections
        bob_qubit = self.apply_corrections(bell_pair[1], classical_bits)
        
        return bob_qubit
    
    def quantum_key_distribution(self, num_bits):
        """BB84 QKD protocol"""
        # Alice prepares random bits in random bases
        alice_bits = np.random.randint(0, 2, num_bits)
        alice_bases = np.random.randint(0, 2, num_bits)
        
        # Bob measures in random bases
        bob_bases = np.random.randint(0, 2, num_bits)
        
        # Sift key
        matching_bases = alice_bases == bob_bases
        sifted_key = alice_bits[matching_bases]
        
        return sifted_key
```

## References and Further Reading

### Graduate-Level Textbooks
1. **Peterson & Davie** - "Computer Networks: A Systems Approach" (6th Edition)
2. **Kurose & Ross** - "Computer Networking: A Top-Down Approach" (8th Edition)
3. **Bertsekas & Gallager** - "Data Networks" (2nd Edition)
4. **Kleinrock** - "Queueing Systems" (Volumes 1 & 2)

### Research Papers
1. **Congestion Control**
   - Jacobson (1988) - "Congestion Avoidance and Control"
   - Cardwell et al. (2016) - "BBR: Congestion-Based Congestion Control"

2. **Software-Defined Networking**
   - McKeown et al. (2008) - "OpenFlow: Enabling Innovation in Campus Networks"
   - Kreutz et al. (2015) - "Software-Defined Networking: A Comprehensive Survey"

3. **Network Coding**
   - Ahlswede et al. (2000) - "Network Information Flow"
   - Ho et al. (2006) - "A Random Linear Network Coding Approach"

4. **Future Internet Architectures**
   - Jacobson et al. (2009) - "Networking Named Content"
   - Clark et al. (2018) - "The Design Philosophy of the DARPA Internet Protocols"

### Advanced Research Topics
1. **Programmable Networks** - P4, eBPF, XDP
2. **Time-Sensitive Networking** - Deterministic latency
3. **Network Verification** - Formal methods for networks
4. **Machine Learning for Networks** - AI-driven optimization
5. **Edge Computing** - MEC, fog computing architectures

## See Also
- [Cybersecurity](cybersecurity.html) - Network security details
- [AWS](aws.html) - Cloud networking
- [Docker](docker.html) - Container networking
- [Kubernetes](kubernetes.html) - Cluster networking
- [Quantum Computing](quantumcomputing.html) - Quantum communication