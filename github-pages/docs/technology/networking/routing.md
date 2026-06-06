---
layout: docs
title: "Networking: Routing & Switching"
permalink: /docs/technology/networking/routing.html
toc: true
toc_sticky: true
hide_title: true
---

[Networking](./)

<!-- Custom styles are now loaded via main.scss -->

The internet is a massive graph of routers and links, and getting data across it efficiently is a path-finding problem. This page starts with the graph algorithms behind routing, scales up to the protocols that route traffic within and between organizations (OSPF and BGP), covers how routers actually make decisions (static, dynamic, and NAT), and ends with VLANs for segmenting networks on shared hardware.

## Finding the Best Path: Graph Theory in Networks

The internet is essentially a massive graph where routers are nodes and connections are edges. Finding efficient paths through this graph is crucial for performance.

### Why Path Selection Matters

When you connect to a website hosted on another continent, your data doesn't take a direct path. It hops through multiple networks, each making routing decisions. Poor routing can double or triple your latency, making applications feel sluggish.

The implementations in this section are *illustrative* — compact versions of the classic algorithms that real routers use, written to show the logic rather than to ship in a router. Let's implement the algorithms that routers use to find optimal paths:
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

### Beyond Shortest Paths: Network Capacity

Finding the shortest path is only part of the story. We also need to consider capacity—how much traffic can flow through the network. This is like planning routes for delivery trucks: the shortest path might be a narrow street that can't handle many vehicles.

#### Calculating Maximum Network Capacity
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

## How the Internet Routes Traffic: Advanced Protocols

We've seen how routers find paths within a network. But how does traffic flow between the 70,000+ independent networks that form the internet? This is where BGP comes in—the protocol that literally holds the internet together.

### BGP: The Internet's Routing Protocol
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

### OSPF: Smart Routing Within Organizations

While BGP connects different organizations, OSPF (Open Shortest Path First) optimizes routing within a single organization. It's like having a real-time traffic map for your corporate network.

```python
class OSPFRouter:
    """OSPF builds a complete map of the network for optimal routing.
    
    Unlike distance-vector protocols that only know their neighbors,
    OSPF routers share their complete view, enabling better decisions.
    """
    
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

## Making Routing Decisions: From Simple to Complex

Now that we understand addressing and protocols, let's see how routers actually decide where to send packets.

### Static Routing: Manual Control
Sometimes you know exactly where traffic should go. Static routes are like putting up permanent road signs.

```bash
# Add route
ip route add 10.0.0.0/8 via 192.168.1.1

# Delete route
ip route del 10.0.0.0/8

# Show routing table
ip route show
```

### Dynamic Routing: Networks That Adapt

Static routes work for small networks, but imagine manually updating routes for the entire internet! Dynamic protocols automatically discover paths and adapt to changes.

**Within Organizations (IGP):**
- **RIP**: Simple but limited (counts hops, max 15)
  - Good for: Small networks, lab environments
  - Problem: Treats all links equally (1Gbps same as 10Mbps)
  
- **OSPF**: Smarter routing based on link speed
  - Good for: Large corporate networks
  - Advantage: Considers bandwidth, builds complete network map
  
- **EIGRP**: Cisco's enhanced protocol
  - Good for: Cisco-only environments
  - Advantage: Fast convergence, multiple metrics

**Between Organizations (EGP):**
- **BGP**: The internet's routing protocol
  - Exchanges routes between ISPs, companies, countries
  - Makes policy decisions (prefer certain providers, avoid others)
  - Handles 900,000+ routes in the global routing table

### NAT (Network Address Translation)
Translates private IPs to public IPs.

**Types**:
- Static NAT: One-to-one mapping
- Dynamic NAT: Pool of public IPs
- PAT/Overload: Many-to-one using ports

## VLANs: Virtual Networks on Physical Hardware

Imagine you need separate networks for different departments, but running separate cables is expensive. VLANs create multiple logical networks on the same physical switches—like having multiple virtual highways on the same road.

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

---

## Continue

**Previous:** [Transport & Application Protocols](transport-and-protocols.html) — what rides on top of routed packets. &nbsp;**Next:** [Performance, QoS & Security](performance-and-security.html) — how fast and how safely it all moves.

### See Also

- [Layers & Addressing](fundamentals.html) — IP addressing and CIDR subnetting that routing operates on.
- [Modern & Future Networking](modern-architecture.html) — SDN and programmable forwarding that re-imagine routing.
- [Cybersecurity](../cybersecurity/) — BGP hijacking and routing security.
