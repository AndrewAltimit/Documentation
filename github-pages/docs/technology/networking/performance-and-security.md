---
layout: docs
title: "Networking: Performance, QoS & Security"
permalink: /docs/technology/networking/performance-and-security.html
toc: true
toc_sticky: true
hide_title: true
---

[Networking](./)

<!-- Custom styles are now loaded via main.scss -->

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Performance, QoS &amp; Security</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Why networks are fast or slow, how to defend them, and how to keep them healthy</p>
</div>

What makes a network feel fast or slow comes down to queues filling at bottleneck links. This page builds the queueing models that explain latency, then turns to defending the network (firewalls, VPNs, ACLs), prioritizing traffic with quality of service, and the tools and metrics for troubleshooting and monitoring.

## Understanding Network Performance

When network engineers talk about performance, they're often dealing with queues—just like lines at a coffee shop.

### Why Networks Need Queues

Imagine a router as a busy intersection. Packets arrive from multiple sources, but the router can only forward them one at a time. When packets arrive faster than they can be processed, they must wait in a queue. This waiting time directly impacts your experience—it's why video calls sometimes freeze or web pages load slowly.

This real-world problem motivates queueing theory. By modeling network devices as queuing systems, we can predict and optimize performance.

### Modeling Network Queues

The Python models below are *illustrative* — small M/M/1 and queueing-network models that make the formulas concrete.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class NetworkQueue:
    """Model a network device (like a router) as a queue to predict performance.
    
    This M/M/1 model helps us understand:
    - How long packets wait in router buffers
    - When a network link becomes congested
    - How to size buffers appropriately
    """
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

# Example: Analyzing a home router handling video streaming
# Your router receives 800 packets/second during Netflix streaming
# It can forward 1000 packets/second to your device
router = NetworkQueue(arrival_rate=800, service_rate=1000)
print(f"Average packets waiting: {router.average_queue_length():.2f}")
print(f"Average delay added: {router.average_queue_time()*1000:.2f} ms")

# This 4ms delay might not seem like much, but it adds up across multiple hops!
```

### When Single Queues Aren't Enough

Real networks have multiple devices, each adding its own delays. To understand end-to-end performance, we need to model networks of queues. This is particularly important for cloud applications where data might traverse dozens of devices.
```python
class MultiHopNetwork:
    """Model traffic flow through multiple network devices.
    
    Use this to analyze:
    - Cloud application latency (web server → load balancer → app server → database)
    - Content delivery networks
    - Multi-datacenter architectures
    """
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
                queue = NetworkQueue(arrival_rates[i], self.service_rates[i])
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

> Congestion control (TCP Reno and BBR, in [Transport & Application Protocols](transport-and-protocols.html)) exists precisely to keep these queues from filling up and overflowing.

## Securing Networks: Defense in Depth

Every network connection is a potential security risk. Let's explore how networks are protected at multiple layers.

### Firewalls: The Network's Bouncer
Firewalls examine traffic and block anything suspicious, like a bouncer checking IDs at a club.

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

> For threat models, zero-trust, and attack techniques, see [Cybersecurity](../cybersecurity/).

## Quality of Service: Managing Network Traffic

Not all traffic is equal. Would you rather have your video call drop or your background download slow down? QoS lets networks make these decisions intelligently.

### How QoS Works
- Classification and marking
- Queuing
- Policing and shaping
- Congestion avoidance

**Common QoS Models**:
- Best Effort
- IntServ (Integrated Services)
- DiffServ (Differentiated Services)

## Troubleshooting Networks: Tools and Techniques

When networks fail, you need to diagnose problems quickly. Here are the essential tools every network engineer uses.

### Essential Diagnostic Tools

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

### Systematic Troubleshooting

**When things don't work, follow the OSI model from bottom to top:**

**Layer 1 - Physical**: Is it plugged in?
- Check cable connections
- Look for damaged cables
- Verify link lights on switches

**Layer 2 - Data Link**: Can you reach local devices?
- Ping your default gateway
- Check ARP cache (`arp -a`)
- Verify VLAN configuration

**Layer 3 - Network**: Can you reach remote networks?
- Ping external IPs (8.8.8.8)
- Traceroute to see where packets stop
- Check routing table (`ip route`)

**Layer 4+ - Transport/Application**: Are services working?
- Test specific ports with telnet/nc
- Check firewall rules
- Verify DNS resolution
- Look at application logs

**Slow Performance**:
1. Check bandwidth utilization
2. Look for packet loss
3. Measure latency
4. Check for duplex mismatch
5. Verify MTU settings

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

---

## Continue

<div class="see-also-card">
  <h4>Previous / Next</h4>
  <ul>
    <li><strong>Previous:</strong> <a href="routing.html">Routing &amp; Switching</a> — the paths whose performance we measure here.</li>
    <li><strong>Next:</strong> <a href="modern-architecture.html">Modern &amp; Future Networking</a> — programmable, cloud, and research-frontier networks.</li>
  </ul>
</div>

<div class="see-also-card">
  <h4>See Also</h4>
  <ul>
    <li><a href="transport-and-protocols.html">Transport &amp; Application Protocols</a> — congestion control, the response to queue buildup.</li>
    <li><a href="../cybersecurity/">Cybersecurity</a> — deeper coverage of network defense and zero trust.</li>
    <li><a href="../aws/">AWS</a> — security groups, NACLs, and cloud monitoring with CloudWatch.</li>
  </ul>
</div>
