---
layout: docs
title: "Networking: Modern Architecture & Frontiers"
permalink: /docs/technology/networking/modern-architecture.html
toc: true
toc_sticky: true
---

[Networking](./) &raquo; Modern Architecture &amp; Frontiers

<!-- Custom styles are now loaded via main.scss -->

The fixed, hardware-defined networks of the past have given way to software, and the design of those software networks rests on two things this page covers: an honest model of how traffic actually behaves, and an eye on the research that will reshape the internet next. We start by replacing textbook traffic assumptions with realistic ones (Poisson, self-similar, heavy-tailed, and modulated processes), then survey the research frontier — content-centric networking, network coding, quantum networking and QKD, and 6G. The three pages that grew out of this section — programmable, cloud, and wireless networking — are linked from the hub below.

The Python listings throughout this page are *illustrative* — compact models that show the logic of each idea rather than production code.

## The Modern-Networking Map

This page sits at the apex of a small family of pages, each of which started life as a section here and has since grown into its own full treatment. Before diving into traffic and research, here is where each piece now lives:

- **[Programmable Networks](programmable-networks.html)** — SDN and the control/data-plane split, NFV and service chaining, P4 and the programmable data plane, plus MPLS, Segment Routing, SRv6, and the IPv6 transition.
- **[Cloud Networking](cloud-networking.html)** — VPCs, subnets and route tables, Layer-4/7 load balancing, CDNs and anycast, NAT and egress, private service networking, and the shared-responsibility model.
- **[Wireless & Mobile](wireless-and-mobile.html)** — Wi-Fi (802.11), cellular evolution to 5G, spectrum and modulation, the 5G core and network slicing, edge/IoT, and mobility management.

The common thread linking all four is **disaggregation and programmability** — every layer of the network, from the control plane down to the radio, is becoming software you can change. This page is the part of that story that is less about *building* the network and more about *understanding the load it carries* and *where the research is heading*.

## Understanding Real-World Network Behavior

Textbook models assume nice, predictable traffic patterns. Real networks are messier — video streams create bursts, IoT devices chirp periodically, and users create flash crowds. Modeling realistic traffic is what lets capacity planning hold up under real load rather than idealized averages. Get the model wrong and you size buffers, links, and servers for an average that the network almost never experiences; get it right and you can predict tail latency, loss, and the buffer occupancy that actually drives user-visible delay.

The history of traffic modeling is essentially a story of one assumption being overturned. For decades the telephone network's **Poisson** model was carried over to data networks. Then a landmark measurement study — Leland, Taqqu, Willinger & Wilson, "On the Self-Similar Nature of Ethernet Traffic" (1994) — showed that real LAN traffic is **self-similar**: bursty across *every* timescale, not just the short one Poisson predicts. That single result reshaped how we provision and buffer networks.

### Poisson Arrivals: The Classical Baseline

The Poisson process is the natural starting point: arrivals are independent, and the number in any interval depends only on the interval's length. Inter-arrival times are then exponentially distributed with rate $\lambda$, so $P(T > t) = e^{-\lambda t}$ and the mean gap is $1/\lambda$. It is the model behind the classic Erlang and M/M/1 queueing results, and it is *memoryless* — knowing how long you have waited tells you nothing about how much longer you will wait.

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
```

Poisson's great virtue is **aggregation smoothing**: superpose many independent Poisson streams and you get a smoother Poisson stream, with relative variance shrinking as you add sources. That is exactly why it worked for telephone trunks — thousands of independent callers — and exactly why it *fails* for packet data, where a single video flow's bursts do not average away no matter how many flows you aggregate.

### Self-Similar Traffic: Burstiness at Every Scale

Self-similar traffic looks statistically the same whether you plot it per millisecond, per second, or per minute — zoom out and the burstiness does not smooth away the way Poisson predicts. This **long-range dependence** is captured by a single number, the **Hurst parameter** $H \in (0.5, 1)$:

- $H = 0.5$ corresponds to ordinary (short-range-dependent) traffic — Poisson-like, burstiness averages out.
- $0.5 < H < 1$ means long-range dependence: the closer $H$ is to 1, the burstier the traffic stays at large timescales.

The autocorrelation of a self-similar process decays only **polynomially** (like $k^{2H-2}$) rather than exponentially, so correlations persist across long lags — a busy period now makes a busy period much later more likely. The practical consequence is harsh: buffers fill far more often than Poisson predicts, queue lengths have heavy tails, and simply adding buffer space yields diminishing returns instead of the exponential payoff Poisson would promise.

The listing below synthesizes self-similar traffic from **Fractional Gaussian Noise (FGN)** with a chosen Hurst parameter, using the exact-covariance (Cholesky) method:

```python
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

The covariance entry above is exactly the increment covariance of **fractional Brownian motion**: the term

$$
\gamma(k) = \tfrac{1}{2}\left( |k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H} \right)
$$

reduces to $0$ for all $k \neq 0$ only when $H = 0.5$ (independent increments); for $H > 0.5$ it stays positive and decays slowly, which is the mathematical signature of long-range dependence. (The Cholesky method is exact but $O(n^2)$ in memory; production simulators use faster spectral or wavelet methods such as Davies–Harte for large $n$.)

### Heavy-Tailed Distributions: Where Self-Similarity Comes From

Self-similarity at the aggregate is not magic — it emerges from **heavy tails** at the level of individual sources. If flow or file sizes follow a heavy-tailed distribution, then superposing many ON/OFF sources with heavy-tailed ON periods provably produces self-similar aggregate traffic. The web is the canonical example: most files are tiny, but a few enormous ones (videos, downloads) dominate the bytes.

The **Pareto distribution** is the standard heavy-tail model. Its tail falls off polynomially,

$$
P(X > x) = \left( \frac{x_m}{x} \right)^{\alpha}, \qquad x \ge x_m,
$$

with shape parameter $\alpha$. The key feature is *infinite variance* when $\alpha < 2$ (and even infinite mean when $\alpha < 1$): a vanishing fraction of flows carry a dominant fraction of the traffic. This is why "average flow size" is a misleading planning number — the variance, not the mean, drives buffer and link behavior.

```python
    def heavy_tail_traffic(self, alpha=1.5, x_min=1.0, num_flows=10000):
        """Generate heavy-tailed (Pareto) flow sizes.

        alpha < 2  => infinite variance: a few flows dominate the bytes.
        alpha < 1  => infinite mean: 'average flow size' is meaningless.
        """
        # Inverse-transform sampling of a Pareto(alpha, x_min) tail
        u = np.random.uniform(0, 1, num_flows)
        flow_sizes = x_min / (u ** (1.0 / alpha))
        return flow_sizes
```

### MMPP: Modeling Correlated Bursts

Many real sources are neither purely random nor purely self-similar but **modulated** — they switch between regimes (a video call alternating talk-spurts and silence, a sensor toggling active and idle). The **Markov-Modulated Poisson Process (MMPP)** captures this: a hidden Markov chain moves between states, and the *Poisson arrival rate depends on the current state*. In a busy state arrivals come fast; in an idle state they trickle.

```python
    def mmpp_traffic(self, rates, transition_matrix, duration):
        """Generate Markov-Modulated Poisson Process traffic.

        rates[s]               : Poisson arrival rate while in state s
        transition_matrix[s]   : state-transition probabilities out of s
        The arrival rate is piecewise-constant, switching when the
        underlying Markov chain changes state.
        """
        timestamps = []
        state = 0
        t = 0.0

        while t < duration:
            # Arrivals are Poisson at the current state's rate
            interval = np.random.exponential(1.0 / rates[state])
            t += interval
            if t < duration:
                timestamps.append((t, state))
            # Possibly hop to a new modulating state
            state = np.random.choice(len(rates), p=transition_matrix[state])

        return timestamps
```

MMPP is popular precisely because it is **analytically tractable** — it plugs into matrix-analytic queueing methods (MAP/M/1 queues) the way plain Poisson plugs into M/M/1 — yet it reproduces the short-range burst correlation that Poisson misses. It is a pragmatic middle ground: not as faithful as a full self-similar model over many decades of timescale, but far more workable for closed-form analysis, which is why it remains a staple in voice, video, and IoT-traffic modeling.

### Which Model to Use

| Model | Captures | Good for | Misses |
|-------|----------|----------|--------|
| **Poisson** | Independent, memoryless arrivals | Highly aggregated, independent sources (call arrivals); analytic baselines | Burstiness, correlation — *too optimistic* for packet data |
| **MMPP** | Correlated bursts via hidden states | Voice/video on-off behavior, tractable queueing analysis | Long-range dependence across many timescales |
| **Self-similar (FGN)** | Long-range dependence, scale-invariant bursts | LAN/WAN aggregate traffic, buffer/tail-latency studies | Closed-form queueing results (mostly simulation-only) |
| **Heavy-tailed (Pareto)** | Dominant-flow / mouse-elephant structure | Flow- and file-size modeling, the *source* of self-similarity | The time-correlation structure (it models sizes, not arrivals) |

The headline lesson for capacity planning: **provision for the tail, not the mean.** Self-similar, heavy-tailed traffic means the network spends a meaningful fraction of time far above its average load, so buffers sized to a Poisson model will overflow and latency-sensitive traffic will suffer. The queueing foundations these models feed into — Little's law, M/M/1, and buffer sizing — live in [Performance, QoS & Security](performance-and-security.html).

## The Cutting Edge: Research Changing Networking

The internet was designed for connecting computers. Today's challenges — content delivery, IoT, quantum computing, AI inference at the edge — require fundamentally new approaches. Recent developments include deterministic networking for industrial IoT, network digital twins, and AI-native protocols. The sections below cover four of the most consequential research directions, then the trends already crossing from lab to production.

### Information-Centric Networking: Content Over Connections

Today's internet cares about WHERE (which server), but users care about WHAT (which video). **Information-Centric Networking (ICN)** reimagines networking around content, not locations. Its most developed form is **Named Data Networking (NDN)** (and its predecessor CCN), which replaces the source/destination-address model with *named, signed content* that any node may cache and serve.

In NDN there are just two packet types and three tables. A consumer sends an **Interest** for a name (e.g. `/andrew/video/lecture3/seg7`); the network forwards it toward a producer, and a matching **Data** packet flows back along the reverse path, getting cached at every hop. The three per-router tables are:

- **Content Store (CS)** — an in-network cache of Data packets seen recently.
- **Pending Interest Table (PIT)** — the breadcrumb trail of Interests still awaiting Data, recording which interface(s) each came from so the Data knows where to return.
- **Forwarding Information Base (FIB)** — name-prefix-based forwarding state, the NDN analogue of an IP routing table.

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

The payoffs are structural. **Caching is native** (any router can answer a repeated Interest), so popular content is served close to consumers without a bolt-on CDN. **Multicast is automatic** — the PIT naturally aggregates multiple Interests for the same name into one upstream request and fans the Data back to all of them. And because each Data packet is **signed at the object level**, trust attaches to the content itself rather than to the channel, so a cached copy is as trustworthy as the original. The open problems are equally structural: routing on an unbounded, application-defined namespace, PIT scalability at line rate, and cache-management and privacy concerns. NDN remains a leading research platform (and the intellectual root of why CDNs and edge caches dominate today's web).

### Network Coding: Breaking the Store-and-Forward Paradigm

Traditional routers just forward packets. What if they could combine packets mathematically, increasing throughput and reliability? **Network coding** — introduced by Ahlswede, Cai, Li & Yeung, "Network Information Flow" (2000) — proved that allowing intermediate nodes to *mix* incoming packets (rather than only store-and-forward) can achieve the multicast capacity of a network, something pure routing provably cannot.

The canonical illustration is the **butterfly network**, where a single bottleneck link must carry two flows to two receivers. Routing forces the link to serialize the two packets; with coding, the bottleneck node sends one combined packet $a \oplus b$, and each receiver — already holding one of the originals from another path — recovers the other by XOR. Two packets' worth of information crosses a one-packet link.

```python
class NetworkCoding:
    """Mix packets mathematically instead of just forwarding them.

    Benefits:
    - Increased throughput in multicast
    - Better reliability in wireless networks
    - Reduced retransmissions

    Think of it like this: Instead of carrying individual letters,
    the postal service could carry mathematical combinations that
    let recipients reconstruct any lost letters.
    """

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

The practical workhorse is **Random Linear Network Coding (RLNC)**, shown above: a node combines a *generation* of $n$ packets with random coefficients drawn from a finite field $\mathrm{GF}(q)$, and a receiver decodes by Gaussian elimination as soon as it collects *any* $n$ linearly independent coded packets — it does not matter *which* ones. This is what makes coding shine on lossy and wireless links: there is no need to retransmit a *specific* lost packet, since any fresh coded packet is equally useful, which slashes feedback and retransmission overhead. The cost is the encode/decode computation and a small coefficient-vector header overhead, plus the decoding-delay penalty of waiting for a full generation. Coding ideas now appear in distributed storage (regenerating codes), multipath transport, and wireless mesh systems.

### Quantum Networking: Unhackable Communications

Quantum mechanics enables fundamentally secure communication. By encoding information in quantum states, we can detect any eavesdropping attempt — security that rests on *physics* (the no-cloning theorem and measurement disturbance) rather than on the computational hardness that classical cryptography assumes.

```python
class QuantumNetwork:
    """Implement quantum communication protocols.

    Revolutionary properties:
    - Unconditional security (physics, not math)
    - Detection of eavesdropping
    - Quantum teleportation of states

    Current challenges:
    - Limited distance (~100km)
    - Requires special hardware
    - Very low data rates
    """

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

**Quantum Key Distribution (QKD)** is the one piece already deployed. In **BB84** (Bennett & Brassard, 1984), Alice encodes random bits in randomly chosen bases (rectilinear or diagonal); Bob measures in his own random bases; they publicly compare *which bases* they used (never the bit values) and keep only the matching positions to form a shared **sifted key**. Because any eavesdropper must measure — and measuring in the wrong basis disturbs the state — listening in *introduces detectable errors*. Alice and Bob sacrifice a subset of the key to estimate the **Quantum Bit Error Rate (QBER)**; if it exceeds a threshold, they abort, otherwise they apply classical error correction and privacy amplification to distill a provably secret key.

**Deployment reality (2024–2025):** QKD has moved well past the lab but remains niche. Metropolitan fiber QKD networks operate in several cities, China's Beijing–Shanghai backbone spans ~2,000 km using **trusted relay** nodes, and the **Micius** satellite demonstrated entanglement-based QKD over more than 1,000 km — sidestepping fiber loss by going through space. The hard limits keep it specialized: photon loss caps terrestrial fiber at roughly 100–200 km without repeaters, key rates are low, and trusted-relay nodes are a security weak point. The missing piece is the **quantum repeater** (entanglement swapping plus quantum memory), which would enable end-to-end entanglement over arbitrary distance and a true **quantum internet** carrying *entanglement* rather than just keys. Notably, many security agencies now favor **post-quantum cryptography** (classical algorithms resistant to quantum attack) over QKD for most use cases, because it needs no special hardware — making QKD's near-term role a complement, not a replacement.

> The physics behind these protocols — superposition, measurement, the no-cloning theorem, and entanglement — is developed in [Quantum Mechanics](../../physics/quantum-mechanics/) and [Quantum Computing](../quantumcomputing.html).

## Current Networking Trends

Several of the ideas above are already moving from research into production:

### QUIC and HTTP/3 Adoption
- Major websites now use HTTP/3 by default
- QUIC provides faster connections and better mobile performance
- Built-in encryption and multiplexing

> The transport mechanics of QUIC and HTTP/3 are covered in [Transport & Application Protocols](transport-and-protocols.html).

### eBPF Revolution
- Programmable kernel networking without modules
- Used in load balancers, firewalls, observability tools
- Projects: Cilium, Katran, Pixie

### AI/ML in Networking
- Predictive network maintenance
- Automated troubleshooting
- Traffic pattern analysis (directly atop the traffic models above)
- DDoS detection and mitigation

### 6G Research (2024–2030)

6G targets the early 2030s and is being shaped now. The headline research themes:

- **Terahertz communications** — 100+ Gbps wireless links using the sub-THz / THz bands above mmWave, trading enormous bandwidth for extreme path loss and line-of-sight-only reach.
- **AI-native air interface** — learned waveforms, coding, and protocols, where the radio's physical layer is partly trained rather than hand-designed.
- **Integrated Sensing and Communication (ISAC)** — networks that *sense* their environment (radar-like imaging, positioning) while they communicate, reusing the same spectrum and hardware.
- **Non-terrestrial networks (NTN)** — tight integration of LEO satellite constellations, high-altitude platforms, and ground networks for truly ubiquitous coverage.
- **Sustainability and energy efficiency** as a first-class design goal rather than an afterthought.

> The 5G foundations 6G builds on — massive MIMO, the cloud-native 5G core, network slicing, and mmWave — are covered in [Wireless & Mobile](wireless-and-mobile.html).

### What Else Is Active

Beyond the four deep dives above, several adjacent research areas are worth tracking:

**Ultra-low latency:**
- **Deterministic Networking (DetNet)** — guaranteed bounded latency for industrial IoT.
- **Time-Sensitive Networking (TSN)** — IEEE 802.1 standards for real-time Ethernet.

**Verification and security:**
- **Network verification** — automated correctness proofs using formal methods.
- **Post-quantum network security** — classical algorithms hardened against quantum attack, now the mainstream answer to the quantum threat.

**New computing paradigms:**
- **In-network computing** — P4-programmable switches and computational storage (see [Programmable Networks](programmable-networks.html)).
- **Network digital twins** — real-time simulation and prediction of a live network, fed by exactly the traffic models above.

## Continuing Your Networking Journey

Networking is a vast field that continues to evolve rapidly. Here are resources to deepen your understanding.

### Foundational Textbooks
1. **Peterson & Davie** — "Computer Networks: A Systems Approach" (6th Edition, 2022)
2. **Kurose & Ross** — "Computer Networking: A Top-Down Approach" (8th Edition, 2021)
3. **Bertsekas & Gallager** — "Data Networks" (2nd Edition)
4. **Kleinrock** — "Queueing Systems" (Volumes 1 & 2)
5. **Tanenbaum & Wetherall** — "Computer Networks" (6th Edition, 2021)

### Landmark Papers That Shaped Networking

**The Problems That Started It All:**
- **Jacobson (1988)** — "Congestion Avoidance and Control"
  *Why it matters:* Saved the internet from collapse in the 1980s.

- **Cardwell et al. (2016)** — "BBR: Congestion-Based Congestion Control"
  *Why it matters:* Made YouTube and Google services noticeably faster.

**Measuring What Traffic Really Looks Like:**
- **Leland, Taqqu, Willinger & Wilson (1994)** — "On the Self-Similar Nature of Ethernet Traffic"
  *Why it matters:* Overturned the Poisson assumption and reshaped how we buffer and provision networks.

**Revolutionizing How We Build Networks:**
- **McKeown et al. (2008)** — "OpenFlow: Enabling Innovation in Campus Networks"
  *Why it matters:* Launched the SDN revolution, making networks programmable (see [Programmable Networks](programmable-networks.html)).

**Breaking Theoretical Limits:**
- **Ahlswede et al. (2000)** — "Network Information Flow"
  *Why it matters:* Showed that mixing data beats store-and-forward.

**Reimagining the Internet:**
- **Jacobson et al. (2009)** — "Networking Named Content"
  *Why it matters:* Proposed focusing on what we want, not where it is.

### What's Next: Active Research Areas

**Making Networks Smarter:**
- **AI-Native Networks** — self-optimizing networks using transformer models and reinforcement learning.
- **Intent-Based Networking (IBN)** — declarative networking with natural-language interfaces.
- **Digital Twin Networks** — real-time network simulation and prediction.

**Ultra-Low Latency:**
- **Deterministic Networking (DetNet)** — guaranteed bounded latency for industrial IoT.
- **Edge Computing** — ETSI MEC standards, 5G edge integration.
- **Time-Sensitive Networking (TSN)** — IEEE 802.1 standards for real-time Ethernet.

**Verification and Security:**
- **Network Verification** — automated correctness proofs using formal methods.
- **Zero Trust Network Access (ZTNA)** — modern perimeter-less security.
- **SASE (Secure Access Service Edge)** — converged network and security services.
- **Post-Quantum Network Security** — preparing for quantum computing threats.

**New Computing Paradigms:**
- **In-Network Computing** — P4-programmable switches, computational storage.
- **Quantum Internet** — quantum-key-distribution networks operational in multiple countries.
- **Blockchain-Based Networking** — decentralized DNS, routing security.
- **Neuromorphic Networking** — brain-inspired packet processing.

---

## Continue

**Previous:** [Performance, QoS & Security](performance-and-security.html) — the queueing theory these traffic models feed. &nbsp;**Up:** [Networking](./) — overview and navigation hub.

### See Also

- [Programmable Networks](programmable-networks.html) — SDN, NFV, P4, MPLS, and SRv6.
- [Cloud Networking](cloud-networking.html) — VPCs, load balancing, CDNs, and the shared-responsibility model.
- [Wireless & Mobile](wireless-and-mobile.html) — Wi-Fi, 5G, the 5G core, and the wireless foundations 6G builds on.
- [Performance, QoS & Security](performance-and-security.html) — queueing models, Little's law, and buffer sizing.
- [AWS](../aws/) — VPC, load balancing, and CloudFront in a real cloud.
- [Quantum Mechanics](../../physics/quantum-mechanics/) — the physics behind QKD and quantum networking.
