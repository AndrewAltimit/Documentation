---
layout: docs
title: "Optimization: Network & I/O Optimization"
description: "Latency, bandwidth, and concurrency; TCP, QUIC, and HTTP/1.1-3; batching, compression, connection pooling; storage queue depth, the page cache, io_uring, and zero-copy."
permalink: /docs/optimization/network-io-optimization.html
toc: true
toc_sticky: true
hide_title: true
---

# Network & I/O Optimization

[Performance Optimization](./) &raquo; Network &amp; I/O Optimization

Many slow systems are not short of CPU; they spend their time **waiting**: for a packet to cross the network, for a storage device to complete a read, for the kernel to copy a buffer, or for a connection handshake. This page covers optimizing those wait states. It starts with the quantities that govern every I/O channel (latency, bandwidth, and concurrency), then covers protocol choice, batching and pipelining, compression, connection pooling, storage access patterns, the page cache, and zero-copy data paths.

The recurring rules:

| Rule | Why |
|------|-----|
| Remove round trips before tuning anything else | Latency has a physical floor; bandwidth can be bought |
| Keep many operations in flight | By Little's Law, concurrency is how throughput rises at fixed latency |
| Batch small operations | Per-operation overhead dominates small transfers |
| Avoid copying bytes you only forward | Each copy costs memory bandwidth, cache, and CPU |
| Treat storage like a network | Device latency, queue depth, and caching obey the same arithmetic |

## Latency, Bandwidth, and Concurrency

Every I/O channel, whether a TCP connection, an NVMe SSD, or a PCIe link, has two independent characteristics:

- **Latency**: the time for one operation to complete, such as a round trip (RTT) or the time to first byte of a read. It is set by distance, queuing, and per-operation overhead.
- **Bandwidth** (throughput): the rate at which data flows once it is flowing, in bytes per second.

A trans-Pacific 10 Gbit/s link has high bandwidth and a round-trip time of well over 100 ms; a datacenter link might have a 50-200 µs RTT. Light in fiber travels about 200 km per millisecond, so no protocol or hardware change reduces a New York-London round trip much below about 55 ms. Latency is attacked with **fewer round trips and more concurrency**; bandwidth with **larger transfers, compression, and parallel streams**.

### Reference Latencies

Approximate figures, useful for back-of-envelope estimates; measure your own environment.

| Operation | Approximate time |
|-----------|------------------|
| Main-memory access | ~100 ns |
| Syscall (simple, no I/O) | ~0.1-1 µs |
| NVMe SSD random 4 KB read | ~50-100 µs |
| Round trip within a datacenter | ~50-500 µs |
| HDD seek + rotational latency | ~5-10 ms |
| Round trip within a continent | ~20-60 ms |
| Round trip across an ocean | ~70-200 ms |
| Round trip over a mobile network | ~30-100+ ms, highly variable |

### Bandwidth-Delay Product

The amount of data that must be in flight to keep a link full is the **bandwidth-delay product** (BDP):

$$\text{BDP} = \text{bandwidth} \times \text{RTT}$$

For a 1 Gbit/s link with a 100 ms round trip:

$$\text{BDP} = 125 \times 10^{6} \ \text{B/s} \times 0.1 \ \text{s} = 12.5 \ \text{MB}$$

A TCP sender can have at most one window of unacknowledged data outstanding, so the achievable throughput is bounded by:

$$\text{throughput} \le \frac{\text{window}}{\text{RTT}}$$

A 64 KB window on that 100 ms path caps throughput at about 5 Mbit/s regardless of link capacity. Window scaling (RFC 7323) allows windows up to 1 GB, and Linux autotunes socket buffers, but the maximums in `net.ipv4.tcp_rmem` / `tcp_wmem` (and application-set `SO_RCVBUF`, which disables autotuning) must still be large enough for long, fat paths.

### Little's Law

Throughput, latency, and concurrency are related by Little's Law, which holds for any stable queueing system:

$$L = \lambda W \qquad \Longleftrightarrow \qquad \lambda = \frac{L}{W}$$

$L$ is the mean number of requests in flight, $\lambda$ the throughput, and $W$ the mean latency. A 10 ms request handled one at a time yields 100 requests/s; with 50 in flight it yields 5,000 requests/s at the same latency. Connection pools, pipelining, HTTP/2 multiplexing, asynchronous I/O, and deep storage queues are all ways of raising $L$.

### Tail Latency

Averages hide the requests users notice. In a fan-out system, one slow dependency makes the whole response slow: if a request touches 100 backends and each has a 1% chance of being slow, then

$$P(\text{at least one slow}) = 1 - 0.99^{100} \approx 0.63$$

so most requests experience some backend's p99. Defenses include bounded queues (shed load rather than queue unboundedly), deadlines propagated through the call graph, **hedged requests** (send a duplicate to another replica after the p95 latency elapses and take whichever answers first), and avoiding head-of-line blocking at every layer.

## Protocol Choice

The transport and application protocols determine how many round trips precede useful data, how loss affects unrelated requests, and the per-message overhead.

### TCP vs. UDP vs. QUIC

| Property | TCP | UDP | QUIC (RFC 9000) |
|----------|-----|-----|-----------------|
| Delivery | Reliable, ordered byte stream | Best-effort datagrams | Reliable, ordered *per stream*; optional unreliable datagrams (RFC 9221) |
| Setup | 1 RTT handshake (+ TLS) | None | 1 RTT including TLS 1.3; 0-RTT on resumption |
| Encryption | Separate (TLS) | None (DTLS optional) | Always (TLS 1.3 integrated) |
| Head-of-line blocking | Yes, across the whole connection | No | Only within the affected stream |
| Congestion control | Kernel (CUBIC default on Linux, BBR available) | Application's responsibility | Userspace library (CUBIC, BBR, others) |
| Connection identity | IP/port 4-tuple | None | Connection ID; survives address changes |
| Implementation | Kernel, with hardware offloads | Kernel | Usually a userspace library over UDP |

TCP's reliability means a single lost segment holds back every byte behind it until it is retransmitted. UDP avoids this but leaves reliability, ordering, and congestion control to the application. QUIC provides them in userspace on top of UDP, with independent streams. The trade-off is CPU cost: QUIC stacks cannot use the kernel's mature TCP offloads as fully, and historically cost more CPU per byte than TCP+TLS, although UDP GSO/GRO and improved stacks have narrowed the gap.

### Handshake Cost

Setup is paid in round trips before the first byte of application data:

| Stack | Round trips before request can be sent |
|-------|----------------------------------------|
| TCP + TLS 1.2 (full handshake) | 3 (1 TCP + 2 TLS) |
| TCP + TLS 1.3 | 2 (1 TCP + 1 TLS) |
| TCP + TLS 1.3 resumption with 0-RTT early data | 1 |
| QUIC (first connection) | 1 |
| QUIC resumption with 0-RTT | 0 |

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    Note over C,S: TCP + TLS 1.3 (2 RTT to first request)
    C->>S: SYN
    S->>C: SYN-ACK
    C->>S: ACK + ClientHello
    S->>C: ServerHello, certificate, Finished
    C->>S: Finished + HTTP request
    S->>C: HTTP response
    Note over C,S: QUIC (1 RTT to first request)
    C->>S: Initial (ClientHello)
    S->>C: Initial + Handshake (ServerHello, certificate, Finished)
    C->>S: Handshake Finished + HTTP/3 request
    S->>C: HTTP/3 response
```

On a 100 ms path, TLS 1.2 over TCP spends 300 ms before sending the request. Reusing connections amortizes this cost across many requests; that is why keep-alive and pooling usually matter more than any protocol tuning. Note that 0-RTT early data can be replayed by an attacker, so servers should accept it only for idempotent requests.

### HTTP/1.1, HTTP/2, and HTTP/3

| | HTTP/1.1 | HTTP/2 (RFC 9113) | HTTP/3 (RFC 9114) |
|---|----------|-------------------|-------------------|
| Transport | TCP | TCP | QUIC over UDP |
| Concurrency per connection | One outstanding request (pipelining unusable in practice) | Many multiplexed streams | Many multiplexed streams |
| Header compression | None | HPACK | QPACK |
| Loss behavior | Affects one request | One lost segment stalls **all** streams | Stalls only the affected stream |
| Typical client behavior | ~6 parallel connections per host | One connection per origin | One connection per origin; migrates across networks |

- **HTTP/1.1** serializes requests on a connection, so browsers open several connections per host, each paying its own handshake and slow-start ramp.
- **HTTP/2** multiplexes streams over one TCP connection, but TCP delivers bytes strictly in order, so a single loss stalls every stream. On lossy networks, one HTTP/2 connection can underperform six HTTP/1.1 connections. **Server push** has been removed from major browsers (Chrome disabled it in 2022); use `103 Early Hints` or preload links instead.
- **HTTP/3** moves the same semantics onto QUIC, eliminating transport-level head-of-line blocking and enabling connection migration (Wi-Fi to cellular without reconnecting). All major browsers and CDNs support it; servers advertise it through the `Alt-Svc` header or DNS HTTPS records.

### Choosing

| Workload | Recommended |
|----------|-------------|
| Web and API traffic to browsers and mobile clients | HTTP/2 everywhere, plus HTTP/3 for lossy and mobile networks |
| Internal service-to-service RPC | Pooled, long-lived HTTP/2 connections (e.g. gRPC) |
| Bulk transfer within a datacenter | TCP with tuned buffers; parallel streams for long, fat paths |
| Real-time media and game state | UDP with application-level reliability, WebRTC, or QUIC datagrams; conceal loss rather than wait for retransmission |
| Browser-to-server low-latency messaging | WebSocket today; WebTransport (over HTTP/3) where supported |
| Tiny request/response within a datacenter | Persistent connection; handshake cost dominates everything else |

### Transport Tuning Knobs

| Setting | Effect |
|---------|--------|
| `TCP_NODELAY` | Disables Nagle's algorithm, which delays small writes to coalesce them. Set it for request/response protocols; batch writes yourself instead. |
| Congestion control (`net.ipv4.tcp_congestion_control`) | CUBIC is the Linux default. BBR models bandwidth and RTT instead of reacting to loss and performs much better on lossy long-haul paths; widely deployed by large content providers. Pair it with the `fq` qdisc. |
| TCP Fast Open (RFC 7413) | Carries data in the SYN on repeat connections; limited by middlebox interference. TLS 1.3 0-RTT and QUIC achieve similar results more reliably. |
| Socket buffer limits (`tcp_rmem`, `tcp_wmem`) | Must accommodate the BDP of the longest paths served |
| Initial congestion window | Linux default is 10 segments (RFC 6928); determines how much a new connection can send in its first RTT |
| NIC offloads (TSO/GSO, GRO, checksum), RSS, IRQ affinity | Reduce per-packet CPU cost; spread load across cores |

## Batching and Pipelining

The cheapest round trip is one that is never made.

- **Batching** merges many logical operations into one physical request: one multi-row `INSERT` instead of 1,000 single-row inserts. At 1 ms RTT, 1,000 sequential round trips take at least a second; one batch takes about a millisecond plus transfer time.
- **Pipelining** sends further requests without waiting for earlier responses, so the RTT is paid once per window rather than once per request.

```python
import redis

r = redis.Redis()

# Anti-pattern: one round trip per command (10,000 x RTT).
for i in range(10_000):
    r.set(f"key:{i}", i)

# Pipelined: commands are buffered and sent together (a few RTTs total).
with r.pipeline(transaction=False) as pipe:
    for i in range(10_000):
        pipe.set(f"key:{i}", i)
    pipe.execute()
```

The same pattern appears as `executemany` and `COPY` in databases, bulk APIs in object stores, request coalescing in GraphQL (DataLoader), and batch publish in message queues.

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    Note over C,S: Sequential: N round trips
    C->>S: GET a
    S-->>C: a
    C->>S: GET b
    S-->>C: b
    Note over C,S: Pipelined: about one round trip
    C->>S: GET a
    C->>S: GET b
    C->>S: GET c
    S-->>C: a
    S-->>C: b
    S-->>C: c
```

**Batching trades latency for throughput.** Waiting to fill a batch delays the first item in it. Production batchers flush on **whichever comes first: a maximum batch size or a maximum delay** (for example, 100 items or 5 ms), so a quiet period never strands a request. Kafka's `linger.ms` and `batch.size` producer settings implement exactly this rule.

## Compression

Compression trades CPU for bytes. It helps when the link is the bottleneck and the data is compressible; it hurts when the CPU is the bottleneck or the data is already compressed (images, video, encrypted data, most archives).

| Codec | Ratio | Speed | Typical use |
|-------|-------|-------|-------------|
| gzip / DEFLATE | Moderate | Moderate | Universal HTTP compatibility |
| Brotli | Best for text at high levels | Slow at high levels; fast decompression | Static web assets compressed once at build time |
| Zstandard (RFC 8878) | Good, tunable (negative "fast" levels through 22) | Fast compression, very fast decompression; dictionaries | RPC, logs, storage, databases, HTTP (`Content-Encoding: zstd` supported in Chrome and Firefox since 2024) |
| LZ4 | Lower | Several GB/s per core | Hot paths: in-memory, page-level, and block-storage compression |
| Snappy | Lower | Very fast | Legacy big-data systems (being replaced by zstd and LZ4 in many) |

**Decision rule.** Compress when

$$t_{\text{compress}} + t_{\text{transfer, compressed}} + t_{\text{decompress}} < t_{\text{transfer, uncompressed}}$$

On a 25-100 Gbit/s datacenter link, a high-ratio codec can be slower than sending raw bytes; LZ4 or no compression wins. Over a slow WAN or mobile link, even expensive compression pays for itself.

Practical guidance:

- **Precompress static assets** at maximum Brotli or zstd levels at build time and serve the stored file; never recompress per request.
- **Use a fast level for dynamic responses** (gzip 4-6, Brotli 4-5, zstd 1-3) so CPU cost stays bounded under load.
- **Skip incompressible payloads** and very small ones (below about 1 KB), where framing overhead exceeds savings.
- **Use dictionaries for many small, similar messages** (JSON records, log lines, protobufs). A trained zstd dictionary lets a 200-byte message compress well on its own. For the web, **Compression Dictionary Transport** (RFC 9842, 2025) lets a previously downloaded resource serve as the dictionary for the next version, so an updated JavaScript bundle is sent as a small delta (`dcb` for Brotli, `dcz` for zstd).
- **Beware compression side channels.** Compressing secrets together with attacker-controlled input over TLS enables CRIME/BREACH-style attacks.

## Connection Pooling

A new connection costs a TCP handshake, a TLS handshake, a slow-start ramp, and for databases authentication and session setup (a PostgreSQL connection is a forked backend process). A **connection pool** keeps established connections open and lends them to callers.

```python
from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg://user:pass@host/db",
    pool_size=10,         # persistent connections kept open
    max_overflow=5,       # temporary extras during bursts
    pool_timeout=5,       # seconds a caller waits before failing
    pool_recycle=1800,    # replace connections older than 30 minutes
    pool_pre_ping=True,   # validate a connection before lending it out
)

with engine.connect() as conn:                  # returned to the pool on exit
    rows = conn.execute(text("SELECT 1")).all()
```

**Sizing.** From Little's Law, the pool needs about $\lambda \times W_{\text{hold}}$ connections, where $W_{\text{hold}}$ is how long each operation holds a connection. For databases, more is not better: past a small multiple of the server's core count, extra connections add lock contention and memory without adding throughput. With many application instances, put a pooler such as PgBouncer (transaction mode) in front of PostgreSQL so thousands of client connections share a few dozen server connections. PgBouncer 1.21 and later support protocol-level prepared statements in transaction mode.

For HTTP clients, create one client (and one pool) per process and reuse it; constructing a client per request discards the pool and pays a new TLS handshake every time.

| Hazard | Symptom | Defense |
|--------|---------|---------|
| Stale connections | Errors after idle periods; a load balancer or NAT dropped the connection silently | Pre-ping or validation, idle timeouts shorter than the middlebox's |
| Pool exhaustion | Callers block until `pool_timeout`; latency cliff | Statement timeouts, always return connections (context managers), alert on wait time |
| Connection age | Server-side limits, accumulated session state | `pool_recycle`, maximum-lifetime settings |
| Uneven load after scaling | New backends receive no traffic on long-lived HTTP/2 connections | Maximum connection age, client-side load balancing |

## Storage I/O Patterns

Storage follows the same latency, bandwidth, and concurrency rules as the network.

### Sequential vs. Random

- **Sequential** access streams contiguous blocks. HDDs avoid seeks, SSDs spread work across flash channels, and readahead can fetch ahead.
- **Random** access pays the device's per-operation latency on each request, including a mechanical seek of several milliseconds on HDDs. On an HDD, sequential throughput can exceed random throughput by more than 100x; on NVMe SSDs the gap is much smaller at high queue depths but remains significant, especially for writes.

This is why storage engines prefer large nodes (B-trees) and append-only writes (write-ahead logs, LSM-trees): converting random writes into sequential ones is often the largest storage optimization available.

### IOPS, Throughput, and Queue Depth

Devices are rated in **IOPS** (operations per second, the limit for small random I/O) and **throughput** (bytes per second, the limit for large sequential I/O):

$$\text{throughput} = \text{IOPS} \times \text{I/O size}$$

An NVMe SSD achieves its rated IOPS only with many requests outstanding. Little's Law again gives the arithmetic:

| Access pattern | Outstanding I/Os | Per-I/O latency | Result |
|----------------|------------------|-----------------|--------|
| Synchronous 4 KB reads, one thread | 1 | ~80 µs | ~12,500 IOPS, ~50 MB/s |
| Asynchronous 4 KB reads | 32 | ~80-100 µs | ~300,000-400,000 IOPS, ~1.2-1.6 GB/s |

A fast device driven by a single synchronous thread leaves most of its capability idle. Measure with `fio` at several queue depths to find where latency begins to rise sharply.

### Buffered, Direct, and Asynchronous I/O

| Mode | Behavior | Use when |
|------|----------|----------|
| Buffered (default) | Goes through the page cache; reads may hit RAM, writes are deferred | General-purpose file access, repeated reads |
| Direct (`O_DIRECT`) | Bypasses the page cache; requires aligned buffers, offsets, and sizes | Databases and caches that manage their own buffer pool |
| Memory-mapped (`mmap`) | File pages appear in the address space; faults load them | Read-mostly random access to large files |
| Asynchronous (`io_uring`, Windows IOCP / IoRing) | Many operations submitted without blocking a thread each | High-IOPS storage and high-connection-count networking |

**`io_uring`** (Linux 5.1 and later, still gaining features) uses a pair of ring buffers shared between the application and the kernel: the application places submission entries on one ring and reaps completions from the other, so many operations cost few or no syscalls. It covers files, sockets, timers, and more, and supports registered buffers and files to avoid per-operation setup. It has largely superseded `libaio`, which worked reliably only with `O_DIRECT`.

```mermaid
flowchart LR
    subgraph U["User space"]
        A["Application"]
    end
    subgraph SH["Shared memory"]
        SQ["Submission queue<br/>(SQEs)"]
        CQ["Completion queue<br/>(CQEs)"]
    end
    subgraph K["Kernel"]
        IO["io_uring<br/>worker / driver"]
        D["NVMe / NIC"]
    end
    A -->|"write SQEs"| SQ
    SQ -->|"io_uring_enter<br/>or SQPOLL thread"| IO
    IO --> D
    D --> IO
    IO -->|"post CQEs"| CQ
    CQ -->|"reap, no syscall"| A
```

Because `io_uring` exposes a large kernel attack surface, some environments restrict it (several container runtimes' default seccomp profiles, some Android and ChromeOS configurations, and a sysctl `kernel.io_uring_disabled` since Linux 6.6). Check the deployment target before depending on it.

## The Page Cache

The page cache is the kernel's in-memory cache of file contents. Understanding it separates "the disk is slow" from "the workload is thrashing the cache."

```mermaid
flowchart TD
    R["read()"] --> H{"Page in cache?"}
    H -->|hit| U["Copy to user buffer"]
    H -->|miss| DEV["Device read"] --> PC["Insert into page cache<br/>+ readahead next pages"] --> U
    W["write()"] --> DP["Copy into page cache<br/>mark dirty, return"]
    DP --> FL["Background writeback<br/>or fsync()"] --> DEV2["Device write"]
```

- **Reads** check the cache first. On a miss, the kernel reads from the device and, if the access looks sequential, **reads ahead** to prefetch subsequent pages.
- **Writes** copy into the cache, mark pages dirty, and return immediately. Data reaches the device during background writeback or on `fsync`/`fdatasync`. A successful `write()` does not mean the data is durable.

Implications:

- **Cache pollution.** Streaming a file larger than RAM (a backup, a full scan) evicts everyone else's hot data. Use `posix_fadvise(POSIX_FADV_DONTNEED)` after consuming data, or `O_DIRECT` for one-shot scans.
- **`fsync` is the durability-latency trade-off.** Flushing after every write is safe but slow. **Group commit** (collect many transactions, then issue one `fsync`) amortizes the flush; it is how write-ahead logs achieve high transaction rates.
- **`mmap`** removes explicit read calls and the read copy, and suits random access over large read-mostly files. It is a poor fit for write-heavy workloads or where the application needs control over eviction and I/O errors, which is why many databases that began with `mmap` moved to their own buffer pools.
- **Readahead** helps sequential scans and wastes bandwidth on random access. Tune per file with `posix_fadvise(POSIX_FADV_SEQUENTIAL / POSIX_FADV_RANDOM)` or per device with `blockdev --setra`.

## Zero-Copy

Serving a file over a socket with `read()` and `write()` copies the data through user space:

```mermaid
flowchart LR
    subgraph T["read() + write(): 2 CPU copies, 4 user/kernel transitions"]
        direction LR
        D1["Disk"] -->|DMA| P1["Page cache"] -->|CPU copy| U1["User buffer"] -->|CPU copy| S1["Socket buffer"] -->|DMA| N1["NIC"]
    end
    subgraph Z["sendfile(): 0 CPU copies with scatter-gather NIC"]
        direction LR
        D2["Disk"] -->|DMA| P2["Page cache"] -->|"DMA (descriptors only)"| N2["NIC"]
    end
```

Each CPU copy consumes memory bandwidth and pollutes caches; zero-copy paths let the kernel or hardware move data without touching it with the CPU.

| Mechanism (Linux) | What it does | Notes |
|-------------------|--------------|-------|
| `sendfile()` | File to socket entirely in the kernel | The classic static-file fast path |
| `splice()` / `tee()` / `vmsplice()` | Move data between file descriptors through a pipe | Zero-copy proxying and pipelines |
| Kernel TLS (kTLS) | TLS record encryption in the kernel or NIC after the handshake | Lets `sendfile` work for HTTPS; with NIC TLS offload, encryption is also removed from the CPU |
| `MSG_ZEROCOPY` (Linux 4.14+) | `send()` from user memory without copying into the socket buffer | Completion notifications required; pays off for sends larger than about 10 KB |
| `io_uring` `SEND_ZC` (Linux 6.0+) | Zero-copy send through the ring | Pairs with registered buffers |
| `io_uring` zero-copy receive (zcrx, Linux 6.15+) | Receives payloads directly into user memory | Requires NIC header/data split and flow steering |
| RDMA, DPDK, AF_XDP | Bypass or partially bypass the kernel network stack | Specialized high-throughput and low-latency deployments |

```c
#include <sys/sendfile.h>

// Serve a static file: bytes flow page cache -> NIC without entering user space.
off_t offset = 0;
while (offset < file_size) {
    ssize_t n = sendfile(client_fd, file_fd, &offset, file_size - offset);
    if (n <= 0) { /* handle EAGAIN on non-blocking sockets, or errors */ break; }
}
```

Zero-copy pays off for large transfers (file servers, video delivery, proxies, log shipping). For small messages, notification and setup overhead can exceed the cost of the copy; apply it where a profile shows copying is significant.

## Profiling I/O

| Question | Tools |
|----------|-------|
| Is the process waiting on I/O or CPU? | `pidstat -d`, `top` (I/O wait), off-CPU flame graphs (`offcputime` from BCC/bpftrace) |
| What is the device doing? | `iostat -x` (utilization, queue size, await), `biolatency`, `biosnoop` |
| Which syscalls are slow? | `strace -T -c` (high overhead), `perf trace`, `syscount` |
| What is on the wire? | `tcpdump`, Wireshark, `ss -ti` (per-socket RTT, congestion window, retransmits) |
| Where is network latency added? | `tcplife`, `tcpretrans`, distributed tracing (OpenTelemetry) |
| Maximum achievable throughput | `iperf3` (network), `fio` (storage) |

## Checklist

```mermaid
flowchart TD
    S["Slow I/O-bound path"] --> Q{"Latency-bound or<br/>bandwidth-bound?"}
    Q -->|"latency: idle, waiting"| L1["Reuse connections<br/>(keep-alive, pools)"]
    L1 --> L2["Batch and pipeline"]
    L2 --> L3["Raise concurrency<br/>(async I/O, deeper queues)"]
    L3 --> L4["Fewer handshakes<br/>(TLS 1.3, QUIC, 0-RTT where safe)"]
    Q -->|"bandwidth: link or device saturated"| B1["Compress, if CPU allows"]
    B1 --> B2["Larger, sequential transfers"]
    B2 --> B3["Remove copies<br/>(sendfile, kTLS, io_uring)"]
    B3 --> B4["Protect the page cache<br/>(fadvise, O_DIRECT, group commit)"]
```

1. **Classify the bottleneck.** Is the path idle and waiting on round trips (latency-bound), or is a link or device saturated (bandwidth-bound)?
2. **Remove round trips.** Pool connections, enable keep-alive, batch, and pipeline. This is usually the largest win and needs no hardware.
3. **Match the protocol to the network.** HTTP/2 on stable links, HTTP/3 for lossy and mobile clients, UDP-based protocols for real-time data, pooled connections for internal RPC.
4. **Compress only when bandwidth-bound**, with a codec matched to link speed and data type.
5. **Make storage access sequential and concurrent.** Append-only writes, deep asynchronous queues.
6. **Respect the page cache.** Avoid pollution from one-shot scans; group `fsync` calls.
7. **Eliminate copies** on large transfers once profiling shows they matter.

## See Also

- [Performance Optimization](./) - section hub and the optimization loop
- [CPU Optimization](cpu-optimization.html) - threading and concurrency models that underlie asynchronous servers
- [Memory Optimization](memory-optimization.html) - streaming, memory-mapped assets, and huge pages
- [Algorithmic Optimization](algorithmic-optimization.html) - data structures and caching that reduce I/O volume
- [Transport and Protocols](../technology/networking/transport-and-protocols.html) - TCP, UDP, and QUIC mechanics in depth
- [Network Performance and Security](../technology/networking/performance-and-security.html) - network-level performance tuning
- [Distributed Tracing](../observability/tracing.html) - attributing latency across services
- [Docker](../technology/docker/) - container networking and storage driver performance
- [Kubernetes](../technology/kubernetes/) - service networking and ingress at scale
- [Distributed Systems Theory](../advanced/distributed-systems-theory/) - latency, consistency, and coordination across the network
