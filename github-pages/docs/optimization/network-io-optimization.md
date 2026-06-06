---
layout: docs
title: "Optimization: Network & I/O Optimization"
permalink: /docs/optimization/network-io-optimization.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Network &amp; I/O Optimization</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Tame latency and bandwidth, pick the right protocol, and move bytes efficiently between the network, the kernel, and the disk.</p>
</div>

[Performance Optimization](./) &raquo; Network &amp; I/O Optimization

Most "slow" systems are not slow because the CPU cannot keep up — they are slow because they spend their time *waiting*: for a packet to cross the network, for a disk head to seek, for the kernel to copy a buffer, or for a connection to be established. This page is about the optimization discipline for that waiting. It covers the two physical constraints you can never repeal (latency and bandwidth), the protocol choices that determine how efficiently you use them, and the software techniques — batching, pipelining, compression, connection pooling, and zero-copy — that shrink the gap between theoretical throughput and what your application actually achieves.

<div class="key-insights">
  <div class="insight-card"><i class="fas fa-stopwatch"></i><h4>Latency is a floor you cannot cross</h4><p>Bandwidth can be bought; round-trip latency is bounded by the speed of light. Design to amortize round trips, not to "make the network faster."</p></div>
  <div class="insight-card"><i class="fas fa-layer-group"></i><h4>Batch and pipeline before you tune</h4><p>One request of 1000 items beats 1000 requests of one item by orders of magnitude. Coalescing work hides latency for free.</p></div>
  <div class="insight-card"><i class="fas fa-copy"></i><h4>Every copy is wasted bandwidth</h4><p>A naive read-then-send touches data four times across two address spaces. Zero-copy paths (`sendfile`, `splice`, `io_uring`) eliminate most of it.</p></div>
  <div class="insight-card"><i class="fas fa-hdd"></i><h4>Disk is just another network</h4><p>Seeks are round trips; the page cache is your connection pool. The same latency-vs-bandwidth reasoning applies to storage.</p></div>
</div>

## Latency vs Bandwidth: The Two Fundamental Limits

Every I/O channel — a TCP socket, an NVMe SSD, a spinning disk, a PCIe lane — is characterized by two independent numbers, and conflating them is the most common source of bad optimization decisions.

- **Latency** is the time for a single operation to complete end to end: how long one packet takes to reach the other side and come back, or how long one read takes to return its first byte. It is dominated by distance, queuing, and per-operation overhead.
- **Bandwidth** (throughput) is the rate at which data flows once it *is* flowing: bytes per second on a saturated link, or sustained MB/s from a streaming read.

These are independent. A trans-Pacific 10 Gbit/s link has enormous bandwidth and terrible latency (~150 ms RTT). A local loopback has low bandwidth ceilings relative to the CPU but near-zero latency. You optimize them with different techniques: latency with **fewer, coalesced round trips**; bandwidth with **compression, larger transfers, and parallel streams**.

### The Bandwidth-Delay Product

The amount of data "in flight" on a link at any instant is its capacity, and it sets the minimum buffer size needed to keep the pipe full:

$$\text{BDP} = \text{bandwidth} \times \text{RTT}$$

For a 1 Gbit/s link with a 100 ms round-trip time:

$$\text{BDP} = 125 \times 10^6 \ \text{B/s} \times 0.1 \ \text{s} = 12.5 \ \text{MB}$$

If your TCP send/receive window is smaller than 12.5 MB, the sender stalls waiting for acknowledgements and you will *never* reach line rate no matter how much bandwidth you bought. This is why long-fat networks (LFNs) require window scaling (RFC 7323) and why a default 64 KB window caps a 100 ms link at roughly 5 Mbit/s.

### Little's Law: Concurrency Hides Latency

Throughput, latency, and concurrency are tied together by Little's Law:

$$L = \lambda W$$

where $L$ is the average number of in-flight requests, $\lambda$ is throughput (requests/sec), and $W$ is average latency (seconds). Rearranged:

$$\lambda = \frac{L}{W}$$

To raise throughput you either lower per-request latency $W$ or raise concurrency $L$. A request that takes 10 ms served strictly one-at-a-time yields 100 req/s; with 50 in-flight requests it yields 5000 req/s at the *same* latency. This is the entire justification for connection pools, pipelining, and async I/O: you cannot make light faster, so you keep many requests in flight.

### Tail Latency

Averages lie. User-facing systems are judged by their tail — the p99 and p99.9 latencies — because a single slow dependency in a fan-out makes the whole response slow. If a request touches 100 backends and each has a 1% chance of a slow response, roughly $1 - 0.99^{100} \approx 63\%$ of requests hit at least one slow backend. Optimizing the network means optimizing the tail: bounded queues, hedged/redundant requests, and timeouts that fail fast rather than letting one stalled connection block a thread.

## Protocol Choice: TCP, UDP, QUIC, HTTP/1.1, HTTP/2, HTTP/3

The transport and application protocol you choose dictates how many round trips you pay before useful data flows, how head-of-line blocking behaves, and how much per-message overhead you carry.

### TCP vs UDP

| Property | TCP | UDP |
|----------|-----|-----|
| Delivery | Reliable, ordered, retransmitted | Best-effort, may drop/reorder |
| Connection | Stateful, 3-way handshake (1 RTT) | Connectionless, zero setup |
| Congestion control | Built-in (CUBIC, BBR) | None — you implement it |
| Head-of-line blocking | Yes (one lost segment stalls the stream) | No (each datagram independent) |
| Overhead | 20-byte header + handshake + ACKs | 8-byte header |
| Best for | Bulk transfer, RPC, anything needing ordering | Real-time media, games, DNS, custom reliability |

TCP's reliability is not free: a single lost segment blocks delivery of every byte behind it until retransmission completes (head-of-line blocking). UDP avoids this but pushes reliability, ordering, and congestion control into your application — which is exactly what QUIC does, in userspace, on top of UDP.

### The Handshake Cost

Connection setup is pure latency tax, paid in round trips before any application data moves:

- **TCP**: 1 RTT for the SYN / SYN-ACK / ACK handshake.
- **TCP + TLS 1.2**: 1 RTT (TCP) + 2 RTT (TLS) = **3 RTT** before the first byte.
- **TCP + TLS 1.3**: 1 RTT (TCP) + 1 RTT (TLS) = **2 RTT**, with optional 0-RTT resumption.
- **QUIC**: combines transport and crypto into **1 RTT** on first connect, and **0-RTT** for resumed sessions.

On a 100 ms link, TLS-1.2-over-TCP spends 300 ms doing nothing useful. This is why connection reuse (pooling, keep-alive) matters so much: amortizing one handshake over thousands of requests turns a per-request 300 ms tax into a per-request rounding error.

### HTTP/1.1

One request per connection at a time. Pipelining (sending the next request before the response arrives) is specified but suffers from head-of-line blocking and is effectively unused. Browsers worked around the serialization by opening 6+ parallel connections per host, multiplying handshakes and congestion-control state. Each connection independently ramps up from TCP slow start, so short-lived HTTP/1.1 connections rarely reach full bandwidth.

### HTTP/2

Multiplexes many concurrent **streams** over a *single* TCP connection, with header compression (HPACK) and server push. This eliminates application-level head-of-line blocking and the wasteful connection sprawl of HTTP/1.1. But because it still rides on one TCP stream, it reintroduces head-of-line blocking at the *transport* layer: one lost TCP segment stalls *every* multiplexed stream, because TCP must deliver bytes in order before HTTP/2 can demultiplex them. On lossy networks, HTTP/2 over a single connection can be *worse* than HTTP/1.1 over six.

### HTTP/3 and QUIC

HTTP/3 runs over **QUIC**, which runs over UDP. QUIC implements reliability, ordering, congestion control, and TLS 1.3 in userspace, with one crucial difference: each stream has its own delivery state, so a lost packet only stalls the stream it belonged to, not the others. This finally solves transport-layer head-of-line blocking. QUIC also enables 0-RTT resumption and connection migration (a connection survives an IP change, e.g. Wi-Fi to cellular, via a connection ID rather than the 4-tuple).

```text
HTTP/1.1     HTTP/2              HTTP/3
  one          many streams        many streams
 request       over one TCP        over QUIC/UDP
per conn       (transport HOL      (per-stream
               blocking on loss)    delivery, no HOL)

   TCP            TCP                 QUIC
    |              |                   |
   IP             IP                 UDP / IP
```

### Choosing

- **Bulk, ordered, lossless transfer (file sync, RPC, databases):** TCP, ideally with HTTP/2 multiplexing or a binary RPC framing (gRPC).
- **Web traffic at scale, mobile/lossy networks:** HTTP/3 (QUIC) for faster setup, no transport HOL blocking, and connection migration.
- **Real-time / latency-over-reliability (VoIP, video conferencing, game state):** UDP, or QUIC/WebRTC datagrams, accepting and concealing occasional loss rather than waiting for retransmission.
- **Tiny request/response on the same datacenter:** plain UDP (DNS-style) or a pooled persistent TCP connection — the handshake dominates everything else.

## Batching and Pipelining

The cheapest round trip is the one you never make. Both techniques attack latency by amortizing per-operation overhead.

**Batching** combines many logical operations into one physical request. Instead of 1000 single-row inserts (1000 round trips), send one multi-row insert (one round trip). The latency win is dramatic: at 1 ms RTT, 1000 sequential round trips cost 1 second; one batched round trip costs ~1 ms plus transfer time.

**Pipelining** keeps multiple requests in flight without waiting for each response, exploiting Little's Law directly. Redis pipelining and HTTP/2 multiplexing both work this way: you fire request 2 before response 1 arrives, so the RTT is paid once across the whole batch rather than once per request.

```python
import redis

r = redis.Redis()

# Anti-pattern: one round trip per command (N x RTT)
for i in range(10_000):
    r.set(f"key:{i}", i)

# Pipelined: commands buffered and flushed together (~1 x RTT)
with r.pipeline(transaction=False) as pipe:
    for i in range(10_000):
        pipe.set(f"key:{i}", i)
    pipe.execute()   # single network round trip for all 10,000 commands
```

On a 1 ms link this turns ~10 seconds of cumulative round-trip time into a handful of milliseconds. The same pattern applies to database `executemany`, S3/object-store bulk operations, GraphQL/REST request coalescing (DataLoader-style), and message queues that support batch publish.

**The batching trade-off is latency vs throughput.** Holding requests to accumulate a batch adds latency to the first item in the batch. Production systems bound this with a **max-batch-size OR max-delay** flush rule (e.g. "flush when 100 items queued *or* 5 ms elapses, whichever comes first") so a quiet period never strands a request indefinitely. This is the Nagle algorithm's logic generalized — and Nagle itself (which coalesces small TCP writes) is why latency-sensitive protocols disable it with `TCP_NODELAY`.

## Compression

Compression trades CPU for bandwidth — sensible when the link is the bottleneck and the data is compressible, harmful when the CPU is the bottleneck or the data is already compressed (JPEG, video, encrypted bytes).

| Codec | Ratio | Speed | Use case |
|-------|-------|-------|----------|
| gzip / DEFLATE | Good | Moderate | Universal HTTP, broad compatibility |
| Brotli | Best (text) | Slow to compress, fast to decompress | Static web assets compressed once, served many times |
| Zstandard (zstd) | Good, tunable (levels 1-22) | Very fast, dictionary support | RPC, logs, storage, real-time streams |
| LZ4 | Lower | Extremely fast (GB/s) | Hot paths where CPU is scarce — memory, page-cache, DB pages |
| Snappy | Lower | Very fast | Big-data shuffles (Kafka, Cassandra, RPC) |

**Decision rule.** Compress when `compression_time + transfer_time_compressed < transfer_time_uncompressed`. On a fast LAN (10+ Gbit/s) heavy compression often *slows* transfer because the CPU cannot compress as fast as the wire moves bytes; LZ4/Snappy or no compression wins. On a slow WAN, even Brotli's CPU cost is dwarfed by the bytes saved.

Practical guidance:
- **Compress once, serve many:** precompress static assets at the highest ratio (Brotli) at build time; never recompress per request.
- **Compress dynamic responses with a fast codec** (gzip level 4-6, or zstd) so the CPU cost stays bounded under load.
- **Never compress already-compressed payloads** — you pay CPU for ~0% gain and sometimes a slightly larger output.
- **Set a minimum size threshold** (e.g. only compress responses > 1 KB); small payloads cost more in framing overhead than they save.
- **Use shared dictionaries** (zstd dictionaries) when transferring many small, similar messages (JSON records, protobufs) — the dictionary primes the compressor so even tiny messages compress well.

## Connection Pooling

Opening a connection is expensive: a TCP handshake, a TLS handshake, congestion-control slow start, and (for databases) authentication and session setup. A **connection pool** keeps a set of established connections alive and hands them out to callers, amortizing all of that across thousands of operations.

```python
# Without pooling: full TCP + TLS + auth handshake on EVERY query
def query_unpooled(sql):
    conn = connect(host, user, password)   # ~3 RTT + auth
    result = conn.execute(sql)
    conn.close()                            # connection discarded
    return result

# With pooling: handshake paid once, reused for the connection's lifetime
from sqlalchemy import create_engine
engine = create_engine(
    "postgresql://user:pass@host/db",
    pool_size=20,         # persistent connections kept warm
    max_overflow=10,      # temporary extras under burst load
    pool_timeout=30,      # how long a caller waits for a free connection
    pool_recycle=1800,    # recycle connections older than 30 min
    pool_pre_ping=True,   # validate a connection before handing it out
)
```

**Sizing the pool** follows from Little's Law: you need roughly `throughput x average_hold_time` connections to avoid queuing. For databases, more is not better — each Postgres connection is a backend process consuming memory and contending on locks; pools of 2-3x the core count, fronted by a proxy like PgBouncer, usually beat huge per-app pools. For HTTP clients, the pool prevents the latency disaster of a fresh TLS handshake per call; always enable keep-alive and reuse a single client instance rather than constructing one per request.

**Pool hazards to guard against:**
- **Stale connections:** a peer or load balancer silently dropped the connection. Use `pool_pre_ping` / validation queries or bounded idle timeouts.
- **Pool exhaustion:** all connections checked out and held (e.g. a slow query or a leaked handle) causes callers to block on `pool_timeout`. Cap query time and always return connections in a `finally`/context manager.
- **Connection age:** long-lived connections accumulate state and can outlive server-side timeouts; `pool_recycle` rotates them.

## Disk I/O Patterns

Storage obeys the same latency-vs-bandwidth duality as the network, and the gap between access patterns is enormous.

### Sequential vs Random

- **Sequential access** reads/writes contiguous blocks. On an HDD the head streams without seeking; on an SSD it maximizes parallelism across NAND chips and lets the prefetcher work. Sequential throughput can be 100x+ random throughput on HDDs and several-x on SSDs.
- **Random access** scatters small reads across the device. Each one pays the device's access latency, and on an HDD that includes a mechanical seek (~5-10 ms). This is why a B-tree designed for disk uses large nodes (one big sequential read) rather than many pointer chases.

### IOPS vs Throughput

A device is rated by both **IOPS** (operations/sec, the limit for small random I/O) and **throughput** (MB/s, the limit for large sequential I/O). A workload of 4 KB random reads is IOPS-bound; a workload of 1 MB sequential reads is throughput-bound. The relationship:

$$\text{throughput} = \text{IOPS} \times \text{I/O size}$$

Larger I/O sizes amortize per-operation overhead. This is why databases and log systems prefer large, append-only sequential writes (LSM-trees, write-ahead logs) over in-place random updates: turning random writes into sequential ones can be the single biggest storage win available.

### Buffered vs Direct, Sync vs Async

- **Buffered I/O** (the default) routes through the kernel page cache: reads may hit cache and return instantly; writes are absorbed into dirty pages and flushed later. Great for repeated access, but it costs a memory copy and gives up control over timing.
- **Direct I/O** (`O_DIRECT`) bypasses the page cache to move data straight between the device and a user buffer. Databases use it because they manage their own cache and do not want double-buffering; it requires aligned buffers and careful sizing.
- **Asynchronous I/O** lets you submit many requests without blocking a thread per request — essential for keeping a fast SSD's deep queue full. On Linux, `io_uring` is the modern interface: a submission/completion ring shared with the kernel that batches syscalls and supports zero-copy operations, far outperforming the older `libaio` and thread-pool approaches.

```c
// Queue depth matters: one outstanding 4 KB read at a time on an NVMe SSD
// leaves >90% of its IOPS idle. Async submission keeps the device busy.
//
// Synchronous (queue depth 1): throughput = 1 / latency
//   4 KB read latency ~ 80 us  ->  ~12,500 IOPS  ->  ~50 MB/s
//
// Asynchronous (queue depth 32 via io_uring):
//   device serves 32 in flight  ->  ~400,000 IOPS  ->  ~1.6 GB/s
```

The lesson mirrors Little's Law from the network: a fast device with deep parallelism is wasted by a serial, one-at-a-time access pattern. Keep the queue full.

## Filesystem and Page-Cache Behavior

The page cache is the kernel's RAM cache of file contents, and understanding it is what separates "the disk is slow" from "we are thrashing the cache."

### How It Works

A buffered `read()` first checks the page cache; a hit returns from RAM with no device access. A miss triggers a device read, the page is cached, and (often) the kernel **readahead** heuristic prefetches subsequent pages, betting on sequential access. A buffered `write()` marks pages **dirty** and returns immediately; a background flusher (or `fsync`) writes them to the device later. This write-back behavior is why an un-`fsync`'d write can be lost on power failure even though the syscall "succeeded" — durability requires an explicit flush.

### Implications for Optimization

- **The cache is shared and finite.** Streaming a file far larger than RAM (a backup, a full table scan) can evict the hot working set of everything else on the box — *cache pollution*. Advise the kernel with `posix_fadvise(POSIX_FADV_DONTNEED)` after streaming, or use `O_DIRECT`, so a one-shot scan does not blow away useful cache.
- **`fsync` is the durability/latency knob.** Calling `fsync` per write is safe but slow (it forces a device flush and waits). Batching writes and `fsync`-ing once per group — the WAL group-commit pattern — amortizes the flush across many transactions.
- **Memory-mapped files (`mmap`)** map file pages directly into the address space; access faults pages in on demand and the page cache *is* your buffer, eliminating explicit read syscalls and the read copy. Excellent for random access over large read-mostly files (indexes, embedded databases); less ideal for write-heavy or strictly-sequential streaming where readahead and explicit buffering win.
- **Readahead tuning** helps sequential scans (raise it) and hurts random workloads (lower it), because aggressive prefetch on random access wastes bandwidth on pages you never use.

## Zero-Copy

The classic "read a file and send it over a socket" path copies the data four times and crosses the user/kernel boundary four times:

```text
Traditional read() + write():

  disk --DMA--> [kernel page cache] --CPU copy--> [user buffer]
       --CPU copy--> [kernel socket buffer] --DMA--> NIC

  4 copies (2 of them CPU-driven), 4 context switches
```

Every CPU copy burns memory bandwidth and cache, and the context switches add latency. **Zero-copy** removes the redundant trips through userspace by letting the kernel move data device-to-device:

```text
sendfile() / splice():

  disk --DMA--> [kernel page cache] --(DMA, scatter-gather)--> NIC

  0 CPU copies, data never enters userspace, 2 context switches
```

Mechanisms on Linux:
- **`sendfile(out_fd, in_fd, ...)`** copies between two file descriptors entirely in the kernel — the canonical web-server and file-server fast path for serving static content.
- **`splice()` / `vmsplice()` / `tee()`** move data between fds via a kernel pipe buffer, enabling zero-copy pipelines (e.g. file -> socket, or socket -> socket proxying).
- **`MSG_ZEROCOPY`** lets `send()` transmit directly from a user buffer without copying into the kernel socket buffer (useful for large sends; has setup overhead, so it pays off above a size threshold).
- **`io_uring`** supports zero-copy submission and registered buffers, unifying async I/O and zero-copy in one ring interface.

The payoff is largest for large transfers where the copies dominate (file servers, video streaming, proxies, log shippers). For tiny messages the syscall and setup overhead can outweigh the saved copy, so zero-copy — like compression and batching — is a technique you apply where the profiler shows the copy actually costs you.

```c
// Static-file server hot path: serve a file to a socket without
// ever bringing its bytes into userspace.
off_t offset = 0;
ssize_t sent = sendfile(client_socket, file_fd, &offset, file_size);
// Bytes flow disk -> page cache -> NIC entirely inside the kernel.
// No read() into a user buffer, no write() copy back out.
```

## Putting It Together: A Profiling-Driven Checklist

Optimizing I/O is the same disciplined loop as the rest of [performance work](./) — measure, find the bottleneck, fix it, re-measure — applied to the wait states rather than the compute.

1. **Classify the bottleneck.** Is the request latency-bound (idle waiting on round trips) or bandwidth-bound (link/device saturated)? `latency` charts vs `MB/s` charts tell you which family of fixes applies.
2. **Kill round trips first.** Pool connections, enable keep-alive, batch and pipeline. This is usually the single biggest win and costs no extra hardware.
3. **Pick the protocol for the network.** HTTP/2 on stable links, HTTP/3/QUIC on lossy/mobile, UDP for real-time, pooled TCP for datacenter RPC.
4. **Trade CPU for bandwidth only when bandwidth-bound.** Compress with a codec matched to the link speed and the data; never recompress incompressible bytes.
5. **Make disk access sequential and parallel.** Convert random writes to append-only sequential ones; keep a deep async queue (`io_uring`) to saturate fast SSDs.
6. **Respect the page cache.** Avoid cache pollution from one-shot scans; batch `fsync`; consider `mmap` for read-mostly random access.
7. **Eliminate redundant copies.** Use `sendfile`/`splice`/zero-copy on large transfers once the profiler shows copies dominate.

## See Also

- [Performance Optimization](./) - The section hub: profiling-driven methodology and the optimization loop
- [CPU Optimization](./cpu-optimization.html) - Cache hierarchy, multithreading, and allocation patterns that underlie async I/O
- [Memory Optimization](./memory-optimization.html) - Streaming, page-cache interaction, and memory-mapped assets
- [Algorithmic Optimization](./algorithmic-optimization.html) - Spatial structures and complexity wins that reduce I/O volume
- [Docker](../technology/docker/) - Container networking and storage driver performance
- [Kubernetes](../technology/kubernetes/) - Service networking, ingress, and resource-aware I/O at scale
- [Distributed Systems Theory](../advanced/distributed-systems-theory/) - Foundations of latency, consistency, and coordination across the network
