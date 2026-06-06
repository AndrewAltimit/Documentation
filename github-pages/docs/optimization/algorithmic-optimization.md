---
layout: docs
title: "Optimization: Algorithmic Optimization"
permalink: /docs/optimization/algorithmic-optimization.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Algorithmic Optimization</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Complexity analysis in practice, choosing the right data structure, spatial partitioning, caching, memoization, and amortization</p>
</div>

[Performance Optimization](./) &raquo; Algorithmic Optimization

<div class="code-example" markdown="1">
The single most valuable optimization is rarely a faster line of code — it is a *better algorithm*. A change from $O(n^2)$ to $O(n \log n)$ shrinks a one-second operation on a million elements to a few milliseconds, a speedup no amount of constant-factor tuning can match. This page covers how complexity analysis plays out in real systems: reading the asymptotics that actually matter, choosing data structures by their operation profile, partitioning space to avoid all-pairs work, and trading memory for time through caching, memoization, and amortization. For the constant-factor and hardware-level work that comes *after* the algorithm is right, see [CPU Optimization](./cpu-optimization.html) and [Memory Optimization](./memory-optimization.html).
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Complexity Analysis in Practice

### Why Big O Dominates

Asymptotic complexity describes how an algorithm's cost grows with input size $n$. Because it ignores constant factors and lower-order terms, it tells you which algorithm *wins as the problem scales* — and in practice problems always scale. The table below shows operation counts for common growth rates; note how the gap between classes explodes:

| $n$ | $\log n$ | $n$ | $n \log n$ | $n^2$ | $2^n$ |
|------|----------|------|------------|--------|--------|
| 16 | 4 | 16 | 64 | 256 | 65,536 |
| 1,024 | 10 | 1,024 | ~10,240 | ~1.05M | ~$10^{308}$ |
| 1,000,000 | ~20 | $10^6$ | ~$2 \times 10^7$ | $10^{12}$ | overflow |

A constant-factor optimization that makes your code 2x faster moves the $n^2$ row's $10^{12}$ down to $5 \times 10^{11}$ — still astronomically worse than the $n \log n$ alternative at $2 \times 10^7$. This is the core lesson: **fix the complexity class before tuning constants.**

The standard algorithmic upgrades, drawn from the same operations a profiler will flag as hot:

| Operation | Naive | Optimized |
|-----------|-------|-----------|
| Find in list | O(n) linear scan | O(1) hash table |
| Sort | O(n²) bubble/insertion | O(n log n) merge/quick/heap |
| Nearest neighbor | O(n) brute force | O(log n) spatial tree |
| Path finding | O(n²) Dijkstra (dense) | O(n log n) A* with heap |
| Collision detection | O(n²) all-pairs | O(n log n) broad phase |
| Membership test (repeated) | O(n) per query | O(1) hash set |
| Range/min query | O(n) per query | O(log n) segment/Fenwick tree |

### When Big O Lies: Constants and Cache

Asymptotics are necessary but not sufficient. Three caveats decide whether the theoretically faster algorithm wins on real hardware:

1. **Small $n$ favors simple algorithms.** Insertion sort beats quicksort for arrays of ~16 elements because its constant factor and cache behavior are excellent. Production sorts (`std::sort`, Timsort) switch to insertion sort below a threshold for exactly this reason.

2. **Memory locality is a hidden constant.** A linear scan over a contiguous array can outrun a "faster" pointer-chasing structure (linked list, tree) because every cache miss costs ~200 cycles. Big O counts operations, not cache misses. See [Memory Optimization](./memory-optimization.html) for how data layout interacts with the cache hierarchy.

3. **Amortized vs worst-case matters for latency.** An $O(1)$ amortized hash insert can stall for $O(n)$ on a resize — fine for throughput, fatal for a real-time frame budget. Knowing *which* bound you need is part of the analysis (covered under [Amortized Analysis](#amortized-analysis) below).

### A Worked Example: De-quadratifying a Hot Loop

A common $O(n^2)$ pattern is "for each item, search a list for a match":

```cpp
// O(n*m): for each event, linearly find its handler by id.
for (const Event& e : events) {              // n events
    for (Handler& h : handlers) {            // m handlers
        if (h.id == e.target_id) { h.process(e); break; }
    }
}
```

If `events` and `handlers` are both large, this is quadratic. Building a hash index once turns the inner scan into an $O(1)$ lookup, making the whole loop $O(n + m)$:

```cpp
// O(n + m): index handlers by id once, then look up directly.
std::unordered_map<Id, Handler*> by_id;
by_id.reserve(handlers.size());
for (Handler& h : handlers) by_id[h.id] = &h;

for (const Event& e : events) {
    if (auto it = by_id.find(e.target_id); it != by_id.end())
        it->second->process(e);
}
```

The pattern generalizes: **whenever you see a search nested inside a loop, ask whether the searched collection can be pre-indexed** into a hash map, sorted array (for binary search), or spatial structure.

## Choosing Data Structures

A data structure is a contract over the *operations* you perform and their costs. Picking the right one is often the entire optimization, because it changes the complexity of every operation that touches it. Profile your access pattern first: are you mostly inserting, searching, iterating in order, querying ranges, or removing from the middle?

| Structure | Lookup | Insert | Delete | Ordered iteration | Notes |
|-----------|--------|--------|--------|-------------------|-------|
| Dynamic array (`vector`) | O(n) search / O(1) index | O(1) amortized (back) | O(n) (middle) | Yes (sorted) | Best cache locality; ideal default |
| Hash map (`unordered_map`) | O(1) avg | O(1) avg | O(1) avg | No | Random-access lookups; resize stalls |
| Balanced BST (`map`) | O(log n) | O(log n) | O(log n) | Yes | Ordered + range queries |
| Linked list | O(n) | O(1) at node | O(1) at node | Yes | Poor locality; rarely worth it |
| Binary heap (priority queue) | O(1) peek | O(log n) | O(log n) pop-min | No | Scheduling, Dijkstra/A* frontier |
| Bitset | O(1) | O(1) | O(1) | Yes | Dense small-integer sets, very fast |
| B-tree / B+tree | O(log n) | O(log n) | O(log n) | Yes | Disk/page-friendly; database indexes |

### Practical Heuristics

- **Default to a contiguous array.** Its cache behavior beats node-based structures so often that you should justify *not* using one. Linear search over a packed array can outperform a tree up to surprisingly large $n$.
- **Use a hash map for repeated membership/lookup by key**, but reserve capacity up front (`reserve(n)`) to avoid incremental rehashing, and remember it gives no ordering and has worst-case $O(n)$ buckets under collisions.
- **Use a balanced tree (or sorted array + binary search) when you need order**: range queries, nearest-key, or in-order traversal. A sorted `vector` with `lower_bound` beats `std::map` when the data is built once and queried many times, because it stays contiguous.
- **Use a heap for "repeatedly extract the best"**: priority scheduling, event queues, and the open set in A*.
- **Match the structure to the storage tier.** In-memory work favors flat arrays and hash maps; on-disk or paged data favors B-trees, which is why [databases](../technology/database-design/) build B+tree indexes — they minimize page reads, the dominant cost when data lives on disk.

### The Cost of Indirection

Two structures with the same Big O can differ by an order of magnitude in wall-clock time because of pointer chasing. A `std::map` and a sorted `std::vector` are both $O(\log n)$ for lookup, but the vector touches contiguous memory while the map jumps across heap-scattered nodes, missing cache on nearly every step. When you have a choice between equal asymptotics, **prefer the structure with better locality** — this is the bridge from algorithmic optimization into [data-oriented design](./cpu-optimization.html).

## Spatial Data Structures

Many performance problems are "find all objects near a point or region" — collision detection, rendering culling, nearest-neighbor queries, AI perception. Done naively this is $O(n^2)$ (test every pair) or $O(n)$ per query. Spatial data structures partition space so that each query only examines nearby candidates, cutting this to roughly $O(\log n)$ per query or $O(n \log n)$ for the all-pairs broad phase.

### Choosing by Use Case

```
Static geometry (built once, queried often):
- BVH (Bounding Volume Hierarchy)  — ray tracing, static collision
- BSP trees                        — visibility, classic level geometry
- Octrees                          — uniform 3D subdivision

Dynamic objects (rebuilt/updated every frame):
- Spatial hashing                  — uniform-size objects, cheap rebuild
- Uniform grid partitioning        — bounded worlds, fast neighbor lookup
- Loose octrees                    — moving objects spanning cell boundaries

2D / GIS:
- Quadtrees                        — adaptive 2D subdivision
- R-trees                          — bounding-box indexing, spatial databases
- Spatial hashing                  — particles, tile-based games
```

### The Broad-Phase / Narrow-Phase Pattern

The canonical use of spatial structures is collision detection. Instead of testing all $\binom{n}{2}$ pairs:

1. **Broad phase** — a spatial structure (grid, BVH, or sweep-and-prune) cheaply finds *candidate* pairs whose bounding volumes overlap. This is the step that turns $O(n^2)$ into roughly $O(n \log n)$.
2. **Narrow phase** — exact, expensive intersection tests run only on the few candidate pairs the broad phase produced.

### Uniform Spatial Hashing

For evenly distributed, similarly sized objects, a spatial hash is the simplest fast structure: map each object's position to a grid cell and store cell occupants in a hash map. Neighbor queries inspect only the object's cell and its eight (2D) or 26 (3D) neighbors:

```cpp
// Hash a world position into an integer grid cell.
struct CellKey { int x, y, z; bool operator==(const CellKey&) const = default; };

struct CellHash {
    size_t operator()(const CellKey& k) const {
        // Mix the three coordinates; large primes reduce collisions.
        return (k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791);
    }
};

class SpatialHash {
    float cell_size;
    std::unordered_map<CellKey, std::vector<Object*>, CellHash> cells;

    CellKey key_of(const Vector3& p) const {
        return { int(std::floor(p.x / cell_size)),
                 int(std::floor(p.y / cell_size)),
                 int(std::floor(p.z / cell_size)) };
    }
public:
    void insert(Object* o) { cells[key_of(o->position)].push_back(o); }

    // Gather candidates in the object's cell and its 26 neighbors.
    std::vector<Object*> query_neighbors(const Vector3& p) const {
        std::vector<Object*> out;
        CellKey c = key_of(p);
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (auto it = cells.find({c.x+dx, c.y+dy, c.z+dz}); it != cells.end())
                out.insert(out.end(), it->second.begin(), it->second.end());
        return out;
    }
};
```

The `cell_size` is the key tuning parameter: too large and each cell holds too many objects (degenerating toward $O(n^2)$); too small and objects span many cells, wasting memory and broadening queries. A good default is roughly the diameter of a typical object.

### Hierarchical Trees: Octrees and BVHs

When object density is *non-uniform* — clusters of detail amid empty space — a flat grid wastes memory on empty cells. Hierarchical structures adapt:

- **Octree (3D) / Quadtree (2D)** recursively subdivides a cube/square into eight/four children only where objects are present, giving $O(\log n)$ depth in well-balanced scenes. Ideal for static or slowly changing worlds.
- **BVH (Bounding Volume Hierarchy)** wraps objects in nested bounding boxes (or spheres) forming a binary tree. A ray or query descends only into boxes it intersects, pruning whole subtrees. BVHs are the backbone of modern ray tracing and are cheap to refit (update bounds without rebuilding) when objects move slightly. See how this feeds into rendering in [3D Graphics & Rendering](../graphics/3d-rendering.html).

The shared idea across all of these: **convert a global "test everything" into a local "test only what's nearby" by exploiting spatial coherence.**

## Caching and Memoization

When the same expensive computation recurs with the same inputs, store the result and return it on subsequent calls. This trades memory for time — the central trade-off of algorithmic optimization — and is most effective when the function is *pure* (output depends only on inputs) and recomputation is costly relative to a lookup.

### Memoizing a Pure Function

The pattern, preserved and expanded from the optimization hub, wraps an expensive calculation in a cache keyed by its inputs:

```cpp
// Expensive computation caching (memoization)
class ExpensiveComputation {
    mutable std::unordered_map<Key, Result> cache;

public:
    Result compute(const Key& key) const {
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;   // cache hit: O(1)
        }

        Result result = expensive_calculation(key);  // cache miss: full cost
        cache[key] = result;
        return result;
    }

    void invalidate() { cache.clear(); }
};
```

### Memoization and Dynamic Programming

Memoization is the top-down face of dynamic programming: a naive recursive solution that recomputes overlapping subproblems becomes linear once each subproblem is cached. The textbook example is Fibonacci, where naive recursion is $O(2^n)$ but memoized recursion is $O(n)$:

```cpp
long long fib(int n, std::vector<long long>& memo) {
    if (n < 2) return n;
    if (memo[n] != -1) return memo[n];        // already computed
    return memo[n] = fib(n-1, memo) + fib(n-2, memo);
}
```

Each value `fib(k)` is computed once and reused, collapsing the exponential call tree into $n$ distinct evaluations. The recurrence it encodes is:

$$
F(n) = F(n-1) + F(n-2), \quad F(0) = 0, \; F(1) = 1
$$

The same transformation — identify overlapping subproblems, cache them — turns exponential brute force into polynomial time across an enormous class of problems (shortest paths, edit distance, knapsack, sequence alignment).

### Cache Invalidation and Eviction

The two hard parts of caching are *when results stop being valid* and *what to throw away when the cache is full*.

- **Invalidation.** A memoized result is only correct while its inputs are unchanged. Pure functions of immutable inputs never need invalidation; caches over mutable state must be cleared or selectively evicted when that state changes. Stale-cache bugs are notorious precisely because the wrong answer is fast and silent.
- **Eviction (bounded caches).** An unbounded cache is a memory leak. Real caches cap their size and evict under a policy: **LRU** (least-recently-used) for temporal locality, **LFU** (least-frequently-used) when some keys are persistently hot, or simple FIFO/random when cheap is good enough. The hit rate of the policy, not its theoretical elegance, is what to measure.

### Caching Across the Stack

The same store-and-reuse principle recurs at every scale: the [CPU's L1/L2/L3 hierarchy](./cpu-optimization.html) caches memory, web layers cache HTTP responses, [databases](../technology/database-design/) cache query plans and pages, and applications memoize functions. Each level trades cheaper, larger, slower storage for a fast copy of recent results.

## Amortized Analysis

Some operations are usually cheap but occasionally expensive. **Amortized analysis** measures the *average* cost per operation over a long sequence, proving that the rare expensive case is paid for by the many cheap ones. This is the right lens for any structure that does bulk work intermittently.

### The Dynamic Array Doubling Argument

The canonical example is appending to a dynamic array (`std::vector`). Most `push_back` calls are $O(1)$: write to the next slot. But when the array fills, it must allocate a larger buffer and copy every element — an $O(n)$ operation. Is `push_back` therefore $O(n)$?

No. The trick is the **growth factor**: when full, the array doubles its capacity. Starting from capacity 1 and inserting $n$ elements, the total copying work across all resizes is:

$$
1 + 2 + 4 + 8 + \cdots + n = \sum_{k=0}^{\log_2 n} 2^k = 2n - 1 = O(n)
$$

Spread over $n$ insertions, that is $O(1)$ **amortized** per `push_back`. The geometric (not linear) growth is essential: growing by a constant amount instead of a factor would make total work $O(n^2)$ and the amortized cost $O(n)$. This is exactly why you `reserve()` capacity when the final size is known — it eliminates *all* intermediate copies, turning amortized $O(1)$ into guaranteed $O(1)$.

### Why the Distinction Matters

Amortized $O(1)$ is excellent for **throughput** but says nothing about any single operation's **latency**. The one `push_back` that triggers a resize stalls for $O(n)$ — invisible in a server's average request time, but a frame-rate killer in a game that hits it inside the 16 ms budget. The same caveat applies to hash-map rehashing and incremental garbage collection.

The defenses are the same in every case:

- **Pre-size structures** (`reserve`) when the eventual size is known, so the expensive grow never happens on the hot path.
- **Amortize deliberately** by spreading the expensive work across frames, or do it during a load screen or idle period rather than mid-action.
- **Choose worst-case-bounded structures** for hard-real-time paths — a fixed-capacity pool or ring buffer never resizes, trading flexibility for predictable latency. This connects directly to the pooling and frame-allocator techniques in [CPU Optimization](./cpu-optimization.html) and [Memory Optimization](./memory-optimization.html).

### The Three Analysis Methods

Formally, amortized bounds are proven by one of three equivalent techniques:

1. **Aggregate method** — bound the total cost of $n$ operations, then divide by $n$ (the doubling-array sum above).
2. **Accounting (banker's) method** — charge each cheap operation a small surplus "credit" that prepays for future expensive ones; if credit never goes negative, the charged rate is a valid amortized bound.
3. **Potential method** — define a potential function $\Phi$ over the data structure's state; the amortized cost of an operation is its actual cost plus the change in potential, $\hat{c}_i = c_i + \Phi_i - \Phi_{i-1}$.

In practice you rarely write these proofs, but recognizing *which* bound a structure provides — worst-case, average, or amortized — is what lets you choose correctly for throughput-bound versus latency-bound code.

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card"><h4>Complexity class first</h4><p>An O(n²)→O(n log n) change dwarfs any constant-factor tuning at scale. Fix the asymptotics before touching the hot path.</p></div>
  <div class="takeaway-card"><h4>But measure the constants</h4><p>Small n, cache locality, and amortized-vs-worst-case can make the "slower" Big O win on real hardware.</p></div>
  <div class="takeaway-card"><h4>The structure is the algorithm</h4><p>Match the data structure to your operation profile; a contiguous array's locality often beats a node-based structure of equal Big O.</p></div>
  <div class="takeaway-card"><h4>Partition space for locality</h4><p>Grids, octrees, and BVHs turn O(n²) all-pairs queries into O(n log n) by only testing nearby candidates.</p></div>
  <div class="takeaway-card"><h4>Trade memory for time</h4><p>Memoization and caching collapse repeated work — but bound the cache and get invalidation right, or you leak memory and serve stale answers.</p></div>
  <div class="takeaway-card"><h4>Know which bound you need</h4><p>Amortized O(1) is great for throughput and lethal for latency; pre-size or use fixed-capacity structures on real-time paths.</p></div>
</div>

## See Also

- **[Performance Optimization Hub](./)** — the full optimization section index, philosophy, and learning paths
- **[CPU Optimization](./cpu-optimization.html)** — cache-aware data-oriented design, multithreading, and pooling that turn good algorithms into fast code
- **[Memory Optimization](./memory-optimization.html)** — the memory hierarchy and data layout that determine the hidden constants behind Big O
- **[Profiling Best Practices](./cpu-optimization.html#profiling-tools)** — measuring before optimizing, so you target the algorithm that actually dominates
- **[Database Design](../technology/database-design/)** — B-tree indexes, query-plan caching, and page-oriented structures in action
- **[3D Graphics & Rendering](../graphics/3d-rendering.html)** — BVHs and spatial structures driving culling and ray tracing
- **[Distributed Systems Theory](../advanced/distributed-systems-theory/)** — complexity and coordination costs at scale
</content>
</invoke>
