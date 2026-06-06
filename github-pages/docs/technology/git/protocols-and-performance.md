---
layout: docs
title: "Git Internals: Protocols, Packs & Performance"
permalink: /docs/technology/git/protocols-and-performance.html
toc: true
toc_sticky: true
---

[Git Internals](./) ›

How the object store synchronizes over the wire, the pack and index formats, and the tuning that keeps large repositories fast.

## Repository Operations at the Protocol Level

Git's network protocol enables efficient distributed repository synchronization through:

**Protocol Phases:**
1. **Reference Discovery**: Client queries available refs from server
2. **Capability Negotiation**: Client and server agree on protocol features
3. **Pack Negotiation**: Client specifies wants/haves for minimal data transfer
4. **Pack Transfer**: Server sends optimized pack file with only needed objects

**Protocol Features:**
- **Smart HTTP Protocol**: Efficient bidirectional communication
- **Pack Protocol**: Optimized object transfer format
- **Shallow Cloning**: Fetch limited history depth
- **Partial Cloning**: Fetch subset of repository objects
- **Protocol Extensions**: Side-band, progress reporting, atomic push

**Optimization Techniques:**
- Delta compression between similar objects
- Bitmap indexes for reachability queries
- Multi-pack indexes for large repositories

> **Code Reference**: For complete Git protocol implementation including clone, fetch, and push operations, see [`repository_operations.py`](../../../code-examples/technology/git/repository_operations.py)

## Distributed Repository Synchronization

### Remote Protocol Implementation

Git's distributed nature relies on efficient protocols for synchronizing repositories:

**Protocol Capabilities:**
- **multi_ack_detailed**: Optimized negotiation with multiple acknowledgments
- **side-band-64k**: Multiplexed data streams for progress and errors
- **ofs-delta**: Offset-based delta encoding for better compression
- **shallow**: Support for shallow clones and fetches
- **filter**: Partial clone with object filtering
- **atomic**: All-or-nothing push transactions

**Fetch Operation Phases:**
1. **Reference Discovery**: Query remote for available refs and capabilities
2. **Negotiation**: Find common commits to minimize transfer
3. **Pack Transfer**: Receive optimized pack with only missing objects
4. **Reference Update**: Atomically update local refs

**Push Operation Phases:**
1. **Permission Check**: Verify write access to remote
2. **Pack Building**: Create pack with objects missing on remote
3. **Atomic Transaction**: Upload pack and update refs atomically
4. **Hook Execution**: Server-side hooks for policy enforcement

**Delta Compression:**
- Rolling hash for efficient block matching
- Copy/insert instructions for minimal delta size
- Window-based search for similar objects

> **Code Reference**: For complete remote protocol implementation with pack negotiation and delta compression, see [`remote_protocol.py`](../../../code-examples/technology/git/remote_protocol.py)

## Wire Protocol

**Protocol Capabilities:**
- `multi_ack_detailed`
- `side-band-64k`
- `ofs-delta`
- `agent`
- `shallow`
- `filter`
- `atomic`

**Packfile Negotiation:**
1. Reference discovery
2. Capability negotiation
3. Want/have exchange
4. Pack transmission
5. Reference update

## Pack and Index Formats

### Pack File Format

**Structure:**
```
PACK Header (12 bytes)
├── Signature: "PACK"
├── Version: 2 or 3
└── Object count: 32-bit

Objects (variable)
├── Type and size encoding
├── Delta base (if delta)
└── Compressed data

Checksum (20 bytes)
└── SHA-1 of entire pack
```

**Object Types:**
- OBJ_COMMIT (1)
- OBJ_TREE (2)
- OBJ_BLOB (3)
- OBJ_TAG (4)
- OBJ_OFS_DELTA (6)
- OBJ_REF_DELTA (7)

### Index File Format

The on-disk `.git/index` (the staging area) records one entry per tracked path. Its version-2 layout:

```
DIRC Header
├── Signature: "DIRC"
├── Version: 2
└── Entry count

Index entries
├── ctime, mtime
├── dev, ino, mode
├── uid, gid, size
├── SHA-1
├── Flags
└── Path (null-terminated)

Extensions (optional)
├── Tree cache
├── Resolve undo
└── Split index

Checksum
```

> See [Object Model &amp; Storage](object-model.html#index-staging-area-structure) for the C-struct view of these index entries and how the cached `stat` data accelerates `git status`.

## Performance Optimization

### Git Performance Analysis

Git provides sophisticated tools and algorithms for optimizing repository performance:

**Performance Metrics:**
- **Object Database**: Loose objects vs pack files ratio
- **Pack Efficiency**: Compression ratios and delta chain depths
- **Reference Performance**: Loose vs packed refs
- **Working Tree**: Large files and untracked content
- **Network Operations**: Pack negotiation efficiency

**Optimization Strategies:**

**Geometric Repacking:**
- Groups pack files by size in geometric progression
- Each pack ~2x size of previous
- Reduces pack file count while maintaining efficiency
- Optimal for large repositories with many packs

**Bitmap Indexes:**
- EWAH compressed bitmaps for reachability queries
- Dramatically speeds up counting and traversal operations
- Optimal commit selection for coverage
- Used in fetch/push negotiations

**Other Optimization Techniques:**
- Multi-pack indexes for very large repositories
- Commit graphs for faster traversal
- Bloom filters for changed paths
- Delta islands for fork networks

**Performance Tuning Parameters:**
- `pack.window`: Delta search window size
- `pack.depth`: Maximum delta chain length
- `pack.threads`: Parallel compression threads
- `core.preloadIndex`: Parallel index loading

> **Code Reference**: For complete performance analysis and optimization implementations, see [`performance_optimization.py`](../../../code-examples/technology/git/performance_optimization.py)

### Repository Maintenance

```bash
# Regular maintenance
git maintenance start          # Enable automatic maintenance
git maintenance run           # Run maintenance tasks
git maintenance stop          # Disable automatic maintenance

# Manual optimization
git gc --aggressive --prune=now
git repack -a -d -f --depth=50 --window=100
git prune --expire=now

# Performance diagnostics
git count-objects -vH
git rev-list --all --objects | wc -l
du -sh .git/objects/pack/
```

### Large Repository Strategies

```bash
# Partial clone
git clone --filter=blob:none <url>     # Omit blobs
git clone --filter=tree:0 <url>        # Omit trees
git clone --filter=blob:limit=1m <url> # Omit large blobs

# Sparse checkout
git sparse-checkout init --cone
git sparse-checkout set <directory>
git sparse-checkout add <pattern>
git sparse-checkout list

# Shallow operations
git fetch --depth=1
git pull --depth=1
git fetch --unshallow              # Convert to full
```

---

**Previous:** [← Object Model & Storage](object-model.html). **Next:** [Algorithms & Advanced Operations →](algorithms-and-operations.html) — three-way merge, merge strategies, rebase and bisect, and advanced day-to-day operations.

## See Also

- [Object Model &amp; Storage](object-model.html) — the objects and on-disk layout these protocols transfer.
- [Algorithms &amp; Advanced Operations](algorithms-and-operations.html) — the merge, rebase, and bisect algorithms.
- [Git Command Reference](../git-reference.html) — clone, fetch, push, gc, and maintenance command syntax.
