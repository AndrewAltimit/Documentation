---
layout: docs
title: "Database Design: Replication & Consensus"
permalink: /docs/technology/database-design/replication-and-consensus.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">&larr; Database Design</a></p>

# Replication & Consensus

## Why Replicate at All?

A single database server is a single point of failure and a single ceiling on throughput. **Replication** copies the same data onto multiple machines so the system can:

- **Survive failures** — if the primary dies, a replica takes over (high availability).
- **Scale reads** — route read-only queries to replicas instead of overloading the primary.
- **Reduce latency** — place replicas near users in other regions.
- **Take backups without downtime** — back up a replica instead of the live primary.

The catch is that the moment you have more than one copy, you have to answer a hard question: *what does it mean for the copies to "agree"?* That question is the whole subject of this page. We start with the mechanics of moving data between nodes (replication topologies and the wire formats databases use), then build up to **consensus** — the protocols that let a cluster agree even when machines crash and networks split.

## Replication Topologies

A topology describes *who copies from whom*. The choice determines your consistency model, your write throughput, and how painful failover is.

### Single-Leader (Primary/Replica)

The most common arrangement. One node is the **leader** (a.k.a. primary, master); all writes go to it. The leader streams its changes to one or more **followers** (replicas, standbys), which apply them in the same order.

```
            writes
              |
              v
        +-----------+
        |  Leader   |
        +-----------+
         /    |    \      replication stream
        v     v     v
    +------++------++------+
    |Follow||Follow||Follow|   <-- serve reads only
    +------++------++------+
```

**Pros:** writes have a single ordering authority, so conflicts never arise. Simple to reason about.
**Cons:** the leader is a write bottleneck and a failover point. Followers lag behind (see [Read Replicas & Replication Lag](#read-replicas--replication-lag)).

Used by PostgreSQL, MySQL, MongoDB replica sets, and most managed cloud databases (Amazon RDS, Cloud SQL).

### Multi-Leader (Active/Active)

Multiple nodes accept writes, typically one leader per region. Each leader replicates its writes to the others.

```
   Region A                 Region B
  +--------+   bidirectional  +--------+
  | Leader |<---------------->| Leader |
  +--------+   replication    +--------+
```

**Pros:** writes are local in each region (low latency), and the system survives a whole region going offline.
**Cons:** two regions can write to the same row concurrently, producing a **write conflict** that must be resolved (last-write-wins, CRDTs, or application logic). Multi-leader is powerful but operationally subtle — most teams avoid it unless geography forces their hand.

### Leaderless (Quorum)

There is no leader at all. A client (or a coordinator on its behalf) writes to *several* replicas directly and reads from *several* replicas, using quorums to paper over the fact that any individual node might be stale or down. This is the Dynamo-style model used by Cassandra, ScyllaDB, and Riak — covered in detail under [Quorum Reads & Writes](#quorum-reads--writes).

| Topology | Writes go to | Conflict handling | Typical systems |
|----------|--------------|-------------------|-----------------|
| Single-leader | One node | None (single ordering) | PostgreSQL, MySQL, RDS |
| Multi-leader | Several nodes | Required (LWW/CRDT/app) | BDR, CouchDB, multi-region MySQL |
| Leaderless | A quorum of nodes | Read repair + LWW/versions | Cassandra, Riak, DynamoDB |

## Streaming vs. Logical Replication

Once you pick single-leader, the leader still has to *describe* its changes to followers. There are two fundamentally different wire formats, and the distinction shapes what you can do with the stream.

### Physical (Streaming) Replication

The leader ships the **write-ahead log (WAL)** — the same byte-level record of physical page changes it writes for crash recovery (see [Storage Engines & Recovery](storage-internals.html)). Followers replay those WAL records verbatim, reproducing the leader's data files block-for-block.

```
Leader WAL:   "page 4217, offset 96, write 64 bytes: <bytes>"
                      |
                      v  (TCP stream)
Follower:     apply identical byte change to its own page 4217
```

**Properties:**

- **Fast and low-overhead** — no parsing or re-planning, just byte replay.
- **Exact replica** — followers are physically identical, so a follower can be promoted to leader instantly.
- **All-or-nothing** — you replicate the *entire* cluster; you cannot copy just one table.
- **Version-locked** — leader and follower usually must run the same major version and architecture, because WAL is an internal format.

This is PostgreSQL **streaming replication** (the `walsender`/`walreceiver` pair) and MySQL's row-based binlog replication when treated as a physical stream. It is the backbone of read replicas and hot standbys.

```sql
-- PostgreSQL: a standby connects and streams WAL from the primary.
-- On the standby (postgresql.conf / connection):
--   primary_conninfo = 'host=primary port=5432 user=replicator'
-- The standby stays in continuous recovery, replaying WAL as it arrives.

-- Inspect replication state on the primary:
SELECT client_addr, state, sent_lsn, replay_lsn,
       pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replay_lag_bytes
FROM pg_stat_replication;
```

### Logical Replication

Instead of physical bytes, the leader decodes the WAL into a stream of **logical change events**: "INSERT into orders (id=42, total=19.99)", "UPDATE users SET email=… WHERE id=7". Followers apply these as ordinary statements.

```
Leader change stream:  INSERT orders (id=42, total=19.99)
                       UPDATE users SET email='x' WHERE id=7
                              |
                              v
Subscriber:            executes equivalent SQL against its own tables
```

**Properties:**

- **Selective** — replicate specific tables or rows (a *publication*), not the whole cluster.
- **Cross-version / cross-engine** — the subscriber only needs to understand the logical rows, so you can replicate Postgres 14 → 16, or feed the stream into Kafka, Elasticsearch, or a data warehouse.
- **Bidirectional-capable** — the building block for multi-leader and zero-downtime upgrades.
- **More overhead** — decoding and re-executing is costlier than byte replay, and large transactions/DDL need care.

This is **Change Data Capture (CDC)**: tools like Debezium tail the logical stream to publish every row change to downstream systems.

```sql
-- PostgreSQL logical replication: publish a subset on the source...
CREATE PUBLICATION orders_pub FOR TABLE orders, order_items;

-- ...and subscribe on the destination (possibly a different major version):
CREATE SUBSCRIPTION orders_sub
  CONNECTION 'host=source dbname=shop user=repl'
  PUBLICATION orders_pub;
```

> **Rule of thumb.** Use **physical/streaming** replication for high-availability standbys and read replicas of an entire database — it is fastest and yields a byte-identical, instantly-promotable copy. Reach for **logical** replication when you need to replicate a *subset*, cross versions/engines, perform a zero-downtime major upgrade, or feed a CDC pipeline.

## Read Replicas & Replication Lag

The biggest payoff of single-leader replication is **read scaling**: send writes to the leader and fan reads out to followers.

```python
# Route queries by intent
def get_connection(query_is_write: bool):
    if query_is_write:
        return primary_pool.get()        # all writes -> leader
    return random.choice(replica_pools)  # reads -> any follower
```

But replicas are always a little behind. **Replication lag** is the time (or WAL bytes) between a write committing on the leader and that same write becoming visible on a follower. Under steady load it might be a few milliseconds; under a write spike, a long-running query, or a slow network it can balloon to seconds or minutes.

Lag produces the classic anomaly of asynchronous replication, **read-your-own-writes** violations:

```
1. User updates their profile photo   -> committed on leader
2. Page reloads, reads from a replica  -> replica hasn't applied it yet
3. User sees the OLD photo and panics
```

### Strategies to Tame Lag

- **Read-your-writes via stickiness** — for a short window after a user writes, route *that user's* reads to the leader (or to a replica known to be caught up). Cheap and effective for "I just changed it" UX.
- **Monotonic reads** — pin a user's session to a single replica so they never see time go backwards by bouncing between a fresh and a stale node.
- **Bounded-staleness reads** — only use a replica whose lag is under a threshold; databases expose lag (`pg_stat_replication.replay_lag`, `SHOW REPLICA STATUS \G` → `Seconds_Behind_Source`).
- **Synchronous replication for critical writes** — make the leader wait for at least one follower to acknowledge before reporting commit. Eliminates lag for that data at the cost of write latency.

### Synchronous vs. Asynchronous Replication

This is the central durability/latency dial:

| Mode | Leader waits for follower? | On leader failure | Write latency |
|------|----------------------------|-------------------|---------------|
| **Asynchronous** | No — commit returns immediately | Recently-acked writes can be lost | Lowest |
| **Synchronous** | Yes — at least one follower acks | No acknowledged write is lost | Higher (network round-trip) |
| **Semi-sync / quorum** | Waits for *k* of *n* followers | Tunable durability | In between |

Pure synchronous replication to *all* followers is fragile: one slow follower stalls every write. The practical compromise is **semi-synchronous** (PostgreSQL `synchronous_standby_names` with `ANY 1 (...)`, MySQL semi-sync) — block until *one* standby confirms, so you can lose the leader without losing data, but a single straggler can't freeze the cluster.

## Consensus: Getting Distributed Nodes to Agree

Replication mechanics tell you *how* bytes move; they don't tell you how the cluster decides **who the leader is** when machines crash and networks split. That decision is the job of a **consensus** protocol. Consensus lets a set of nodes agree on a single value (or a single ordered log of values) even though some nodes may fail and messages may be delayed or reordered — as long as a *majority* are alive and can talk to each other.

Databases use consensus for two closely related jobs:

1. **Leader election** — pick exactly one leader, and make sure a partitioned-away old leader can't keep accepting writes (avoiding "split-brain").
2. **Replicated log / state machine replication** — agree on the *order* of writes so every replica applies them identically.

### Raft: Consensus Made Understandable

Raft deliberately decomposes the problem into pieces a human can hold in their head: leader election, log replication, and safety.

**The Leader Election Analogy**
Imagine a group project where you need a coordinator:

1. **Everyone starts as a follower** — waiting for a leader.
2. **If no leader speaks up** — someone volunteers (becomes a *candidate*).
3. **Candidates request votes** — "I'll be leader, okay?"
4. **Majority wins** — becomes leader; others go back to following.
5. **Leader sends heartbeats** — "Still here, still in charge!"

**How It Handles Failures:**

```python
# Simplified Raft leader election
class RaftNode:
    def __init__(self):
        self.state = "follower"
        self.term = 0
        self.voted_for = None

    def election_timeout(self):
        # No heartbeat from leader? Start election!
        self.state = "candidate"
        self.term += 1
        self.voted_for = self.id

        votes = 1  # Vote for self
        for node in other_nodes:
            if node.request_vote(self.term, self.id):
                votes += 1

        if votes > len(all_nodes) / 2:
            self.state = "leader"
            self.send_heartbeats()  # Tell everyone I'm leader
```

**Why This Works:**

- Only one leader per term (majority vote — two leaders would each need a majority, and two majorities of the same set must overlap, so they can't both win).
- Split votes resolved by randomized election timeouts (the next timeout to fire usually wins uncontested).
- Old leaders step down when they see a higher term number.
- All changes go through the leader, which appends them to a replicated log; an entry is **committed** once a majority of followers have stored it. This single funnel is what makes consistency tractable.

**Terms as a logical clock.** Each election bumps a monotonically increasing *term* number. Terms let nodes detect stale leaders instantly: any message carrying a smaller term is rejected, and any node seeing a larger term reverts to follower. This is how a partitioned old leader is neutralized — when the partition heals, its writes carry an outdated term and are refused, so it cannot corrupt the committed log.

Raft is the engine inside **etcd** (and therefore Kubernetes' control plane), **Consul**, **TiKV/TiDB**, **CockroachDB** (one Raft group *per data range*), and **MongoDB**'s replica-set election protocol.

### Paxos: The Original (Complex) Solution

Paxos solves the same problem but is notoriously hard to understand — Leslie Lamport originally presented it through an analogy of part-time legislators on the fictional Greek island of Paxos. The key insight is a **two-phase** structure that guarantees safety even when proposers fail or compete:

- **Phase 1 — Prepare:** a proposer picks a ballot number *n* and asks a majority of *acceptors* to promise not to accept anything numbered below *n*. Acceptors reply with the highest-numbered value they have already accepted (if any).
- **Phase 2 — Accept:** the proposer then asks the majority to accept value *v* at ballot *n* — where *v* must be the value from the highest-numbered prior acceptance it heard about, if one exists. This rule is what prevents two different values from ever being chosen.

A value is **chosen** once a majority of acceptors have accepted it at the same ballot. Because any two majorities overlap in at least one acceptor, no two conflicting values can both be chosen — the same overlap argument that makes Raft safe.

Basic ("single-decree") Paxos agrees on *one* value. Real systems need an ordered *log* of values, so they use **Multi-Paxos**, which elects a stable leader to skip Phase 1 on the common path — at which point it looks a lot like Raft, just harder to implement correctly. Google's Chubby lock service and Spanner's per-shard replication use Paxos variants.

<div class="notice--info">
  <p><strong>Raft vs. Paxos in practice.</strong> They provide the same guarantees and the same fault tolerance (survive failure of a minority, i.e. up to <code>floor((n-1)/2)</code> of <code>n</code> nodes). Raft "won" in industry not because it is more capable but because its prescriptive, single-leader design is dramatically easier to implement, test, and operate. Most new systems reach for Raft; Paxos endures in older Google infrastructure and the literature.</p>
</div>

### Why an Odd Number of Nodes?

Consensus needs a **majority (quorum) = floor(n/2) + 1**. The fault tolerance — how many nodes can fail while a majority survives — is what matters:

| Cluster size *n* | Majority needed | Failures tolerated |
|------------------|-----------------|--------------------|
| 3 | 2 | 1 |
| 4 | 3 | 1 |
| 5 | 3 | 2 |
| 6 | 4 | 2 |

Notice that going from 3→4 or 5→6 adds a node but *not* fault tolerance, while making quorum harder to reach (more nodes must agree, slowing writes). That is why consensus clusters almost always run **3 or 5 nodes** — odd sizes give the best failure-tolerance-per-node.

## Failover & High Availability

Consensus solves leader election *inside* the database. But a complete HA story has more moving parts: detecting that the leader is gone, promoting a replacement, and redirecting clients.

### The Failover Sequence

```
1. Detect    Health checks / heartbeats stop arriving from the leader.
2. Decide    Don't act on one missed beat (a GC pause looks like death).
             Confirm via timeout + (ideally) a majority's agreement.
3. Promote   Choose the most up-to-date replica; tell it to become leader.
4. Reconfigure  Point clients/proxy at the new leader; demote the old one.
5. Recover   When the old leader returns, it rejoins as a follower
             (it must NOT keep serving writes -> split-brain).
```

### Automatic vs. Manual Failover

- **Automatic** (etcd/Raft elections, Patroni for PostgreSQL, Orchestrator/Group Replication for MySQL, MongoDB replica sets) — fast recovery, no human in the loop, but risks acting on a false alarm.
- **Manual** — an operator confirms and triggers promotion. Slower, but avoids needless failovers during transient blips.

### The Two Failure Modes to Fear

- **Split-brain** — two nodes both believe they are leader and both accept writes, silently diverging. Consensus prevents this *for committed data* by requiring a majority; the old leader, stuck in a minority partition, cannot reach quorum and so cannot commit. Systems without true consensus (e.g. naive primary/replica with an external failover script) must add **fencing** — actively shutting down or STONITH-ing the old leader — to be safe.
- **Lost writes on failover** — with *asynchronous* replication, a leader can ack writes that no follower has yet received; if it then dies, those writes vanish when a behind replica is promoted. Synchronous/semi-sync replication trades latency to close this gap.

```python
# A health check that resists false positives:
def leader_is_dead(leader, *, misses_required=3, interval_s=1):
    consecutive_misses = 0
    for _ in range(misses_required):
        if leader.heartbeat(timeout=interval_s):
            return False          # any success -> still alive
        consecutive_misses += 1
    # Require a majority of MONITORS to agree before promoting,
    # so a single observer's network blip can't trigger failover.
    return consecutive_misses >= misses_required and quorum_of_monitors_agree()
```

## Quorum Reads & Writes

Leaderless (Dynamo-style) systems skip elections entirely and lean on **quorums** to deliver consistency on demand. The idea: with *N* replicas of each piece of data, a write must be acknowledged by *W* replicas and a read must consult *R* replicas.

The magic inequality is:

$$
W + R > N
$$

When this holds, the set of nodes a write touched and the set a read touches are guaranteed to **overlap in at least one node** — so every read sees at least one replica carrying the latest write. (It's the same majority-overlap argument that makes Raft and Paxos safe, exposed as a tunable knob.) Each stored value carries a version (timestamp or vector clock), so the reader can pick the newest among the replicas it contacted.

```python
# Dynamo-style tunable quorum
N = 3   # replicas per key
W = 2   # acks required for a write to succeed
R = 2   # replicas consulted for a read
# W + R = 4 > N = 3  -> read and write sets always overlap

def write(key, value, version):
    acks = 0
    for replica in replicas_for(key):              # N replicas
        if replica.put(key, value, version):
            acks += 1
        if acks >= W:                              # quorum reached
            return "OK"
    return "FAILED: quorum not met"

def read(key):
    responses = []
    for replica in replicas_for(key):
        responses.append(replica.get(key))         # value + version
        if len(responses) >= R:
            break
    # Newest version wins; stale replicas get fixed (read repair)
    return max(responses, key=lambda r: r.version)
```

### Tuning the Knobs

| Goal | Setting | Effect |
|------|---------|--------|
| Strong-ish consistency | `W + R > N` | Reads always see latest acked write |
| Fast writes, durable | `W=1` (small) | Cheap writes, but reads may miss them unless `R` large |
| Fast reads | `R=1` | One replica answers; may be stale |
| Survive a node loss on write | `W <= N-1` | Don't require the down node |

Cassandra exposes these as per-query *consistency levels* (`ONE`, `QUORUM`, `ALL`, `LOCAL_QUORUM`); DynamoDB offers "eventually consistent" (`R=1`) vs. "strongly consistent" reads. `QUORUM` on both reads and writes (W=R=⌈(N+1)/2⌉) satisfies the overlap rule.

### The Catch: Quorums Aren't Quite Linearizable

`W + R > N` guarantees a read overlaps the latest *successful* write, but it does **not** by itself give you the strong, single-copy illusion (linearizability) that Raft/Paxos provide:

- **Sloppy quorums / hinted handoff** — to stay available during a partition, a write may be accepted by *any* W reachable nodes (not the "home" replicas). Now the overlap guarantee with later reads can break.
- **Concurrent writes** — two clients writing the same key at once produce sibling versions that last-write-wins may silently drop. Vector clocks detect the conflict but the application must resolve it.
- **Partial writes** — a write that reaches some but not W replicas is neither rolled back nor completed; later reads may or may not see it.

**Read repair** and **anti-entropy** (background Merkle-tree comparison between replicas) heal the resulting divergence over time, which is why these systems are described as *eventually consistent*. If you need true linearizable guarantees, use a consensus-backed system (Raft/Paxos, Spanner, CockroachDB) rather than tuning quorums.

<div class="notice--info">
  <p>Quorum replication sits at the <strong>AP</strong> end of the <a href="distributed-and-nosql.html#the-cap-theorem-pick-two">CAP</a> spectrum (available, eventually consistent), while consensus-replicated systems sit at the <strong>CP</strong> end (consistent, refusing writes when they can't reach a majority). Choosing between them is choosing what to do during a network partition.</p>
</div>

> **Code Reference:** For working implementations of leader election, quorum I/O, and the other algorithms here, see [`distributed_systems.py`](../../../code-examples/technology/database-design/distributed_systems.py).

## See Also

- **Closely related:** [Distributed Databases & NoSQL](distributed-and-nosql.html) — CAP, distributed transactions (2PC/Saga), and the NoSQL data models these techniques underpin.
- **Foundations:** [Storage Engines & Recovery](storage-internals.html) — the write-ahead log that physical replication streams.
- **Correctness:** [Transactions & Concurrency](transactions-and-concurrency.html) — ACID and isolation, the single-node story replication must preserve.
- **Up:** [Database Design hub](./)
- See also: [AWS](../aws/) for managed replication (RDS read replicas, Multi-AZ failover) and [Networking](../networking/) for the protocols beneath these clusters.
