---
layout: docs
title: "Database Design: Distributed & NoSQL Databases"
permalink: /docs/technology/database-design/distributed-and-nosql.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Distributed &amp; NoSQL Databases</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">CAP, consensus, distributed transactions, NoSQL models, and choosing the right database</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## Scaling Beyond One Machine

Eventually a database outgrows a single server. Maybe there is too much data, too many users, or a need for geographic distribution. This is where distributed databases come in — and where the easy guarantees of a single node start to fray. Once data lives on more than one machine, three forces dominate every design decision: the **CAP** trade-off that any partition forces on you, the **consensus** protocols that let nodes agree despite failures, and the **data model** you choose for the access patterns you actually have.

This page is the **hub** for that material. It frames the trade-offs and points you at the deep dives:

- **[Replication & Consensus](replication-and-consensus.html)** — how copies of data are kept in sync, and how Raft/Paxos let a cluster agree on a leader and an ordered log even when machines crash.
- **[Distributed Transactions](distributed-transactions.html)** — making `N` independent participants all-commit-or-all-abort: 2PC/3PC, sagas, the outbox pattern, and idempotency.
- **[NoSQL Data Models](nosql-data-models.html)** — document, key-value, wide-column, graph, and time-series stores, and how to model for each.

<div class="command-grid">
  <a href="replication-and-consensus.html" class="nav-card">
    <h4><i class="fas fa-network-wired"></i> Replication &amp; Consensus</h4>
    <p>Replication topologies, streaming vs. logical replication, Raft and Paxos, read replicas and lag, failover, and quorum reads/writes.</p>
  </a>
  <a href="distributed-transactions.html" class="nav-card">
    <h4><i class="fas fa-code-branch"></i> Distributed Transactions</h4>
    <p>Two-phase commit and why it blocks, 3PC, sagas with compensation, the transactional outbox, idempotency, and exactly-once semantics.</p>
  </a>
  <a href="nosql-data-models.html" class="nav-card">
    <h4><i class="fas fa-database"></i> NoSQL Data Models</h4>
    <p>Document, key-value, wide-column, graph, and time-series stores — their core abstractions, sweet spots, and query-first modeling.</p>
  </a>
</div>

## The CAP Theorem: Pick Two

In 2000, Eric Brewer observed that distributed systems face a fundamental trade-off. You can have at most two of:

**Consistency**: Everyone sees the same data
- Example: a bank account balance is identical at all branches
- Cost: might need to wait for all nodes to agree

**Availability**: the system always responds
- Example: a shopping cart always works, even during Black Friday
- Cost: might show slightly outdated data

**Partition Tolerance**: the system survives network failures
- Example: an East-Coast datacenter loses connection to the West Coast
- Cost: must choose between C and A when the split happens

<div class="notice--warning">
  <p><strong>"Pick two" is a simplification.</strong> In any real distributed system, network partitions <em>will</em> happen, so partition tolerance (P) is not optional — you cannot trade it away. The genuine choice is therefore <strong>what to do during a partition: stay Consistent (CP) or stay Available (AP)</strong>. The "CA" corner only describes a single-node or single-datacenter system that simply isn't distributed.</p>
</div>

### CP vs. AP in Practice

The interesting decision is what a system does the moment a partition cuts it in half.

**CP — refuse to diverge.** A bank ledger or a configuration store would rather return an error than risk two halves disagreeing. During a partition, the minority side stops accepting writes (it cannot reach a majority), so no contradictory state can be committed. Better to say "ATM temporarily unavailable" than to allow an overdraft.

```python
# CP: a withdrawal must confirm against a quorum before deducting
def withdraw(account_id, amount):
    # Reaching a majority of replicas may fail during a partition
    if quorum_balance(account_id) >= amount:
        deduct_on_quorum(account_id, amount)
        return "Success"
    return "Unavailable: could not reach a quorum"
```

**AP — stay up, reconcile later.** A social feed or a shopping cart would rather serve slightly stale data than go dark. Each side keeps accepting reads and writes during the partition and reconciles the divergence afterward (last-write-wins, vector clocks, or CRDTs).

```python
# AP: always answer, even if the freshest copy is unreachable
def get_feed(user_id):
    try:
        return latest_feed(user_id)
    except NetworkPartition:
        return cached_feed(user_id)   # may be a few minutes old
```

The mechanics behind both choices — quorum tuning for AP, leader election and replicated logs for CP — are the subject of [Replication & Consensus](replication-and-consensus.html).

<div class="notice--info">
  <p>"CA" really means "not partition-tolerant." A config service whose nodes all live in one rack behaves like CA only because it assumes the network never splits. The instant it spans datacenters, the partition becomes inevitable and you must pick CP or AP.</p>
</div>

## Consensus in One Paragraph

The hard core of a CP system is **consensus**: getting a set of nodes to agree on a single value — or a single *ordered log* of values — even though some may crash and messages may be delayed or reordered. Databases use it for two jobs: **leader election** (pick exactly one leader so a partitioned-away old leader can't keep writing) and **state-machine replication** (agree on the order of writes so every replica applies them identically). Both **Raft** and **Paxos** achieve this by requiring a **majority** to act: because any two majorities of the same node set must overlap in at least one node, two conflicting decisions can never both be committed. That single overlap argument is why consensus clusters run an **odd number of nodes (3 or 5)** — and why a minority partition simply stops, rather than diverging.

| Protocol | Reputation | Used by |
|----------|------------|---------|
| **Raft** | Same guarantees as Paxos, far easier to implement and operate | etcd (Kubernetes), Consul, CockroachDB, TiKV, MongoDB elections |
| **Paxos / Multi-Paxos** | The original; notoriously subtle to get right | Google Chubby, Spanner per-shard replication |

That is the whole story at hub altitude. Leader election, log replication, terms-as-a-clock, the safety arguments, and Dynamo-style quorums all live in **[Replication & Consensus](replication-and-consensus.html)**. Making a *transaction* atomic across several such systems — a different and harder problem — lives in **[Distributed Transactions](distributed-transactions.html)**.

## When to Use What: A Decision Guide

There is no universally best database; there is only the database whose trade-offs match your workload. Use the questions below as a triage, then follow the deep dives for the details.

### Decision Tree

```
Start Here
    |
    v
Is your data naturally relational (fixed schema, rich queries, joins)?
    |                                        \
   Yes                                        No
    |                                          |
    v                                          v
Need strict ACID across the dataset?     What shape is the data / access?
    |                  \                      |  nested aggregate -> Document (MongoDB)
   Yes                  No                    |  pure key lookup   -> Key-Value (Redis, DynamoDB)
    |                    |                     |  huge write-heavy  -> Wide-Column (Cassandra)
    v                    v                     |  many-hop links    -> Graph (Neo4j, Neptune)
Scale needs?         Consider NoSQL           |  time-stamped feed -> Time-Series (TimescaleDB)
    |          \      (see the right branch)
 Single        Multi-region
 server         |
    |           v
    v       CockroachDB / Spanner (NewSQL: SQL + ACID at scale)
PostgreSQL / MySQL

Special cases:
  Full-text search -> Elasticsearch, OpenSearch
  Analytics / OLAP -> ClickHouse, Apache Druid, BigQuery
  Vector / similarity (AI) -> pgvector, Pinecone, Weaviate, Qdrant
  Embedded / edge -> SQLite, RocksDB, Cloudflare D1
```

### Database Comparison Matrix

| Database | Type | Best For | Avoid When | Scale Limit |
|----------|------|----------|------------|-------------|
| PostgreSQL | Relational | General purpose, complex queries | Petabyte scale | ~10TB comfortable |
| MySQL | Relational | Web apps, simple queries | Complex analytics | ~5TB comfortable |
| MongoDB | Document | Flexible schema, rapid development | Strong cross-document consistency | ~100TB |
| Cassandra | Wide Column | Time-series, write-heavy | Complex ad-hoc queries | Petabytes |
| Redis | Key-Value | Caching, real-time | Primary system of record | RAM size |
| Neo4j | Graph | Relationship traversal | Tabular/aggregate workloads | ~10B nodes |
| ClickHouse | Column | Analytics, aggregations | OLTP workloads | Petabytes |
| CockroachDB / Spanner | NewSQL | Global ACID at scale | Single-node simplicity is enough | Petabytes |
| SQLite | Embedded | Mobile, desktop, edge apps | Concurrent writers | ~100GB |

### Rules of Thumb

- **Start relational.** PostgreSQL or MySQL is the right default for most applications; reach for NoSQL only when a specific access pattern or scale requirement forces it.
- **Model the queries, not the data.** In most NoSQL stores there are no joins at read time — decide your access patterns first, then shape (and often duplicate) the data so each pattern is one cheap lookup. See [NoSQL Data Models](nosql-data-models.html).
- **Need ACID *and* horizontal scale?** That is the NewSQL niche (CockroachDB, Spanner) — SQL semantics on a consensus-replicated, sharded core.
- **Polyglot persistence is normal.** Large systems mix engines: Postgres for the system of record, Redis for caching, Cassandra for the firehose, Elasticsearch for search. Use each where it is strong.

## The Future of Databases

Database technology continues to evolve rapidly. A few of the directions worth watching:

### NewSQL: Best of Both Worlds

NewSQL databases provide SQL and ACID guarantees at massive scale, putting a familiar relational interface on top of the consensus and replication machinery covered in the deep dives.

**Google Spanner**: the pioneer

```sql
-- Looks like regular SQL
CREATE TABLE users (
    user_id INT64 NOT NULL,
    email STRING(255),
    created_at TIMESTAMP
) PRIMARY KEY (user_id);

-- But runs across continents:
-- synchronous replication globally,
-- external consistency via TrueTime.
```

Spanner uses atomic clocks and GPS to bound clock uncertainty globally (TrueTime), enabling externally-consistent transactions across the planet.

**CockroachDB**: Spanner for mortals

```sql
-- Familiar PostgreSQL syntax
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL,
    total DECIMAL(10,2),
    region STRING AS (CASE
        WHEN country IN ('US', 'CA') THEN 'NA'
        WHEN country IN ('GB', 'FR', 'DE') THEN 'EU'
        ELSE 'OTHER'
    END) STORED  -- computed column for partitioning
);

-- Automatically distributed, survives datacenter failures.
```

### Machine Learning in Databases

Databases are beginning to use ML to optimize themselves.

**Learned indexes**: replacing B+ trees with a model that predicts a record's position directly.

```python
# Traditional B+ tree: follow pointers
def btree_lookup(key):
    node = root
    while not node.is_leaf:
        node = node.find_child(key)
    return node.find_position(key)

# Learned index: predict position, then search a small window
def learned_lookup(key):
    predicted_pos = model.predict(key) * num_records   # learns the CDF
    lo = max(0, predicted_pos - error_bound)
    hi = min(num_records, predicted_pos + error_bound)
    return binary_search(data[lo:hi], key)
```

For some read-heavy workloads this has shown large memory savings and faster lookups, though it remains an active research area rather than a default.

**Self-tuning** systems observe the live query mix and automatically create or drop indexes, and **learned query optimizers** (often graph neural networks over the query plan) produce better cardinality estimates than static statistics — both already used in production at Microsoft and Google.

### AI-Native and New Architectures

- **Vector databases** (pgvector, Pinecone, Weaviate, Qdrant) index high-dimensional embeddings for similarity search, the storage backbone of retrieval-augmented generation.
- **Natural-language-to-SQL** (Text2SQL with LLMs) lowers the barrier to ad-hoc querying.
- **Disaggregated storage and compute** (Snowflake, Databricks) and **serverless databases** (Neon, PlanetScale, Fauna) decouple capacity from cost.
- **Edge databases** (Cloudflare D1, Fly.io, Turso) push data close to users for low latency.

## Real-World Case Studies

### Case Study 1: Instagram's Cassandra Migration

**The challenge**: PostgreSQL couldn't keep up with Instagram's growth — billions of photos, likes, and follows, plus a need for geographic distribution.

**The solution**: a wide-column schema designed around the dominant query ("show me user X's feed, newest first").

```cql
CREATE TABLE user_feed (
    user_id BIGINT,
    post_timestamp TIMESTAMP,
    post_id BIGINT,
    author_id BIGINT,
    post_data TEXT,
    PRIMARY KEY (user_id, post_timestamp, post_id)
) WITH CLUSTERING ORDER BY (post_timestamp DESC);
```

**Lessons learned**:
- NoSQL isn't always better — Instagram kept PostgreSQL for user data.
- Design the schema around query patterns, not entities.
- Denormalization is fine when you need scale.

### Case Study 2: Uber's Schemaless

**The challenge**: hundreds of microservices with different data needs, rapid development requiring schema flexibility, and strong consistency in some cases.

**The solution**: "Schemaless" — a MySQL backend exposed through a JSON-like, append-only, per-cell-versioned interface.

```json
{
  "row_key": "rider:123:profile",
  "cells": [
    { "column": "name",   "value": "Alice Smith", "version": 1234567890 },
    { "column": "rating", "value": 4.8,           "version": 1234567891 }
  ]
}
```

**Benefits**: schema changes without migrations, per-cell versioning for consistency, and MySQL's operational reliability under a NoSQL-flavored API.

### Case Study 3: Discord's Message Storage

**The challenge**: billions of messages across millions of channels, queryable by channel and time, where old messages are accessed rarely but must remain available.

**The solution**: a wide-column store partitioned by channel and time bucket, with a hot/cold tiering strategy.

```cql
CREATE TABLE messages (
    channel_id BIGINT,
    bucket INT,            -- time bucket (e.g. a day)
    message_id BIGINT,
    author_id BIGINT,
    content TEXT,
    PRIMARY KEY ((channel_id, bucket), message_id)
);
-- Cassandra, later ScyllaDB for better tail latency;
-- cold messages migrate to object storage.
```

**Architecture**: write to the wide-column store immediately; after a retention window, migrate cold data to object storage; a query router checks both tiers.

## Best Practices from the Trenches

### Design Principles That Scale

**1. Design for 10x growth**

```sql
-- Bad: works today, fails at scale
CREATE TABLE users (
    id INT PRIMARY KEY,  -- runs out near 2 billion
    email VARCHAR(50)    -- some emails are longer
);

-- Good: room to grow
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_created (created_at)
);
```

**2. Make schemas self-documenting**

```sql
-- Bad: cryptic names
CREATE TABLE usr_prch_hist (u_id INT, p_id INT, ts INT);

-- Good: clear intent
CREATE TABLE user_purchase_history (
    user_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    purchased_at TIMESTAMP NOT NULL,
    quantity INT NOT NULL DEFAULT 1,
    unit_price DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    INDEX idx_user_purchases (user_id, purchased_at DESC)
);
```

<div class="tip-card">
  <h4>Dialect note: inline index syntax</h4>
  <p>The inline <code>INDEX idx_name (cols)</code> clause inside <code>CREATE TABLE</code> above is MySQL-only and is invalid in PostgreSQL. In PostgreSQL, create the table without those clauses and add each index with a separate <code>CREATE INDEX</code> statement.</p>
</div>

**3. Plan for maintenance**

```sql
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    -- Business columns
    customer_id BIGINT NOT NULL,
    total DECIMAL(10,2) NOT NULL,

    -- Maintenance columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    version INT DEFAULT 1,            -- optimistic locking
    is_deleted BOOLEAN DEFAULT FALSE  -- soft deletes
);
```

### Common Pitfalls to Avoid

1. **N+1 queries**: load related data in one query, not one per row.
2. **Missing indexes on foreign keys**: every FK that you filter or join on should have an index.
3. **Storing calculated values**: use generated columns or views instead of duplicating derivable data.
4. **Ignoring time zones**: store UTC, convert for display.
5. **Not planning for deletes**: soft deletes are often better than hard deletes for auditability.

## Build Your Own Database

The best way to understand databases is to build one. A practical progression:

### Project 1: Key-Value Store (a weekend)

```python
class SimpleKVStore:
    def __init__(self):
        self.data = {}
        self.log = []  # for durability

    def set(self, key, value):
        self.log.append(f"SET {key} {value}")
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def snapshot(self):
        with open("snapshot.db", "w") as f:
            json.dump(self.data, f)
```

### Project 2: B+ Tree Index (1–2 weeks)

```python
class BPlusTree:
    def __init__(self, order=4):
        self.root = LeafNode()
        self.order = order

    def insert(self, key, value):
        # find the leaf, split if full, push splits up to parents
        pass

    def range_query(self, start, end):
        # find the start leaf, scan linked leaves until end
        pass
```

### Project 3: Simple SQL Engine (a month)

```python
class MiniSQL:
    def execute(self, query):
        ast = parse_sql(query)
        if ast.type == "SELECT":
            table = self.scan_table(ast.table)
            filtered = self.apply_where(table, ast.where)
            return self.project(filtered, ast.columns)
        elif ast.type == "CREATE TABLE":
            self.create_table(ast.table_name, ast.columns)
```

### Project 4: Add Transactions (a couple of months)

- Implement write-ahead logging.
- Add simple two-phase locking (2PL) for isolation.
- Build a recovery manager.
- Handle concurrent access.

Each project builds on the last, gradually introducing the complexity real databases manage.

> **Code Reference**: For working implementations of the distributed algorithms referenced here, see [`distributed_systems.py`](../../../code-examples/technology/database-design/distributed_systems.py) and [`modern_databases.py`](../../../code-examples/technology/database-design/modern_databases.py).

## See Also

<div class="see-also-card">
  <h4>Continue the deep dive</h4>
  <ul>
    <li><strong>Mechanics:</strong> <a href="replication-and-consensus.html">Replication &amp; Consensus</a> — how copies stay in sync and how Raft/Paxos agree under failure.</li>
    <li><strong>Atomicity at scale:</strong> <a href="distributed-transactions.html">Distributed Transactions</a> — 2PC/3PC, sagas, outbox, and idempotency.</li>
    <li><strong>Data models:</strong> <a href="nosql-data-models.html">NoSQL Data Models</a> — document, key-value, wide-column, graph, and time-series stores.</li>
    <li><strong>Foundations:</strong> <a href="storage-internals.html">Storage Engines &amp; Recovery</a> and <a href="transactions-and-concurrency.html">Transactions &amp; Concurrency</a> — the single-node story distribution must preserve.</li>
    <li><strong>Up:</strong> <a href="./">Database Design hub</a></li>
    <li>See also: <a href="../aws/">AWS</a> for managed and serverless database services, and <a href="../networking/">Networking</a> for the protocols behind distributed systems.</li>
  </ul>
</div>
