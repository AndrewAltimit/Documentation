---
layout: docs
title: "Database Design: NoSQL Data Models"
permalink: /docs/technology/database-design/nosql-data-models.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">NoSQL Data Models</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Document, key-value, wide-column, graph, and time-series stores — and how to model for each</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## NoSQL: When Relational Isn't the Right Fit

Not all data fits neatly into tables. NoSQL databases emerged to handle specific use cases where the relational model struggles — variable schemas, extreme write volume, deeply connected data, or access patterns dominated by a single key. "NoSQL" is an umbrella for several genuinely different data models, each optimized around a particular shape of data and a particular set of queries.

The single most important idea in NoSQL data modeling is a reversal of the relational habit. In a relational database you model the *data* (normalize to remove redundancy) and trust the query planner to assemble whatever queries arrive later via joins. In most NoSQL stores there are **no joins at query time**, so you model the *queries* instead: you decide the access patterns first, then shape the data — often duplicating it — so each pattern is satisfied by a single, cheap lookup.

<div class="notice--info">
  <p><strong>Query-first, not schema-first.</strong> Before choosing a NoSQL store, write down the exact reads and writes your application must serve and how often each occurs. The right model is the one that turns your most frequent access pattern into a single-partition lookup. Pick the model to fit the access pattern, not by hype.</p>
</div>

The five families covered below are:

| Model | Core abstraction | Sweet spot |
|-------|------------------|------------|
| **Document** | Self-contained JSON-like documents | Variable schemas, nested aggregates, rapid development |
| **Key-value** | Opaque value behind a single key | Caching, sessions, leaderboards, sub-millisecond lookups |
| **Wide-column** | Rows of sparse, partitioned columns | Write-heavy time-series, huge tables, predictable queries |
| **Graph** | Nodes and relationships as first-class objects | Many-hop relationship traversal, recommendations, fraud |
| **Time-series** | Append-only measurements stamped in time | Metrics, IoT telemetry, monitoring, sensor feeds |

## Document Stores: Natural for Nested Data

A document store keeps each record as a self-contained, hierarchical document (JSON, BSON, or XML). Related sub-objects and arrays live *inside* the document rather than in separate joined tables, so a single read returns the whole aggregate.

**When to Use**: Variable schemas, nested data, rapid development

**MongoDB Example - Product Catalog**:
```javascript
// Products have wildly different attributes
db.products.insertOne({
    name: "Gaming Laptop",
    price: 1299,
    specs: {
        cpu: "Intel i7",
        ram: "16GB",
        gpu: "RTX 3060",
        display: {
            size: "15.6 inches",
            resolution: "1920x1080",
            refresh_rate: "144Hz"
        }
    },
    reviews: [
        {user: "gamer123", rating: 5, text: "Runs everything!"},
        {user: "techie99", rating: 4, text: "Great but runs hot"}
    ]
});

// Query nested fields naturally
db.products.find({
    "specs.ram": "16GB",
    "specs.display.refresh_rate": "144Hz"
});
```

In SQL, this would require multiple tables and joins!

### Modeling pattern: embed vs. reference

The central modeling decision in a document store is whether related data should be **embedded** in the parent document or **referenced** by storing its id and reading it separately.

```javascript
// EMBED: one read returns everything. Good when the child is owned by,
// and always read with, the parent (reviews belong to a product).
{
    _id: "prod_42",
    name: "Gaming Laptop",
    reviews: [ { user: "gamer123", rating: 5 } ]   // child lives inside parent
}

// REFERENCE: store the id, fetch separately. Good when the child is large,
// shared, or grows without bound (an author referenced by many books).
{
    _id: "book_7",
    title: "Database Internals",
    author_id: "auth_99"        // resolve with a second query / $lookup
}
```

Rules of thumb:

- **Embed** when the relationship is *contains* / one-to-few, the embedded data is read together with the parent, and it is updated as a unit. Embedding gives you atomic single-document writes and the fastest reads.
- **Reference** when the relationship is many-to-many, when the child is shared across parents (so duplication would be expensive to keep consistent), or when an embedded array would grow **unboundedly** — documents have a hard size cap (16 MB in MongoDB), and an ever-growing array eventually breaks that and hurts write performance.
- **Bucketing** is a hybrid for high-volume one-to-many data: instead of one document per event, store a fixed-size *bucket* of events (e.g. one document per sensor per hour holding up to 200 readings). This caps document growth while keeping reads aggregated.

### Schema flexibility is a contract, not an excuse

"Schemaless" does not mean structureless. Because the database no longer enforces a schema, your application code must. Use **schema validation** (MongoDB's `$jsonSchema`, for example) to pin down required fields and types, and version your documents (a `schema_version` field) so old and new shapes can coexist while you migrate. The flexibility is real, but unmanaged heterogeneity becomes an analytics and bug-hunting liability.

**Representative engines**: MongoDB, Couchbase, Amazon DocumentDB, RavenDB, Firestore.

## Key-Value Stores: Speed Above All

A key-value store is the simplest model: a giant, distributed hash map. You hand it a key, it hands back an opaque value (a string, blob, or — in richer stores like Redis — a typed data structure). There is no query language over the *contents* of the value; access is by key.

**When to Use**: Caching, sessions, real-time features

**Redis Example - Gaming Leaderboard**:
```bash
# Update player score (atomic operation)
ZINCRBY game:leaderboard 100 "player:alice"

# Get top 10 players instantly
ZREVRANGE game:leaderboard 0 9 WITHSCORES

# Cache expensive database query
SET cache:top_products '[{"id":1,"name":"Laptop"}...]' EX 300
```

Millions of operations per second, sub-millisecond latency!

### Modeling pattern: structured keys are your schema

Because you can only retrieve by key, the **key namespace is the data model**. You design a consistent, hierarchical key convention so that every access pattern maps to a deterministic key:

```bash
user:1234:profile          # a hash of profile fields
user:1234:sessions         # a set of active session ids
session:abc123  ->  {...}   # value carries a TTL so it self-expires
cart:1234                   # a hash: product_id -> quantity
game:leaderboard           # a sorted set: member -> score
```

Key design principles:

- **Encode the access pattern in the key.** If you need "all sessions for a user," do not scan — keep a `user:{id}:sessions` set and maintain it on write. There is no secondary index; you build one by hand as another key.
- **Pick the value type to fit the operation.** Redis offers strings, hashes, lists, sets, sorted sets, streams, and HyperLogLogs. A leaderboard is a *sorted set* (ranges and ranks are O(log n)); a cart is a *hash* (field-level updates without rewriting the whole value); a deduplicated counter is a *HyperLogLog*.
- **Use TTLs for ephemeral data.** Sessions, caches, and rate-limit counters should expire automatically (`EX`/`EXPIRE`) rather than relying on a sweep job.
- **Beware large values and hot keys.** A single multi-megabyte value or a key every request touches becomes a hotspot. Split large aggregates across multiple keys and shard hot counters.

### When key-value is the wrong primary store

Key-value shines as a cache, session store, or real-time scratchpad. It is a poor *system of record* for data you need to query by attribute ("all orders over $100"), because there is no way to ask that without scanning every key or maintaining every index yourself.

**Representative engines**: Redis, Memcached, Amazon DynamoDB (in its pure KV mode), Riak, etcd, RocksDB.

## Wide-Column Stores: Big, Sparse, Write-Heavy Tables

A wide-column (column-family) store looks tabular but behaves very differently from a relational table. Data is organized by a **partition key** that decides which node owns the row, and within a partition rows are physically sorted by **clustering columns**. Each row can have a different, sparse set of columns, and the engine is optimized for very high write throughput and queries that hit a single partition.

**When to Use**: Time-series data, write-heavy workloads, analytics with known access paths

**Cassandra Example - IoT Sensor Data**:
```cql
-- Optimized for time-series queries
CREATE TABLE sensor_data (
    sensor_id UUID,
    timestamp TIMESTAMP,
    temperature DOUBLE,
    humidity DOUBLE,
    PRIMARY KEY (sensor_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Fast writes from millions of sensors
INSERT INTO sensor_data (sensor_id, timestamp, temperature, humidity)
VALUES (123e4567-e89b-12d3-a456-426614174000, now(), 22.5, 45.2);

-- Efficient time-range queries
SELECT * FROM sensor_data
WHERE sensor_id = 123e4567-e89b-12d3-a456-426614174000
AND timestamp > '2024-01-01' AND timestamp < '2024-01-02';
```

### Modeling pattern: design the table per query

In Cassandra and its kin you **model one table per query**, and you do it backwards from the query:

1. The `WHERE` clause must restrict on the full partition key (so the coordinator knows exactly which node holds the data).
2. The clustering columns define the sort order and the only range predicates you may apply.
3. If you have a second access pattern, you create a **second table** holding the same data shaped differently, and you write to both on every insert.

```cql
-- Query A: "messages in a channel, newest first" -> partition by channel
CREATE TABLE messages_by_channel (
    channel_id BIGINT,
    bucket INT,            -- time bucket caps partition size (e.g. one day)
    message_id TIMEUUID,
    author_id BIGINT,
    content TEXT,
    PRIMARY KEY ((channel_id, bucket), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);

-- Query B: "every message a user wrote" -> a SECOND, duplicated table
CREATE TABLE messages_by_author (
    author_id BIGINT,
    message_id TIMEUUID,
    channel_id BIGINT,
    content TEXT,
    PRIMARY KEY (author_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```

Key constraints and patterns:

- **Denormalization and duplication are the norm**, not a code smell. There are no joins; you trade storage for single-partition reads.
- **Bound your partition size.** An unbounded partition (e.g. every message in a busy channel forever) creates a "wide row" that destroys read and repair performance. Introduce a **bucket** (day, week, or hashed sub-key) into the partition key so each partition stays a few MB / tens of thousands of rows.
- **Avoid hotspots with good partition keys.** A monotonic key (a raw timestamp, an auto-incrementing id) funnels all current writes to one node. Prefix or hash it so writes spread across the cluster.
- **No ad-hoc queries.** You cannot filter on a non-key column without `ALLOW FILTERING` (a full scan) or a secondary index (which is a partition-local lookup, not a global one). If you might need it, model a table for it up front.

### Wide-column vs. the columnar analytics engines

Beware a naming collision. *Wide-column* stores like Cassandra and HBase are **row-oriented** OLTP-style stores partitioned by key. *Columnar* analytics engines like ClickHouse, Apache Druid, and Parquet-backed warehouses physically store each column separately to make aggregations over billions of rows fast — a different design for a different (OLAP) job. Cassandra answers "give me this sensor's last hour"; ClickHouse answers "average temperature across all sensors this year."

**Representative engines**: Apache Cassandra, ScyllaDB, Apache HBase, Google Bigtable, Amazon Keyspaces.

## Graph Databases: It's All About Relationships

A graph database makes **relationships first-class citizens**. Data is stored as *nodes* (entities) connected by *edges* (relationships), each able to carry properties. Unlike a relational join — which computes the relationship at query time by matching keys — a graph database stores the connection directly (often called *index-free adjacency*), so traversing from one node to its neighbors is a constant-time pointer hop regardless of total dataset size.

**When to Use**: Social networks, recommendations, fraud detection — anywhere queries follow many hops of relationships.

**Neo4j Example - Friend Recommendations**:
```cypher
// Find friends of friends who aren't already friends
MATCH (me:Person {name: 'Alice'})-[:FRIENDS_WITH]->(friend)
      -[:FRIENDS_WITH]->(foaf:Person)
WHERE NOT (me)-[:FRIENDS_WITH]-(foaf) AND me <> foaf
RETURN foaf.name, COUNT(*) as mutual_friends
ORDER BY mutual_friends DESC
LIMIT 10;
```

Try writing this in SQL — it's a recursive nightmare!

### Why graphs win at deep traversal

The advantage is asymptotic. In a relational schema, "friends of friends of friends" is a self-join of the `friendships` table three times; each join scans or index-probes a table whose size grows with the *whole* network, so cost climbs steeply with both depth and dataset size. In a graph, each hop visits only the *local* neighborhood of the current node, so a fixed-depth traversal costs roughly the same whether the graph has a thousand nodes or a billion. The deeper the traversal, the larger the win.

### Modeling pattern: nodes, edges, and edge direction

```cypher
// Model nouns as nodes (with a label) and verbs as relationships.
// Put attributes that describe the RELATIONSHIP on the edge itself.
CREATE (alice:Person {name: 'Alice'})
CREATE (acme:Company {name: 'Acme'})
CREATE (alice)-[:WORKS_AT {since: 2021, role: 'Engineer'}]->(acme)
```

Modeling guidance:

- **Nodes are entities, edges are relationships, and edge *direction* and *type* matter.** `(:User)-[:FOLLOWS]->(:User)` is directional; `[:FRIENDS_WITH]` is usually queried as undirected. Choose deliberately and stay consistent.
- **Push relationship attributes onto the edge.** A `RATED` edge from a user to a movie should carry the `stars` and `at` properties; this keeps nodes clean and lets you filter on the relationship.
- **Reify rich relationships into nodes** when a relationship itself has structure or its own relationships — e.g. model an `Order` as a node connecting a `Customer` and many `Product`s, rather than as a single edge, when the order has line items, a status, and a timestamp.
- **Use distinct labels and indexes for entry points.** Traversal is cheap once you are *in* the graph; finding the starting node (`MATCH (me:Person {name:'Alice'})`) still needs an index on that property.

### Property graph vs. RDF

Two graph models dominate. The **property graph** (Neo4j, Amazon Neptune, Memgraph) attaches key-value properties to nodes and edges and is queried with Cypher or Gremlin — the pragmatic choice for application development. **RDF triple stores** (also Neptune, plus Blazegraph, Stardog) represent everything as `(subject, predicate, object)` triples queried with SPARQL — the choice for linked-data, semantic-web, and ontology-driven domains.

**Representative engines**: Neo4j, Amazon Neptune, Memgraph, TigerGraph, JanusGraph.

## Time-Series Databases: Measurements Stamped in Time

A time-series database (TSDB) is purpose-built for data that is **append-only and indexed primarily by time**: a stream of measurements, each tagged with metadata and a timestamp. Metrics, IoT telemetry, application monitoring, financial ticks, and sensor feeds all share this shape — overwhelmingly writes, almost no updates, and reads that are time-range scans and downsampled aggregations.

**When to Use**: Server and application metrics, IoT/sensor telemetry, monitoring and observability, anything where "value over time" is the question.

You *can* store time-series in a wide-column store (the Cassandra IoT example above does exactly that), and for some workloads that is the right call. A dedicated TSDB adds time-aware features on top: automatic time-partitioning, columnar compression tuned for slowly changing series, downsampling/continuous aggregates, and **retention/expiry policies** that age out raw data automatically.

**InfluxDB Example - Server Metrics**:
```sql
-- A "measurement" with tags (indexed metadata) and fields (the values)
-- tags: host, region   fields: usage_user, usage_system
INSERT cpu,host=web-01,region=us-east usage_user=23.1,usage_system=8.4

-- Downsample raw points into 5-minute averages on the fly
SELECT MEAN(usage_user)
FROM cpu
WHERE region = 'us-east' AND time > now() - 1h
GROUP BY time(5m), host;
```

**TimescaleDB Example - PostgreSQL with hypertables**:
```sql
-- TimescaleDB is a Postgres extension: full SQL, but time-partitioned storage
CREATE TABLE metrics (
    time        TIMESTAMPTZ NOT NULL,
    device_id   INT,
    temperature DOUBLE PRECISION
);
SELECT create_hypertable('metrics', 'time');   -- transparently partition by time

-- Continuous aggregate keeps a rolling hourly rollup materialized
CREATE MATERIALIZED VIEW metrics_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS hour,
       device_id,
       AVG(temperature) AS avg_temp
FROM metrics
GROUP BY hour, device_id;
```

### Modeling pattern: tags, fields, cardinality, and retention

- **Separate indexed *tags* from unindexed *fields*.** Tags (host, region, device_id) are the dimensions you filter and group by; fields are the numeric values you aggregate. The product of distinct tag values is the **series cardinality** — and runaway cardinality (e.g. putting a unique request-id or raw user-id in a tag) is the number-one way to blow up a TSDB's memory. Keep tag sets bounded.
- **Lean on time-partitioning.** Data is chunked by time window so writes always hit the newest chunk (cheap) and old chunks can be compressed or dropped wholesale.
- **Downsample and set retention.** Keep raw resolution for days, hourly rollups for months, daily rollups for years. Continuous aggregates / downsampling tasks plus a retention policy do this automatically so storage stays bounded.
- **Write in time order when you can.** Most TSDBs are far happier with monotonically increasing timestamps; large out-of-order backfills are more expensive.

**Representative engines**: InfluxDB, TimescaleDB, Prometheus (monitoring-focused), Amazon Timestream, QuestDB, VictoriaMetrics.

## Choosing the Right Model

There is no universally "best" NoSQL store — only the best fit for a specific access pattern. Work from the queries inward.

### Decision guide

```
What dominates your workload?
    |
    ├─ Look up / cache by a single key, sub-ms?       -> Key-Value (Redis, Memcached)
    |
    ├─ Whole nested aggregate, flexible schema?       -> Document (MongoDB, Firestore)
    |
    ├─ Huge write volume, queries by known key,
    |  time-series at OLTP scale?                      -> Wide-Column (Cassandra, Bigtable)
    |
    ├─ Many-hop relationships / recommendations /
    |  fraud rings?                                    -> Graph (Neo4j, Neptune)
    |
    ├─ Metrics / telemetry over time, downsampling?   -> Time-Series (InfluxDB, Timescale)
    |
    └─ Complex ad-hoc queries, joins, strong ACID?    -> Stay relational (Postgres/MySQL)
```

### Comparison matrix

| Model | Query by | Joins | Schema | Write profile | Classic use case | Avoid when |
|-------|----------|-------|--------|---------------|------------------|------------|
| **Document** | document fields | $lookup only | flexible per-doc | balanced | product catalog, user profiles, CMS | you need cross-document transactions and joins everywhere |
| **Key-value** | the key, only | none | opaque value | very fast point ops | cache, sessions, leaderboards | you must query by attribute / value |
| **Wide-column** | partition + clustering key | none | sparse columns | extremely write-heavy | feeds, messaging, IoT at scale | ad-hoc queries, frequent updates of single columns across partitions |
| **Graph** | pattern traversal | native (the point) | flexible | moderate | social graph, recommendations, fraud | data is tabular and unconnected; bulk analytics |
| **Time-series** | time range + tags | limited | tags + fields | append-only firehose | metrics, telemetry, monitoring | data isn't time-ordered; you need updates |

### Polyglot persistence is normal

Real systems rarely pick one. A single product might use **PostgreSQL** as the system of record for orders and users, **Redis** for sessions and a cache in front of it, **Elasticsearch** for full-text product search, **Cassandra** or a TSDB for the event/metrics firehose, and **Neo4j** for the recommendation graph. This is called *polyglot persistence*: choose each store for the access pattern it serves best, and accept the operational cost of running several. The relational engine stays the default for anything needing rich ad-hoc queries and strong ACID transactions; reach for a specialized NoSQL model when a specific access pattern outgrows what the relational store does well.

> **Code Reference**: For working data-model examples, see [`distributed_systems.py`](../../../code-examples/technology/database-design/distributed_systems.py).

## See Also

<div class="see-also-card">
  <h4>Related pages</h4>
  <ul>
    <li><a href="distributed-and-nosql.html">Distributed Databases &amp; NoSQL</a> — CAP, consensus, distributed transactions, and the NewSQL stores that bring SQL back to scale</li>
    <li><a href="modeling.html">Data Modeling &amp; Normalization</a> — the relational counterpart: normalization, relationships, and design patterns</li>
    <li><a href="storage-internals.html">Storage Engines &amp; Recovery</a> — B+ trees vs. LSM trees, the storage engines underneath many of these stores</li>
    <li><a href="indexing-and-queries.html">Indexing &amp; Query Execution</a> — how indexes and planners work in the relational world</li>
    <li><a href="./">Database Design hub</a></li>
    <li><a href="../aws/">AWS</a> — managed NoSQL services (DynamoDB, DocumentDB, Neptune, Timestream)</li>
  </ul>
</div>
