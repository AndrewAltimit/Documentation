---
layout: docs
title: Database Design
permalink: /docs/technology/database-design/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.25rem;">Database Design</h1>
  <p style="font-size: 1.1rem; margin-top: 0.75rem; opacity: 0.9;">Relational modeling, indexing, and distributed database architecture</p>
</div>

Every application needs to store data, and they all face the same questions: how should data be organized, how can many users access it at once, and what happens when the system crashes? This is the **deep-dive companion** to the crash course — relational modeling and normalization, indexing internals, query execution, transactions, storage engines, and distributed/NoSQL architecture.

> **New to databases?** If you just need tables, SQL basics, and enough to be productive, read the [Database Crash Course](../database-crash-course.html) first — tables, SQL, relationships, indexes, and transactions in five minutes — then come back here for the theory, internals, and scaling concerns.

## Explore Database Design

| Area | Guide | What it covers |
|------|-------|----------------|
| **Modeling** | [Data Modeling & Normalization](modeling.html) | From files to the relational model, ACID, normalization (1NF–3NF), modeling relationships, star/snowflake/EAV patterns, and design anti-patterns |
| **Querying** | [Indexing & Query Execution](indexing-and-queries.html) | Index types and strategies, how the planner parses, optimizes, and executes queries, plus optimizer, memory, and lock internals |
| **Transactions** | [Transactions & Concurrency](transactions-and-concurrency.html) | The concurrency problem, locking vs MVCC, serializability, isolation levels, practical locking patterns, and database security |
| **Storage** | [Storage Engines & Recovery](storage-internals.html) | Pages, the buffer pool, B+ trees and LSM trees, write-ahead logging, backup and recovery, troubleshooting, and performance tuning |
| **Distributed** | [Distributed Databases & NoSQL](distributed-and-nosql.html) | Sub-hub: CAP theorem, consensus overview, NoSQL landscape, the future of databases, case studies, and a selection guide |
| **Distributed** | [Replication & Consensus](replication-and-consensus.html) | Replication topologies, streaming & logical replication, Raft/Paxos, read replicas, failover, and quorums |
| **Distributed** | [Distributed Transactions](distributed-transactions.html) | 2PC/3PC, sagas, the outbox pattern, idempotency, distributed deadlocks, and exactly-once semantics |
| **Distributed** | [NoSQL Data Models](nosql-data-models.html) | Document, key-value, wide-column, graph, and time-series stores — and how to model for each |
| **Operations** | [Operations & Monitoring](operations-and-monitoring.html) | Backups & PITR, disaster recovery, VACUUM, connection pooling, observability, and incident response |
| **Operations** | [ORMs & Data-Access Patterns](orm-patterns.html) | Object-relational mapping, the impedance mismatch, the N+1 problem, and when to drop to SQL |
| **Operations** | [Schema Evolution & Migrations](schema-evolution-and-migrations.html) | Migration tooling, zero-downtime expand–contract, backfills, online schema change, and safe rollbacks |

## Why Databases Matter

Imagine building an online store. You start by storing product information in files:

```python
# products.json
[
    {"id": 1, "name": "Laptop", "price": 999, "stock": 50},
    {"id": 2, "name": "Mouse", "price": 29, "stock": 200}
]
```

This works initially, but problems emerge quickly:
- What if two customers buy the same product simultaneously?
- How do you ensure stock never goes negative?
- What if the server crashes during a purchase?
- How do you find all products under $50 efficiently?

Databases solve these problems through carefully designed systems that have evolved over decades. The guides below explore how they work, starting with practical needs and building up to the theory that makes modern databases possible.

## Key Takeaways

- **Model for integrity first.** Normalization removes redundant data so updates can't leave the database in a contradictory state; denormalize deliberately, for performance.
- **Indexes trade writes for reads.** A B+ tree index turns a full-table scan into a logarithmic lookup, but every index adds cost to inserts, updates, and storage.
- **ACID guarantees reliability.** Atomicity, consistency, isolation, and durability let many users hit the same data concurrently without corruption or lost work.
- **The query planner is your ally.** SQL is declarative — you describe the result and the optimizer chooses the access path. Read `EXPLAIN` output to understand and tune it.
- **Scaling forces trade-offs.** Replication and sharding add capacity but invoke the CAP theorem: under a partition you choose between consistency and availability.
- **Pick the model to fit the access pattern.** Relational, document, key-value, graph, and vector stores each optimize different queries. Choose by how the data is read, not by hype.

## Glossary of Database Terms

**ACID**: Atomicity, Consistency, Isolation, Durability - properties that guarantee reliable transactions

**B-Tree/B+ Tree**: Balanced tree data structure used in most database indexes

**CAP Theorem**: States you can have at most 2 of: Consistency, Availability, Partition tolerance

**Cardinality**: Number of unique values in a column (affects index efficiency)

**Deadlock**: When two transactions wait for each other indefinitely

**Foreign Key**: Column that references primary key in another table

**Index**: Data structure that speeds up queries

**MVCC**: Multi-Version Concurrency Control - allows concurrent access without locking

**Normalization**: Process of organizing data to reduce redundancy

**OLTP/OLAP**: Online Transaction Processing vs Online Analytical Processing

**Primary Key**: Unique identifier for each row

**Query Planner**: Component that decides how to execute queries efficiently

**Replication**: Copying data to multiple servers for availability

**Sharding**: Splitting data across multiple servers horizontally

**Transaction**: Group of operations that succeed or fail together

**WAL**: Write-Ahead Logging - ensures durability by logging before applying changes

## References

### Essential Literature

**Foundational Texts**:
- Kleppmann, M. (2017). *Designing Data-Intensive Applications* - Best modern overview
- Karwin, B. (2010). *SQL Antipatterns* - Learn from common mistakes

**Going Deeper**:
- Ramakrishnan & Gehrke (2003). *Database Management Systems* - Solid textbook
- Petrov, A. (2019). *Database Internals* - How databases actually work

**Research Frontiers**:
- Recent SIGMOD, VLDB, and ICDE conference proceedings
- [The Morning Paper](https://blog.acolyer.org/) - Database paper summaries

### Online Resources

**Interactive Learning**:
- [Use The Index, Luke](https://use-the-index-luke.com/) - SQL indexing tutorial
- [PostgreSQL Exercises](https://pgexercises.com/) - Practice SQL
- [Mystery: SQL Murder Mystery](https://mystery.knightlab.com/) - Learn SQL solving a mystery

**Talks and Videos**:
- [CMU Database Group](https://www.youtube.com/c/CMUDatabaseGroup) - Excellent lectures
- [Designing Data-Intensive Applications](https://www.youtube.com/watch?v=PdtlXdse7pw) - Kleppmann's talks

### Hands-On Projects

1. **Build a Mini Database**: Implement B+ tree, buffer pool, and simple queries
2. **Benchmark Different Databases**: Compare PostgreSQL, MySQL, MongoDB for your use case
3. **Distributed System**: Build a simple distributed key-value store with Raft
4. **Query Optimizer**: Write a cost-based optimizer for simple queries

## See Also

- [Database Crash Course](../database-crash-course.html) — the fast on-ramp to tables and SQL
- [AWS](../aws/) — managed database services and DynamoDB internals
- [Docker](../docker/) — containerizing databases for local development
- [Cybersecurity](../cybersecurity/) — database security and encryption
- [Networking](../networking/) — protocols behind distributed databases
