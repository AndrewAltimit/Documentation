---
layout: docs
title: Database Design
permalink: /docs/technology/database-design/
toc: false
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Database Design</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Relational modeling, indexing, and distributed database architecture</p>
</div>

<div class="intro-card">
  <p class="lead-text">Every application needs to store data. Whether you're building a social network, an e-commerce platform, or an analytics system, you'll face the same fundamental questions: how should data be organized, how can many users access it at once, and what happens when the system crashes? This is the <strong>deep-dive companion</strong> — it works through relational modeling, normalization, indexing internals, query execution, transactions, and distributed/NoSQL architecture.</p>
</div>

<div class="tip-card">
  <h4>New to databases? Start with the crash course</h4>
  <p>If you just need tables, SQL basics, and enough to be productive, read the <a href="../database-crash-course.html">Database Crash Course</a> first, then come back here for the theory and scaling concerns.</p>
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-table"></i>
    <h4>Modeling</h4>
    <p>Schemas, relationships, and normalization that keep data consistent</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-bolt"></i>
    <h4>Performance</h4>
    <p>Indexes, query planning, and execution that make reads fast</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-network-wired"></i>
    <h4>Scale</h4>
    <p>Replication, sharding, transactions, and the NoSQL trade-offs</p>
  </div>
</div>

## Quick Start

New to databases? The fastest on-ramp — tables, SQL, relationships, indexes, and transactions in five minutes — lives in the [Database Crash Course](../database-crash-course.html). Come back here for the theory, internals, and scaling concerns.

## Explore Database Design

<div class="command-grid">
  <a href="modeling.html" class="nav-card">
    <h4><i class="fas fa-table"></i> Data Modeling &amp; Normalization</h4>
    <p>From files to the relational model, ACID, normalization (1NF–3NF), modeling relationships, star/snowflake/EAV patterns, and design anti-patterns.</p>
  </a>
  <a href="indexing-and-queries.html" class="nav-card">
    <h4><i class="fas fa-bolt"></i> Indexing &amp; Query Execution</h4>
    <p>Index types and strategies, how the planner parses, optimizes, and executes queries, plus optimizer, memory, and lock internals.</p>
  </a>
  <a href="transactions-and-concurrency.html" class="nav-card">
    <h4><i class="fas fa-code-branch"></i> Transactions &amp; Concurrency</h4>
    <p>The concurrency problem, locking vs MVCC, serializability, isolation levels, practical locking patterns, and database security.</p>
  </a>
  <a href="storage-internals.html" class="nav-card">
    <h4><i class="fas fa-hdd"></i> Storage Engines &amp; Recovery</h4>
    <p>Pages, the buffer pool, B+ trees and LSM trees, write-ahead logging, backup and recovery, troubleshooting, and performance tuning.</p>
  </a>
  <a href="distributed-and-nosql.html" class="nav-card">
    <h4><i class="fas fa-network-wired"></i> Distributed Databases &amp; NoSQL</h4>
    <p>The CAP theorem, consensus (Raft/Paxos), 2PC and sagas, NoSQL models, the future of databases, case studies, and a selection guide.</p>
  </a>
  <a href="distributed-and-nosql.html#best-practices-from-the-trenches" class="nav-card">
    <h4><i class="fas fa-clipboard-check"></i> Best Practices &amp; Build Your Own</h4>
    <p>Design principles that scale, common pitfalls, and a hands-on progression for building your own database from a key-value store up.</p>
  </a>
</div>

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

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Model for integrity first</h4>
    <p>Normalization removes redundant data so updates can't leave the database in a contradictory state; denormalize deliberately, for performance.</p>
  </div>
  <div class="takeaway-card">
    <h4>Indexes trade writes for reads</h4>
    <p>A B+ tree index turns a full-table scan into a logarithmic lookup, but every index adds cost to inserts, updates, and storage.</p>
  </div>
  <div class="takeaway-card">
    <h4>ACID guarantees reliability</h4>
    <p>Atomicity, consistency, isolation, and durability let many users hit the same data concurrently without corruption or lost work.</p>
  </div>
  <div class="takeaway-card">
    <h4>The query planner is your ally</h4>
    <p>SQL is declarative — you describe the result and the optimizer chooses the access path. Read `EXPLAIN` output to understand and tune it.</p>
  </div>
  <div class="takeaway-card">
    <h4>Scaling forces trade-offs</h4>
    <p>Replication and sharding add capacity but invoke the CAP theorem: under a partition you choose between consistency and availability.</p>
  </div>
  <div class="takeaway-card">
    <h4>Pick the model to fit the access pattern</h4>
    <p>Relational, document, key-value, graph, and vector stores each optimize different queries. Choose by how the data is read, not by hype.</p>
  </div>
</div>

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

<div class="see-also-card">
  <h4>Related pages</h4>
  <ul>
    <li><a href="../database-crash-course.html">Database Crash Course</a> — the fast on-ramp to tables and SQL</li>
    <li><a href="../aws/">AWS</a> — managed database services and DynamoDB internals</li>
    <li><a href="../docker/">Docker</a> — containerizing databases for local development</li>
    <li><a href="../cybersecurity/">Cybersecurity</a> — database security and encryption</li>
    <li><a href="../networking/">Networking</a> — protocols behind distributed databases</li>
  </ul>
</div>
