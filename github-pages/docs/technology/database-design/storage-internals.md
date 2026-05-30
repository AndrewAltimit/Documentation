---
layout: docs
title: "Database Design: Storage Engines & Recovery"
permalink: /docs/technology/database-design/storage-internals.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Storage Engines &amp; Recovery</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Pages, buffer pool, B+/LSM trees, write-ahead logging, backups, and tuning</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## How Databases Store Your Data

Understanding database internals helps you design better schemas and debug performance issues.

### The Storage Hierarchy

Databases carefully manage the journey from SQL to disk:

```
SQL Query
    ↓
Buffer Pool (RAM) - "Hot" data cached here
    ↓
Storage Engine - Manages pages and files
    ↓  
File System - Database files
    ↓
Disk - Actual persistent storage
```

### Pages: The Building Blocks

Databases don't read individual rows from disk—they read pages (typically 8KB or 16KB):

```
+------------------Page 1 (8KB)-------------------+
| Header (checksum, LSN, free space pointer)      |
|--------------------------------------------------||
| Row 1: {id: 1, name: "Alice", email: "..."}    |
| Row 2: {id: 2, name: "Bob", email: "..."}      |
| Row 3: {id: 3, name: "Charlie", email: "..."}  |
| ... (more rows) ...                             |
| Free Space                                       |
+--------------------------------------------------+
```

**Why Pages Matter**:
- Disk I/O is slow; reading 8KB isn't much slower than reading 100 bytes
- Related data stored together (spatial locality)
- Enables efficient caching in memory

### Buffer Pool: Your Database's Cache

The buffer pool is why databases can serve queries from memory:

```python
class BufferPool:
    def __init__(self, size_mb):
        self.pages = {}  # page_id -> page_data
        self.lru = OrderedDict()  # for eviction
        self.max_pages = size_mb * 1024 // 8  # 8KB pages
    
    def get_page(self, page_id):
        if page_id in self.pages:
            # Cache hit! Move to end (most recently used)
            self.lru.move_to_end(page_id)
            return self.pages[page_id]
        else:
            # Cache miss - read from disk
            page = read_page_from_disk(page_id)
            self.add_to_cache(page_id, page)
            return page
```

**Tuning Buffer Pool**:
```sql
-- PostgreSQL: Check cache hit ratio
SELECT 
    sum(blks_hit)/(sum(blks_hit)+sum(blks_read)) as cache_hit_ratio
FROM pg_stat_database;
-- Want > 0.95 (95% from cache)

-- Increase if too low
ALTER SYSTEM SET shared_buffers = '4GB';
```

<div class="notice--info">
  <p>A frame-and-page-table view of the same buffer pool, alongside the clock eviction algorithm and work-memory areas, appears in <a href="indexing-and-queries.html#memory-management-internals">Indexing &amp; Query Execution</a>.</p>
</div>

### B+ Trees: The Workhorse Index Structure

B+ trees power most database indexes. Think of them as a multi-level phone book:

```
                    [M]
                   /   \
            [D,G,J]     [P,S,V]
           /   |   \     /   |   \
        [A-C][D-F][G-I][M-O][P-R][S-U][V-Z]
         ↓    ↓    ↓    ↓    ↓    ↓    ↓
      (actual data rows in leaf nodes)
```

**Why B+ Trees Work Well**:
1. **Shallow**: Even with millions of rows, only 3-4 levels deep
2. **Cache-friendly**: Each node fits in a page
3. **Range queries**: Leaf nodes linked for scanning
4. **Predictable**: Always balanced, consistent performance

**Following a Search**:
```python
# Finding "John" in a B+ tree index on names
1. Root: "John" < "M", go left
2. Level 2: "John" > "G", go to middle child  
3. Leaf: Scan "G-I" page, find "John" -> row location
4. Fetch actual row from heap file
```

**Insert Example**:
```python
def insert(tree, key, value):
    leaf = find_leaf(tree.root, key)
    
    if leaf.has_space():
        leaf.insert(key, value)
    else:
        # Split leaf into two
        new_leaf = leaf.split()
        middle_key = new_leaf.keys[0]
        
        # Propagate split up the tree
        insert_into_parent(leaf.parent, middle_key, new_leaf)
```

### LSM Trees: Built for Big Data Writes

While B+ trees update in place, LSM trees use a different strategy perfect for write-heavy workloads:

**The Big Idea**: Buffer writes in memory, flush to disk in batches

```
Writes go to:
    MemTable (in RAM)
         ↓ (when full)
    SSTable Level 0 (on disk)
         ↓ (compaction)
    SSTable Level 1 (larger, sorted)
         ↓ (compaction)
    SSTable Level 2 (even larger)
```

**Write Path Example**:
```python
class LSMTree:
    def write(self, key, value):
        # 1. Log for crash recovery
        self.wal.append(f"SET {key} = {value}")
        
        # 2. Add to in-memory table
        self.memtable[key] = value
        
        # 3. Flush when full
        if self.memtable.size() > THRESHOLD:
            self.flush_to_disk()
    
    def flush_to_disk(self):
        # Sort and write to new SSTable file
        sorted_data = sorted(self.memtable.items())
        sstable = create_sstable(sorted_data)
        self.sstables[0].append(sstable)
        self.memtable.clear()
```

**Why Cassandra/RocksDB Use LSM**:
- Sequential writes are 100x faster than random writes
- Great for time-series data (always appending)
- Compaction happens in background

**The Trade-off**:
- Writes: Super fast (just append)
- Reads: Slower (might check multiple files)
- Solution: Bloom filters to skip files that definitely don't have the key

### Write-Ahead Logging: Surviving Crashes

How do databases maintain ACID's durability when power fails mid-transaction? Write-Ahead Logging (WAL).

**The Rule**: Log changes before applying them

```python
class WriteAheadLog:
    def __init__(self):
        self.log_file = open("database.wal", "ab")  # Append, binary
        self.lsn = 0  # Log Sequence Number
    
    def log_update(self, tx_id, table, row_id, old_value, new_value):
        entry = {
            "lsn": self.lsn,
            "tx_id": tx_id,
            "type": "UPDATE",
            "table": table,
            "row_id": row_id,
            "old_value": old_value,  # For undo
            "new_value": new_value   # For redo
        }
        self.log_file.write(serialize(entry))
        self.log_file.flush()  # Force to disk
        self.lsn += 1
        
        # Only now safe to update actual data
        return self.lsn
```

**Recovery After Crash**:
```python
def recover():
    # Phase 1: Analysis - What was happening?
    committed_txns = set()
    active_txns = set()
    
    for entry in read_log_from_checkpoint():
        if entry.type == "BEGIN":
            active_txns.add(entry.tx_id)
        elif entry.type == "COMMIT":
            active_txns.remove(entry.tx_id)
            committed_txns.add(entry.tx_id)
    
    # Phase 2: Redo - Replay committed transactions
    for entry in read_log_from_checkpoint():
        if entry.tx_id in committed_txns:
            apply_change(entry)
    
    # Phase 3: Undo - Rollback incomplete transactions
    for entry in reversed(read_log_from_checkpoint()):
        if entry.tx_id in active_txns:
            undo_change(entry)
```

**Why This Works**:
- Log is sequential (fast writes)
- Log records are small
- Can reconstruct any state from log
- Checkpoints limit recovery time

> **Code Reference**: For working implementations, see [`storage_engines.py`](../../../code-examples/technology/database-design/storage_engines.py)

## Backup and Recovery

### Backup Strategies

**Full Backup**:
```bash
# PostgreSQL
pg_dump dbname > backup.sql

# MySQL
mysqldump --all-databases > backup.sql
```

**Incremental Backup**:
- Only changes since last backup
- Requires less storage
- Faster backup, slower restore

**Point-in-Time Recovery**:
- Transaction logs
- Restore to specific moment

### High Availability

**Replication**:
- Master-slave
- Master-master
- Multi-master

**Clustering**:
- Active-passive
- Active-active
- Shared storage

## Troubleshooting Common Database Issues

### Debugging Slow Queries

**Step 1: Identify the Culprit**
```sql
-- PostgreSQL: Enable slow query logging
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1 second
SELECT pg_reload_conf();

-- MySQL: Check slow query log
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;

-- Find currently running queries
SELECT pid, now() - query_start as duration, query 
FROM pg_stat_activity 
WHERE state = 'active' 
ORDER BY duration DESC;
```

**Step 2: Analyze the Query Plan**
```sql
-- Look for these red flags in EXPLAIN output:
EXPLAIN (ANALYZE, BUFFERS) SELECT ...;

-- Bad signs:
-- - "Seq Scan" on large tables (missing index?)
-- - "Nested Loop" with high row counts (consider hash join)
-- - High "Buffers: shared hit" (data not in cache)
-- - "Sort" with "Disk: ..." (increase work_mem)
```

**Step 3: Common Fixes**
```sql
-- Missing index on JOIN columns
CREATE INDEX idx_orders_customer_id ON orders(customer_id);

-- Statistics out of date
ANALYZE orders; -- PostgreSQL
ANALYZE TABLE orders; -- MySQL

-- Query needs rewriting
-- Bad: Correlated subquery
SELECT * FROM orders o 
WHERE total > (SELECT AVG(total) FROM orders WHERE customer_id = o.customer_id);

-- Good: Window function
WITH customer_avgs AS (
    SELECT customer_id, AVG(total) OVER (PARTITION BY customer_id) as avg_total
    FROM orders
)
SELECT o.* FROM orders o 
JOIN customer_avgs ca ON o.customer_id = ca.customer_id 
WHERE o.total > ca.avg_total;
```

### Connection Pool Issues

**Symptoms**: "Too many connections", intermittent timeouts

**Diagnosis**:
```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- See what connections are doing
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;

-- Find idle connections
SELECT pid, usename, application_name, state_change 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND state_change < NOW() - INTERVAL '10 minutes';
```

**Solutions**:
```python
# Configure connection pooling properly
pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=5,    # Keep some connections ready
    maxconn=20,   # Limit maximum connections
    host="localhost",
    database="mydb"
)

# Use context managers to ensure cleanup
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
```

### Disk Space Issues

**Prevention**:
```sql
-- Monitor table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;

-- Set up automatic cleanup
-- PostgreSQL: Configure autovacuum
ALTER TABLE large_table SET (autovacuum_vacuum_scale_factor = 0.01);

-- Archive old data
CREATE TABLE orders_archive (LIKE orders INCLUDING ALL);
INSERT INTO orders_archive SELECT * FROM orders WHERE created_at < '2023-01-01';
DELETE FROM orders WHERE created_at < '2023-01-01';
```

## Performance Tuning: Making It Fast

Performance tuning is part science, part art. Here's a practical approach:

### Step 1: Measure First

**Find Slow Queries**:
```sql
-- PostgreSQL: Find slowest queries
SELECT 
    mean_exec_time,
    calls,
    total_exec_time,
    query
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Check Cache Performance**:
```sql
-- Cache hit ratio (want > 95%)
SELECT 
    sum(heap_blks_hit) / 
    (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
FROM pg_statio_user_tables;
```

### Step 2: Tune Configuration

**Memory Settings** (PostgreSQL example):
```ini
# Buffer pool - start with 25% of RAM
shared_buffers = 4GB

# Total memory for queries  
work_mem = 50MB  # Per operation!

# Maintenance operations
maintenance_work_mem = 1GB

# Effective cache - tell planner about OS cache
effective_cache_size = 12GB  # ~75% of RAM
```

### Step 3: Optimize Schema

**Partitioning for Large Tables**:
```sql
-- Partition orders by month
CREATE TABLE orders (
    order_id BIGINT,
    order_date DATE,
    customer_id INT,
    total DECIMAL(10,2)
) PARTITION BY RANGE (order_date);

-- Create monthly partitions
CREATE TABLE orders_2024_01 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Queries on date ranges now scan only relevant partitions!
```

**Materialized Views for Complex Queries**:
```sql
-- Expensive dashboard query
CREATE MATERIALIZED VIEW customer_stats AS
SELECT 
    c.customer_id,
    c.name,
    COUNT(DISTINCT o.order_id) as order_count,
    SUM(o.total) as lifetime_value,
    MAX(o.order_date) as last_order_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;

-- Refresh periodically
CREATE INDEX idx_customer_stats_value ON customer_stats(lifetime_value);

-- Now dashboard query is instant!
```

### Step 4: Monitor and Iterate

**Key Metrics to Watch**:
- **Response time**: 95th percentile latency
- **Throughput**: Queries per second
- **Resource usage**: CPU, memory, disk I/O
- **Lock waits**: Blocked queries
- **Connection pool**: Active vs idle connections

---

## Next Steps

<div class="see-also-card">
  <h4>Continue the deep dive</h4>
  <ul>
    <li><strong>Previous:</strong> <a href="transactions-and-concurrency.html">Transactions &amp; Concurrency</a></li>
    <li><strong>Next:</strong> <a href="distributed-and-nosql.html">Distributed Databases &amp; NoSQL</a> — scaling storage beyond a single machine.</li>
    <li><strong>Up:</strong> <a href="./">Database Design hub</a></li>
    <li>See also: <a href="indexing-and-queries.html">Indexing &amp; Query Execution</a> for how the planner reads these pages.</li>
  </ul>
</div>
