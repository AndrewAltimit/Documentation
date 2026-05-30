---
layout: docs
title: "Database Design: Transactions & Concurrency"
permalink: /docs/technology/database-design/transactions-and-concurrency.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Transactions &amp; Concurrency</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Locking, MVCC, serializability, isolation levels, and database security</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## Transactions and Concurrency: Managing Simultaneous Access

When multiple users access a database simultaneously, chaos can ensue. Transactions and concurrency control bring order to this chaos.

### The Concurrency Problem

Consider this scenario in an online store:

```python
# Two customers buy the last item simultaneously
# Customer A's thread:
stock = db.query("SELECT stock FROM products WHERE id = 123")  # Returns 1
if stock > 0:
    # Context switch to Customer B here!
    db.execute("UPDATE products SET stock = stock - 1 WHERE id = 123")
    db.execute("INSERT INTO orders ...")

# Customer B's thread (running at same time):
stock = db.query("SELECT stock FROM products WHERE id = 123")  # Also returns 1!
if stock > 0:
    db.execute("UPDATE products SET stock = stock - 1 WHERE id = 123")
    db.execute("INSERT INTO orders ...")

# Result: stock = -1, both customers think they got the item!
```

### How Databases Solve This

Databases use two main strategies: pessimistic (locking) and optimistic (versioning).

#### Strategy 1: Locking (Pessimistic)

**Two-Phase Locking (2PL)**: Grab locks, do work, release locks

```sql
BEGIN TRANSACTION;
-- Lock the row for update
SELECT stock FROM products WHERE id = 123 FOR UPDATE;
-- Now only this transaction can modify this row
UPDATE products SET stock = stock - 1 WHERE id = 123;
INSERT INTO orders ...;
COMMIT;  -- Releases all locks
```

**Lock Types**:
- **Shared (S)**: Multiple readers OK (SELECT)
- **Exclusive (X)**: Single writer, no readers (UPDATE/DELETE)

**The Deadlock Problem**:
```
Transaction 1: Lock A, waiting for B
Transaction 2: Lock B, waiting for A
-- Both stuck forever!
```

Databases detect deadlocks and kill one transaction to break the cycle.

#### Strategy 2: Multi-Version Concurrency Control (MVCC)

Instead of locking, keep multiple versions of data. Each transaction sees a consistent snapshot:

```python
# Simplified MVCC concept
class MVCCDatabase:
    def __init__(self):
        self.data = {}  # {key: [(value, timestamp, deleted), ...]}
        self.timestamp = 0
    
    def begin_transaction(self):
        self.timestamp += 1
        return Transaction(self.timestamp)
    
    def read(self, tx, key):
        # Find latest version visible to this transaction
        versions = self.data.get(key, [])
        for value, ts, deleted in reversed(versions):
            if ts <= tx.start_time:
                return None if deleted else value
        return None
    
    def write(self, tx, key, value):
        # Create new version, don't overwrite
        if key not in self.data:
            self.data[key] = []
        self.data[key].append((value, tx.start_time, False))
```

**How PostgreSQL Uses MVCC**:
```sql
-- Transaction 1 (started at time 100)
BEGIN;
SELECT balance FROM accounts WHERE id = 1;
-- Sees balance = 1000 (version from time 50)

-- Transaction 2 (started at time 101)
BEGIN;
UPDATE accounts SET balance = 900 WHERE id = 1;
COMMIT;
-- Creates new version at time 101

-- Transaction 1 still sees old version!
SELECT balance FROM accounts WHERE id = 1;
-- Still sees balance = 1000 (snapshot from time 100)
```

**Benefits**:
- Readers never block writers
- Writers never block readers  
- Great for read-heavy workloads
- Natural time-travel queries ("show me data as of yesterday")

### Understanding Serializability

The gold standard for correctness is serializability: the result should be as if transactions ran one at a time, even though they actually ran concurrently.

**Testing for Conflicts**:
```python
# Two transactions operating on same data
T1: READ(A), WRITE(B)
T2: WRITE(A), READ(B)

# Conflicts:
# T1.READ(A) conflicts with T2.WRITE(A)  (Read-Write)
# T2.READ(B) conflicts with T1.WRITE(B)  (Read-Write)

# Build a graph: T1 -> T2 (T1 must come before T2)
#                T2 -> T1 (T2 must come before T1)
# Cycle! Not serializable.
```

**Why This Matters**:
Non-serializable schedules can produce results impossible with serial execution:

```sql
-- Account transfer race condition
-- T1: Transfer $100 from A to B
-- T2: Transfer $100 from B to A

-- Serial execution: No net change
-- Bad concurrent execution: Money appears/disappears!
```

### Isolation Levels: Choosing Your Guarantees

Databases offer different isolation levels—trade-offs between correctness and performance:

**Read Uncommitted**: "I live dangerously"
```sql
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
-- Can see uncommitted changes (dirty reads)
-- Use case: Rough analytics where exactness doesn't matter
```

**Read Committed**: "Show me committed data" (PostgreSQL default)
```sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- Each query sees committed data at query start
-- Problem: Same query can return different results
SELECT COUNT(*) FROM orders;  -- Returns 100
-- Another transaction commits new order
SELECT COUNT(*) FROM orders;  -- Returns 101
```

**Repeatable Read**: "My view stays consistent"
```sql
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- All queries see same snapshot
-- Problem: Phantom reads (new rows matching WHERE)
```

**Serializable**: "Perfect isolation" 
```sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- As if transactions ran one at a time
-- Might fail with "serialization error" - retry needed
```

**Real-World Example**: Seat Booking
```sql
-- With READ COMMITTED: Two people might book same seat
-- With SERIALIZABLE: One succeeds, one gets error to retry

BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT seat_id FROM seats 
WHERE flight_id = 123 AND status = 'available' 
LIMIT 1 FOR UPDATE;

UPDATE seats SET status = 'booked', customer_id = 456
WHERE seat_id = 789;
COMMIT;
```

> **Code Reference**: For implementations of these concepts, see [`concurrency_control.py`](../../../code-examples/technology/database-design/concurrency_control.py)

### Practical Locking Patterns

**Pattern 1: Preventing Lost Updates**
```sql
-- Problem: Two users editing same document
-- Solution: Optimistic locking with version

-- User A loads document
SELECT content, version FROM documents WHERE id = 123;
-- Returns: content="Hello", version=5

-- User B loads same document  
SELECT content, version FROM documents WHERE id = 123;
-- Returns: content="Hello", version=5

-- User A saves changes
UPDATE documents 
SET content = 'Hello World', version = version + 1
WHERE id = 123 AND version = 5;
-- Success! 1 row updated

-- User B tries to save
UPDATE documents
SET content = 'Hello Everyone', version = version + 1  
WHERE id = 123 AND version = 5;
-- Failure! 0 rows updated (version is now 6)
-- Application shows: "Document was modified by another user"
```

**Pattern 2: Queue Processing**
```sql
-- Multiple workers processing job queue
-- Need to ensure each job processed once

WITH next_job AS (
    SELECT job_id FROM job_queue
    WHERE status = 'pending'
    ORDER BY priority DESC, created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED  -- Key: Skip rows locked by others
)
UPDATE job_queue 
SET status = 'processing', worker_id = 'worker-1'
WHERE job_id = (SELECT job_id FROM next_job)
RETURNING *;
```

## Database Security

### Access Control

```sql
-- Create user
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'secure_password';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON mydb.* TO 'app_user'@'localhost';

-- Revoke permissions
REVOKE DELETE ON mydb.* FROM 'app_user'@'localhost';

-- Role-based access
CREATE ROLE read_only;
GRANT SELECT ON mydb.* TO read_only;
GRANT read_only TO 'analyst'@'localhost';
```

### Data Encryption

**At Rest**:
- Transparent Data Encryption (TDE)
- File system encryption
- Column-level encryption

**In Transit**:
- SSL/TLS connections
- VPN tunnels

### SQL Injection Prevention

```python
# Bad - vulnerable to injection
query = f"SELECT * FROM users WHERE username = '{username}'"

# Good - parameterized query
cursor.execute(
    "SELECT * FROM users WHERE username = %s",
    (username,)
)

# Good - using ORM
user = User.query.filter_by(username=username).first()
```

---

## Next Steps

<div class="see-also-card">
  <h4>Continue the deep dive</h4>
  <ul>
    <li><strong>Previous:</strong> <a href="indexing-and-queries.html">Indexing &amp; Query Execution</a></li>
    <li><strong>Next:</strong> <a href="storage-internals.html">Storage Engines &amp; Recovery</a> — how WAL makes durability and recovery possible.</li>
    <li><strong>Up:</strong> <a href="./">Database Design hub</a></li>
    <li>See also: <a href="../cybersecurity/">Cybersecurity</a> for defense-in-depth beyond the database layer.</li>
  </ul>
</div>
