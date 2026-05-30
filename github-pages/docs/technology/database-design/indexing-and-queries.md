---
layout: docs
title: "Database Design: Indexing & Query Execution"
permalink: /docs/technology/database-design/indexing-and-queries.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Indexing &amp; Query Execution</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Index types, query planning, and the optimizer, memory, and lock internals</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## Indexing: Making Queries Lightning Fast

Imagine finding a word in a dictionary versus a novel. The dictionary has an index (alphabetical order), while the novel requires reading every page. Database indexes work similarly.

### When Indexes Transform Performance

Without an index, finding a customer among millions requires checking every row:
```sql
-- Slow: Full table scan
SELECT * FROM customers WHERE email = 'john@example.com';
-- Time: 5 seconds for 10 million rows
```

With an index:
```sql
CREATE INDEX idx_customers_email ON customers(email);
-- Same query now takes 0.005 seconds!
```

### Types of Indexes and When to Use Them

**B-Tree Index**: Your Swiss Army Knife
- Use for: Most queries, especially ranges
- Example: Finding orders between dates, products under $100
- How it works: Like a phone book - hierarchical, sorted

**Hash Index**: The Speed Demon
- Use for: Exact matches only
- Example: Looking up users by ID
- How it works: Like a hash table - direct lookup
- Note: PostgreSQL provides real, persistent hash indexes (WAL-logged and crash-safe since PostgreSQL 10), and MySQL's MEMORY engine supports them too. Even so, a B-tree is usually the better default — it serves equality *and* range queries.

**Full-Text Index**: The Search Engine
- Use for: Text search, "contains" queries
- Example: Finding products with "wireless" in description
- How it works: Breaks text into searchable tokens

**Bitmap Index**: The Space Saver  
- Use for: Columns with few unique values
- Example: Status fields (active/inactive), categories
- How it works: One bit per row per unique value

### Smart Indexing Strategies

**Composite Indexes**: Order Matters!
```sql
-- This index helps both queries:
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
-- Fast: WHERE customer_id = 123
-- Fast: WHERE customer_id = 123 AND order_date > '2024-01-01'
-- Slow: WHERE order_date > '2024-01-01'  -- Can't use index efficiently!
```

**Covering Indexes**: Include Everything
```sql
-- Index includes all needed columns - no table lookup needed!
CREATE INDEX idx_orders_covering 
ON orders(customer_id, order_date) 
INCLUDE (total, status);
```

**Partial Indexes**: Index Only What You Need
```sql
-- Only index active users - smaller, faster
CREATE INDEX idx_active_users ON users(email) WHERE active = true;
```

<div class="tip-card">
  <h4>Dialect note: defining indexes</h4>
  <p>The examples above use PostgreSQL's standalone <code>CREATE INDEX</code> form. MySQL additionally allows an inline <code>INDEX idx_name (cols)</code> clause inside <code>CREATE TABLE</code>; that inline syntax is MySQL-only and is invalid in PostgreSQL, where you always issue a separate <code>CREATE INDEX</code> statement.</p>
</div>

**Modern Index Types**:
- **BRIN (Block Range Index)**: For time-series data
- **GIN/GiST**: For JSON, arrays, full-text search
- **Vector Indexes**: For AI/ML embeddings (pgvector, Pinecone)
- **Learned Indexes**: AI-predicted data locations (research phase)

### The Cost of Indexes

Indexes aren't free:
- **Storage**: Each index is a data structure that needs disk space
- **Write performance**: Every INSERT/UPDATE must update indexes
- **Maintenance**: Indexes can become fragmented

Rule of thumb: Index based on read patterns, but don't index everything!

**AI-Assisted Index Recommendations**:
Modern databases now use machine learning to suggest indexes:
- **PostgreSQL**: pg_stat_statements + ML advisors
- **MySQL**: Performance Schema with AI insights
- **Cloud Services**: AWS Performance Insights, Azure Intelligent Performance

## How Databases Execute Your Queries

When you write a SQL query, the database performs remarkable optimizations behind the scenes. Understanding this helps you write better queries.

### The Journey of a Query

Let's follow this query through the database:

```sql
SELECT c.name, SUM(o.total) as lifetime_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.country = 'USA'
GROUP BY c.customer_id
HAVING SUM(o.total) > 1000;
```

**Step 1: Parse and Validate**
- Check syntax: Are the SQL keywords correct?
- Verify objects: Do these tables and columns exist?
- Check permissions: Can this user access this data?

**Step 2: Optimize**
The optimizer considers multiple execution strategies:

*Plan A: Scan all customers, then find their orders*
- Cost: 1 million customers × average 10 orders each = expensive!

*Plan B: Use country index, then join*
- Cost: 50,000 US customers × 10 orders = much better!

*Plan C: Start with high-value orders, then find customers*
- Cost: Depends on how many orders > $100...

The optimizer estimates costs using statistics about your data.

**Step 3: Execute**
The chosen plan becomes physical operations:
1. Index seek on customers.country
2. Hash join with orders
3. Aggregate by customer
4. Filter by total > 1000

### Understanding Query Plans

Databases show you their execution strategy:

```sql
EXPLAIN ANALYZE
SELECT c.name, COUNT(*) 
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.name;

-- Output:
HashAggregate (cost=1234.56 rows=1000)
  -> Hash Join (cost=234.56 rows=10000)
        Hash Cond: (o.customer_id = c.customer_id)
        -> Seq Scan on orders o (cost=0.00 rows=10000)
        -> Hash (cost=123.45 rows=1000)
              -> Seq Scan on customers c (cost=0.00 rows=1000)
```

Reading plans bottom-up:
1. Scan customers table (1000 rows)
2. Build hash table
3. Scan orders table (10000 rows) 
4. For each order, probe hash table (fast!)
5. Aggregate results

### The Magic of Relational Algebra

Behind every optimization is relational algebra—a mathematical framework that makes query optimization possible. Just as arithmetic has commutative (a+b = b+a) and associative ((a+b)+c = a+(b+c)) properties, relational operations have rules:

**Pushing Selections Down**
```sql
-- Original: Join everything, then filter
SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE c.country = 'USA';

-- Optimized: Filter first, then join (much less data!)
SELECT * FROM orders o 
JOIN (SELECT * FROM customers WHERE country = 'USA') c 
ON o.customer_id = c.customer_id;
```

The optimizer applies these transformations automatically!

**Join Reordering**
```sql
-- Three-way join: 6 possible orders!
-- A ⋈ B ⋈ C could be:
-- (A ⋈ B) ⋈ C
-- A ⋈ (B ⋈ C)  
-- (A ⋈ C) ⋈ B
-- etc.
```

The optimizer estimates costs for each order and picks the best one.

> **Code Reference**: For implementations of query optimization algorithms, see [`query_processing.py`](../../../code-examples/technology/database-design/query_processing.py)

### Query Optimization in Practice

**Common Performance Killers and Solutions**:

1. **The N+1 Query Problem**
```python
# Bad: 1 query + N queries
customers = db.query("SELECT * FROM customers")
for customer in customers:
    orders = db.query(f"SELECT * FROM orders WHERE customer_id = {customer.id}")
    # If you have 1000 customers, this runs 1001 queries!

# Good: 1 query with JOIN
result = db.query("""
    SELECT c.*, o.*
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
""")
```

2. **Missing Indexes on Foreign Keys**
```sql
-- Orders reference customers, but no index on customer_id!
-- Every join does full table scan
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
-- Now joins are fast
```

3. **Wrong Data Types**
```sql
-- Bad: Storing numbers as strings
CREATE TABLE products (
    price VARCHAR(10)  -- "99.99" stored as string!
);
-- WHERE price > 100 requires converting every row!

-- Good: Use numeric types
CREATE TABLE products (
    price DECIMAL(10,2)  -- Numeric comparisons are fast
);
```

4. **SELECT * Abuse**
```sql
-- Bad: Fetching all columns when you need two
SELECT * FROM users;  -- Transfers unnecessary data

-- Good: Request only what you need  
SELECT id, email FROM users;  -- Less network traffic, less memory
```

## Advanced Database Internals

### Query Optimizer Deep Dive

The query optimizer is the brain of the database. Understanding how it works helps you write better queries.

**Cost Model Components**:
```python
# Simplified cost calculation
def estimate_cost(plan):
    # I/O cost: Reading pages from disk
    seq_page_cost = 1.0
    random_page_cost = 4.0  # Random I/O is slower
    
    # CPU cost: Processing rows
    cpu_tuple_cost = 0.01
    cpu_operator_cost = 0.0025
    
    # Network cost (for distributed databases)
    network_tuple_cost = 0.1
    
    total_cost = (
        plan.seq_pages * seq_page_cost +
        plan.random_pages * random_page_cost +
        plan.rows * cpu_tuple_cost +
        plan.operators * cpu_operator_cost +
        plan.network_rows * network_tuple_cost
    )
    return total_cost
```

**Join Algorithm Selection**:
```sql
-- Nested Loop Join: Good for small tables or indexed lookups
-- Cost: O(n * m)
SELECT * FROM small_table s JOIN large_table l ON s.id = l.foreign_id;

-- Hash Join: Good for medium tables without indexes  
-- Cost: O(n + m)
SELECT * FROM medium1 m1 JOIN medium2 m2 ON m1.id = m2.id;

-- Merge Join: Good for pre-sorted data
-- Cost: O(n log n + m log m) if sorting needed
SELECT * FROM sorted1 s1 JOIN sorted2 s2 ON s1.id = s2.id;
```

**Statistics and Selectivity**:
```sql
-- Database tracks statistics for better estimates
SELECT 
    attname as column,
    n_distinct,
    most_common_vals,
    most_common_freqs,
    histogram_bounds
FROM pg_stats
WHERE tablename = 'orders';

-- Selectivity affects plan choice
-- High selectivity (few rows): Index scan
-- Low selectivity (many rows): Sequential scan
```

### Memory Management Internals

**Buffer Pool Architecture**:
```python
class BufferPoolManager:
    def __init__(self, pool_size_mb):
        self.frames = [None] * (pool_size_mb * 128)  # 8KB pages
        self.page_table = {}  # page_id -> frame_id
        self.free_list = list(range(len(self.frames)))
        self.clock_hand = 0  # For clock replacement
        
    def fetch_page(self, page_id):
        # Check if in memory
        if page_id in self.page_table:
            frame_id = self.page_table[page_id]
            self.frames[frame_id].pin_count += 1
            return self.frames[frame_id]
        
        # Need to load from disk
        frame_id = self._get_free_frame()
        if frame_id is None:
            frame_id = self._evict_page()  # Clock algorithm
            
        # Load page from disk
        page = self._read_from_disk(page_id)
        self.frames[frame_id] = page
        self.page_table[page_id] = frame_id
        return page
```

<div class="notice--info">
  <p>The buffer pool is described in more depth — pages, cache hit ratio, and eviction — in <a href="storage-internals.html#buffer-pool-your-databases-cache">Storage Engines &amp; Recovery</a>.</p>
</div>

**Work Memory Areas**:
```sql
-- Different operations use different memory areas
-- Sort operations
SET work_mem = '256MB';  -- Per operation!
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM large_table ORDER BY created_at;

-- Hash joins
-- Hash table must fit in work_mem or spills to disk
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM t1 JOIN t2 ON t1.id = t2.id;

-- Maintenance operations use different pool
SET maintenance_work_mem = '1GB';
CREATE INDEX idx_large ON large_table(column);
```

### Lock Management Internals

**Lock Compatibility Matrix**:
```python
# PostgreSQL lock modes
LOCK_COMPATIBILITY = {
    #            AS  RS  RE  SUE  S   SSE  E   AE
    'AS':      [ 1,  1,  1,  1,  1,  1,  1,  0], # AccessShare
    'RS':      [ 1,  1,  1,  1,  0,  0,  0,  0], # RowShare  
    'RE':      [ 1,  1,  0,  0,  0,  0,  0,  0], # RowExclusive
    'SUE':     [ 1,  1,  0,  0,  1,  0,  0,  0], # ShareUpdateExcl
    'S':       [ 1,  0,  0,  1,  1,  0,  0,  0], # Share
    'SSE':     [ 1,  0,  0,  0,  0,  0,  0,  0], # ShareRowExcl
    'E':       [ 1,  0,  0,  0,  0,  0,  0,  0], # Exclusive
    'AE':      [ 0,  0,  0,  0,  0,  0,  0,  0], # AccessExclusive
}
```

**Deadlock Detection Algorithm**:
```python
class DeadlockDetector:
    def __init__(self):
        self.wait_graph = {}  # txn -> [waiting_for_txns]
        
    def add_wait(self, waiter, holder):
        if waiter not in self.wait_graph:
            self.wait_graph[waiter] = []
        self.wait_graph[waiter].append(holder)
        
        # Check for cycle
        if self._has_cycle(waiter, holder, {holder}):
            return self._choose_victim()
    
    def _has_cycle(self, start, current, visited):
        if current == start:
            return True
        if current in self.wait_graph:
            for next_txn in self.wait_graph[current]:
                if next_txn not in visited:
                    visited.add(next_txn)
                    if self._has_cycle(start, next_txn, visited):
                        return True
        return False
```

---

## Next Steps

<div class="see-also-card">
  <h4>Continue the deep dive</h4>
  <ul>
    <li><strong>Previous:</strong> <a href="modeling.html">Data Modeling &amp; Normalization</a></li>
    <li><strong>Next:</strong> <a href="transactions-and-concurrency.html">Transactions &amp; Concurrency</a> — how locks and MVCC keep concurrent queries correct.</li>
    <li><strong>Up:</strong> <a href="./">Database Design hub</a></li>
    <li>See also: <a href="storage-internals.html">Storage Engines &amp; Recovery</a> for the pages and buffer pool the optimizer reads from.</li>
  </ul>
</div>
