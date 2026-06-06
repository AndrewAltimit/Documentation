---
layout: docs
title: "Database Design: Indexing & Query Execution"
permalink: /docs/technology/database-design/indexing-and-queries.html
toc: true
toc_sticky: true
hide_title: true
---

<p><a href="./">&larr; Database Design</a></p>

# Indexing & Query Execution

## Indexing: Making Queries Lightning Fast

Imagine finding a word in a dictionary versus a novel. The dictionary has an index (alphabetical order), while the novel requires reading every page. Database indexes work similarly.

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

The rest of this page builds up in the order you would actually reach for these
ideas: first the *kinds* of index and when each one earns its keep, then the cases
where an index makes things **worse**, then how to read the plan the database
actually chose, the everyday query mistakes that defeat your indexes, and finally
the optimizer internals that explain *why* the planner made its choices.

### Types of Indexes and When to Use Them

The choice of index structure follows directly from the *shape* of the predicate
you run most often. The table below is the quick-reference; the prose underneath
explains the trade-offs.

| Index type | Best for | Supports ranges? | Typical use |
|---|---|---|---|
| **B-Tree** | equality **and** ordered ranges | Yes | the safe default for almost everything |
| **Hash** | exact equality only | No | high-volume key lookups |
| **Full-Text** (GIN) | "contains this word" searches | n/a | product/article search |
| **GIN / GiST** | JSON, arrays, geometry, fuzzy text | partial | `@>`, `&&`, trigram `LIKE '%foo%'` |
| **BRIN** | huge, naturally-ordered tables | Yes (coarse) | append-only time-series |
| **Bitmap** (DW engines) | low-cardinality columns | No | warehouse star-schema filters |

**B-Tree Index**: Your Swiss Army Knife
- Use for: Most queries, especially ranges
- Example: Finding orders between dates, products under $100
- How it works: Like a phone book - hierarchical, sorted
- Why it's the default: a B-tree answers `=`, `<`, `>`, `BETWEEN`, `ORDER BY`,
  and `LIKE 'prefix%'` from the *same* structure, and stays balanced as you write.

**Hash Index**: The Speed Demon
- Use for: Exact matches only
- Example: Looking up users by ID
- How it works: Like a hash table - direct lookup
- When *not* to use it: any query with `<`, `>`, `BETWEEN`, or `ORDER BY` — a hash
  has no notion of order, so the planner falls back to a scan.
- Note: PostgreSQL provides real, persistent hash indexes (WAL-logged and crash-safe since PostgreSQL 10), and MySQL's MEMORY engine supports them too. Even so, a B-tree is usually the better default — it serves equality *and* range queries.

**Full-Text Index**: The Search Engine
- Use for: Text search, "contains" queries
- Example: Finding products with "wireless" in description
- How it works: Breaks text into searchable tokens

**Bitmap Index**: The Space Saver  
- Use for: Columns with few unique values
- Example: Status fields (active/inactive), categories
- How it works: One bit per row per unique value

**Modern Index Types**:
- **BRIN (Block Range Index)**: For time-series data — stores only the min/max value
  per block range, so the index is tiny; great when rows arrive in roughly the
  same order as the indexed column.
- **GIN/GiST**: For JSON, arrays, full-text search, and trigram `LIKE '%substr%'`.
- **Vector Indexes**: For AI/ML embeddings (pgvector, Pinecone) — approximate
  nearest-neighbour search over high-dimensional vectors.
- **Learned Indexes**: AI-predicted data locations (research phase).

### Smart Indexing Strategies

Once you've picked the *structure*, the next decisions are about *which columns*
and *which rows* to cover.

**Composite Indexes**: Order Matters!
```sql
-- This index helps both queries:
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
-- Fast: WHERE customer_id = 123
-- Fast: WHERE customer_id = 123 AND order_date > '2024-01-01'
-- Slow: WHERE order_date > '2024-01-01'  -- Can't use index efficiently!
```

The rule is the **leftmost-prefix rule**: a composite index on `(a, b, c)` can
serve predicates on `a`, on `a, b`, and on `a, b, c`, but *not* a predicate on `b`
alone — the index is sorted by `a` first, so a query that doesn't constrain `a`
has no contiguous slice to seek into. Put the column you filter by equality first,
the column you range-scan or sort by second.

**Covering Indexes**: Include Everything
```sql
-- Index includes all needed columns - no table lookup needed!
CREATE INDEX idx_orders_covering 
ON orders(customer_id, order_date) 
INCLUDE (total, status);
```

A *covering* index lets the query be answered entirely from the index — the
planner reports an **Index Only Scan** and never touches the heap. The `INCLUDE`
columns are stored in the leaf pages but are *not* part of the sort key, so they
add no ordering cost.

**Partial Indexes**: Index Only What You Need
```sql
-- Only index active users - smaller, faster
CREATE INDEX idx_active_users ON users(email) WHERE active = true;
```

> **Dialect note: defining indexes.** The examples above use PostgreSQL's standalone `CREATE INDEX` form. MySQL additionally allows an inline `INDEX idx_name (cols)` clause inside `CREATE TABLE`; that inline syntax is MySQL-only and is invalid in PostgreSQL, where you always issue a separate `CREATE INDEX` statement.

## When Indexes Hurt

Indexing is a trade-off, not a free win. Every index you add is a second data
structure the database must keep perfectly in sync, and there are query shapes an
index actively slows down. Knowing when *not* to index is as important as knowing
when to.

**Indexes aren't free:**
- **Storage**: Each index is a separate B-tree on disk; a wide multi-column index
  can rival the size of the table itself.
- **Write amplification**: Every `INSERT`, `UPDATE` (of an indexed column), and
  `DELETE` must update *every* affected index. A table with eight indexes turns one
  logical write into nine physical ones.
- **Maintenance / bloat**: Indexes fragment as rows churn; B-trees develop dead
  entries that need `REINDEX` or autovacuum to reclaim.
- **Planner overhead**: More candidate indexes means more plans for the optimizer
  to cost out before it even starts executing.

**Cases where an index is the wrong tool:**

1. **Low selectivity** — when the predicate matches a large fraction of the table.
```sql
-- gender has 2 values; matching ~50% of rows.
-- An index scan would do millions of random reads PLUS heap fetches;
-- a sequential scan reads the table once, in order. Seq scan wins.
SELECT * FROM users WHERE gender = 'F';
```
A useful heuristic: if a predicate returns more than roughly 5–10% of the table,
a sequential scan usually beats an index scan because random I/O (`random_page_cost`)
is several times more expensive than sequential I/O.

2. **A predicate that hides the column inside a function** — this makes the index
unusable, because the index stores `column`, not `f(column)`:
```sql
-- Index on created_at is IGNORED: the planner can't match YEAR(created_at)
WHERE YEAR(created_at) = 2024;

-- Rewrite as a range so the index applies:
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
-- (or build an expression index: CREATE INDEX ... ON t (EXTRACT(YEAR FROM created_at)))
```

3. **Leading wildcards** — `LIKE '%term'` can't use an ordinary B-tree because
there is no known prefix to seek to. Use a trigram (GIN) index or full-text search
instead.

4. **Tiny tables** — for a few hundred rows the whole table fits in one or two
pages; a sequential scan is faster than the index lookup plus heap fetch, and the
planner will (correctly) ignore your index.

5. **Write-heavy hot paths** — a high-ingest logging table may be better off with
*few* indexes (or BRIN) so writes stay cheap; add read indexes only on the columns
you actually query.

> **Rule of thumb.** Index based on read patterns, but don't index everything. Start from the predicates and join keys in your hottest queries, confirm with `EXPLAIN` that each index is actually used, and drop indexes that never appear in a plan — they cost you on every write for no read benefit.

**AI-Assisted Index Recommendations**:
Modern databases now use machine learning to suggest indexes:
- **PostgreSQL**: pg_stat_statements + ML advisors
- **MySQL**: Performance Schema with AI insights
- **Cloud Services**: AWS Performance Insights, Azure Intelligent Performance

Treat these as *candidates*, not gospel — they optimize the workload they've seen,
and will happily recommend an index that a single weekly report would use but that
your write-heavy OLTP path pays for every second.

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

### Reading EXPLAIN and EXPLAIN ANALYZE

`EXPLAIN` shows the plan the optimizer *chose* and its *estimates*. `EXPLAIN
ANALYZE` actually **runs** the query and adds the *measured* timings and row
counts, so you can compare what the planner predicted against what really happened.
That comparison is the single most useful debugging skill for slow queries.

Start with the simple form:

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

A plan is a **tree**, and it executes from the **leaves up**: each node pulls rows
from its children, transforms them, and hands them to its parent. Read it
bottom-up to follow the data, top-down to read the *goal*:
1. Scan customers table (1000 rows)
2. Build hash table
3. Scan orders table (10000 rows) 
4. For each order, probe hash table (fast!)
5. Aggregate results

#### Annotating a real plan

The summary form above hides the numbers you actually use in practice. Here is the
*full* output of `EXPLAIN (ANALYZE, BUFFERS)` for a realistic query, with every
field annotated. Suppose we run:

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT c.name, SUM(o.total) AS lifetime_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.country = 'USA'
GROUP BY c.name
HAVING SUM(o.total) > 1000;
```

```text
GroupAggregate  (cost=12850.42..14102.88 rows=842 width=40)
                (actual time=88.214..142.905 rows=796 loops=1)
  Group Key: c.name
  Filter: (sum(o.total) > 1000)
  Rows Removed by Filter: 1204
  Buffers: shared hit=512 read=4188
  ->  Sort  (cost=12850.42..13061.05 rows=84252 width=18)
            (actual time=88.190..101.337 rows=84252 loops=1)
        Sort Key: c.name
        Sort Method: external merge  Disk: 2304kB
        Buffers: shared hit=512 read=4188, temp read=288 written=290
        ->  Hash Join  (cost=1842.00..5921.66 rows=84252 width=18)
                       (actual time=12.004..58.221 rows=84252 loops=1)
              Hash Cond: (o.customer_id = c.customer_id)
              Buffers: shared hit=512 read=4188
              ->  Seq Scan on orders o  (cost=0.00..3258.00 rows=200000 width=12)
                                        (actual time=0.009..18.402 rows=200000 loops=1)
                    Buffers: shared read=3258
              ->  Hash  (cost=1620.00..1620.00 rows=17760 width=14)
                        (actual time=11.880..11.881 rows=17760 loops=1)
                    Buckets: 32768  Batches: 1  Memory Usage: 1099kB
                    Buffers: shared hit=512 read=930
                    ->  Seq Scan on customers c  (cost=0.00..1620.00 rows=17760 width=14)
                                                 (actual time=0.014..7.901 rows=17760 loops=1)
                          Filter: (country = 'USA'::text)
                          Rows Removed by Filter: 82240
                          Buffers: shared read=930
Planning Time: 0.624 ms
Execution Time: 144.880 ms
```

Now walk it field by field — this is what each piece is telling you:

- **`cost=12850.42..14102.88`** — two numbers in arbitrary cost units. The first is
  the *startup cost* (work before the first row can be returned; high for `Sort`
  because nothing emits until everything is sorted), the second is the *total cost*
  to return the last row. These are estimates, used only to *rank* plans.
- **`rows=842` vs `actual ... rows=796`** — the estimated vs. real row count. They're
  close here (842 vs 796), which means the planner's statistics are healthy. **A
  large gap between estimated and actual rows is the number-one cause of bad plans**
  — it usually means stale statistics; fix it with `ANALYZE customers;`.
- **`actual time=88.214..142.905`** — wall-clock milliseconds: time to the first row,
  then to the last. Subtract a child's total time from its parent's to see how much
  time a node *itself* spent.
- **`loops=1`** — how many times the node ran. **Critical gotcha: `actual time` and
  `rows` are reported *per loop*.** On the inner side of a nested loop you must
  multiply by `loops` to get the true totals; a "0.05 ms" node run 50,000 times is
  2.5 seconds.
- **`Rows Removed by Filter: 82240`** on the customers scan — the `country = 'USA'`
  predicate threw away 82,240 of 100,000 rows. That is the smoking gun: **an index
  on `customers(country)` would let the scan read only the ~18k US rows** instead of
  the whole table.
- **`Sort Method: external merge  Disk: 2304kB`** — the sort did **not** fit in
  `work_mem` and *spilled to disk*. The matching `temp read=288 written=290` under
  `Buffers` confirms temp-file I/O. Raising `work_mem` so the sort stays in memory
  (`Sort Method: quicksort  Memory: ...`) would remove that disk traffic.
- **`Buffers: shared hit=512 read=4188`** — pages served from cache (`hit`) versus
  pages fetched from disk (`read`). 4,188 disk reads dominate the runtime; on a
  warm cache the same plan would be far faster. Always run `EXPLAIN ANALYZE` twice
  and compare `read` vs `hit` to tell a cold cache apart from a genuinely
  I/O-bound plan.
- **`Hash` node — `Batches: 1  Memory Usage: 1099kB`** — the hash table fit in one
  batch in memory. If you ever see `Batches: 8` (a power of two > 1), the build side
  overflowed `work_mem` and the join spilled to disk — another `work_mem` signal.
- **`Planning Time` vs `Execution Time`** — planning is sub-millisecond here, so the
  problem is purely execution. If planning ever dominates (common with very many
  partitions or prepared-statement churn), that's a different class of fix.

The takeaway from this one plan: two concrete actions — add `customers(country)` to
kill the 82k-row filter, and raise `work_mem` to keep the sort in memory — both of
which you could *only* have known by reading the annotated output, not the SQL.

> **Output formats and engine differences.** Add `(FORMAT JSON)` for machine-readable output you can diff in CI, or `(VERBOSE)` to see output column lists. MySQL's `EXPLAIN ANALYZE` reports the same idea in a different shape — `actual time=first..last` and `loops=N` per node — while SQL Server exposes the live plan through `SET STATISTICS PROFILE ON` or the graphical "Include Actual Execution Plan". The vocabulary differs; the reading discipline (estimate vs actual, watch the loops, find the spill) is identical.

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
-- A JOIN B JOIN C could be:
-- (A JOIN B) JOIN C
-- A JOIN (B JOIN C)
-- (A JOIN C) JOIN B
-- etc.
```

The optimizer estimates costs for each order and picks the best one.

> **Code Reference**: For implementations of query optimization algorithms, see [`query_processing.py`](../../../code-examples/technology/database-design/query_processing.py)

## Common Query Problems

The fastest query is the one you never send. Most real-world slowness traces back
to a handful of recurring anti-patterns — and now that you can read a plan, you can
*see* each of them in `EXPLAIN ANALYZE` output.

### 1. The N+1 Query Problem

The most common performance killer in application code: one query to fetch a list,
then *one more query per row* to fetch each item's related data.

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

It's called "N+1" because *N* rows from the first query trigger *N* follow-up
queries, plus the original — 1001 round-trips where one JOIN (or one `WHERE
customer_id IN (...)`) would do. The damage is mostly **network latency**: even at
0.5 ms per round-trip, 1000 extra queries add half a second of pure waiting before
any work happens.

ORMs are the usual culprit because lazy-loading a relationship inside a loop hides
the extra queries behind innocent-looking attribute access. The fix is **eager
loading** — telling the ORM to fetch the related rows up front:

```python
# SQLAlchemy: one extra query for ALL orders, not one per customer
from sqlalchemy.orm import selectinload
customers = session.query(Customer).options(selectinload(Customer.orders)).all()

# Django: prefetch_related / select_related collapse the N follow-ups
customers = Customer.objects.prefetch_related("orders").all()
```

See [ORM Patterns](orm-patterns.html) for a deeper treatment of eager vs. lazy
loading and how to spot N+1 in query logs.

### 2. Missing Indexes on Foreign Keys

```sql
-- Orders reference customers, but no index on customer_id!
-- Every join does full table scan
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
-- Now joins are fast
```

A foreign key constraint does **not** automatically create an index on the
referencing column (PostgreSQL and MySQL/InnoDB differ here — InnoDB auto-creates
one, PostgreSQL does not). Without it, every join and every `ON DELETE CASCADE`
re-scans the child table.

### 3. Functions and Type Mismatches That Disable Indexes

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

The same trap appears as an *implicit cast* in the predicate: comparing an indexed
`bigint` column to a quoted `'123'`, or a `varchar` column to a number, can force a
per-row conversion that disables the index. Match the literal's type to the
column's, and avoid wrapping the column in a function (see *When Indexes Hurt*
above).

### 4. SELECT * Abuse

```sql
-- Bad: Fetching all columns when you need two
SELECT * FROM users;  -- Transfers unnecessary data

-- Good: Request only what you need  
SELECT id, email FROM users;  -- Less network traffic, less memory
```

Beyond the wasted bandwidth, `SELECT *` defeats **covering indexes**: a query that
asked only for indexed columns could be an Index-Only Scan, but pulling every
column forces a heap fetch for each row.

### 5. OFFSET-based Pagination at Depth

```sql
-- Slow at high offsets: must read and discard 100000 rows
SELECT * FROM events ORDER BY created_at LIMIT 20 OFFSET 100000;

-- Fast: keyset / seek pagination remembers the last row seen
SELECT * FROM events
WHERE created_at < '2024-05-01T10:00:00'   -- the last row of the previous page
ORDER BY created_at DESC LIMIT 20;
```

`OFFSET n` still *reads* the first `n` rows just to throw them away, so deep pages
get linearly slower. **Keyset pagination** seeks straight to the boundary using the
index and stays fast at any depth.

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

- **Previous:** [Data Modeling & Normalization](modeling.html)
- **Next:** [Transactions & Concurrency](transactions-and-concurrency.html) — how locks and MVCC keep concurrent queries correct.
- **Up:** [Database Design hub](./)
- See also: [Storage Engines & Recovery](storage-internals.html) for the pages and buffer pool the optimizer reads from.
