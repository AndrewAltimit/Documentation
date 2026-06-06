---
layout: docs
title: "Database Design: Operations & Monitoring"
permalink: /docs/technology/database-design/operations-and-monitoring.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Operations &amp; Monitoring</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Backups &amp; PITR, disaster recovery, VACUUM, pooling, observability, and incident response</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## Running a Database in Production

Designing a correct schema and choosing the right indexes gets you a database
that *works*. Keeping it fast, available, and recoverable at 3 a.m. on a holiday
weekend is a different discipline: **operations**. A database is the one
component of most systems that genuinely cannot be rebuilt from source — the data
*is* the value — so the operational practices around it (backups, recovery
drills, capacity planning, and monitoring) matter more than for any stateless
service.

This page is the practical, runbook-oriented companion to
[Storage Engines &amp; Recovery](storage-internals.html). Where that page explains
*how* the write-ahead log and buffer pool work, this one explains how to operate
them: how to take and verify backups, restore to an exact point in time, keep
autovacuum healthy, pool connections, read the `pg_stat_*` views, plan capacity,
and work through an incident calmly. Examples are PostgreSQL-centric because its
tooling is mature and open, but the *principles* — RPO/RTO, log shipping,
saturation metrics, pool sizing — apply to MySQL, SQL Server, and the managed
cloud equivalents alike.

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-shield-alt"></i>
    <h4>Recoverability</h4>
    <p>A backup you have never restored is a hypothesis, not a backup.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-broom"></i>
    <h4>Maintenance</h4>
    <p>MVCC produces garbage; autovacuum is what keeps tables from bloating.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-chart-line"></i>
    <h4>Observability</h4>
    <p>You cannot tune what you cannot see — instrument before you optimize.</p>
  </div>
</div>

## Backup and Restore

Everything in operations ultimately serves one goal: when something goes wrong,
get the correct data back, quickly. Two numbers frame every backup decision.

- **RPO — Recovery Point Objective**: how much data, measured in *time*, you can
  afford to lose. A nightly dump gives an RPO of up to 24 hours; continuous WAL
  archiving gives an RPO of seconds.
- **RTO — Recovery Time Objective**: how long you can afford to be *down* while
  restoring. A 2 TB logical dump might take hours to reload; a pre-warmed replica
  promotes in seconds.

$$
\text{RPO} = t_{\text{failure}} - t_{\text{last recoverable state}},
\qquad
\text{RTO} = t_{\text{service restored}} - t_{\text{failure}}
$$

You design the backup strategy backwards from the RPO/RTO the business actually
requires — over-engineering recovery is as wasteful as under-engineering it is
dangerous.

### Logical vs Physical Backups

| Aspect | Logical (`pg_dump`) | Physical (`pg_basebackup`) |
|---|---|---|
| Format | SQL/`COPY` statements | Byte-for-byte copy of data files |
| Granularity | Single table → whole DB | Whole cluster only |
| Portability | Cross-version, cross-arch | Same major version/arch |
| Restore speed | Slow (replays SQL, rebuilds indexes) | Fast (copy files, replay WAL) |
| Enables PITR | No | Yes (with WAL archiving) |
| Best for | Small DBs, migrations, single-table recovery | Large DBs, full-cluster DR, PITR |

```bash
# Logical: portable, schema-aware, slow to restore.
# Use the custom format (-Fc) so you can restore selectively and in parallel.
pg_dump -Fc -d mydb -f mydb.dump
pg_restore -d mydb_new -j 4 mydb.dump        # 4 parallel workers

# Restore a single table out of a full dump:
pg_restore -d mydb_new -t orders mydb.dump

# Physical: a consistent on-disk snapshot of the whole cluster.
pg_basebackup -D /backups/base -Ft -z -X stream -P
```

<div class="notice--warning">
  <p><strong>The cardinal rule of backups:</strong> an untested backup does not
  exist. Schedule a periodic restore into a scratch instance and run a checksum or
  row-count smoke test. Most "we had backups" outages are really "we had backup
  <em>files</em> that turned out to be empty, truncated, or unrestorable."</p>
</div>

### Point-in-Time Recovery (PITR)

A nightly base backup alone can only restore you to last midnight. **Point-in-Time
Recovery** combines a base backup with the stream of WAL segments written since, so
you can replay forward to *any* moment — crucially, to the instant *before* a bad
`DELETE` or a deployment that corrupted data.

```
  Base backup            WAL archive (continuous)         Recovery target
       |======================================================|
   Sun 02:00          ... segment 0A 0B 0C 0D 0E ...      Mon 14:32:07
       |                                                      |
       +---- restore data files ----+---- replay WAL up to here, then stop
```

The mechanism is just the crash-recovery replay from
[Storage Engines &amp; Recovery](storage-internals.html#write-ahead-logging-surviving-crashes),
extended: instead of replaying the WAL to its end, you stop at a chosen LSN, time,
named restore point, or transaction ID.

```ini
# postgresql.conf — turn on continuous archiving
wal_level = replica
archive_mode = on
archive_command = 'test ! -f /archive/%f && cp %p /archive/%f'
```

```bash
# 1. Take the periodic base backup (this is your starting point)
pg_basebackup -D /backups/base -X stream

# 2. WAL segments accumulate in /archive via archive_command

# 3. To recover to a point in time: restore the base, then set the target.
#    PostgreSQL replays archived WAL and pauses at the target.
```

```ini
# postgresql.conf on the recovery instance
restore_command = 'cp /archive/%f %p'
recovery_target_time = '2026-06-06 14:32:00+00'
recovery_target_action = 'promote'   # become primary once target reached
```

```bash
# Signal that this is an archive recovery, then start.
touch /var/lib/postgresql/data/recovery.signal
pg_ctl start
```

<div class="notice--info">
  <p>Recovery targets compose. You can stop at a <code>recovery_target_time</code>,
  a <code>recovery_target_lsn</code>, a <code>recovery_target_xid</code>, or a
  named <code>recovery_target_name</code> (created with
  <code>pg_create_restore_point('before-migration')</code>). Wrapping a risky
  migration in a named restore point gives you a precise, labeled rollback line.</p>
</div>

### Disaster Recovery

Backups protect against data loss; **disaster recovery (DR)** protects against the
loss of a whole site — a region outage, a ransomware event, an accidental
`DROP DATABASE` in production. A credible DR posture has three legs:

1. **Geographic redundancy.** Ship WAL or stream replication to a standby in a
   *different* region/availability zone. If the primary's data center disappears,
   the standby is promoted.
2. **Immutable / offsite copies.** Backups on the same host (or same cloud account)
   as the database are not DR — a compromised account can delete both. Push copies
   to object storage with versioning and an object-lock retention policy so even an
   attacker with credentials cannot erase history.
3. **The 3-2-1 rule.** Keep **3** copies of the data, on **2** different media/types,
   with **1** copy offsite. It is decades old and still the cheapest insurance you
   can buy.

```
   Primary (us-east)                    Standby (us-west)
   +-------------+   streaming repl     +-------------+
   |  PostgreSQL | ===================> |  PostgreSQL |  (hot standby)
   +------+------+   (async or sync)    +-------------+
          | archive_command
          v
   +---------------------------+
   |  Object storage (versioned,|  <-- offsite, immutable, the "1" in 3-2-1
   |  object-lock, cross-region)|
   +---------------------------+
```

<div class="notice--warning">
  <p><strong>Run game-days.</strong> A DR plan that has never been exercised will
  fail in novel ways under real pressure. Schedule a regular failover drill:
  promote the standby, point a copy of the app at it, verify, and document how long
  the whole thing actually took. That measured number is your real RTO.</p>
</div>

## Maintenance: VACUUM, ANALYZE, and Autovacuum

PostgreSQL's MVCC (see
[Transactions &amp; Concurrency](transactions-and-concurrency.html)) never updates
a row in place — an `UPDATE` writes a new row version and marks the old one dead, and
a `DELETE` just marks the row dead. Those dead tuples linger until something reclaims
them. That something is **VACUUM**.

### Why Dead Tuples Accumulate

```
UPDATE accounts SET balance = 90 WHERE id = 7;

  Heap page before        Heap page after
  +----------------+      +----------------+
  | id=7 bal=100   |      | id=7 bal=100   |  <- dead (old version)
  |                |      | id=7 bal=90    |  <- live (new version)
  +----------------+      +----------------+
```

Old versions must remain visible to transactions that started before the update.
Once no running transaction can see them (they fall below the oldest active
snapshot's *xid horizon*), they are **bloat** — wasted space that slows scans and
inflates the table on disk.

### VACUUM and ANALYZE

```sql
-- VACUUM reclaims dead tuples and makes the space reusable (does not
-- shrink the file, but the freed space is reused by future inserts).
VACUUM (VERBOSE) orders;

-- ANALYZE refreshes the planner's statistics (row counts, value
-- distributions) so EXPLAIN chooses good plans. Stale stats are a
-- top cause of "the query was fast yesterday".
ANALYZE orders;

-- Routine combined maintenance:
VACUUM ANALYZE orders;

-- VACUUM FULL rewrites the table compactly and DOES return disk to the
-- OS -- but takes an ACCESS EXCLUSIVE lock (blocks all reads/writes).
-- Use pg_repack instead for an online rewrite on busy tables.
VACUUM FULL orders;   -- locks the table; schedule a maintenance window
```

There is a second, non-negotiable job VACUUM performs: **freezing** old rows to
prevent transaction-ID wraparound. PostgreSQL's xids are 32-bit and cycle; VACUUM
"freezes" sufficiently old rows so they stay visible across the wraparound. Let this
fall too far behind and the database will, as a last resort, refuse new writes to
protect itself — a notorious and entirely preventable production outage.

### Autovacuum

You should almost never run VACUUM by hand on a schedule — the **autovacuum**
daemon does it automatically, triggering per-table when enough rows have changed.

$$
\text{vacuum threshold} = \text{base} + \text{scale factor} \times n_{\text{live tuples}}
$$

With the defaults (`base = 50`, `scale factor = 0.2`), a table is vacuumed after
roughly 20% of its rows change. That percentage is fine for small tables and far
too lazy for large, hot ones: 20% of a billion-row table is 200 million dead
tuples of bloat before autovacuum even starts. Tune the scale factor down per table:

```sql
-- A large, frequently-updated table: vacuum after ~1% churn, analyze after ~0.5%.
ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor  = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

-- Watch the daemon's progress and per-table health:
SELECT relname,
       n_live_tup,
       n_dead_tup,
       round(100.0 * n_dead_tup / nullif(n_live_tup, 0), 1) AS dead_pct,
       last_autovacuum,
       autovacuum_count
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 20;

-- Find tables creeping toward xid wraparound (act well before 2^31 ~= 2.1B):
SELECT relname, age(relfrozenxid) AS xid_age
FROM pg_class
WHERE relkind = 'r'
ORDER BY xid_age DESC
LIMIT 10;
```

<div class="notice--info">
  <p>If autovacuum "can't keep up," the usual culprits are: a long-running
  transaction or abandoned replication slot pinning the xid horizon (so VACUUM
  cannot remove any tuples newer than it), too few <code>autovacuum_max_workers</code>,
  or an over-conservative <code>autovacuum_vacuum_cost_delay</code> throttling it.
  Check <code>pg_stat_activity</code> for ancient transactions and
  <code>pg_replication_slots</code> for stale slots first.</p>
</div>

## Connection Pooling

Each PostgreSQL connection is a separate OS process with its own memory. A few
hundred is fine; a few thousand idle connections waste gigabytes of RAM and
schedule poorly. Yet web apps and serverless functions love to open a connection
per request. A **connection pooler** sits between them and the database, multiplexing
many short-lived client connections onto a small, stable set of server connections.

```
  Thousands of app                       Small fixed set
  connections (cheap)                    of DB backends (expensive)
   app  ----\                             /---- backend
   app  -----\      +-------------+      /----- backend
   app  ------+---> |  PgBouncer  | ---->+----- backend
   app  -----/      +-------------+      \----- backend
   app  ----/                             \---- (e.g. 20 total)
```

**PgBouncer** is the de-facto lightweight pooler. Its key knob is the pool *mode*:

| Mode | A server connection is returned to the pool... | Notes |
|---|---|---|
| `session` | when the client disconnects | Safe for everything; least multiplexing |
| `transaction` | at the end of each transaction | The sweet spot for web apps |
| `statement` | after each statement | Maximum reuse; forbids multi-statement txns |

```ini
; pgbouncer.ini
[databases]
mydb = host=127.0.0.1 port=5432 dbname=mydb

[pgbouncer]
pool_mode = transaction
max_client_conn = 5000      ; clients PgBouncer will accept
default_pool_size = 20      ; server connections per (user, db)
reserve_pool_size = 5       ; extra connections for bursts
```

<div class="notice--warning">
  <p><strong>Transaction mode breaks session-scoped state.</strong> Because a
  backend is shared across transactions, anything that lives at the session level —
  <code>SET</code> session variables, <code>LISTEN/NOTIFY</code>, advisory
  session locks, server-side prepared statements, <code>WITH HOLD</code> cursors —
  may land on a different backend than you expect. Keep state inside the transaction,
  or use <code>session</code> mode for those clients.</p>
</div>

### Sizing the Pool

More connections is not faster. Once every CPU and disk spindle is busy, extra
concurrent queries just add context-switching and lock contention, so throughput
falls. A widely cited starting point for an OLTP pool is:

$$
\text{pool size} \approx (\text{cores} \times 2) + \text{effective spindle count}
$$

On a 4-core box backed by SSD, that suggests roughly 8–12 server connections — far
smaller than most people guess. Measure under load and adjust: watch query latency
and the count of sessions waiting on locks, not just raw QPS.

```python
# Application-side pooling complements PgBouncer (e.g. SQLAlchemy / psycopg).
# Keep the app pool modest; let PgBouncer absorb the spikes.
engine = create_engine(
    "postgresql+psycopg://user@pgbouncer-host:6432/mydb",
    pool_size=10,        # steady-state connections this process holds
    max_overflow=5,      # temporary extra under burst
    pool_pre_ping=True,  # detect dead connections before using them
    pool_recycle=1800,   # recycle to avoid stale/expired connections
)
```

## Monitoring with the `pg_stat_*` Views

PostgreSQL exposes its internal counters through a family of `pg_stat_*` and
`pg_statio_*` system views. They are the source of truth that dashboards
(Prometheus `postgres_exporter`, Datadog, CloudWatch) ultimately scrape. Knowing
the raw views lets you investigate when a dashboard does not have the panel you need.

A useful mental model is the **USE** method — for each resource, watch
**U**tilization, **S**aturation, and **E**rrors — alongside golden signals of
latency and throughput.

### Live Activity: `pg_stat_activity`

```sql
-- What is every backend doing right now? The first stop in any incident.
SELECT pid,
       usename,
       application_name,
       state,
       wait_event_type,
       wait_event,
       now() - query_start AS runtime,
       left(query, 80)      AS query
FROM pg_stat_activity
WHERE state <> 'idle'
ORDER BY runtime DESC;

-- Find who is blocking whom (the classic "everything is stuck" query):
SELECT blocked.pid          AS blocked_pid,
       blocked.query        AS blocked_query,
       blocking.pid         AS blocking_pid,
       blocking.query       AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking
  ON blocking.pid = ANY(pg_blocking_pids(blocked.pid));
```

### Query Hotspots: `pg_stat_statements`

The single most valuable extension for performance work. It aggregates execution
statistics per normalized query, so you can rank by *total* time spent — the
queries actually worth optimizing are the ones that are slow **times** frequent.

```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Where is the server's time actually going?
SELECT calls,
       round(mean_exec_time::numeric, 2)  AS avg_ms,
       round(total_exec_time::numeric, 0) AS total_ms,
       round(100.0 * shared_blks_hit /
             nullif(shared_blks_hit + shared_blks_read, 0), 1) AS cache_hit_pct,
       left(query, 80) AS query
FROM pg_stat_statements
ORDER BY total_exec_time DESC      -- total, not mean: optimize the big rocks
LIMIT 15;
```

<div class="notice--info">
  <p>Optimize by <em>total</em> time, not average. A 5 ms query called a million
  times an hour costs the server far more than a 2-second report run once a day.
  Rank by <code>total_exec_time</code>, fix the top of the list, then re-measure.</p>
</div>

### Table and I/O Health

```sql
-- Per-table access patterns: seq scans vs index scans, dead tuples, last vacuum.
SELECT relname,
       seq_scan,
       idx_scan,
       n_live_tup,
       n_dead_tup,
       last_autovacuum,
       last_autoanalyze
FROM pg_stat_user_tables
ORDER BY seq_scan DESC;       -- big tables with high seq_scan want an index

-- Cache hit ratio per table (want > 0.99 for hot tables):
SELECT relname,
       heap_blks_hit,
       heap_blks_read,
       round(heap_blks_hit::numeric /
             nullif(heap_blks_hit + heap_blks_read, 0), 4) AS hit_ratio
FROM pg_statio_user_tables
ORDER BY heap_blks_read DESC;

-- Unused indexes (idx_scan = 0): pure write overhead, candidates to drop.
SELECT relname AS table, indexrelname AS index, idx_scan,
       pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

### The Slow Query Log

Aggregated counters tell you *what* is slow on average; the slow query log captures
the *individual* offenders with their exact parameters and timing, which is what you
need to reproduce and `EXPLAIN` them.

```sql
-- Log any statement slower than 500 ms, with the plan for slow ones.
ALTER SYSTEM SET log_min_duration_statement = 500;     -- milliseconds
ALTER SYSTEM SET auto_explain.log_min_duration = 1000; -- needs auto_explain
ALTER SYSTEM SET log_lock_waits = on;                  -- log lock-wait stalls
SELECT pg_reload_conf();
```

```sql
-- Always capture the actual plan when investigating a specific slow query:
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT ...;
-- Red flags: Seq Scan on a large table, a Sort/Hash spilling to "Disk:",
-- a row-estimate that is orders of magnitude off the actual (stale stats),
-- or a Nested Loop driving millions of inner iterations.
```

The per-query plan-reading workflow — what `EXPLAIN` output means and how to fix the
common pathologies — lives in
[Indexing &amp; Query Execution](indexing-and-queries.html); this page is about
*surfacing* the queries worth analyzing.

### Key Metrics to Alert On

| Metric | Source | Why it matters |
|---|---|---|
| Replication lag | `pg_stat_replication` (`replay_lag`) | A lagging standby = data loss on failover |
| Cache hit ratio | `pg_stat_database` | A sudden drop signals working set > RAM |
| Connections vs `max_connections` | `pg_stat_activity` | Nearing the limit means refused logins |
| Longest transaction age | `pg_stat_activity` | Long txns block VACUUM and bloat tables |
| Deadlocks / rollbacks | `pg_stat_database` | Rising counts hint at a contention bug |
| Disk free on the data volume | OS / cloud metric | A full WAL volume can halt the database |
| xid age | `pg_class.relfrozenxid` | Wraparound protection will stop writes |

## Capacity Planning

Capacity planning is forecasting *when* you will run out of headroom on each
resource — CPU, memory, disk, IOPS, connections — so you can scale **before** users
feel it, not during the incident. The method is the same regardless of which
resource you are projecting:

1. **Establish a baseline** from the monitoring views above (current QPS, dataset
   size, peak connections, daily growth).
2. **Project growth** with a simple model. Linear growth is the common case; viral
   or seasonal growth needs an exponential or seasonal model.
3. **Find the saturation point** — the utilization at which latency starts climbing
   (queueing theory says this knee is well before 100%).
4. **Subtract a safety margin** and translate the remaining runway into a date.

For a resource growing at a steady rate, the runway to a target threshold is:

$$
t_{\text{runway}} = \frac{C_{\text{threshold}} - C_{\text{now}}}{r_{\text{growth}}}
$$

where $C$ is capacity used and $r_{\text{growth}}$ is consumption per unit time.

```sql
-- Disk growth: total size of the database now (track this over time).
SELECT pg_size_pretty(pg_database_size(current_database())) AS db_size;

-- Per-table footprint, biggest first -- where growth and bloat concentrate.
SELECT relname,
       pg_size_pretty(pg_total_relation_size(relid)) AS total,
       pg_size_pretty(pg_relation_size(relid))       AS heap,
       pg_size_pretty(pg_indexes_size(relid))        AS indexes
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 20;
```

<div class="example-section">
  <h4>Worked example: disk runway</h4>
  <p>A 400 GB database is growing 12 GB/week. The data volume is 1 TB, and you do
  not want to exceed 80% (820 GB) before adding capacity. Remaining runway:</p>

  $$
  t_{\text{runway}} = \frac{820\,\text{GB} - 400\,\text{GB}}{12\,\text{GB/week}}
  = 35\ \text{weeks}
  $$

  <p>Roughly eight months of headroom — comfortable, but note that adding a WAL
  archive, more indexes, or a busier write workload all steepen the slope, so
  re-run the projection whenever the growth rate changes.</p>
</div>

<div class="notice--info">
  <p>Plan for the resource that runs out <em>first</em>. A database can have ample
  disk but be starved on IOPS, memory, or connection slots. Track every dimension
  and let the nearest ceiling drive the upgrade timeline.</p>
</div>

## Handling a Production Incident

When a database incident hits — queries timing out, replication broken, disk
nearly full — a calm, repeatable procedure beats improvisation. The same loop
applies whether you are an SRE or the lone maintainer.

1. **Assess and stabilize.** Is the database up? Accepting connections? Read the
   golden signals: connection count, longest-running query, replication lag, disk
   free. The first goal is to stop the bleeding, not to find root cause.

   ```sql
   -- One screen of triage:
   SELECT count(*) FILTER (WHERE state = 'active')  AS active,
          count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn,
          count(*)                                   AS total,
          max(now() - xact_start)                    AS longest_txn
   FROM pg_stat_activity;
   ```

2. **Mitigate the immediate harm.** This is reversible first aid, not the fix:
   cancel or terminate a runaway query, kill a stuck `idle in transaction` session
   that is blocking VACUUM, or free disk by archiving WAL that has been backed up.

   ```sql
   SELECT pg_cancel_backend(pid);     -- ask the query to stop (gentle)
   SELECT pg_terminate_backend(pid);  -- kill the whole backend (forceful)
   ```

3. **Diagnose root cause** once the fire is out — using `pg_stat_statements`, the
   slow-query log, and `EXPLAIN`. Resist the urge to skip this step: an
   un-diagnosed incident *will* recur.

4. **Recover or fail over.** If the primary is unrecoverable, promote a standby; if
   data was corrupted or wrongly deleted, this is where PITR earns its keep —
   restore to the moment before the bad change.

5. **Write the postmortem.** Blameless, focused on the *system* gap that let it
   happen. The output is concrete: an alert that would have caught it sooner, an
   autovacuum setting changed, a runbook step added, a capacity threshold lowered.

<div class="notice--warning">
  <p><strong>Slow down to speed up.</strong> Most catastrophic database incidents
  are made worse by a panicked second action — a <code>VACUUM FULL</code> that locks
  the table mid-outage, a restore over the only good copy, a failover with a lagging
  standby that loses committed data. State the change, confirm it is reversible, then
  act.</p>
</div>

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Test your restores</h4>
    <p>An unrestored backup is a guess. Automate periodic test restores and a smoke check so you discover broken backups in a drill, not in a disaster.</p>
  </div>
  <div class="takeaway-card">
    <h4>PITR gives you a time machine</h4>
    <p>A base backup plus continuous WAL archiving lets you rewind to the instant before a bad change — your defense against human error, not just hardware failure.</p>
  </div>
  <div class="takeaway-card">
    <h4>Keep autovacuum healthy</h4>
    <p>MVCC generates dead tuples; tune <code>autovacuum_*_scale_factor</code> down on large hot tables, and never let xid age drift toward wraparound.</p>
  </div>
  <div class="takeaway-card">
    <h4>Pool connections, modestly</h4>
    <p>PgBouncer in transaction mode multiplexes thousands of clients onto a small backend pool. Smaller pools often mean lower latency.</p>
  </div>
  <div class="takeaway-card">
    <h4>Instrument before you tune</h4>
    <p>The <code>pg_stat_*</code> views and <code>pg_stat_statements</code> show where time goes. Optimize by total time, alert on saturation.</p>
  </div>
  <div class="takeaway-card">
    <h4>Have a calm incident loop</h4>
    <p>Stabilize, mitigate, diagnose, recover, learn. Slowing down to confirm reversibility prevents the second mistake that turns an incident into an outage.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Related pages</h4>
  <ul>
    <li><a href="storage-internals.html">Storage Engines &amp; Recovery</a> — the WAL, buffer pool, and checkpoints these operations rest on.</li>
    <li><a href="indexing-and-queries.html">Indexing &amp; Query Execution</a> — reading <code>EXPLAIN</code> to fix the slow queries you surface here.</li>
    <li><a href="transactions-and-concurrency.html">Transactions &amp; Concurrency</a> — why MVCC produces the dead tuples VACUUM reclaims.</li>
    <li><a href="distributed-and-nosql.html">Distributed Databases &amp; NoSQL</a> — replication and failover at scale.</li>
    <li><a href="./">Database Design hub</a> — the rest of the deep dive.</li>
  </ul>
</div>
