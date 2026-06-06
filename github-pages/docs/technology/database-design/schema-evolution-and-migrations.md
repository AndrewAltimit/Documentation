---
layout: docs
title: "Database Design: Schema Evolution & Migrations"
permalink: /docs/technology/database-design/schema-evolution-and-migrations.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Schema Evolution &amp; Migrations</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Migration tooling, zero-downtime expand–contract, backfills, online schema change, and safe rollbacks</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## Why Schemas Change

A schema is never finished. New features need new columns, growth forces new indexes, refactors split and merge tables, and yesterday's "temporary" `VARCHAR(50)` turns out to be too small. The hard part is not writing the `ALTER TABLE` — it is changing a schema that a *running* application depends on, often against a table holding hundreds of millions of rows, without dropping requests or corrupting data.

This page covers the discipline that makes schema change routine instead of terrifying: **versioned migrations** managed by tooling, the **expand–contract** pattern for zero-downtime changes, **backfills** that don't melt the database, **online schema change** mechanics on MySQL and PostgreSQL, and **rollback** strategies that actually work.

> **Mental model.** Treat the schema as code. Every change is a small, ordered, reviewed, version-controlled script that runs forward in every environment in the same sequence. The database's current shape is fully described by *which migrations have been applied*.

## Migrations as Version Control for the Database

A **migration** is a discrete, ordered change to the schema (and sometimes the data) expressed as a script. A migration tool records which migrations a given database has applied, so any environment — a developer laptop, CI, staging, production — can be brought from its current version to the latest by replaying the missing migrations in order.

The tool maintains a bookkeeping table inside the database itself. In Flyway it is `flyway_schema_history`; in Alembic it is `alembic_version`; in Rails/ActiveRecord it is `schema_migrations`. Conceptually:

```sql
CREATE TABLE schema_migrations (
    version      VARCHAR(255) PRIMARY KEY,  -- e.g. "0042" or a hash
    description  TEXT,
    checksum     VARCHAR(64),               -- detects edits to already-applied scripts
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    success      BOOLEAN NOT NULL
);
```

Two broad styles exist:

- **Versioned / sequential migrations** (Flyway, Liquibase, golang-migrate, Rails): each migration has a monotonic version (`V1__`, `V2__`, …). The tool applies anything newer than the recorded high-water mark. Order is total and deterministic.
- **Revision-graph migrations** (Alembic, Django): each migration names its parent revision, forming a directed acyclic graph. This handles parallel feature branches gracefully — two branches each add a migration whose `down_revision` is the same ancestor, and a later **merge migration** reconciles them — but you must resolve "multiple heads" before deploying.

### Golden rules of migrations

1. **Migrations are immutable once merged.** Never edit a migration that has run anywhere beyond your own laptop. Checksums exist precisely to catch this; editing a shipped migration means environments silently diverge. To fix a mistake, write a *new* migration.
2. **Migrations are ordered and idempotent at the tool level.** Re-running the suite must be a no-op once everything is applied.
3. **Every migration is reviewed like code** — in a pull request, with the same scrutiny as application changes, because a bad `ALTER` can lock a production table for hours.
4. **Schema and data migrations are separated where possible.** Mixing a giant `UPDATE` into a DDL migration ties a slow backfill to a transaction that may hold locks.
5. **Forward-only in production is a valid policy.** Many teams never run `down` migrations against production (see [Rollbacks](#rollbacks-and-recovery)); they roll *forward* with a corrective migration instead.

## The Tooling Landscape

The three most common dedicated tools are **Flyway**, **Liquibase**, and **Alembic**, alongside framework-bundled tools (Rails, Django, Entity Framework) and language-agnostic CLIs (golang-migrate, dbmate). They differ mainly in how migrations are *authored* and how much database-independence they promise.

| Tool | Authoring format | Versioning model | Database abstraction | Typical ecosystem |
|---|---|---|---|---|
| **Flyway** | Plain SQL (`.sql`) or Java | Sequential `V{n}__desc.sql` | None — you write native SQL | JVM, polyglot via CLI |
| **Liquibase** | XML / YAML / JSON / SQL "changesets" | Ordered changelog file | High — abstract change types generate dialect SQL | JVM, enterprise |
| **Alembic** | Python (`upgrade()`/`downgrade()`) | Revision DAG (`down_revision`) | Via SQLAlchemy `op.*` | Python / SQLAlchemy |
| **Rails / ActiveRecord** | Ruby DSL | Timestamped sequential | High, via the ORM | Ruby on Rails |
| **golang-migrate / dbmate** | Paired `.up.sql` / `.down.sql` | Sequential numbered | None | Go, polyglot |

### Flyway

Flyway favors **plain SQL** and convention over configuration. You drop versioned files into a migrations directory and run `flyway migrate`.

```
db/migration/
  V1__create_users.sql
  V2__add_email_index.sql
  V3__add_orders_table.sql
  R__refresh_reporting_view.sql   -- repeatable: re-runs whenever its checksum changes
```

```sql
-- V2__add_email_index.sql
CREATE UNIQUE INDEX CONCURRENTLY idx_users_email ON users (email);
```

The naming prefix carries the semantics: `V` = versioned (run once, in order), `U` = undo (paid feature), `R` = repeatable (re-applied whenever the file's checksum changes — ideal for views, stored procedures, and grants where you want *the latest definition* rather than a diff). Flyway records each `V` migration's checksum; if you edit a file it has already applied, `flyway validate` fails.

### Liquibase

Liquibase models change as a **changelog** of independent **changesets**, each identified by `id` + `author` + file path. Its differentiator is database-agnostic change types: you describe *intent* (`addColumn`, `createIndex`) and Liquibase emits the right dialect SQL.

```yaml
# changelog.yaml
databaseChangeLog:
  - changeSet:
      id: add-phone-to-users
      author: andrew
      changes:
        - addColumn:
            tableName: users
            columns:
              - column: { name: phone, type: VARCHAR(32) }
      rollback:
        - dropColumn: { tableName: users, columnName: phone }
```

Liquibase supports **preconditions** (skip or fail a changeset unless the DB is in an expected state), **contexts/labels** (run subsets per environment), and explicit `rollback` blocks per changeset. The abstraction is powerful for multi-database products but leaks for anything dialect-specific (partial indexes, `CONCURRENTLY`, generated columns), where you fall back to raw `sql` changes anyway.

### Alembic

Alembic (the migration component of SQLAlchemy) authors migrations in **Python** and links them in a **revision graph**. It can *autogenerate* a migration by diffing your ORM models against the live database — a huge convenience, but the output must always be reviewed (autogenerate misses things like server defaults, `CHECK` constraints, and renames, which it sees as drop+add).

```python
# alembic/versions/8f3a_add_status_to_orders.py
revision = "8f3a"
down_revision = "7b21"

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column(
        "orders",
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
    )
    op.create_index("ix_orders_status", "orders", ["status"])

def downgrade():
    op.drop_index("ix_orders_status", table_name="orders")
    op.drop_column("orders", "status")
```

Because revisions form a DAG, two feature branches can each add a migration off the same parent. After merging both, `alembic heads` shows two heads; you reconcile with `alembic merge` to produce an empty merge revision that depends on both, restoring a single head.

## What `ALTER TABLE` Actually Costs

Before designing zero-downtime changes you must know which DDL is cheap and which rewrites the whole table or takes a blocking lock. The answer is engine-specific.

**PostgreSQL** takes an `ACCESS EXCLUSIVE` lock for most DDL, but many operations are now *metadata-only* and therefore near-instant once the lock is acquired:

- **Cheap (metadata only):** `ADD COLUMN` with no default *or* with a constant default (PG 11+ stores the default in catalog metadata), `DROP COLUMN`, renaming columns/tables, adding a `CHECK ... NOT VALID` constraint, dropping a constraint.
- **Expensive (table rewrite):** changing a column type in a way that needs reformatting, `ADD COLUMN ... DEFAULT <volatile expression>`, setting `NOT NULL` on an existing column (pre-PG 12 scans the whole table; PG 12+ can use a validated `CHECK` to skip the scan).
- **The lock-duration trap:** even a metadata-only change must *acquire* `ACCESS EXCLUSIVE`. If a long-running query holds the table, your `ALTER` queues behind it — and every new query queues behind *your* `ALTER*. Always run DDL with a short `lock_timeout` so it backs off instead of stampeding:

```sql
SET lock_timeout = '3s';
ALTER TABLE orders ADD COLUMN status TEXT;  -- retried by your migration runner if it times out
```

**Indexes** are the classic blocker. A plain `CREATE INDEX` locks writes for the whole build. PostgreSQL's `CREATE INDEX CONCURRENTLY` (and `DROP INDEX CONCURRENTLY`) builds without blocking writes, at the cost of two table scans and the inability to run inside a transaction — so such migrations must be marked non-transactional in your tool (`transactional: false` in Flyway, `op.create_index(..., postgresql_concurrently=True)` with an autocommit block in Alembic).

**MySQL/InnoDB** classifies each DDL as `INSTANT`, `INPLACE`, or `COPY` (the `ALGORITHM=` clause). `INSTANT` (8.0+) adds/drops columns via metadata only; `INPLACE` rebuilds indexes without copying the table but may still block briefly; `COPY` rebuilds the entire table and blocks writes for its duration. The `LOCK=` clause (`NONE`/`SHARED`/`EXCLUSIVE`) controls concurrent DML. When the native `INPLACE`/`INSTANT` path isn't available, teams reach for external online-schema-change tools (below).

## Zero-Downtime Migrations: The Expand–Contract Pattern

The central technique for changing schema under a live application is **expand–contract** (also called *parallel change* or *expand–migrate–contract*). The insight: you cannot change the schema and the application in the same instant, so you make the change in stages where **old and new code both work against the database** at all times.

```
Time ──────────────────────────────────────────────────────────►

  [EXPAND]            [MIGRATE / BACKFILL]        [CONTRACT]
  Add the new,        Dual-write + backfill;      Remove the old
  backward-           switch reads to new         structure once
  compatible          once data is consistent     nothing uses it
  structure
   │                        │                          │
   ▼                        ▼                          ▼
 old code OK            old code OK                old code GONE
 new code OK            new code OK                new code OK
```

Each phase is independently deployable and independently reversible, and **at no point** is there a moment where the currently deployed application is incompatible with the current schema. That property is what gives you zero downtime.

### Worked example: rename `users.name` to `users.full_name`

A naive `ALTER TABLE users RENAME COLUMN name TO full_name` is fatal: the running app still issues `SELECT name`, which now errors. Expand–contract turns it into five safe, ordered steps:

1. **Expand (migration).** Add the new column. Cheap metadata change.
   ```sql
   ALTER TABLE users ADD COLUMN full_name TEXT;
   ```
2. **Dual-write (deploy app v2).** Application writes both columns; reads still come from `name`.
   ```sql
   UPDATE users SET name = $1, full_name = $1 WHERE id = $2;
   ```
3. **Backfill (migration/job).** Copy historical rows in batches (see next section).
   ```sql
   UPDATE users SET full_name = name WHERE full_name IS NULL;  -- batched
   ```
4. **Switch reads (deploy app v3).** Reads now use `full_name`; the app stops depending on `name`.
5. **Contract (migration).** Once no deployed version references `name`, drop it.
   ```sql
   ALTER TABLE users DROP COLUMN name;
   ```

If anything looks wrong before step 5, you simply *stop advancing* — the old column is still populated, so rolling back the app is harmless. The contract step is the only irreversible one, and you take it last, after the new path has soaked in production.

### Other changes in the expand–contract frame

- **Adding a `NOT NULL` column:** add it nullable (or with a default) → backfill → add the `NOT NULL` constraint (on PostgreSQL, add a `CHECK (col IS NOT NULL) NOT VALID`, `VALIDATE CONSTRAINT` in a separate step to avoid a long lock, then optionally set `NOT NULL`). Never add `NOT NULL` with no default in one shot against a populated table.
- **Splitting a column** (`address` → `street`, `city`, `zip`): expand with three new columns, dual-write the parsed parts, backfill, switch reads, contract the old column.
- **Changing a column's type** (`INT` → `BIGINT` for an id about to overflow): add a new `BIGINT` column, dual-write, backfill, switch reads/writes, then swap. For a primary key this is a major operation; many teams instead provision `BIGINT` ids up front precisely to avoid it.
- **Moving a column to another table / introducing a foreign key:** add the FK column as nullable and *not validated*, backfill, `VALIDATE CONSTRAINT` separately, then enforce.

### The compatibility rule

Expand–contract works only if you respect one rule at every step:

> **Each deployed version of the application must be compatible with both the schema before and the schema after the migration that ships alongside it.**

Concretely: a column you intend to drop must first stop being *read*, then stop being *written*, and only then be dropped — three separate deploys/migrations, never one. Adding a `NOT NULL` column the old code doesn't know about will break the old code's `INSERT`s unless the column has a default. The pattern is mechanical once internalized, and most migration mistakes are simply a violation of this single rule.

## Backfills

A **backfill** populates a newly added column (or new table) for the rows that already exist. The danger is volume: `UPDATE big_table SET new_col = f(old_col)` as one statement takes a single long transaction that bloats the WAL/undo log, holds locks, blocks autovacuum, and can lock out writers for minutes. Backfills must be **batched, throttled, and resumable**.

```python
# Batched, resumable backfill driven by the primary key.
# Each batch is its own transaction so locks are released between batches.
BATCH = 5_000
last_id = 0
while True:
    rows = db.execute(
        """
        UPDATE users
           SET full_name = name
         WHERE id > :last_id
           AND id <= :last_id + :batch
           AND full_name IS NULL
        RETURNING id
        """,
        last_id=last_id, batch=BATCH,
    )
    if not rows:
        break
    last_id += BATCH
    db.commit()
    time.sleep(0.05)   # throttle: give other transactions and replication room
```

Key properties of a safe backfill:

- **Bounded batches** keyed on an indexed column (usually the PK) so each statement touches a small, predictable range and releases locks promptly.
- **Resumable** — if it dies at id 4,000,000 you restart from there; the `WHERE new_col IS NULL` guard keeps it idempotent.
- **Throttled** with a small sleep and ideally *adaptive* — watch replication lag and back off when replicas fall behind, because a fast backfill on the primary can starve read replicas serving live traffic.
- **Run outside the DDL migration.** Add the column in a fast migration; run the backfill as a separate job (a one-off script, a worker task, or a dedicated "data migration" your framework distinguishes from schema migrations). This keeps schema deploys fast and lets you pause/resume the slow part.
- **Mind triggers and dual-writes.** During the migrate phase, new writes already set `new_col`; the backfill only needs to fill *historical* rows. The `IS NULL` predicate naturally avoids fighting concurrent writes.

For very large tables, consider building the backfilled state in a shadow table and swapping, which is exactly what online-schema-change tools automate.

## Online Schema Change Tools

When the engine's native online DDL isn't enough — typically a `COPY`-algorithm `ALTER` on a huge MySQL/InnoDB table — external tools perform the change by the **shadow-table-and-swap** method:

1. Create an empty **ghost/shadow** table with the *desired* new schema.
2. Copy existing rows into it in throttled batches.
3. Capture concurrent changes to the original (via triggers, or by tailing the binary log) and apply them to the ghost table so it stays current.
4. When the ghost has caught up, perform an **atomic rename** to swap it in for the original; drop the old table.

The two dominant MySQL tools:

- **pt-online-schema-change** (Percona Toolkit) uses **triggers** on the original table to mirror writes into the ghost table. Simple and battle-tested, but adds trigger overhead to every write and conflicts with pre-existing triggers.
- **gh-ost** (GitHub) is **triggerless**: it reads the **binary log** to replay changes onto the ghost table, so it imposes no write-path trigger overhead, is more easily **pausable/throttleable**, and can run the copy from a replica. It has become the default choice for large MySQL fleets.

```bash
# gh-ost: add a column to a 500M-row table with no write downtime
gh-ost \
  --host=primary.db --database=shop --table=orders \
  --alter="ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending'" \
  --max-load=Threads_running=25 \   # throttle when the DB is busy
  --chunk-size=1000 \
  --cut-over=atomic \
  --execute
```

**PostgreSQL** rarely needs this because most changes are metadata-only or covered by `CREATE INDEX CONCURRENTLY`. For the cases it doesn't cover (rewriting a column type on a hot table), `pg_repack` rebuilds tables and indexes online by the same shadow-and-swap idea, and tools like `pgroll` implement expand–contract with versioned views so old and new schema versions are queryable simultaneously during the transition.

> **Trade-off.** Online tools turn a blocking `ALTER` into a slow, throttled background copy. You pay in elapsed time and extra disk (a full second copy of the table) to buy availability. Schedule them off-peak, watch replication lag, and keep the binlog/WAL retention long enough to cover the whole run.

## Versioning, Branching, and CI/CD

Migrations are part of the deployment pipeline, not a manual afterthought.

- **Single source of order.** Sequential tools enforce a total order by version number; teams on different branches must rebase so versions don't collide. DAG tools (Alembic) tolerate parallel development but require resolving multiple heads before merge — make "no multiple heads" a CI check.
- **Run migrations in CI** against a throwaway database on every PR: apply all migrations from empty, then assert the resulting schema matches the ORM/models. This catches autogenerate misses and dialect bugs early.
- **Deploy ordering with expand–contract.** Because each phase ships with compatible code, the natural pipeline is: run *expand* migrations *before* the app deploy that depends on them, run *contract* migrations *after* every old app version is gone. Many teams run migrations as a separate, gated step (a Kubernetes Job/init container, or a manual approval) rather than inside the app boot, so a slow or failed migration doesn't crash-loop every pod.
- **Guard against destructive DDL.** Lint migrations in CI for dangerous patterns — `DROP COLUMN`/`DROP TABLE`, type changes, `NOT NULL` without default, plain `CREATE INDEX` on big tables. Tools like Squawk (Postgres) and `gh-ost`-aware linters fail the build or require an explicit override.
- **Pin a `lock_timeout` and `statement_timeout`** in the migration session so a DDL that can't get its lock fails fast and is retried, instead of jamming the table behind a queue of blocked queries.

## Rollbacks and Recovery

When a migration goes wrong, you have three recovery strategies, in rough order of preference.

### 1. Roll forward (preferred)

Most mature teams treat production migrations as **forward-only**: they do not run `down`/`downgrade` scripts in production. If a change is bad, they write and deploy a *new* corrective migration. The reasons are practical:

- `down` scripts are written but rarely *tested* against production-scale data, so they fail exactly when you need them most.
- Many forward changes are not cleanly reversible: a backfill that has lost the original values, a dropped column whose data is gone, a merge that can't be un-merged.
- Expand–contract already gives you a safe abort: if you haven't reached the *contract* step, the old structure is intact and you simply redeploy the previous app version. No DDL undo needed.

### 2. Reverse migration (`down`)

Versioned and DAG tools support paired down-migrations (`downgrade()` in Alembic, `.down.sql` in golang-migrate, `rollback` blocks in Liquibase). These are genuinely useful in development and CI — `alembic downgrade -1` to step back, test, redo. In production they work only when the change is *truly* reversible and *no data has been lost*: dropping an index you just added, removing a still-empty column. Treat a down-migration as a convenience for additive changes, not a safety net for destructive ones.

```python
def downgrade():
    # Safe to reverse: the column was new and we only ever wrote to it additively.
    op.drop_index("ix_orders_status", table_name="orders")
    op.drop_column("orders", "status")
```

### 3. Restore from backup / PITR (last resort)

If a migration *destroyed or corrupted data* and there is no forward fix, you fall back to the storage layer: restore from a backup and replay the write-ahead log to a point in time just before the bad migration (**point-in-time recovery**). This is covered in depth in [Storage Engines &amp; Recovery](storage-internals.html); the relevant discipline here is to **take a backup (or a PITR-capable snapshot) immediately before any irreversible migration**, and to verify you can restore it.

### Making rollback safe by design

- **Wrap reversible DDL in a transaction** where the engine allows it. PostgreSQL is *transactional DDL* — `BEGIN; ALTER ...; ALTER ...; COMMIT;` either applies fully or not at all, so a failed multi-statement migration leaves no half-applied mess. MySQL is **not** transactional for DDL: each statement auto-commits, so a multi-statement migration can partially apply and you must hand-write the cleanup. Keep MySQL migrations to one DDL statement where practical.
- **Never combine the irreversible step with anything else.** A migration that drops a column should drop *only* that column, so its blast radius is exactly one understood change.
- **Sequence destructive changes last and separately**, after a soak period, so a rollback before that point never has to undo data loss.

## Best Practices Checklist

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>Treat schema as versioned code</h4>
    <p>Every change is an ordered, reviewed, immutable migration recorded in a history table. To fix a mistake, write a new migration — never edit a shipped one.</p>
  </div>
  <div class="takeaway-card">
    <h4>Expand before you contract</h4>
    <p>Add new structure, dual-write, backfill, switch reads, then remove the old structure — across multiple deploys so old and new code always work.</p>
  </div>
  <div class="takeaway-card">
    <h4>Batch and throttle backfills</h4>
    <p>Update in small, resumable, key-ranged batches with their own transactions and a sleep; watch replication lag and back off.</p>
  </div>
  <div class="takeaway-card">
    <h4>Build indexes concurrently</h4>
    <p>Use <code>CREATE INDEX CONCURRENTLY</code> (Postgres) or gh-ost/pt-osc (MySQL) so index builds don't block writes; mark such migrations non-transactional.</p>
  </div>
  <div class="takeaway-card">
    <h4>Set a short lock timeout</h4>
    <p>DDL must back off rather than stampede. <code>lock_timeout</code> stops an <code>ALTER</code> from queuing the whole table behind it.</p>
  </div>
  <div class="takeaway-card">
    <h4>Prefer rolling forward</h4>
    <p>Production rollback is a corrective migration or, for data loss, a PITR restore. Down-scripts are for dev; back up before anything irreversible.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Continue the deep dive</h4>
  <ul>
    <li><a href="modeling.html">Data Modeling &amp; Normalization</a> — the schema you are evolving and the integrity rules migrations must preserve.</li>
    <li><a href="indexing-and-queries.html">Indexing &amp; Query Execution</a> — why <code>CREATE INDEX CONCURRENTLY</code> matters and how indexes are built.</li>
    <li><a href="transactions-and-concurrency.html">Transactions &amp; Concurrency</a> — locking and MVCC, the machinery that makes online DDL and backfills safe.</li>
    <li><a href="storage-internals.html">Storage Engines &amp; Recovery</a> — WAL, backups, and point-in-time recovery for migration rollback.</li>
    <li><a href="./">Database Design hub</a> — the rest of the deep-dive series.</li>
  </ul>
</div>
