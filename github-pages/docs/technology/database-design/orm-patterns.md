---
layout: docs
title: "Database Design: ORMs & Data-Access Patterns"
permalink: /docs/technology/database-design/orm-patterns.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">ORMs &amp; Data-Access Patterns</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Object-relational mapping, the impedance mismatch, the N+1 problem, and when to drop to SQL</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## Why an ORM Exists

Applications model the world as **objects** — a `Customer` with a list of `Order`s, each holding `LineItem`s. Relational databases model the world as **flat tables** joined by keys. An **Object-Relational Mapper (ORM)** sits between these two worlds, translating rows into objects and method calls into SQL.

```python
# Without an ORM: you write SQL strings and hydrate objects by hand
rows = db.execute(
    "SELECT id, name, email FROM customers WHERE country = %s", ("USA",)
).fetchall()
customers = [Customer(id=r[0], name=r[1], email=r[2]) for r in rows]

# With an ORM: you describe the mapping once and query in your own language
customers = session.query(Customer).filter_by(country="USA").all()
```

An ORM buys you three things:

- **Productivity** — CRUD, joins, and pagination become method calls instead of hand-written SQL and result parsing.
- **Portability** — the same code can target PostgreSQL, MySQL, or SQLite because the ORM emits dialect-specific SQL.
- **Safety** — parameters are bound automatically, which closes off most SQL-injection vectors (see [Transactions &amp; Concurrency](transactions-and-concurrency.html) for the security discussion).

It also has a cost: the abstraction hides what SQL actually runs, and a careless query in object-space can explode into thousands of database round-trips. The rest of this page is about getting the productivity without paying that cost.

## The Object-Relational Impedance Mismatch

The "impedance mismatch" is the structural friction between the object model and the relational model. The two were designed under different assumptions, and the ORM has to bridge each gap:

<div class="comparison-table">

| Object world | Relational world | The friction |
|---|---|---|
| Object identity (`a is b`) | Primary-key equality | Two queries can return two distinct objects for the same row unless an identity map deduplicates them. |
| References / pointers | Foreign keys + joins | Following `order.customer` may trigger a hidden query. |
| Inheritance & polymorphism | Flat tables, no subtype | Must be faked: single-table, joined-table, or concrete-table mapping. |
| Collections (`list`, `set`) | Rows in a child table | Loading a collection is a separate `SELECT`; mutating it is a diff. |
| Encapsulation / private state | Public columns | The schema leaks structure the object tried to hide. |
| Methods / behavior | Data only | Behavior lives in the app; the database sees only state. |
| Nested / graph structure | Normalized, decomposed | Saving one aggregate may touch many tables in one transaction. |

</div>

### Mapping inheritance

Inheritance is the sharpest corner of the mismatch. Suppose `Employee` has subtypes `Manager` and `Engineer`. Three classic strategies:

1. **Single-table inheritance (STI)** — one table with all columns plus a discriminator. Fast (no joins) but sparse: columns that belong only to `Manager` are `NULL` for every engineer.
2. **Joined-table (class-table) inheritance** — a base `employees` table plus one table per subtype, joined on the shared key. Normalized but every read of a subtype needs a join.
3. **Concrete-table inheritance** — one independent table per concrete class with all columns duplicated. No joins, but querying "all employees" requires a `UNION`.

```python
# SQLAlchemy single-table inheritance
class Employee(Base):
    __tablename__ = "employees"
    id   = Column(Integer, primary_key=True)
    type = Column(String(20))          # discriminator column
    name = Column(String(100))
    __mapper_args__ = {"polymorphic_on": type, "polymorphic_identity": "employee"}

class Manager(Employee):
    reports_to = Column(Integer)        # NULL for non-managers
    __mapper_args__ = {"polymorphic_identity": "manager"}
```

There is no universally correct choice — it is a normalization-vs-join-cost trade-off, exactly the tension covered in [Data Modeling &amp; Normalization](modeling.html).

## The Major ORMs at a Glance

ORMs cluster into two design philosophies, named by Martin Fowler:

- **Active Record** — a model object *is* a row and knows how to save itself (`user.save()`). Simple and discoverable; couples domain objects to persistence.
- **Data Mapper** — a separate session/repository moves data between plain objects and the database. More ceremony, cleaner domain model.

<div class="comparison-table">

| ORM | Language | Pattern | Notable trait |
|---|---|---|---|
| **SQLAlchemy** | Python | Data Mapper (Core + ORM) | Powerful query builder; you can drop to SQL expressions without leaving the API. |
| **Django ORM** | Python | Active Record | Tightly integrated with Django; lazy `QuerySet`s; migrations built in. |
| **Sequelize** | JavaScript/Node | Active Record | Promise-based; broad SQL-dialect support. |
| **Prisma** | TypeScript/Node | Query builder + client | Schema-first; generates a fully typed client; no lazy loading (explicit `include`). |
| **Hibernate / JPA** | Java | Data Mapper | The JPA reference implementation; persistence context, dirty checking, HQL/JPQL. |

</div>

The same query in four of them:

```python
# SQLAlchemy (2.0 style)
session.execute(
    select(Order).where(Order.total > 1000)
).scalars().all()
```

```python
# Django ORM
Order.objects.filter(total__gt=1000)
```

```javascript
// Sequelize
await Order.findAll({ where: { total: { [Op.gt]: 1000 } } });
```

```javascript
// Prisma
await prisma.order.findMany({ where: { total: { gt: 1000 } } });
```

All four emit essentially the same SQL — `SELECT ... FROM orders WHERE total > 1000` — but Django and Sequelize objects can also save themselves, while SQLAlchemy and Prisma keep persistence in a separate session/client.

## Query Builders vs Raw SQL

Most ORMs expose a **query builder**: a fluent, composable API that produces SQL programmatically. It sits between full ORM object-mapping and raw strings.

```python
# SQLAlchemy Core query builder — returns rows, not mapped objects
stmt = (
    select(Customer.name, func.sum(Order.total).label("ltv"))
    .join(Order, Order.customer_id == Customer.id)
    .where(Customer.country == "USA")
    .group_by(Customer.id)
    .having(func.sum(Order.total) > 1000)
)
```

A query builder is the right tool when you need **dynamic SQL** — filters, sorts, or joins assembled at runtime — because you compose Python/JS objects instead of concatenating strings (which is both unsafe and unreadable):

```python
stmt = select(Product)
if max_price is not None:
    stmt = stmt.where(Product.price <= max_price)   # conditionally add clauses
if category:
    stmt = stmt.where(Product.category == category)
stmt = stmt.order_by(Product.name)
```

**Raw SQL** is still the right tool when:

- The query uses dialect-specific features the ORM cannot express (window functions with complex frames, recursive CTEs, `LATERAL` joins, vendor JSON operators).
- You are tuning a hot path and need exact control over the plan.
- The query is reporting/analytics-shaped and maps poorly onto objects.

Every mature ORM provides a safe escape hatch that **still binds parameters** — never interpolate user input into the string:

```python
# SQLAlchemy — bound parameters via :name placeholders
rows = session.execute(
    text("SELECT * FROM orders WHERE total > :min ORDER BY total DESC"),
    {"min": 1000},
).all()
```

```javascript
// Prisma — tagged template binds ${} as parameters, NOT string interpolation
const orders = await prisma.$queryRaw`
  SELECT * FROM "Order" WHERE total > ${minTotal}
`;
```

<div class="tip-card">
  <h4>The cardinal rule</h4>
  <p>Use the ORM/query builder for the 95% of queries that are routine CRUD and simple joins; reach for parameterized raw SQL for the 5% that are performance-critical or use features the ORM can't express. Never build SQL by string concatenation of user input — that is the classic SQL-injection bug.</p>
</div>

## The N+1 Query Problem

The single most common ORM performance bug is the **N+1 problem**. It appears when you load a list of N parent objects and then touch a related attribute inside a loop, triggering one extra query per parent.

```python
# 1 query to load the customers...
customers = session.query(Customer).all()
for c in customers:
    # ...then N MORE queries, one per customer, hidden behind attribute access
    print(c.name, len(c.orders))     # each c.orders fires its own SELECT
```

If `customers` returns 1,000 rows, this executes **1 + 1,000 = 1,001** queries. Each query is fast, but the round-trip latency dominates: at 1 ms per round-trip that loop spends a full second waiting on the network, almost all of it avoidable.

### Why it hides so well

The fix-free version *works correctly* and is fast in development with ten test rows. It only collapses in production with real data volume, which is what makes N+1 so insidious. The cause is **lazy loading**: by default many ORMs defer loading a relationship until the attribute is first accessed, and an access inside a loop becomes a query inside a loop.

### Eager loading: the fix

**Eager loading** tells the ORM to fetch the related rows up front, collapsing N+1 into a small constant number of queries. Two main mechanics:

- **JOIN-based** — one query with a `JOIN` returns parents and children together. Best for to-one relationships; can multiply rows for to-many.
- **Second-query (`IN` / `SELECT IN`)** — load parents, collect their keys, then `WHERE child.parent_id IN (...)` in one follow-up query. Best for to-many relationships; total queries = 2 regardless of N.

```python
# SQLAlchemy: selectinload issues 2 queries total (parents, then children IN keys)
from sqlalchemy.orm import selectinload
customers = (
    session.query(Customer)
    .options(selectinload(Customer.orders))
    .all()
)
for c in customers:
    print(c.name, len(c.orders))     # no extra queries — orders already loaded

# joinedload instead emits a single LEFT JOIN
from sqlalchemy.orm import joinedload
session.query(Customer).options(joinedload(Customer.orders)).all()
```

```python
# Django: select_related (JOIN, to-one) and prefetch_related (2nd query, to-many)
customers = Customer.objects.prefetch_related("orders")        # to-many -> 2 queries
orders   = Order.objects.select_related("customer")            # to-one  -> 1 JOIN
```

```javascript
// Sequelize: include eager-loads the association
await Customer.findAll({ include: [{ model: Order }] });

// Prisma: include is explicit and required (Prisma has no lazy loading)
await prisma.customer.findMany({ include: { orders: true } });
```

<div class="notice--info">
  <p><strong>Prisma sidesteps lazy loading entirely.</strong> Because relations are only loaded when you explicitly <code>include</code> or <code>select</code> them, Prisma cannot accidentally trigger N+1 through attribute access — the trade-off is that you must remember to ask for related data up front.</p>
</div>

### Choosing JOIN vs second-query loading

| | JOIN-based (`joinedload`, `select_related`) | Second-query (`selectinload`, `prefetch_related`) |
|---|---|---|
| Query count | 1 | 2 (parents + children) |
| To-one relations | Ideal | Works, slightly wasteful |
| To-many relations | Row explosion (parent repeated per child) | Ideal — no duplication |
| Large parent payload | Duplicated across joined rows | Sent once |

A common mistake is `joinedload` on a to-many relationship: a customer with 50 orders is returned 50 times in the result set, inflating bytes transferred and forcing the ORM to deduplicate. Prefer second-query loading for collections.

### Detecting N+1

You cannot fix what you cannot see. Make the SQL visible:

```python
# SQLAlchemy: echo every statement to logs
engine = create_engine(url, echo=True)
```

```python
# Django: read connection.queries (DEBUG=True), or use django-debug-toolbar
from django.db import connection
print(len(connection.queries))      # spikes reveal N+1
```

Many test suites add an assertion that a given endpoint runs **at most K queries**, turning a silent performance regression into a failing test.

## Transactions in ORMs

A [transaction](transactions-and-concurrency.html) groups operations so they commit or roll back together. ORMs wrap transactions in two related concepts:

- The **session / persistence context / unit of work** — an in-memory staging area that tracks which objects are new, dirty, or deleted.
- **Flush vs commit** — *flush* emits the pending SQL (`INSERT`/`UPDATE`/`DELETE`) within the transaction; *commit* makes it durable and ends the transaction.

```python
# SQLAlchemy: the context manager commits on success, rolls back on exception
with Session(engine) as session:
    with session.begin():                  # one transaction
        account_a.balance -= 100
        account_b.balance += 100
        # dirty objects are flushed and the transaction commits at block exit
# any exception inside the block triggers ROLLBACK automatically
```

```python
# Django: atomic() is the transaction boundary
from django.db import transaction
with transaction.atomic():
    account_a.balance -= 100; account_a.save()
    account_b.balance += 100; account_b.save()
    # exception -> the whole block rolls back
```

```javascript
// Sequelize: managed transaction commits/rolls back automatically
await sequelize.transaction(async (t) => {
  await accountA.decrement("balance", { by: 100, transaction: t });
  await accountB.increment("balance", { by: 100, transaction: t });
});

// Prisma: $transaction runs the array atomically
await prisma.$transaction([
  prisma.account.update({ where: { id: a }, data: { balance: { decrement: 100 } } }),
  prisma.account.update({ where: { id: b }, data: { balance: { increment: 100 } } }),
]);
```

### Dirty checking and the unit of work

Data-Mapper ORMs (SQLAlchemy, Hibernate) track loaded objects and compare them against their original state at flush time — **dirty checking**. You mutate objects in memory; the ORM works out the minimal set of `UPDATE`s and orders them to respect foreign keys. This is the **Unit of Work** pattern: changes accumulate and are written in one coordinated batch at commit.

```python
order = session.get(Order, 42)
order.status = "shipped"        # no SQL yet — just a tracked change
session.commit()               # now: UPDATE orders SET status='shipped' WHERE id=42
```

### Optimistic concurrency

Holding a database lock across a user's "think time" does not scale (see the locking discussion in [Transactions &amp; Concurrency](transactions-and-concurrency.html)). ORMs instead offer **optimistic locking**: add a `version` column, and the `UPDATE` carries the version the app last read.

$$
\text{UPDATE orders SET status = '?', version = version + 1}
$$
$$
\text{WHERE id = ? AND version = ?}
$$

If another transaction already bumped the version, the `WHERE` matches zero rows, the ORM detects the lost update, and raises a `StaleDataError` (SQLAlchemy) / `OptimisticLockException` (Hibernate) for the app to retry. The probability that a given write conflicts is roughly

$$
P_{\text{conflict}} \approx 1 - \left(1 - \frac{1}{N}\right)^{W}
$$

for $W$ concurrent writers contending over $N$ equally-likely rows — low when writes are spread across many rows, high on a hot row, which is exactly when you should reconsider the data model.

## When to Drop to SQL

ORMs are excellent at the common case and frustrating at the edges. Drop to SQL (or a thin query builder) when:

1. **Bulk operations.** Looping to update 100,000 rows one object at a time is 100,000 round-trips. A single `UPDATE ... WHERE` is one statement.

   ```python
   # Bad: load, mutate, save each row (N round-trips)
   for p in session.query(Product).filter_by(discontinued=True):
       p.active = False
   # Good: one set-based statement
   session.query(Product).filter_by(discontinued=True).update({"active": False})
   ```

2. **Analytical/reporting queries** with window functions, `ROLLUP`/`CUBE`, or recursive CTEs that map awkwardly onto objects.
3. **Vendor-specific features** — PostgreSQL `JSONB` operators, full-text search, `pgvector` similarity, `INSERT ... ON CONFLICT` upserts.
4. **Hot paths** where you have profiled the ORM-generated SQL and need exact control over the join order or index usage. Read the plan with `EXPLAIN ANALYZE` (see [Indexing &amp; Query Execution](indexing-and-queries.html)).
5. **Complex multi-table joins** where the ORM's eager-loading strategy generates a worse plan than a hand-written query.

The healthy posture is a **spectrum**, not a religion:

```
ORM objects  →  query builder  →  parameterized raw SQL  →  stored procedures
(most code)     (dynamic SQL)     (hot/edge queries)        (rare, DB-owned logic)
```

Stay as high on the ladder as the workload allows, and step down deliberately — with a profiler or `EXPLAIN` output justifying the move — rather than rewriting everything in SQL out of habit.

## ORM Anti-Patterns and Pitfalls

<div class="comparison-table">

| Pitfall | Symptom | Fix |
|---|---|---|
| N+1 queries | Query count scales with row count | Eager-load (`selectinload`/`prefetch_related`/`include`) |
| `joinedload` on collections | Result set has duplicated parents | Use second-query loading for to-many |
| Loading whole objects to count | Slow `len(query.all())` | Use a `COUNT(*)` aggregate, not `len()` |
| Row-by-row writes | Thousands of `UPDATE`s | One set-based statement / bulk insert |
| Over-fetching columns | `SELECT *` everywhere | Project only needed columns |
| Leaky lazy loads after session close | `DetachedInstanceError` / lazy load outside transaction | Eager-load before the session closes |
| Fighting the ORM for a hard query | Convoluted method chains | Drop to raw SQL for that one query |

</div>

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card">
    <h4>The mapping is leaky on purpose</h4>
    <p>An ORM bridges objects and tables but cannot erase the impedance mismatch — inheritance, identity, and collections all need explicit mapping decisions.</p>
  </div>
  <div class="takeaway-card">
    <h4>Lazy loading causes N+1</h4>
    <p>Touching a relationship inside a loop fires one query per row. Eager-load related data up front to collapse 1+N queries into a small constant.</p>
  </div>
  <div class="takeaway-card">
    <h4>JOIN for to-one, second-query for to-many</h4>
    <p><code>joinedload</code>/<code>select_related</code> suit single related rows; <code>selectinload</code>/<code>prefetch_related</code> avoid row explosion for collections.</p>
  </div>
  <div class="takeaway-card">
    <h4>The session is the unit of work</h4>
    <p>Dirty checking batches in-memory mutations into a minimal, correctly-ordered set of writes committed in one transaction.</p>
  </div>
  <div class="takeaway-card">
    <h4>Stay high on the ladder, drop down deliberately</h4>
    <p>Use objects and the query builder for routine work; reach for parameterized raw SQL only for bulk, analytical, or vendor-specific queries — never concatenate user input.</p>
  </div>
</div>

## See Also

<div class="see-also-card">
  <h4>Continue in Database Design</h4>
  <ul>
    <li><a href="modeling.html">Data Modeling &amp; Normalization</a> — the schema the ORM maps onto, and the inheritance/normalization trade-offs.</li>
    <li><a href="indexing-and-queries.html">Indexing &amp; Query Execution</a> — reading <code>EXPLAIN</code> output to tune the SQL your ORM generates, including the N+1 discussion.</li>
    <li><a href="transactions-and-concurrency.html">Transactions &amp; Concurrency</a> — locking, MVCC, and isolation behind the ORM's <code>commit()</code> and optimistic locking.</li>
    <li><a href="storage-internals.html">Storage Engines &amp; Recovery</a> — what flush and commit actually do to pages and the write-ahead log.</li>
    <li><a href="./">Database Design hub</a> — the full deep-dive series.</li>
  </ul>
</div>
