---
layout: docs
title: "Database Design: Distributed Transactions"
permalink: /docs/technology/database-design/distributed-transactions.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Distributed Transactions</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">2PC/3PC, sagas, the outbox pattern, idempotency, distributed deadlocks, and exactly-once semantics</p>
</div>

<p><a href="./">&larr; Database Design</a></p>

## Atomicity Across Machines

A single-node database gives you atomicity for free: the storage engine writes a commit record to the write-ahead log, and either the whole transaction is durable or none of it is. Once data spans **multiple machines**—shards, microservices, or independent databases—that guarantee evaporates. Each node has its own log, its own clock, and its own failure modes, and the network between them can drop, delay, or reorder messages at any moment.

The central question of distributed transactions is: **how do you make `N` independent participants either all commit or all abort, despite crashes and network partitions?** There is no perfect answer—the [FLP impossibility result](#why-2pc-blocks) proves you cannot have a protocol that is simultaneously safe, live, and fault-tolerant in an asynchronous network. Every technique below is a different point on that trade-off curve.

<div class="notice--info">
  <p><strong>Two philosophies.</strong> You can either coordinate a single atomic commit across all participants (<strong>2PC/3PC</strong>—strong consistency, blocking, tight coupling), or accept that each participant commits independently and stitch the steps together with application logic (<strong>sagas, outbox, idempotency</strong>—eventual consistency, non-blocking, loose coupling). Modern microservice architectures overwhelmingly choose the second.</p>
</div>

## Two-Phase Commit (2PC)

Two-phase commit is the classic protocol for an *atomic* distributed commit. A designated **coordinator** drives the protocol; the other nodes are **participants** (also called cohorts or resource managers). It works like a wedding ceremony: first everyone is asked whether they consent, then—only if all agree—the union is finalized.

**Phase 1 — Prepare (voting):**

```python
# Coordinator (the officiant)
def prepare_transaction(tx_id, participants):
    responses = []
    for participant in participants:
        # "Can you commit? Promise not to back out."
        response = participant.prepare(tx_id)
        responses.append(response)

    if all(r == "YES" for r in responses):
        decision = "COMMIT"
    else:
        decision = "ABORT"

    log_decision(tx_id, decision)  # Write to disk BEFORE telling anyone
    return decision
```

When a participant votes `YES`, it does something crucial: it makes the changes durable in its own log (so it can survive a crash) but does **not** release locks. It is now **in doubt**—bound by its promise, unable to act unilaterally until the coordinator tells it the outcome.

**Phase 2 — Commit/Abort (decision):**

```python
def finish_transaction(tx_id, participants, decision):
    for participant in participants:
        participant.apply(tx_id, decision)  # COMMIT or ABORT
        # Participant applies changes (or rolls back) and releases locks
```

### The commit point

The single most important moment in 2PC is when the coordinator **forces its decision to its own log**. Before that write, the default is abort; after it, the default is commit. This is the "point of no return." If the coordinator crashes after logging `COMMIT` but before notifying participants, recovery reads the log and resends the decision. This is why both the coordinator and the participants must use durable, recoverable logging—2PC is built on top of each node's local WAL, not a replacement for it.

```
   Coordinator                Participant A            Participant B
       |                           |                        |
       |------ PREPARE ----------->|                        |
       |------ PREPARE ------------------------------------>|
       |                           | (flush to log, lock)   |
       |<----- YES ----------------|                        |
       |<----- YES -----------------------------------------|
       |                           |                        |
   [force COMMIT to log]  <-- commit point                  |
       |                           |                        |
       |------ COMMIT ------------>| (apply, release locks) |
       |------ COMMIT ------------------------------------->|
       |<----- ACK ----------------|                        |
       |<----- ACK -----------------------------------------|
```

### Why 2PC blocks

2PC is **safe** (it never lets some nodes commit while others abort) but it is **not live** under failure. Consider the coordinator crashing right after every participant has voted `YES` but before sending any decision:

- Participants are **stuck holding locks** ("standing at the altar"), in doubt.
- They cannot commit—the coordinator might have decided to abort.
- They cannot abort—the coordinator might have decided to commit, and other participants might already have applied it.
- They cannot even ask each other, because a peer that voted `YES` knows no more than they do.

This is the famous **blocking problem**. In-doubt transactions hold locks indefinitely, stalling unrelated work, until the coordinator recovers. Worse, a *participant* crash during phase 2 leaves the coordinator unable to complete until that node returns. For this reason 2PC is best confined to a single datacenter with a highly available coordinator, and it is the reason most internet-scale systems avoid it.

<div class="notice--warning">
  <p><strong>2PC couples availability.</strong> A transaction across N participants is only as available as the <em>least</em> available participant <em>and</em> the coordinator. The probability that all are up multiplies down quickly: five 99.9%-available services give roughly 99.5% combined availability for a cross-service commit. This multiplicative fragility, not just the blocking window, is why 2PC scales poorly.</p>
</div>

## Three-Phase Commit (3PC)

Three-phase commit was designed to make distributed commit **non-blocking** by inserting an extra round. The insight: in 2PC, a participant moves directly from "voted yes" to "committed," so a recovering node can't tell which side of the commit point the cluster was on. 3PC adds an intermediate **pre-commit** state that all participants reach *before* anyone commits.

**The three phases:**

1. **CanCommit** — coordinator asks, participants vote yes/no (like 2PC prepare, but participants do not yet lock resources irrevocably).
2. **PreCommit** — if all voted yes, the coordinator sends `PRECOMMIT`; participants acknowledge and enter a "prepared to commit" state. Reaching this state means *everyone agreed*.
3. **DoCommit** — coordinator sends `DOCOMMIT`; participants commit and release locks.

```
Phase 1: CanCommit?  --->  yes/no votes
Phase 2: PreCommit   --->  acks   (now everyone knows everyone said yes)
Phase 3: DoCommit    --->  commit
```

Because the `PRECOMMIT` state is reached by all participants before any commit, a recovering participant can reason about the global state using a **timeout-based default**: if it has received `PRECOMMIT`, it can safely commit even without hearing from the coordinator (everyone must have agreed); if it has not, it can safely abort. This removes the indefinite blocking of 2PC.

<div class="notice--warning">
  <p><strong>Why nobody uses 3PC.</strong> 3PC only achieves non-blocking under a synchronous network with reliable failure detection and no network partitions. Real networks <em>do</em> partition, and under a partition 3PC can violate safety: a partitioned group that has seen <code>PRECOMMIT</code> commits while another group times out and aborts—a split-brain. The extra round-trip also adds latency. In practice, production systems skip 3PC and instead use a partition-tolerant <strong>consensus protocol</strong> (Raft, Paxos, or Spanner's Paxos-over-TrueTime) to replicate the commit decision itself, which gives both safety and availability with a majority quorum.</p>
</div>

> **See also:** Consensus protocols like Raft and Paxos—covered in [Distributed Databases &amp; NoSQL](distributed-and-nosql.html)—are the modern replacement for a fragile single coordinator. A consensus group can lose a minority of nodes and still make progress, eliminating the single point of failure that makes 2PC block.

## The Saga Pattern

For business processes that span multiple services or run for a long time (booking a trip, fulfilling an order, onboarding a customer), holding distributed locks for the whole duration—as 2PC requires—is untenable. The **saga pattern** trades atomicity for availability: a saga is a *sequence of local transactions*, each of which commits independently. If a later step fails, the saga runs **compensating transactions** to semantically undo the earlier steps.

```python
class TripBookingSaga:
    def execute(self):
        flight_id = hotel_id = car_id = None
        try:
            flight_id = book_flight()        # Step 1 (local commit)
            hotel_id = book_hotel()          # Step 2 (local commit)
            car_id = book_rental_car()       # Step 3 (local commit)
            send_confirmation()              # Step 4
        except Exception as e:
            # Compensate in REVERSE order
            if car_id:    cancel_rental_car(car_id)
            if hotel_id:  cancel_hotel(hotel_id)
            if flight_id: cancel_flight(flight_id)
            raise e
```

### Compensation is semantic, not physical

A compensating transaction does not "roll back" in the storage-engine sense—the original local transaction already committed and is visible to everyone. Instead it performs a *new* transaction that **semantically reverses** the effect: a `cancel_flight` that issues a refund and frees the seat, not a magical un-booking. This has consequences:

- **Compensations may themselves be complex.** Refunding money is not the inverse of charging it; it may incur fees, trigger notifications, or be impossible after a cutoff.
- **Some actions are irreversible.** You cannot un-send an email or un-launch a missile. For these, the saga must place the irreversible step **last** (a *pivot transaction*), or split it into a reversible reservation plus a separate, idempotent confirmation.
- **Compensations must commute with concurrent activity.** Between booking and cancelling, another process may have read the booking. Sagas therefore provide weaker isolation than a single ACID transaction—see "lack of isolation" below.

### Orchestration vs. choreography

Sagas come in two coordination styles:

| | **Orchestration** | **Choreography** |
|---|---|---|
| Control | A central orchestrator tells each service what to do next | Each service emits events; others react |
| Coupling | Orchestrator knows the whole flow | No central knowledge; flow is emergent |
| Visibility | Easy to see/trace the whole saga | Hard to reason about end-to-end |
| Risk | Orchestrator is a focal point of logic | Cyclic event dependencies, hard to debug |
| Good for | Complex flows, many steps, clear ownership | Simple flows, highly decoupled teams |

**Orchestrated saga** (a stateful coordinator drives the steps and records progress in a saga log):

```python
class OrderSagaOrchestrator:
    def handle(self, saga_id):
        state = self.load_state(saga_id)   # durable saga log

        if state.step == "START":
            self.invoke("payment.charge", saga_id)
            self.save_state(saga_id, step="CHARGED")

        elif state.step == "CHARGED":
            self.invoke("inventory.reserve", saga_id)
            self.save_state(saga_id, step="RESERVED")

        elif state.step == "RESERVED":
            self.invoke("shipping.dispatch", saga_id)
            self.save_state(saga_id, step="COMPLETE")

    def on_failure(self, saga_id, failed_step):
        # walk the saga log backward, invoking compensations
        for step in reversed(self.completed_steps(saga_id)):
            self.invoke(self.compensation_for(step), saga_id)
```

**Choreographed saga** (services react to each other's events, no central brain):

```
OrderCreated      -> Payment service charges, emits PaymentCompleted
PaymentCompleted  -> Inventory service reserves, emits StockReserved
StockReserved     -> Shipping service dispatches, emits OrderShipped

# Failure path:
PaymentFailed     -> Order service marks order failed (nothing to compensate yet)
StockReservationFailed -> Payment service refunds (compensation), Order cancels
```

### Sagas lack isolation

Because each step commits independently, a saga has **no isolation** by default—intermediate states are visible to other transactions. A trip might briefly show a booked flight before the hotel step fails and the flight is cancelled; a reader in that window sees a state that never "really" existed. Mitigations borrow from the literature on long-lived transactions:

- **Semantic lock** — mark a record as "pending"/"reserved" so other sagas know it is in flight (a flag, not a database lock).
- **Commutative updates** — design operations so order does not matter (increment/decrement rather than absolute set).
- **Reread / version check** — before compensating, re-read and verify nothing incompatible happened (optimistic concurrency, see [Transactions &amp; Concurrency](transactions-and-concurrency.html)).
- **By-pass the dirty read at the app layer** — hide pending entities from other users' views until the saga completes.

## The Outbox Pattern

Sagas, choreography, and event-driven systems all share a deceptively hard sub-problem: **how do you update your database AND publish a message atomically?** A service that does this—

```python
# BROKEN: dual write, no atomicity
def place_order(order):
    db.insert(order)              # commits to Postgres
    broker.publish("OrderCreated", order)  # sends to Kafka/RabbitMQ
```

—has a *dual-write* bug. If the process crashes between the two lines, you get one of two corruptions:

- DB commit succeeds, publish fails ⇒ order exists but **no event** (downstream never reacts; lost message).
- Publish succeeds, DB commit rolls back ⇒ event exists but **no order** (downstream acts on a phantom; ghost message).

You cannot fix this by reordering the two operations or wrapping them in a try/except—the broker and the database are separate systems with no shared transaction. (You *could* use 2PC/XA across them, but that reintroduces all the blocking and coupling problems above, and many brokers don't support it.)

### The transactional outbox

The fix is to make the message part of the **same local database transaction** as the state change, by writing it to an `outbox` table:

```sql
BEGIN;
  INSERT INTO orders (id, customer_id, total, status)
  VALUES ('ord-1', 'cust-9', 49.99, 'created');

  INSERT INTO outbox (id, aggregate_id, topic, payload, created_at)
  VALUES (gen_random_uuid(), 'ord-1', 'OrderCreated',
          '{"orderId":"ord-1","total":49.99}', now());
COMMIT;   -- both rows commit together, atomically, via the local WAL
```

A separate **relay** (message relay / publisher) then reads unpublished rows from the outbox and pushes them to the broker, marking them sent (or deleting them) afterward:

```python
def relay_outbox():
    rows = db.query(
        "SELECT * FROM outbox WHERE published_at IS NULL "
        "ORDER BY created_at LIMIT 100 FOR UPDATE SKIP LOCKED")
    for row in rows:
        broker.publish(row.topic, row.payload, key=row.aggregate_id)
        db.execute("UPDATE outbox SET published_at = now() WHERE id = %s",
                   (row.id,))
```

Two ways to drive the relay:

- **Polling publisher** — a worker polls the table on an interval (simple, but adds query load and latency; `FOR UPDATE SKIP LOCKED` lets many workers share the table safely—see the queue pattern in [Transactions &amp; Concurrency](transactions-and-concurrency.html)).
- **Change Data Capture (CDC)** — a tool like Debezium tails the database's WAL/binlog and emits an event for every committed outbox row. No polling, lower latency, no extra query load, but more infrastructure.

<div class="notice--info">
  <p><strong>Outbox guarantees at-least-once, not exactly-once.</strong> The relay may crash <em>after</em> publishing but <em>before</em> recording the row as sent, so the same message can be published again. That is fine—and intentional. The outbox guarantees the message is <strong>never lost</strong>; making the redelivered message harmless is the job of <a href="#idempotency-keys">idempotency</a> on the consumer side. Together, outbox (no loss) + idempotent consumer (no duplicate effect) yield effective <a href="#exactly-once-semantics">exactly-once processing</a>.</p>
</div>

The mirror image on the receiving side is the **inbox pattern**: a consumer records each processed message id in an `inbox` table inside the same transaction as its effect, and skips messages whose id it has already seen—turning at-least-once delivery into exactly-once processing.

## Idempotency Keys

An operation is **idempotent** if applying it more than once has the same effect as applying it once. In distributed systems, retries are unavoidable—timeouts, redeliveries, relays, and client retries all cause the same request to arrive twice—so idempotency is the workhorse that makes those retries safe.

Some operations are *naturally* idempotent: `SET status = 'shipped'`, `PUT /users/42` with a full body, `DELETE`. Others are not: `balance = balance + 100`, "create order," "charge card," "send email." For the non-idempotent ones you attach an **idempotency key**—a unique client-generated identifier for the *logical* operation—and the server deduplicates on it.

```python
def charge(idempotency_key, amount, customer):
    # 1. Has this key been seen? Return the stored result if so.
    existing = db.query(
        "SELECT response FROM idempotency WHERE key = %s", (idempotency_key,))
    if existing:
        return existing.response       # same answer, no double charge

    # 2. First time: do the work and record key + result atomically.
    with db.transaction():
        result = payment_gateway.charge(amount, customer)
        db.execute(
            "INSERT INTO idempotency (key, response, created_at) "
            "VALUES (%s, %s, now())",
            (idempotency_key, result))
    return result
```

### Getting idempotency right

- **Key uniqueness is on the client.** The client must generate the key (a UUID) once and reuse it across retries of the *same* logical request—if it generates a new key per retry, dedup is impossible. Stripe's API, for example, takes an `Idempotency-Key` header for exactly this.
- **Insert the key in the same transaction as the effect.** If you record the key separately, a crash in between can either double-apply (effect committed, key not recorded) or lose the operation (key recorded, effect not). A `UNIQUE` constraint on the key column turns a concurrent duplicate into a constraint violation you can catch and treat as "already done."
- **Store the response, not just the key.** A retry should get the *same answer* (same order id, same charge id), not a fresh one or a generic "duplicate" error.
- **Scope and expire keys.** Keys are usually scoped per endpoint/operation and may expire (e.g., 24h) once the window for retries has passed.
- **Beware non-deterministic side effects.** If the original attempt half-completed (charged the card but crashed before recording the key), reconciliation against the downstream system is needed—idempotency keys reduce, but do not entirely remove, the need for reconciliation.

## Distributed Deadlocks

A [deadlock](transactions-and-concurrency.html) on one node is a cycle in the local lock graph, and the database detects it by inspecting that graph and killing a victim. Across multiple nodes the cycle can span machines, and **no single node can see the whole graph**:

```
Node 1:  T1 holds row A, waits for row B (on Node 2)
Node 2:  T2 holds row B, waits for row A (on Node 1)

# Node 1's local graph:  T1 -> (waiting for B, off-node)   -- no cycle visible
# Node 2's local graph:  T2 -> (waiting for A, off-node)   -- no cycle visible
# Global graph:          T1 -> T2 -> T1                     -- a cycle nobody sees
```

Three families of solution:

**1. Timeout-based (the pragmatic default).** Give every distributed lock/transaction a deadline; if it waits too long, abort and retry with backoff. Simple and partition-tolerant, but tuning is hard—too short causes spurious aborts of slow-but-live transactions, too long means deadlocks linger. Most distributed databases (including CockroachDB's earlier versions and many app-level lock managers) lean on timeouts.

**2. Global deadlock detection.** Periodically aggregate every node's local *wait-for* graph into a global graph and search it for cycles, then kill a victim. This is precise but expensive and adds latency; the aggregated graph can also be stale (a "phantom deadlock"—a cycle that has already resolved by the time it's detected), so detectors confirm before killing.

**3. Deadlock prevention via ordering.** Avoid cycles by construction. Two classic timestamp schemes assign each transaction a start timestamp and only ever let an *older* transaction wait on a *younger* one (or vice versa), guaranteeing no cycle:

- **Wait-Die** (non-preemptive): if an older transaction requests a lock held by a younger one, it **waits**; if a younger one requests a lock held by an older one, it **dies** (aborts and retries with its *original* timestamp, so it eventually becomes oldest and wins).
- **Wound-Wait** (preemptive): if an older transaction wants a lock held by a younger one, it **wounds** (aborts) the younger; if a younger one wants a lock held by an older one, it **waits**.

Keeping the original timestamp on retry prevents starvation: a transaction only gets older relative to its peers, so it cannot be aborted forever.

<div class="tip-card">
  <h4>Cheapest mitigation: consistent lock ordering</h4>
  <p>Most application-level distributed deadlocks come from two code paths acquiring the same resources in opposite orders (A-then-B vs. B-then-A). Enforce a <strong>global ordering</strong> (e.g., always lock rows in ascending primary-key order, always lock accounts by lowest id first) and the cycle becomes impossible — no detector or timeout needed. This single discipline eliminates the majority of real-world deadlocks.</p>
</div>

## Exactly-Once Semantics

"Exactly-once delivery" is the most misunderstood phrase in distributed systems. Over an unreliable network it is **provably impossible** to deliver a message exactly once: the sender cannot distinguish "the message was lost" from "the message arrived but the ack was lost," so it must either risk loss (at-most-once) or risk duplication (at-least-once). You only ever get to pick:

| Guarantee | Mechanism | Risk |
|---|---|---|
| **At-most-once** | Fire and forget; never retry | Messages can be **lost** |
| **At-least-once** | Retry until acknowledged | Messages can be **duplicated** |
| **Exactly-once** | *Not* a delivery guarantee — see below | (achieved at the processing layer) |

What is actually achievable, and what people mean when they say "exactly-once," is **exactly-once *processing***: the message may be *delivered* many times, but its *effect* is applied exactly once. You get there by combining the building blocks already covered:

```
  at-least-once delivery        guarantees no message is lost
+ idempotent / deduplicated processing  guarantees no duplicate effect
= exactly-once effect (effective exactly-once)
```

### How to build it

1. **No loss on the producer side** — use the [outbox pattern](#the-outbox-pattern) so the state change and the message commit atomically. The message *will* eventually be published (at-least-once).
2. **Dedup on the consumer side** — give each message a stable id and process it with an [idempotency key](#idempotency-keys) or the inbox pattern, so reprocessing the same id is a no-op that returns the prior result.
3. **Atomic "consume + act + ack"** — the consumer must record "I processed message X" in the *same transaction* as the effect of X. If ack and effect are separate, a crash in between reintroduces either loss or duplication.

```python
def handle(message):
    with db.transaction():
        # Dedup: have we already processed this message id?
        if db.exists("SELECT 1 FROM inbox WHERE msg_id = %s", message.id):
            return                      # duplicate delivery -> no-op

        apply_business_effect(message)              # the real work
        db.insert("inbox", msg_id=message.id)       # record in SAME tx
    # only now ack the broker; redelivery before ack is harmless (dedup catches it)
    broker.ack(message)
```

<div class="notice--info">
  <p><strong>"Exactly-once" platforms still rely on this.</strong> Kafka's exactly-once semantics (idempotent producer + transactional writes that bind the consumed-offset commit to the produced output in one transaction) is exactly this pattern implemented inside the platform: at-least-once delivery underneath, deduplication and atomic offset+output commit on top. There is no magic that makes the network reliable — only careful placement of idempotency and atomic commits around an at-least-once core.</p>
</div>

### Choosing a guarantee

- **At-most-once** is fine for high-volume, loss-tolerant data: metrics, logs, telemetry where an occasional dropped sample doesn't matter.
- **At-least-once + idempotent processing** is the right default for almost everything with side effects: payments, orders, notifications. Duplicates are caught; nothing is lost.
- **True exactly-once delivery** is a marketing claim; ask what the system actually does at the producer and consumer, and you will find the outbox/idempotency pattern underneath.

## Putting It Together: a Distributed Order

A realistic e-commerce checkout combines every technique on this page:

```
1. Order service:  BEGIN; insert order(status=pending);
                          insert outbox(OrderCreated); COMMIT;   <- outbox, no dual-write
2. Relay/CDC publishes OrderCreated (at-least-once)              <- no message lost
3. Payment service consumes it with idempotency key=order_id     <- duplicate deliveries deduped
       charges card, emits PaymentCompleted (via its own outbox)
4. Inventory service reserves stock, emits StockReserved         <- saga step (local commit)
5. If any step fails -> compensations run in reverse:
       refund payment, release stock, mark order failed          <- saga compensation
6. Locks across services are never held -> no distributed 2PC,   <- non-blocking
       so no blocking and no multiplicative availability hit
```

No global coordinator, no distributed locks, no blocking—just local ACID transactions glued together by an at-least-once event backbone with idempotent, compensatable steps. That is how internet-scale systems get *effective* atomicity without paying the price of 2PC.

> **Code Reference:** For working implementations of these algorithms, see [`distributed_systems.py`](../../../code-examples/technology/database-design/distributed_systems.py).

## See Also

<div class="see-also-card">
  <h4>Continue the deep dive</h4>
  <ul>
    <li><strong>Foundations:</strong> <a href="transactions-and-concurrency.html">Transactions &amp; Concurrency</a> — single-node ACID, locking, MVCC, and isolation levels that distributed transactions build on.</li>
    <li><strong>Related:</strong> <a href="distributed-and-nosql.html">Distributed Databases &amp; NoSQL</a> — CAP, Raft/Paxos consensus, and the NewSQL systems that implement distributed commit for you.</li>
    <li><strong>Durability:</strong> <a href="storage-internals.html">Storage Engines &amp; Recovery</a> — the write-ahead log that makes every local commit (and the outbox) atomic and recoverable.</li>
    <li><strong>Up:</strong> <a href="./">Database Design hub</a></li>
    <li>See also: <a href="../networking/">Networking</a> for the protocols beneath distributed coordination, and <a href="../aws/">AWS</a> for managed queues and event services (SQS, EventBridge) that implement the outbox/relay plumbing.</li>
  </ul>
</div>
