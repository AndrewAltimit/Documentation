---
layout: docs
title: "API Design: REST"
permalink: /docs/api-design/rest.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">API Design: REST</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Resources, HTTP semantics, versioning, pagination, hypermedia, idempotency, and the contracts (OpenAPI, RFC 7807) that make a web API durable.</p>
</div>

[API Design](./) &raquo; REST

<div class="intro-card">
  <p class="lead-text">REST is not a protocol or a framework — it is an <em>architectural style</em>: a set of constraints that, when honored, give an API the scaling, evolvability, and cache-friendliness of the web itself. In practice "REST" has come to mean "JSON over HTTP", but the value comes from taking the underlying ideas seriously: model your domain as <strong>resources</strong> with stable identifiers, manipulate them with the <strong>uniform interface</strong> of HTTP methods and status codes, keep every request <strong>stateless</strong>, and let responses describe their own cacheability and next actions. This page covers the constraints that define REST, how to model resources and choose URIs, the precise semantics of HTTP methods and status codes, strategies for versioning, pagination, filtering, idempotency, and rate limiting, the RFC 7807 problem-details error format, and the OpenAPI contracts that turn all of this into tooling.</p>
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-cube"></i>
    <h4>Resources, not actions</h4>
    <p>A REST API exposes nouns (<code>/orders/42</code>) acted on by a fixed verb set, not an open-ended catalog of RPC procedures.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-exchange-alt"></i>
    <h4>The uniform interface is the point</h4>
    <p>Methods, status codes, headers, and media types are a shared contract every client and proxy already understands.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-feather"></i>
    <h4>Stateless scales</h4>
    <p>Each request carries everything needed to serve it, so any server can handle any request and you scale by adding boxes.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-redo"></i>
    <h4>Safety and idempotency are guarantees</h4>
    <p>GET must not change state; PUT/DELETE can be retried safely. Clients and proxies rely on these properties.</p>
  </div>
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## REST Principles & Constraints

REST — **Re**presentational **S**tate **T**ransfer — was defined by Roy Fielding's 2000 dissertation as the architectural style of the web. It is characterized by six constraints. The first five are required for an API to be "RESTful"; the sixth is optional.

1. **Client–Server.** Separate the user interface (client) from data storage (server) so each evolves independently. The client knows resource URIs; the server knows how state is stored.
2. **Stateless.** Each request from client to server must contain all information needed to understand it; the server keeps no client *session* state between requests. Authentication, for instance, travels on every request (a token), not in a server-held session. Statelessness is what lets you put any server behind a load balancer and scale horizontally.
3. **Cacheable.** Responses must declare themselves cacheable or not (via `Cache-Control`, `ETag`, etc.). Caches — in the client, in proxies, in CDNs — can then satisfy later requests without touching the origin, cutting latency and load.
4. **Uniform Interface.** This is the central constraint and what most distinguishes REST. It has four facets:
   - **Identification of resources** — each resource has a stable URI.
   - **Manipulation through representations** — clients act on resources by exchanging representations (a JSON document), not by calling stored procedures.
   - **Self-descriptive messages** — each message carries enough metadata (method, status, `Content-Type`, caching headers) to be understood in isolation.
   - **Hypermedia as the engine of application state (HATEOAS)** — responses include links to the next valid actions, so clients navigate the API rather than hard-coding URLs.
5. **Layered System.** A client cannot tell whether it is connected to the origin server or an intermediary (gateway, proxy, cache). This lets you insert load balancers, caches, and security layers transparently.
6. **Code on Demand (optional).** Servers may extend client functionality by sending executable code (e.g. JavaScript). Rarely treated as part of API design.

> **The Richardson Maturity Model** is a useful ladder for grading how "RESTful" an API is. **Level 0**: a single URI, single method (effectively RPC tunneled through HTTP). **Level 1**: multiple resources with distinct URIs, but still one method. **Level 2**: proper use of HTTP verbs and status codes — where most production "REST" APIs live. **Level 3**: hypermedia controls (HATEOAS) drive the interaction. Each rung adds discoverability and decoupling; few APIs reach Level 3, and that is a defensible engineering choice.

A design that violates statelessness (server-side sessions), ignores the uniform interface (`POST /getUser`, `POST /deleteUser`), or returns `200 OK` with an error body inside is not wrong because it breaks a rule for its own sake — it is wrong because it forfeits the caching, scaling, and tooling that the constraints buy you.

## Resource Modeling

The first design task is to identify the **resources** — the nouns your API exposes — and give each a stable identifier. A resource is anything worth naming: an order, a user, a collection of users, a search result. The transition from a procedural mindset (`createOrder`, `cancelOrder`) to a resource mindset (`POST /orders`, then `DELETE /orders/42` or `PATCH /orders/42 {status: cancelled}`) is the core skill.

### URI Design

URIs name resources. A few durable conventions:

- **Use nouns, not verbs.** The HTTP method *is* the verb. `GET /orders`, not `GET /getOrders`.
- **Use plural collection names.** `/orders`, `/orders/42`, `/users`, `/users/7`. A collection is a resource; an item is a member of it.
- **Model hierarchy with nesting — sparingly.** `/users/7/orders` reads as "user 7's orders". Nest to express ownership, but stop at one or two levels; deep nesting (`/a/1/b/2/c/3/d/4`) is brittle. Once you have an order's ID you can usually address it directly at `/orders/42` rather than `/users/7/orders/42`.
- **Lowercase, hyphenated paths.** `/shipping-addresses`, not `/shippingAddresses` or `/Shipping_Addresses`.
- **Keep verbs out, even for actions.** A "ship this order" action that does not fit CRUD can be modeled as a sub-resource state change (`PATCH /orders/42 {status: "shipped"}`) or, pragmatically, as a controller sub-resource (`POST /orders/42/shipments`). Reserve RPC-style action URIs (`POST /orders/42/cancel`) for genuinely non-CRUD operations where forcing a resource shape would be contrived.

| Goal | Good | Avoid |
|------|------|-------|
| List a collection | `GET /orders` | `GET /getAllOrders` |
| Fetch one item | `GET /orders/42` | `GET /order?id=42` |
| A user's orders | `GET /users/7/orders` | `GET /orders?owner=7` *(acceptable, but loses ownership semantics)* |
| Create | `POST /orders` | `POST /orders/create` |
| Replace | `PUT /orders/42` | `POST /orders/42/update` |
| A non-CRUD action | `POST /orders/42/cancel` | `GET /cancelOrder?id=42` |

### Representations

A resource is distinct from its **representation**. `/orders/42` is the order; the JSON document you `GET` is one representation of it. **Content negotiation** lets the same resource be served as JSON, XML, or CSV depending on the client's `Accept` header:

```http
GET /orders/42 HTTP/1.1
Accept: application/json
```

The server replies with `Content-Type: application/json`. The same URI with `Accept: text/csv` could return a CSV export. In practice most APIs serve one format (JSON), but the principle — one resource, many representations — is why the URI names the resource rather than the file.

### Sub-resources and Relationships

Express relationships as links between resources, not as opaque embedded blobs. An order references its customer; rather than baking the full customer object into every order, link to it (`/customers/7`) and let the client follow the link when it needs the detail. This keeps representations small, avoids stale duplicated data, and is the on-ramp to HATEOAS. (Embedding is still a valid optimization for tightly-coupled, always-needed data — see *Pagination, Filtering & Field Selection* below for `?expand=customer`.)

## HTTP Semantics & Status Codes

The uniform interface lives in the precise meaning of HTTP methods and status codes. Honoring these is what lets caches, proxies, and generic client libraries do the right thing without knowing your domain.

### Methods

| Method | Purpose | Safe? | Idempotent? | Request body | Typical success |
|--------|---------|:-----:|:-----------:|:------------:|-----------------|
| `GET` | Retrieve a representation | Yes | Yes | No | `200 OK` |
| `HEAD` | Like GET, headers only | Yes | Yes | No | `200 OK` |
| `POST` | Create a subordinate / non-idempotent action | No | No | Yes | `201 Created` / `200 OK` |
| `PUT` | Create-or-replace at a known URI | No | Yes | Yes | `200 OK` / `201 Created` |
| `PATCH` | Partial update | No | No* | Yes | `200 OK` / `204 No Content` |
| `DELETE` | Remove the resource | No | Yes | No | `204 No Content` |
| `OPTIONS` | Describe communication options | Yes | Yes | No | `200 OK` |

- **Safe** methods (`GET`, `HEAD`, `OPTIONS`) must not change server state. A crawler that follows every `GET` link must never delete anything — this is why "destructive GET" links are a serious bug.
- **Idempotent** methods produce the same server state whether called once or N times. `DELETE /orders/42` called twice still leaves the order deleted. This property is what makes **retries safe**: a client whose request times out can resend a `PUT` or `DELETE` without fear of double-applying it.
- `PATCH` is *not inherently* idempotent (a JSON-Patch `{"op":"add", "path":"/tags/-", ...}` appends each time), but a `PATCH` that *sets* fields to absolute values usually is. Design patches to be idempotent where you can.

### PUT vs. POST vs. PATCH

- **POST** creates a resource whose URI the *server* assigns. `POST /orders` returns `201 Created` with a `Location: /orders/42` header. Because the client cannot predict the URI, POST is not idempotent — two POSTs make two orders.
- **PUT** creates or replaces a resource at a URI the *client* already knows. `PUT /users/7` with a full user representation either creates user 7 or replaces it wholesale. Idempotent: repeating it yields the same state.
- **PATCH** applies a *partial* modification — send only the fields that change, or a patch document (JSON Merge Patch, RFC 7386; or JSON Patch, RFC 6902). Use it to avoid the read-modify-write race and bandwidth of sending a whole representation to change one field.

### Status Codes

Return the most specific accurate status code. Categories:

- **2xx — Success.**
  - `200 OK` — generic success with a body.
  - `201 Created` — a resource was created; include a `Location` header pointing to it.
  - `202 Accepted` — the request was accepted for *asynchronous* processing (the work is not done yet); useful for long-running jobs, often with a status URL to poll.
  - `204 No Content` — success with no body (typical for `DELETE` and some `PUT`/`PATCH`).
- **3xx — Redirection.**
  - `301 Moved Permanently` / `308 Permanent Redirect` — the resource moved; clients should update.
  - `304 Not Modified` — conditional-GET cache hit (see *Caching*).
- **4xx — Client error.** The request is wrong; retrying unchanged will not help.
  - `400 Bad Request` — malformed syntax or invalid payload.
  - `401 Unauthorized` — missing or invalid authentication (really "unauthenticated").
  - `403 Forbidden` — authenticated but not permitted.
  - `404 Not Found` — no such resource (also used to avoid leaking existence to unauthorized callers).
  - `405 Method Not Allowed` — the URI exists but not for this method; include an `Allow` header.
  - `409 Conflict` — the request conflicts with current state (e.g. an edit against a stale version).
  - `422 Unprocessable Entity` — syntactically valid but semantically invalid (failed business validation).
  - `429 Too Many Requests` — rate limit exceeded (see *Rate Limiting*).
- **5xx — Server error.** The request was fine; the server failed.
  - `500 Internal Server Error` — an unexpected fault.
  - `502 Bad Gateway` / `503 Service Unavailable` / `504 Gateway Timeout` — upstream/availability problems; `503` and `504` are often transient and retryable, ideally with a `Retry-After` header.

> **The cardinal sin** is returning `200 OK` with `{"error": "..."}` in the body. It defeats every generic client, monitoring tool, and proxy that keys on the status line, and it forces every caller to parse the body to learn whether the call succeeded. Let the status code carry the outcome; let the body carry the detail.

## Versioning

An API is a contract. Once clients depend on it, you cannot make breaking changes silently. **Additive** changes (new optional fields, new endpoints) are backward-compatible and need no version bump; **breaking** changes (removing a field, renaming one, tightening validation, changing a type) require a new version so existing clients keep working.

There are three common strategies, each with trade-offs:

| Strategy | Example | Pros | Cons |
|----------|---------|------|------|
| **URI path** | `GET /v1/orders` | Obvious, easy to route, cache-friendly, trivially browsable | URI no longer identifies a single resource across versions; couples version to the path |
| **Query parameter** | `GET /orders?version=1` | Easy to default | Easy to forget; muddies caching |
| **Header / media type** | `Accept: application/vnd.acme.v2+json` | URIs stay clean and version-agnostic; "purest" REST | Invisible in a browser, harder to test, more complex routing |

**URI versioning** is the most widely used in practice because it is the most operationally simple — it is visible, cache-key-friendly, and trivial to route. Header-based **media-type versioning** is theoretically cleaner (the resource URI is stable; only its representation version changes) but pays in discoverability and tooling friction.

Whatever you choose, adopt a **deprecation policy**: announce removals ahead of time, serve a `Deprecation` and `Sunset` header (RFC 8594) on outgoing versions, and support old and new in parallel for a defined window. The goal is that no client is ever broken without warning.

## Pagination, Filtering & Field Selection

Collection endpoints (`GET /orders`) must never return an unbounded list — that risks enormous responses and database scans. Always paginate.

### Pagination

Two dominant styles:

- **Offset / limit (page-based).** `GET /orders?limit=20&offset=40` (or `?page=3&per_page=20`). Simple and supports jumping to an arbitrary page. **Weaknesses:** deep offsets are slow (the database must skip N rows), and the result set *shifts* if items are inserted or deleted between page fetches, causing duplicates or skips.
- **Cursor / keyset (token-based).** `GET /orders?limit=20&after=eyJpZCI6NDJ9`. The server returns an opaque `next` cursor encoding "where the last page ended" (e.g. the last item's sort key). **Strengths:** stable under concurrent inserts and efficient at any depth (it is a `WHERE id > ?` seek, not a skip). **Cost:** no random page access. Cursor pagination is the right default for large, mutating, or infinite-scroll collections.

A paginated response should carry the items plus pagination metadata — total count where feasible, and a link or cursor to the next page (this is also a small dose of HATEOAS):

```json
{
  "data": [ { "id": 41, "...": "..." }, { "id": 42, "...": "..." } ],
  "pagination": {
    "limit": 20,
    "next_cursor": "eyJpZCI6NDJ9",
    "has_more": true
  },
  "links": {
    "self": "/orders?limit=20",
    "next": "/orders?limit=20&after=eyJpZCI6NDJ9"
  }
}
```

The cost of pagination is small but real. For an `offset/limit` scheme, the work the database does to return page $k$ grows with the offset: returning $n$ rows after skipping $\text{offset}$ rows costs on the order of

$$
O(\text{offset} + n)
$$

because the engine still walks and discards the skipped rows. A keyset seek on an indexed sort key reduces this to roughly

$$
O(\log N + n)
$$

for a B-tree of $N$ rows — independent of how deep into the collection the page is. This asymptotic gap is exactly why cursor pagination wins at scale.

### Filtering, Sorting & Field Selection

- **Filtering** narrows the collection via query parameters: `GET /orders?status=shipped&min_total=100`. For complex predicates, define an explicit, documented filter grammar rather than letting clients inject arbitrary expressions (an injection and performance hazard).
- **Sorting** via a `sort` parameter with a sign convention: `GET /orders?sort=-created_at,total` (descending created date, then ascending total).
- **Field selection (sparse fieldsets)** lets clients request only the fields they need to cut payload size: `GET /orders/42?fields=id,total,status`. Conversely, `?expand=customer` can inline a related resource that would otherwise be a link, saving a round trip. These give clients GraphQL-like control over response shape while staying within REST.

## HATEOAS

**Hypermedia as the Engine of Application State** is the constraint that separates Level-3 REST from "JSON over HTTP". The idea: a response includes not just data but **links** describing the valid next actions from the current state, so the client *discovers* what it can do rather than hard-coding URLs and state-transition rules.

Consider an order. Its representation links to the actions available *given its current state*:

```json
{
  "id": 42,
  "status": "pending",
  "total": 149.90,
  "_links": {
    "self":   { "href": "/orders/42" },
    "cancel": { "href": "/orders/42/cancel", "method": "POST" },
    "pay":    { "href": "/orders/42/payment", "method": "POST" }
  }
}
```

Once the order is `shipped`, the server simply *stops* returning the `cancel` and `pay` links and starts returning a `track` link. The client does not need to encode the rule "you can only cancel a pending order" — it follows whatever links the server offers. This decouples client and server: the server can change URLs and add transitions, and a hypermedia-driven client adapts without a code change.

Standard hypermedia formats include **HAL** (`_links`, `_embedded`), **JSON:API** (`links`, `relationships`), and **Siren**. In honest practice, full HATEOAS is rare: most clients are written against documented, stable URLs and ignore embedded links, so the cost of building and consuming hypermedia controls often outweighs the decoupling benefit. It remains valuable for long-lived, widely-consumed, or evolving APIs (and is genuinely the "correct" REST), but treat it as a deliberate choice, not an obligation.

## Idempotency

**Idempotency** — that repeating a request has the same effect as making it once — is the property that makes a distributed API safe to retry. Networks drop responses: a client sends `POST /payments`, the server charges the card, and the `201` is lost in transit. The client, seeing a timeout, retries — and without protection, charges the card twice.

`GET`, `PUT`, and `DELETE` are idempotent by their HTTP semantics, so retrying them is inherently safe. `POST` is the problem child because it creates a new resource each time. The standard remedy is the **idempotency key**: the client generates a unique key (a UUID) per logical operation and sends it in a header; the server records the key with the result of first processing it, and on any retry with the same key returns the stored result instead of re-executing.

```python
# Server-side idempotency for a non-idempotent POST
def create_payment(request):
    key = request.headers.get("Idempotency-Key")
    if key is None:
        return problem(400, "Missing Idempotency-Key header")

    # Atomically claim the key; if it already exists, return the prior result.
    existing = idempotency_store.get(key)
    if existing is not None:
        if existing.status == "in_progress":
            return problem(409, "Request with this key is still processing")
        return existing.response  # replay the original 201, no double charge

    idempotency_store.put(key, status="in_progress")
    try:
        payment = charge_card(request.body)          # the side-effecting work
        response = created(payment, location=f"/payments/{payment.id}")
        idempotency_store.put(key, status="done", response=response)
        return response
    except Exception:
        idempotency_store.delete(key)                # allow a genuine retry
        raise
```

Key design points: the key must be stored *durably* and claimed *atomically* (so two concurrent retries do not both execute), the stored result must be returned verbatim on replay, and keys should expire after a window (24–72 h is common). Stripe's payments API popularized this pattern; it is the canonical way to give POST the retry-safety that PUT and DELETE get for free.

## OpenAPI / Swagger

An API contract that lives only in prose drifts from the implementation. **OpenAPI** (formerly Swagger) is a machine-readable specification — a YAML/JSON document describing every endpoint, parameter, request/response schema, and status code. From one spec you can generate interactive documentation (Swagger UI), typed client SDKs, server stubs, mock servers, and request/response validation middleware.

```yaml
openapi: 3.1.0
info:
  title: Orders API
  version: 1.2.0
paths:
  /orders/{orderId}:
    get:
      summary: Fetch an order
      operationId: getOrder
      parameters:
        - name: orderId
          in: path
          required: true
          schema: { type: integer, format: int64 }
      responses:
        '200':
          description: The order
          content:
            application/json:
              schema: { $ref: '#/components/schemas/Order' }
        '404':
          description: No such order
          content:
            application/problem+json:
              schema: { $ref: '#/components/schemas/Problem' }
components:
  schemas:
    Order:
      type: object
      required: [id, status, total]
      properties:
        id:     { type: integer, format: int64 }
        status: { type: string, enum: [pending, shipped, cancelled] }
        total:  { type: number, format: double }
    Problem:
      type: object
      properties:
        type:   { type: string, format: uri }
        title:  { type: string }
        status: { type: integer }
        detail: { type: string }
```

Treat the OpenAPI document as the **source of truth**, ideally written before or alongside the implementation ("design-first") rather than generated as an afterthought. Validate it in CI, generate client SDKs from it so clients cannot drift, and use it to drive contract tests — the spec then guarantees that documentation, server, and clients all agree.

## Error Formats (RFC 7807)

Ad-hoc error bodies (`{"error": "bad"}` in one place, `{"message": "...", "code": 7}` in another) force every client to special-case every endpoint. **RFC 7807 / RFC 9457 — Problem Details for HTTP APIs** standardizes the shape. The response uses `Content-Type: application/problem+json` and a defined member set:

```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/problem+json

{
  "type": "https://api.acme.com/problems/insufficient-funds",
  "title": "Insufficient funds",
  "status": 422,
  "detail": "Your balance is 30.00 but the order total is 149.90.",
  "instance": "/orders/42/payment",
  "balance": 30.00,
  "required": 149.90
}
```

The standard members:

- **`type`** — a URI identifying the *problem type* (and ideally a page documenting it). This is the stable, machine-readable discriminator clients should branch on.
- **`title`** — a short, human-readable summary of the type; should not vary per occurrence.
- **`status`** — the HTTP status code, duplicated in the body for convenience.
- **`detail`** — a human-readable explanation specific to *this* occurrence.
- **`instance`** — a URI identifying the specific occurrence (e.g. the request that failed).

You may add **extension members** (`balance`, `required` above) for machine-actionable detail. Because the format is standardized, generic client libraries, logging, and error-reporting tools can parse every error the same way — and validation errors can attach a list of per-field problems as an extension. Adopt one error format across the whole API; consistency here is worth more than cleverness.

## Rate Limiting

Public APIs must protect themselves from abusive or runaway clients and ensure fair sharing of capacity. **Rate limiting** caps how many requests a client may make in a window, returning **`429 Too Many Requests`** when the limit is exceeded.

Common algorithms:

- **Fixed window** — count requests per calendar window (e.g. 1000/minute). Simple but suffers *boundary bursts*: a client can fire 1000 at 11:59:59 and 1000 at 12:00:00, doubling the intended rate across the boundary.
- **Sliding window** — smooths the boundary by weighting the previous window's count, approximating a true rolling limit cheaply.
- **Token bucket** — a bucket holds up to *B* tokens and refills at rate *r* tokens/second; each request spends a token, and requests with no token available are rejected (or queued). This naturally allows short bursts up to *B* while enforcing an average rate *r* — the most common production choice.
- **Leaky bucket** — requests enter a fixed-capacity queue drained at a constant rate, smoothing bursts into a steady outflow.

For the token bucket, capacity $B$ and refill rate $r$ together bound throughput: after the bucket is drained, the long-run sustainable rate is $r$ requests per second, while the largest instantaneous burst is at most $B$ requests. The time to refill an empty bucket to capacity is

$$
t_{\text{refill}} = \frac{B}{r}
$$

so a 100-token bucket refilling at 10 tokens/second is fully replenished in 10 seconds. Tuning $B$ trades burst tolerance against how hard a client can spike the backend.

Communicate limits via response headers so well-behaved clients can self-throttle:

```http
HTTP/1.1 429 Too Many Requests
RateLimit-Limit: 1000
RateLimit-Remaining: 0
RateLimit-Reset: 30
Retry-After: 30
Content-Type: application/problem+json

{ "type": "https://api.acme.com/problems/rate-limited",
  "title": "Rate limit exceeded", "status": 429,
  "detail": "Try again in 30 seconds." }
```

`Retry-After` tells the client exactly how long to wait; a client that honors it (with jitter) avoids hammering a limited endpoint. Limits are usually keyed per API key or per authenticated user, sometimes tiered by plan.

## Caching

HTTP has a rich, built-in caching model — one of the biggest payoffs of honoring REST's *cacheable* constraint. There are two complementary mechanisms.

### Expiration (freshness)

The server tells caches how long a response stays fresh with `Cache-Control`:

```http
Cache-Control: public, max-age=300
```

For `max-age=300` seconds, any cache (browser, proxy, CDN) may serve the stored response without contacting the origin at all. `public` allows shared caches; `private` restricts to the end-client; `no-store` forbids caching entirely (use for sensitive data). Expiration is the cheapest cache: a hit avoids the network round trip completely.

### Validation (conditional requests)

When a cached copy expires, the cache need not re-download an unchanged resource — it **revalidates**. The server stamps responses with a validator:

- **`ETag`** — an opaque version tag (often a content hash). The client later sends `If-None-Match: "<etag>"`; if the resource is unchanged the server returns a tiny **`304 Not Modified`** with no body, and the cache reuses its copy.
- **`Last-Modified`** — a timestamp; the client revalidates with `If-Modified-Since`.

```http
# First response
HTTP/1.1 200 OK
ETag: "a1b2c3"
Cache-Control: max-age=60

# Client revalidates after expiry
GET /orders/42 HTTP/1.1
If-None-Match: "a1b2c3"

# Unchanged → no body re-sent
HTTP/1.1 304 Not Modified
```

ETags pull double duty: they also power **optimistic concurrency** for writes. A client that read `ETag: "a1b2c3"` can send `If-Match: "a1b2c3"` on its `PUT`/`PATCH`; if another writer changed the resource in the meantime its ETag differs, the precondition fails, and the server returns **`412 Precondition Failed`** instead of clobbering the newer state — the lost-update problem solved with a single header. This is the read-side counterpart to idempotency keys on the write side.

## See Also

- **[API Design Hub](./)** — section overview and the other API styles and cross-cutting concerns
- **[Microservices & Event-Driven Architecture](../distributed-systems/microservices-and-event-driven.html)** — API gateways, synchronous vs. asynchronous communication, and where REST fits between services
- **[Networking](../technology/networking/)** — the HTTP, TLS, and transport layer beneath every REST call
- **[Database Design](../technology/database-design/)** — the data stores behind your resources, and how pagination maps to indexed queries
- **[Cybersecurity Basics](../technology/cybersecurity/)** — authentication, authorization, and securing the API surface
