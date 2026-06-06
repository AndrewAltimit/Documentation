---
layout: docs
title: "Game Dev: Multiplayer Networking"
permalink: /docs/gamedev/multiplayer-networking.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Multiplayer Networking</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Authoritative servers, client-side prediction, lag compensation, and the netcode tradeoffs behind responsive online play.</p>
</div>

[Game Development](./) &raquo; Multiplayer Networking

Multiplayer networking is the discipline of making many machines agree on the state of a shared world while the network between them is slow, lossy, and out of order. Unlike a single-player game where the simulation is the truth, an online game has to reconcile what each player *sees* with what *actually happened* on a remote authority — and it has to do so within a budget of milliseconds, or the game feels sluggish and cheaters thrive. This page covers the network topologies (client-server vs peer-to-peer), the authoritative-server model that underpins competitive games, the prediction-and-reconciliation loop that hides latency from the local player, lag compensation for fair hit detection, interpolation and extrapolation for smooth remote entities, the three data models for transmitting state (snapshots, delta state-sync, and RPC), the fundamental netcode tradeoffs, and the basics of matchmaking.

<div class="key-insights">
  <div class="insight-card"><i class="fas fa-server"></i><h4>The server is the only truth</h4><p>In competitive games, clients merely <em>predict</em>; the authoritative server simulates the real world and corrects clients. This is the only robust defense against cheating.</p></div>
  <div class="insight-card"><i class="fas fa-bolt"></i><h4>Hide latency, never wait for it</h4><p>Round-trip time is bounded by physics. Client-side prediction shows the local player an instant result and reconciles when the server's authoritative answer arrives.</p></div>
  <div class="insight-card"><i class="fas fa-history"></i><h4>Fairness means rewinding time</h4><p>Lag compensation rewinds the server's world to when the shooter actually fired, so a player who aimed correctly on their screen lands the shot despite latency.</p></div>
  <div class="insight-card"><i class="fas fa-stream"></i><h4>Send deltas, not the world</h4><p>Bandwidth is finite. Transmit only what changed since the last acknowledged state, quantize aggressively, and prioritize entities by relevance.</p></div>
</div>

## Network Topologies: Client-Server vs Peer-to-Peer

The first architectural decision is *where the simulation lives*. Two families dominate, with a hybrid in between.

### Client-Server (Dedicated and Listen Servers)

Every client talks only to a central **server**, which owns the canonical simulation. Clients send *inputs* (or input intentions) upward and receive *state* downward. There are two flavors:

- **Dedicated server**: a headless process, usually in a datacenter, with no local player. Standard for competitive and large-scale games (Counter-Strike, Valorant, Overwatch).
- **Listen server**: one of the players' machines also acts as the host. Cheaper (no server cost) but gives the host a latency advantage and a single point of failure (host migration is needed if they leave).

```
        Client A ----\
                      \
        Client B ------>  [ Authoritative Server ]  -- owns world state
                      /
        Client C ----/

   Up:   inputs (move, fire, ...)
   Down: snapshots / state updates
```

**Strengths:** a single authority makes cheating hard (the server validates everything), state stays consistent, and the model scales to many players. **Weaknesses:** every action incurs at least one round trip of latency before it is "real," and dedicated servers cost money to run.

### Peer-to-Peer (P2P)

Every peer connects to every other peer; there is no central authority. Two sub-models exist:

- **Lockstep / deterministic P2P**: peers exchange only *inputs* and each independently runs the *exact same deterministic simulation*. Used heavily in RTS games (StarCraft, Age of Empires) where thousands of units would be too expensive to stream as state. Requires bit-exact determinism across machines and stalls if any peer's input is late (everyone waits for the slowest).
- **Replicated-state P2P**: peers exchange state directly. Simpler but bandwidth scales as O(N²) with peer count and trust is distributed (any peer can lie).

```
        Peer A <----> Peer B
           ^   \      /   ^
           |    \    /    |
           v     \  /     v
        Peer C <--\/--> Peer D     (full mesh: N*(N-1)/2 links)
```

For N peers a full mesh has N(N-1)/2 connections:

$$
L = \frac{N(N-1)}{2}
$$

so 8 peers need 28 links and each peer uploads to 7 others — upstream bandwidth becomes the bottleneck quickly.

**Strengths:** no server cost, potentially lower latency for small groups (one hop instead of two), and lockstep is extremely bandwidth-efficient. **Weaknesses:** trust is hard (any peer can cheat), NAT traversal is fiddly, and it scales poorly past a handful of players.

### Hybrid

Many shipping games mix the two: dedicated servers handle matchmaking, persistence, and anti-cheat, while a relay or P2P path carries the latency-critical gameplay traffic. Fighting games (rollback netcode, see below) often run P2P gameplay over a matchmaking backend. Battle-royale and MMO titles are almost always pure client-server.

### Choosing a Topology

| Criterion | Client-Server (dedicated) | Listen Server | Lockstep P2P | Replicated P2P |
|-----------|---------------------------|---------------|--------------|----------------|
| Cheat resistance | High (server authority) | Medium (host trusted) | Low–Medium | Low |
| Latency to act | 1 RTT (predicted: ~0) | 0 for host, 1 RTT others | Input-delay bound | ~0.5 RTT |
| Bandwidth | Server fans out state | Host fans out | Tiny (inputs only) | O(N²) |
| Player count | Hundreds+ | Tens | Tens (input-bound) | Handful |
| Operating cost | High | None | None | None |
| Typical genre | FPS, MMO, BR | Co-op, casual | RTS | Fighting (small) |

## Authoritative Servers

In an **authoritative** model the server's simulation is definitionally correct; the client's view is an *optimistic guess* that the server may overrule. The client never directly sets its own position on the server — it sends the *input* ("I am holding W and pressed jump on tick 4012") and the server runs the movement code to decide where the player actually ends up.

This separation is the foundation of anti-cheat. A client that sends "my position is (9999, 9999), I have 999 health, I just got 50 kills" is simply ignored, because the server computes those values itself. The threat model shrinks to *information* cheats (wallhacks from over-sent state) and *input* cheats (aimbots that send legitimate-but-superhuman inputs), both of which have their own mitigations (relevancy filtering and statistical detection).

```
Server tick loop (fixed timestep, e.g. 64 Hz):

  for each tick t:
      collect_inputs_for_tick(t)       # from all clients, buffered
      for each player:
          apply_movement(player, input) # server is the physics authority
      run_game_logic(t)                 # combat, scoring, pickups
      for each client:
          send_snapshot(client, t)      # only what is relevant to them
```

The server runs at a **fixed tick rate** (commonly 20–128 Hz). A higher tick rate reduces the granularity of error between client and server and improves hit registration, at the cost of CPU and bandwidth. CS2 and Valorant run effectively 64–128 Hz subtick/tick simulations; many MMOs run far lower (10–20 Hz) because precise sub-frame timing matters less.

**Server-side validation** typically includes: input sanity (no impossible acceleration), rate limiting, line-of-sight checks before revealing entities, and physics constraints (you cannot walk through a wall because the server runs the same collision code the client does).

## Client-Side Prediction and Server Reconciliation

If the client waited a full round trip every time the player pressed a key, a 100 ms ping would mean 100 ms of input lag — unplayable for an FPS. **Client-side prediction** solves this: the client runs the *same movement simulation locally* and shows the result immediately, then corrects itself when the authoritative answer arrives. This is the single most important technique in modern netcode and is the architecture popularized by the Quake 3 / Source ("Valve") networking model.

### The Prediction Loop

1. The client samples input for tick `t`, tags it with a **sequence number**, and sends it to the server.
2. **Immediately**, the client applies that same input to its *local predicted state* and renders the result — zero perceived latency.
3. The client stores the input (and the resulting predicted state) in a ring buffer of *unacknowledged* inputs.
4. The server processes inputs in order and returns authoritative state stamped with the **last input sequence number it has applied** (the acknowledgment).

### Reconciliation

When a snapshot arrives acknowledging input `N`, the client compares its *predicted* state at tick `N` against the *authoritative* state:

- If they match (within an epsilon), the prediction was correct — discard inputs up to `N` and carry on.
- If they differ (a missed jump, a collision the client didn't predict, knockback from another player), the client **snaps to the authoritative state** and then **re-simulates** every still-unacknowledged input `N+1, N+2, ...` on top of it. This "replay" brings the prediction back up to the present without throwing away inputs the server hasn't seen yet.

```
Client input buffer (unacked):  [N+1][N+2][N+3] -> present

Snapshot arrives: authoritative state @ N

   1. Rewind: local state := authoritative_state[N]
   2. Replay: apply input N+1, N+2, N+3 in order
   3. Result: corrected predicted state at present tick
```

```python
# Pseudocode: client reconciliation
def on_server_snapshot(snapshot):
    # Discard inputs the server has already processed
    pending.drop_until(snapshot.last_acked_input)

    predicted = local_state_at(snapshot.last_acked_input)
    if distance(predicted, snapshot.state) > EPSILON:
        # Misprediction: snap to authority, then replay unacked inputs
        local_state = snapshot.state.copy()
        for inp in pending:           # in sequence order
            local_state = simulate(local_state, inp, FIXED_DT)
    # else: prediction was correct, nothing to do
```

The key invariant: **client and server must run identical movement code** over the same fixed timestep. If they diverge (different gravity constant, different collision epsilon), reconciliation will constantly snap and the player will rubber-band. This is why prediction is usually applied only to the *local player's own movement* — predicting other players' actions is unreliable, so those are handled by interpolation instead.

### Rollback Netcode (the P2P variant)

Fighting games use a closely related idea called **rollback**: each peer predicts the opponent's input (usually "same as last frame"), simulates forward immediately, and when the real input arrives, rolls the whole game state back to that frame and re-simulates. Because fighting games are deterministic and short-horizon, this hides one-to-several frames of latency with no perceived delay, and is the gold standard (GGPO) for competitive fighters.

## Lag Compensation

Prediction fixes the *local player's* feel, but it creates a fairness problem for *interactions between players*. Suppose you have 100 ms of latency and you fire at an enemy who is, on your screen, standing still in a doorway. By the time your "fire" packet reaches the server, ~50 ms have passed and the server's version of that enemy has already moved out of the doorway. Without compensation, your perfectly-aimed shot misses through no fault of yours.

**Lag compensation** (server-side rewind) fixes this. The server keeps a short **history buffer** of every entity's position over the last few hundred milliseconds. When it receives a "fire" command, the command carries the client's view timestamp (or the server estimates it from the client's known RTT and interpolation delay). The server then:

1. Computes the time the shooter actually saw the world: roughly `now - (RTT/2) - interpolation_delay`.
2. **Rewinds** all relevant entities to their positions at that historical time.
3. Performs the hit test against that rewound world.
4. Restores the present state and applies the result (damage) at the current tick.

```
Real timeline on server:

   t-150ms     t-100ms     t-50ms       now
     |-----------|-----------|------------|
   enemy@A    enemy@B    enemy@C      enemy@D

Shooter (100ms RTT, ~50ms interp) saw the enemy at ~B.
On "fire", server rewinds enemy to B, tests the ray, then restores to D.
```

The total rewind amount is bounded for sanity (you cannot rewind seconds, or a high-ping player would be shooting into the distant past):

$$
t_{\text{rewind}} = \frac{\text{RTT}}{2} + t_{\text{interp}}
$$

Lag compensation trades one unfairness for another: the *shooter* gets a fair shot, but the *target* can experience "I was already behind cover and still died" (being shot around a corner). This is an inherent, unavoidable tradeoff — you cannot make both the shooter's and the target's local views simultaneously authoritative when they disagree by a round trip. Designers tune the maximum rewind window to balance the two, and competitive shooters generally favor the shooter.

## Interpolation and Extrapolation

The local player is predicted; **remote** entities are a different problem. The client receives discrete snapshots of remote players at the server tick rate (say every 33 ms at 30 Hz). Rendering them only on snapshot arrival would look like a 30 FPS slideshow on a 144 Hz monitor, with visible stutter and teleporting on packet loss. Two techniques smooth this.

### Entity Interpolation

Instead of rendering remote entities at the *latest* received position, the client deliberately renders them slightly **in the past** — typically one or two snapshot intervals behind. It keeps a buffer of recent snapshots and, for the render time `t`, finds the two snapshots that bracket it and **linearly interpolates** between them:

$$
P(t) = P_0 + (P_1 - P_0)\,\frac{t - t_0}{t_1 - t_0}
$$

This guarantees smooth motion because the client is always interpolating between two known-good positions rather than guessing. The cost is a small, fixed **interpolation delay** (e.g. 100 ms in older Source games): remote players are shown 100 ms behind reality. This delay is exactly what lag compensation must account for when rewinding.

```python
def interpolated_position(buffer, render_time):
    # render_time = now - interpolation_delay
    s0, s1 = buffer.bracket(render_time)   # two snapshots around render_time
    alpha = (render_time - s0.time) / (s1.time - s0.time)
    return lerp(s0.pos, s1.pos, alpha)     # smooth, always between knowns
```

Interpolation delay is the classic latency/smoothness tradeoff: a larger buffer hides more jitter and packet loss but makes remote players laggier relative to truth; a smaller buffer is more responsive but stutters under loss.

### Extrapolation (Dead Reckoning)

When a snapshot is *late or lost*, the client has nothing to interpolate toward, so it must **extrapolate** — predict where the entity *probably* is by projecting its last known velocity (and sometimes acceleration) forward:

$$
P(t) = P_0 + v_0\,(t - t_0) + \tfrac{1}{2}\,a_0\,(t - t_0)^2
$$

This **dead reckoning** keeps motion fluid through brief dropouts, but it *guesses*, so it is wrong whenever the entity changes direction. When the real snapshot finally arrives, the client must smoothly correct the error (an instant snap looks like a teleport; a blend over a few frames hides it). Extrapolation is bounded to a short window (e.g. 250 ms) — extrapolating too far produces players sliding through walls.

| | Interpolation | Extrapolation |
|---|---|---|
| Uses | Past snapshots (between two knowns) | Last snapshot + velocity (guess forward) |
| Accuracy | Exact at snapshot points | Wrong on direction changes |
| Cost | Fixed render delay | Visible correction snaps |
| When | Normal operation | Packet loss / late snapshots |

In practice clients interpolate by default and fall back to extrapolation only to cover gaps, blending back smoothly once fresh data arrives.

## Data Models: Snapshots vs State-Sync vs RPC

There are three distinct ways to get information across the wire, and most games use all three for different jobs.

### Snapshots (Full / Delta)

A **snapshot** is the complete relevant world state at a given tick — every entity's position, rotation, health, animation state. Sending the *full* snapshot every tick is simple but bandwidth-hungry. The standard optimization is **delta compression**: the server sends a full baseline occasionally, then transmits only what *changed* relative to the last snapshot the client **acknowledged**.

```
Baseline @ tick 100:  {p1: full, p2: full, p3: full}
Delta @ tick 101:     {p2: pos changed}            # p1, p3 omitted
Delta @ tick 102:     {p1: health changed, p3: pos changed}
```

Snapshots are robust to packet loss (each delta is computed against an *acked* baseline, so a lost packet just makes the next delta larger rather than corrupting state) and pair naturally with interpolation. This is the Quake/Source model.

### State Synchronization (Replicated Properties)

**State-sync** frameworks (Unreal's replication, Unity Netcode, Photon) let you mark individual variables as *replicated*. The engine watches those properties on the server and automatically pushes changes to relevant clients — you write `replicated float Health;` and the framework handles the wire format, dirty-tracking, and relevancy. It is essentially per-property delta compression with ergonomic tooling, and it is the dominant model in commercial engines because it is declarative and integrates with the gameplay framework.

### Remote Procedure Calls (RPC / Events)

An **RPC** is a one-shot *event*: "play this explosion sound," "the round just ended," "fire weapon." Unlike replicated state (which describes *what is*), RPCs describe *what happened* at a moment. They are used for discrete, non-recurring events where streaming a continuous property makes no sense. RPCs come in directions — *client-to-server* (request an action, always validated), *server-to-client* (notify a result), and *multicast* (server to all relevant clients).

| Model | Describes | Loss behavior | Best for |
|-------|-----------|---------------|----------|
| Snapshot/delta | Continuous world state | Self-healing via acked baseline | Positions, physics, everything-at-once |
| State-sync | Per-property values | Engine-managed, eventually consistent | Health, ammo, scores, doors |
| RPC/event | Discrete events | Must be made reliable if important | Sounds, hits, round events, abilities |

A subtlety: RPCs are usually sent over a **reliable-ordered** channel (they can't just be lost — a missed "you died" is unacceptable), while snapshots ride an **unreliable** channel (a lost position update is irrelevant because a newer one is 16 ms behind it). This is why UDP-based game protocols implement *selective* reliability per message rather than using raw TCP.

## Netcode Tradeoffs

Every netcode decision is a tradeoff between four scarce resources: **latency**, **bandwidth**, **CPU**, and **fairness/consistency**. There is no globally optimal point; the right choice depends on genre.

- **Transport: UDP vs TCP.** Real-time games almost universally use **UDP** (or QUIC) because TCP's in-order, fully-reliable delivery causes **head-of-line blocking** — a single lost packet stalls *all* subsequent data until it is retransmitted, which is catastrophic for a stream of position updates where the newest packet matters most. Games build their own lightweight reliability layer on UDP, marking each message reliable or unreliable as appropriate. See [Network & I/O Optimization](../optimization/network-io-optimization.html) for the deeper protocol comparison.

- **Tick rate.** Higher tick rate = lower error and crisper hit registration, but linearly more CPU and bandwidth. 64–128 Hz for competitive FPS; 10–30 Hz for MMOs and casual games.

- **Bandwidth budget.** Quantize floats (positions to fixed-point, rotations to compressed quaternions), use **relevancy / area-of-interest** filtering so a client only hears about nearby entities, and prioritize updates (a player shooting at you matters more than one across the map). Over-sending also leaks information that fuels wallhacks, so relevancy filtering is *both* a bandwidth and an anti-cheat measure.

- **Prediction vs determinism.** Client-prediction/reconciliation (FPS) hides local latency but requires shared movement code and tolerates floating-point drift. Deterministic lockstep (RTS) is bandwidth-tiny but demands *bit-exact* determinism and stalls on the slowest peer.

- **Input delay vs rollback.** Fighting games choose between adding a few frames of **input delay** (everyone feels a little laggy, but no visual corrections) or **rollback** (instant response, occasional visual snaps when predictions miss). Modern competitive fighters favor rollback.

- **Authority placement.** Full server authority maximizes cheat resistance but costs a round trip and server money; giving clients authority over their own movement is cheaper and snappier but trusts the client. Most games split the difference: clients are authoritative over *cosmetic* state and *predicted* over movement, while the server stays authoritative over anything that affects other players (damage, scoring, pickups).

<div class="key-insights">
  <div class="insight-card"><i class="fas fa-balance-scale"></i><h4>No free lunch</h4><p>You cannot simultaneously minimize latency, bandwidth, and CPU while maximizing fairness. Pick the two that matter for your genre and tune the rest.</p></div>
  <div class="insight-card"><i class="fas fa-shield-alt"></i><h4>Relevancy is dual-purpose</h4><p>Only sending nearby entities saves bandwidth <em>and</em> denies cheaters the data wallhacks need.</p></div>
</div>

## Matchmaking Basics

Before any of the above runs, players must be grouped into sessions. **Matchmaking** is the service layer that takes a population of waiting players and assigns them to game servers, balancing several competing objectives.

### Core Goals

- **Skill balance.** Produce close games. A rating system such as **Elo**, **Glicko-2**, or Microsoft's **TrueSkill** estimates each player's skill (and uncertainty), and the matchmaker pairs players of similar rating. After a match, ratings update based on the (un)expected result — beating a much stronger player gains more than beating a weaker one. The classic Elo expected-score formula for player A against player B is:

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

- **Connection quality.** Pick a server (region/datacenter) that minimizes the *maximum* ping across the lobby, so no player is badly disadvantaged. This often trades against skill balance — a perfectly skill-matched lobby spread across three continents plays worse than a slightly looser one in a single region.

- **Wait time.** Tighter constraints (skill window, ping window, party size) mean fewer eligible opponents and longer queues. Matchmakers therefore **relax constraints over time**: a player who waits expands their acceptable skill and ping ranges until a match is found.

### A Typical Pipeline

```
Player presses "Find Match"
        |
        v
[ Queue / Ticket ]  -- skill, region, party, game mode
        |
        v
[ Matchmaker ]  -- forms balanced lobby within current constraints
        |        (relaxes skill/ping window as wait grows)
        v
[ Session / Server Allocation ]  -- spin up or assign a dedicated server
        |
        v
[ Players connect ]  -- now the netcode loop above takes over
```

Additional concerns at scale include **party handling** (groups must stay together, and a 5-stack of friends should be matched against comparable coordination), **backfill** (replacing players who leave mid-match), **anti-toxicity / role queues**, and **server allocation** (orchestrating fleets of dedicated game servers — often on the same [Kubernetes](../technology/kubernetes/)-style infrastructure as any other backend, with autoscaling tied to queue depth). Matchmaking is ultimately a constrained optimization problem balanced live against player population.

## Putting It Together

A modern competitive online game stitches all of these layers into one loop:

1. **Matchmaking** groups skill-balanced players onto a low-ping dedicated server.
2. The **authoritative server** runs the canonical fixed-tick simulation; clients send only inputs.
3. **Client-side prediction** shows the local player instant results, and **reconciliation** replays unacked inputs against authoritative corrections.
4. **Lag compensation** rewinds the server's history buffer so well-aimed shots register fairly.
5. Remote entities are **interpolated** from buffered snapshots, **extrapolated** briefly through packet loss.
6. State crosses the wire as **delta snapshots** (continuous world) and **RPCs** (discrete events), over **UDP** with per-message reliability and **relevancy filtering** to cap bandwidth and starve cheaters of data.

Every one of those choices is a tradeoff dialed in for the genre — and getting them right is what separates netcode that feels instant from netcode that feels like playing underwater.

## See Also

- [Game Development](./) - The section hub: engines, core systems, and the multiplayer overview
- [Network & I/O Optimization](../optimization/network-io-optimization.html) - Latency vs bandwidth, UDP/QUIC vs TCP, batching, and zero-copy transports underlying game netcode
- [Networking Fundamentals](../technology/networking/) - Low-level packet, protocol, and NAT concepts beneath multiplayer transport
- [Performance Optimization](../optimization/) - Profiling-driven methodology for hitting tick-rate and bandwidth budgets
- [Game AI](../ai-ml/game-ai.html) - Server-side behavior that must stay deterministic and authoritative in multiplayer
- [Distributed Systems Theory](../advanced/distributed-systems-theory/) - Consistency, latency, and coordination foundations behind authoritative state
