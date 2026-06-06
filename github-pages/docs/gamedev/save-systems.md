---
layout: docs
title: "Game Dev: Save Systems & Persistence"
permalink: /docs/gamedev/save-systems.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Save Systems & Persistence</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Serializing game state durably and safely — formats, save-game architecture, versioned migration, autosave and checkpoints, cloud sync, and atomic writes that survive a crash mid-save.</p>
</div>

[Game Development](./) &raquo; Save Systems & Persistence

<div class="hub-intro">
  <p class="lead">A save system is the boundary between the in-memory world your game simulates and the durable bytes a player will reload weeks later — possibly after they have patched the game, switched devices, or pulled the power cord the instant you were halfway through writing the file. Getting it right is less about "writing some JSON" and more about treating the save file as a small database with a schema, a migration path, a corruption-recovery story, and a transactional write protocol. This page covers serialization format choices, the architecture that separates persistent state from runtime state, versioning and migration of old saves, autosave and checkpoint strategies, cloud-save synchronization and conflict resolution, and the atomic-write and anti-corruption techniques that keep a save from being destroyed by an ill-timed crash.</p>
</div>

<div class="key-insights">
  <div class="insight-card">
    <i class="fas fa-database"></i>
    <h4>A save file is a tiny database</h4>
    <p>It has a schema, needs migrations when that schema changes, and must survive concurrent or interrupted writes. Treat it with the same discipline.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-code-branch"></i>
    <h4>Version from day one</h4>
    <p>Every save must carry a version number. Without it, the first content patch that adds a field can brick every existing save in the wild.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-shield-alt"></i>
    <h4>Never write in place</h4>
    <p>Write to a temp file, fsync, then atomically rename over the real save. A crash then leaves either the old file or the new one — never a half-written one.</p>
  </div>
  <div class="insight-card">
    <i class="fas fa-cloud"></i>
    <h4>Cloud saves are a sync problem</h4>
    <p>Two devices, last-writer-wins, and offline play create conflicts. You need timestamps, generation counters, and a conflict-resolution policy.</p>
  </div>
</div>

## What Goes in a Save

The first design decision is *what state to persist*, and it is more subtle than "everything." Game state divides into three categories:

- **Persistent state** — the authoritative facts the player expects to keep: inventory, quest flags, position, unlocked abilities, world mutations (the chest you opened stays open), settings, statistics. This is what the save file holds.
- **Derived state** — anything reconstructable from persistent state at load time: the navmesh, spatial partitions, rendered meshes, cached pathfinding, UI layout. Saving it wastes space and creates a second source of truth that can disagree with the authoritative data. Recompute it on load instead.
- **Transient state** — per-frame and per-session data that should *not* survive a reload: particle systems mid-emission, in-flight projectiles (usually), input buffers, network sockets, audio voices. Capturing these makes saves brittle and large for no player-visible benefit.

The discipline of explicitly classifying each field is the single biggest determinant of how maintainable the save system becomes. A common failure mode is serializing the entire scene graph — every transform, every component — which couples the save format to engine internals and breaks the moment you reorganize a prefab.

```
                  Game State
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
 Persistent       Derived          Transient
 (save this)   (recompute on    (discard at
                   load)         session end)

 inventory      navmesh           particle FX
 quest flags    pathfinding       projectiles
 world deltas   render caches     audio voices
 player pos     spatial grids     input buffers
 settings       LOD selection     network state
```

## Serialization Formats

A serializer turns an in-memory object graph into a byte stream (and back). The format choice trades human-readability, size, speed, schema-evolution friendliness, and security against each other.

### Format comparison

| Format | Readable | Size | Speed | Schema evolution | Notes |
|--------|----------|------|-------|------------------|-------|
| JSON | Yes | Large | Slow | Easy (named, optional fields) | Great for debugging; verbose; no binary blobs |
| Binary (custom) | No | Smallest | Fastest | Manual / fragile | Total control; you own versioning |
| Protocol Buffers | No | Small | Fast | Excellent (field numbers, defaults) | Tag-based, forward/backward compatible |
| FlatBuffers / Cap'n Proto | No | Small | Fastest (zero-copy) | Good | Read without parsing; great for big saves |
| MessagePack / CBOR | No | Small | Fast | Easy (like binary JSON) | Compact, schema-less, self-describing |
| XML | Yes | Largest | Slowest | Easy | Mostly legacy; avoid for new saves |
| Engine native (Unity `JsonUtility`, UE `FArchive`) | Varies | Varies | Fast | Engine-coupled | Convenient but ties save to engine version |

### Choosing a format

The right answer depends on the *failure mode you most want to avoid*:

- **You will be patching the game for years** → choose a tag-based format (Protocol Buffers, MessagePack, or a custom binary format with explicit field IDs). Tag-based formats let new code read old data and old code skip unknown fields, which is the foundation of painless migration.
- **Saves are large (open-world, lots of entities)** → prefer a zero-copy format (FlatBuffers) so you can memory-map the save and read only what you touch, and consider compression (below).
- **You are debugging or building tools** → JSON during development is invaluable; you can diff saves, hand-edit them, and read crash reports. Many studios ship a debug build with JSON and a release build with binary, gated behind the same schema.
- **Security matters** (cloud saves, leaderboards) → never use a format with built-in arbitrary-type deserialization (Java/.NET `BinaryFormatter`, Python `pickle`, naive C# `BinaryFormatter`). These let a tampered save execute code. Use a data-only format and validate every field on load.

### A self-describing binary header

Whatever the payload format, prefix the file with a small fixed header you can read before committing to a full parse. This lets you reject foreign files, detect corruption early, and route to the right migration path:

```
struct SaveHeader {
    char     magic[4];     // "ASAV" — reject anything else immediately
    uint16_t format;       // 0 = JSON, 1 = msgpack, 2 = custom-binary
    uint16_t version;      // schema version → drives migration
    uint32_t payload_len;  // bytes of payload that follow
    uint32_t checksum;     // CRC32 / xxHash of the payload
    uint64_t timestamp;    // unix millis, for cloud conflict resolution
    uint64_t generation;   // monotonically increasing save counter
}
```

The `magic` bytes guard against reading the wrong file type; the `version` field is what makes migration possible; the `checksum` is the first line of corruption defense; and `timestamp`/`generation` feed cloud-save conflict resolution.

## Save-Game Architecture

A maintainable save system keeps a hard wall between the **runtime object model** (rich, behavior-laden objects the game logic uses) and the **save model** (plain data, no methods, designed for serialization). Mixing them is the original sin of save systems: it couples your on-disk format to internal refactors, makes versioning impossible, and tempts you to serialize pointers or engine handles that mean nothing after a reload.

### The serialize / deserialize boundary

```
Runtime World                Save Model (DTOs)            Bytes
┌──────────────┐  capture   ┌───────────────────┐  encode ┌────────┐
│ Player obj   │ ─────────► │ PlayerData {       │ ──────► │ header │
│  + behavior  │            │   pos, hp, items   │         │ +      │
│ Entity refs  │            │ }                  │         │ payload│
│ Engine hdls  │            │ WorldDeltas {...}  │         │ +      │
└──────────────┘            └───────────────────┘         │ crc    │
        ▲          restore           ▲           decode    └────────┘
        └─────────────────────────────┴────────────────────────┘
```

Each persistent system exposes two functions: one to *capture* its state into a plain data object, and one to *restore* itself from that object. The save manager orchestrates them but knows nothing about their internals.

```csharp
// Each system implements a small, data-only contract.
public interface ISaveable
{
    string  Key { get; }                 // stable id, e.g. "inventory"
    object  Capture();                   // returns a plain DTO
    void    Restore(object data);        // rebuilds runtime state
}

public sealed class SaveManager
{
    private readonly List<ISaveable> _systems;

    public SaveDocument BuildSnapshot(int schemaVersion)
    {
        var doc = new SaveDocument { Version = schemaVersion };
        foreach (var s in _systems)
            doc.Sections[s.Key] = s.Capture();   // capture is pure: no I/O
        return doc;
    }

    public void ApplySnapshot(SaveDocument doc)
    {
        var migrated = Migrations.UpgradeToCurrent(doc);  // see Versioning
        foreach (var s in _systems)
            if (migrated.Sections.TryGetValue(s.Key, out var data))
                s.Restore(data);
    }
}
```

Two properties make this scale:

1. **`Capture()` does no I/O.** Building the snapshot is a fast, synchronous, allocation-bounded operation. The slow part (encoding + writing bytes) happens separately, which is what lets you do non-blocking autosaves (capture on the game thread, write on a worker thread).
2. **Sections are keyed and independent.** A save is a dictionary of named sections, not one monolithic blob. New systems add a section without touching old ones; a missing section just means "use defaults," which is automatically forward/backward compatible.

### Handling object references

The hardest part of restoring is reconnecting references. The player's "current quest" is a pointer to a quest object; you cannot serialize the pointer. The standard solution is **stable IDs**:

- Give every persistent entity a stable identifier (a GUID baked into the level, or a runtime-assigned ID recorded in the save).
- Serialize references *as IDs*, never as pointers or array indices (indices break the moment ordering changes).
- Restore in two passes: first instantiate all entities and register their IDs in a lookup table, then re-resolve every ID reference into a live pointer.

```
Pass 1 (instantiate):   id 0x4F → Player,  id 0x91 → Quest, id 0xA2 → Chest
Pass 2 (relink):        Player.activeQuest = lookup[0x91]
                        Chest.owner        = lookup[0x4F]
```

This two-pass approach also gracefully handles cycles (A references B references A) that would otherwise cause infinite recursion in a naive depth-first serializer.

## Versioning & Migration

The day you ship, your save format is frozen for every existing player. The next patch that adds a feature almost always changes the schema, and those players must be able to load their old saves. Versioning is therefore not optional polish — it is a launch-day requirement.

### The version number is sacred

Every save carries a `version` integer (the one in the header). It identifies the schema the file was written with. On load, you compare it to the *current* code version and apply a chain of migrations to bridge the gap.

```
saved with v3 ──► migrate v3→v4 ──► migrate v4→v5 ──► current (v5) ──► load
```

Migrations are written **once, additively, and never modified afterward**. A `v3→v4` migration is run by every player who ever had a v3 save; if you edit it later you risk corrupting saves that already passed through it. The rule is: append new migration steps, never rewrite old ones.

```csharp
public static class Migrations
{
    // An ordered list: index i upgrades version i → i+1.
    private static readonly Func<SaveDocument, SaveDocument>[] Steps =
    {
        V1ToV2,   // [0]: v1 → v2
        V2ToV3,   // [1]: v2 → v3
        V3ToV4,   // [2]: v3 → v4
    };

    public const int CurrentVersion = 4;

    public static SaveDocument UpgradeToCurrent(SaveDocument doc)
    {
        if (doc.Version > CurrentVersion)
            throw new SaveTooNewException(doc.Version);   // downgrade is unsafe

        while (doc.Version < CurrentVersion)
        {
            doc = Steps[doc.Version - 1](doc);
            doc.Version++;
        }
        return doc;
    }

    // Example: v3 split a combined "name" field into "first"/"last".
    private static SaveDocument V3ToV4(SaveDocument d)
    {
        var p = (PlayerData)d.Sections["player"];
        var parts = (p.LegacyName ?? "").Split(' ', 2);
        p.FirstName = parts.Length > 0 ? parts[0] : "";
        p.LastName  = parts.Length > 1 ? parts[1] : "";
        p.LegacyName = null;
        return d;
    }
}
```

### Strategies for schema evolution

| Change | Safe approach |
|--------|---------------|
| **Add a field** | Default it on load. Tag-based formats do this for free; for JSON, treat missing as the default. |
| **Remove a field** | Stop reading it; tag-based formats let old data carry the dead field harmlessly. Never reuse its tag/ID. |
| **Rename a field** | Add the new field, write a migration that copies old→new, keep reading the old name for one version. |
| **Change a type** | Add a new field of the new type; migrate by conversion; deprecate the old field. |
| **Restructure** | Write an explicit migration step (e.g. the name-split above). This is why you want the migration framework. |

The golden rule of tag-based formats: **field IDs are forever.** Once `field 7 = stamina`, you never reassign `7` to something else, even after removing stamina — old saves still have a value tagged `7`, and reusing it would misinterpret their data.

### Forward compatibility

Backward compatibility (new code reads old saves) is mandatory. **Forward compatibility** (old code reads *newer* saves) is a bonus that tag-based formats provide: old code skips unknown fields. It matters when players have multiple installs (cloud sync across two devices on different patch levels), or when you want a save written on a beta build to still load on the release build. When forward compatibility is impossible, refuse the load cleanly (`SaveTooNewException` above) rather than crashing or silently dropping data.

## Autosave & Checkpoints

Autosaving is the difference between a frustrated player who lost an hour and one who barely noticed a crash. The design space spans *when* to save, *how* to avoid stutter, and *how many* saves to keep.

### When to autosave

| Trigger | Pros | Cons |
|---------|------|------|
| **Checkpoints** (designer-placed) | Predictable, safe states; tunable difficulty | Lost progress between checkpoints |
| **Timed interval** (every N min) | Bounded data loss | May save in a bad/unsafe state |
| **Event-driven** (zone change, boss defeated) | Saves at natural breakpoints | Needs care to define "safe" moments |
| **On quit / focus loss** | Captures latest state on exit | Unreliable on hard crashes/power loss |

Robust systems combine these: event-driven autosaves at natural boundaries, a timed safety net, and a save-on-pause/quit. The key constraint is to autosave only in a **consistent, safe state** — not mid-cutscene, mid-transition, or while a quest is in a half-applied state — so that reloading always lands the player somewhere playable.

### Non-blocking autosave

A 50 ms hitch every autosave is a quality bug. The fix is to split the operation across threads, exploiting the capture/encode/write separation from the architecture section:

```
Game thread:    [ Capture() ]  ── fast, synchronous snapshot (deep-copy data)
                      │  hand off immutable snapshot
                      ▼
Worker thread:  [ Encode → Compress → Atomic write ]  ── slow, off the hot path
```

The capture must produce an **immutable snapshot** (a deep copy of the persistent data) so the worker can serialize it while the game thread keeps mutating the live world. Serializing the live world directly from a worker thread is a data race that produces torn, corrupt saves.

### Rotating slots and ring buffers

Never overwrite the only save during an autosave — a crash mid-write would destroy it (the atomic-write section addresses this, but rotation is defense in depth). Keep a small ring of autosave slots:

```
autosave_0  autosave_1  autosave_2   ← rotate, oldest overwritten
     ▲
   newest

A crash that corrupts autosave_2 still leaves _1 and _0 intact.
```

Rotation also gives players an "undo a mistake" affordance and protects against a *logically* bad save (e.g. you autosaved while soft-locked) by letting them step back a slot.

## Cloud Saves & Sync

Cloud saves move the same save file between devices through a remote store (Steam Cloud, PlayStation/Xbox cloud, iCloud, Google Play Games, or your own backend). The moment two devices can write, you have a distributed-systems problem: **conflict resolution**.

### The conflict

```
Device A (offline)  ──play──►  save gen 5, t=10:00  ──┐
                                                       ├──► conflict at sync
Device B (offline)  ──play──►  save gen 5, t=10:30  ──┘
```

Both devices started from generation 5 and diverged. Neither is strictly "newer in causal terms" — they are concurrent edits. Strategies, from crudest to safest:

- **Last-writer-wins by timestamp.** Pick the file with the later modified time. Simple, but clocks lie (device clock skew, time-zone changes) and it silently destroys the loser's progress.
- **Generation counter + timestamp.** Track a monotonically increasing `generation` (in the header). The cloud stores the highest generation seen; a device must base its write on the latest generation or be told to reconcile. This catches stale writes that timestamps miss.
- **Player-prompted resolution.** When a true conflict is detected (two saves with the same parent generation), show the player both options — "Device A: Level 12, 4h played" vs "Device B: Level 14, 5h played" — and let them choose. This is the gold standard for single-player games because it never silently discards progress.
- **Mergeable state.** For some data (unlocked achievements, max-level-reached, total currency earned) you can merge by taking the union/maximum rather than choosing a winner. Design save sections to be merge-friendly where possible (CRDT-like monotonic fields).

### A practical sync protocol

1. On launch, fetch the cloud header (cheap: just metadata).
2. Compare cloud `generation` to the local `generation`.
3. If cloud > local and local is unmodified since last sync → download (cloud is authoritative).
4. If local > cloud and they share a parent → upload (local is ahead).
5. If both advanced from the same parent → **conflict**: prompt the player or merge.
6. After a successful play session, write locally (atomically), then upload with the new generation; the cloud rejects the upload if its generation has advanced underneath you (optimistic concurrency), forcing a re-check.

This is optimistic concurrency control — the same pattern relational databases use for lost-update prevention. See [Database Design](../technology/database-design/) for the underlying theory of transactions and concurrency. Always keep cloud sync *advisory*: a network failure must never block the player from playing offline against the local save.

## Anti-Corruption & Atomic Writes

The most important property of a save system is that it **never destroys a good save**. A player who loses an hour to a missed autosave is annoyed; a player whose 80-hour file is corrupted is gone for good. The threats are: a crash or power loss mid-write, partial writes, OS/filesystem caching that reorders writes, and bit-rot/tampering.

### Atomic write via temp-file-and-rename

Writing a file in place is the cardinal sin. If the process dies after truncating the old file but before finishing the new contents, you are left with a half-written, useless save *and* the original is already gone. The fix is the **write-temp-then-rename** protocol, which exploits the fact that `rename()` is atomic on POSIX and Windows (on the same volume):

```
1. Write the full new save to  save.tmp
2. flush() + fsync(save.tmp)        ← force bytes to physical disk, not just cache
3. fsync(directory)                 ← ensure the temp file's directory entry is durable
4. rename(save.tmp → save.dat)      ← ATOMIC: an observer sees old OR new, never partial
5. fsync(directory)                 ← make the rename itself durable
```

```csharp
public static void AtomicWrite(string path, ReadOnlySpan<byte> bytes)
{
    string tmp = path + ".tmp";
    using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write,
                                   FileShare.None))
    {
        fs.Write(bytes);
        fs.Flush(flushToDisk: true);   // fsync the data
    }
    // File.Replace performs an atomic rename and (optionally) keeps a backup.
    if (File.Exists(path))
        File.Replace(tmp, path, path + ".bak");  // old save → .bak
    else
        File.Move(tmp, path);
}
```

The crucial subtlety is the **`fsync` before the rename**. Without it, the OS may have the rename durable but the file *contents* still in a write-back cache; a power loss then leaves a renamed-but-empty file — the worst outcome. The order "fsync contents, then rename, then fsync directory" is what guarantees that the rename only ever exposes fully-written data.

After this protocol, a crash at *any* point leaves the filesystem in exactly one of two valid states:

```
crash before rename:  save.dat = old (intact),  save.tmp = garbage (discard)
crash after  rename:  save.dat = new (intact),  save.bak = old   (recoverable)
```

### Integrity checking

A checksum in the header (CRC32 for speed, xxHash for a better speed/collision tradeoff, or a cryptographic hash if tamper-resistance matters) lets you detect corruption *before* feeding bad data into the game. On load:

```
1. Read header, verify magic bytes.            → reject foreign files
2. Recompute checksum over payload, compare.   → detect bit-rot / truncation
3. On mismatch: fall back to .bak, then to an  → graceful degradation,
   older rotating autosave slot.                 never a hard crash
```

The combination of a checksum, a `.bak` backup from the atomic rename, and rotating autosave slots gives a layered recovery story: a single corrupted file almost never means lost progress.

### Defending against tampering

For competitive or monetized games, players will edit local saves to grant items or currency. You cannot make a client-side save truly tamper-proof (the player owns the machine), but you can raise the cost:

- **Keyed hash (HMAC):** store an HMAC of the payload using a key embedded in the binary. A naive editor that changes a value but not the HMAC is rejected. (Determined attackers extract the key, so this only stops casual cheating.)
- **Server authority:** for anything that matters competitively, the *server* holds the authoritative state and the local save is just a cache. This is the only robust answer. See [Cybersecurity Basics](../technology/cybersecurity/) for the principle that the client is never trusted.
- **Encryption is not integrity:** encrypting a save hides its contents but does not prevent a player from restoring an older, legitimately-encrypted save (a "rollback" to before they spent a resource). Pair encryption with the generation counter to detect rollbacks.

## Worked Example: End-to-End Save Flow

Tying the pieces together, here is the full lifecycle of one autosave, showing where each technique applies:

```
1. Trigger fires (event-driven: boss defeated, in a safe state)
        │
2. SaveManager.BuildSnapshot()           ── architecture: capture, no I/O
        │   → immutable deep copy of all persistent sections
        │   → stamps version = CurrentVersion, generation++, timestamp
        │
3. Hand snapshot to worker thread        ── autosave: non-blocking
        │
4. Encode (MessagePack) + compress       ── format: compact, schema-evolving
        │
5. Prepend header (magic, version,        ── format/integrity
        │   checksum, generation, timestamp)
        │
6. AtomicWrite to autosave_2.tmp          ── anti-corruption
        │   → fsync, rename over autosave_2.dat, old → .bak
        │
7. Rotate ring (newest = slot 2)          ── autosave: rotation
        │
8. Upload to cloud with generation        ── cloud: optimistic concurrency
            (cloud rejects if its gen advanced → re-sync)
```

On the next launch, the load path runs in reverse: read header, verify magic and checksum, fall back to `.bak`/older slot on failure, decode the payload, run the migration chain `version → CurrentVersion`, then `ApplySnapshot` restores each system, instantiating entities and relinking ID references in two passes.

## Key Takeaways

<div class="takeaway-grid">
  <div class="takeaway-card"><h4>Classify your state</h4><p>Persist the authoritative facts, recompute derived data on load, discard transient state. Serializing the whole scene graph is the original sin.</p></div>
  <div class="takeaway-card"><h4>Separate save model from runtime model</h4><p>Capture into plain data objects, restore from them. Serialize references as stable IDs and relink in a two-pass load. Never serialize pointers or engine handles.</p></div>
  <div class="takeaway-card"><h4>Version and migrate, additively</h4><p>Every save carries a version. Write migrations once, chain them v→v+1, and never reuse field IDs. This is a launch-day requirement, not polish.</p></div>
  <div class="takeaway-card"><h4>Autosave off the hot path</h4><p>Capture an immutable snapshot on the game thread, encode and write on a worker. Rotate slots so one bad write never loses everything.</p></div>
  <div class="takeaway-card"><h4>Cloud saves need conflict resolution</h4><p>Generation counters plus optimistic concurrency catch stale writes; prompt the player on true conflicts and keep sync advisory so offline play still works.</p></div>
  <div class="takeaway-card"><h4>Write atomically, verify on load</h4><p>Temp-file, fsync, rename — a crash leaves old or new, never partial. A checksum plus a .bak backup turns corruption into recoverable degradation.</p></div>
</div>

## See Also

- **[Game Development Hub](./)** — section overview: engines, the game loop, ECS, and the systems a save file must capture
- **[Database Design](../technology/database-design/)** — transactions, concurrency control, and the durability guarantees a save system borrows
- **[Cybersecurity Basics](../technology/cybersecurity/)** — why the client is never trusted, and how tamper-resistance and server authority work
- **[Networking Fundamentals](../technology/networking/)** — the transport layer beneath cloud-save synchronization
- **[Memory Optimization](../optimization/memory-optimization.html)** — asset streaming and the serialization-adjacent cost of large in-memory state
- **[Game AI](../ai-ml/game-ai.html)** — behavior and world state that frequently needs to be persisted across sessions

---

*This page is part of the Game Development guide. For suggestions or contributions, visit our [GitHub repository](https://github.com/AndrewAltimit/Documentation).*
