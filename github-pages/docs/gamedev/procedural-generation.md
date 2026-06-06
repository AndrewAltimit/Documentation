---
layout: docs
title: "Game Dev: Procedural Content Generation"
permalink: /docs/gamedev/procedural-generation.html
toc: true
toc_sticky: true
hide_title: true
---

# Procedural Content Generation

[Game Development](./) &raquo; Procedural Content Generation

Procedural Content Generation (PCG) is the algorithmic creation of game content — terrain, dungeons, textures, vegetation, items, even quests — rather than authoring it by hand. PCG trades authored control for variety, replayability, and scale: a few kilobytes of code and a seed can expand into a planet's worth of terrain. The craft is choosing algorithms whose *statistical* output is controllable enough to be fun while remaining unpredictable enough to be interesting.

**Read this page as a pipeline.** Most generators are a base signal (noise) shaped by *rules* (automata, grammars, constraints), then *interpreted* into playable space, all driven by a *seed* that makes the whole chain reproducible:

- **Noise & fractals.** Coherent noise (Perlin, Simplex, value) and fractal summation are the raw material for natural-looking terrain and textures.
- **Generative systems.** Cellular automata, L-systems, and wave-function collapse turn local rules into coherent global structure.
- **Determinism.** Seeded PRNGs make generation reproducible — the same seed always yields the same world.

## Why Generate Content?

Hand-authored content is precise but expensive: every cave, tree, and texture costs artist and designer time. Procedural generation shifts that cost into *code that runs once per playthrough*.

| Goal | What PCG buys you | Examples |
|------|-------------------|----------|
| Replayability | Fresh layouts each run | *Spelunky*, *Hades*, roguelikes |
| Scale | Vastly more content than could be authored | *Minecraft*, *No Man's Sky*, *Elite: Dangerous* |
| Compression | Tiny data expands into huge worlds | *.kkrieger* (96 KB 3D shooter), demoscene |
| Variety | Non-repeating textures, foliage, crowds | SpeedTree, substance materials |
| Adaptivity | Content tuned to the player at runtime | dynamic difficulty, mission generators |

The central tension — explored throughout this page — is **control vs. randomness**. Pure randomness is uniform noise: technically infinite variety, zero meaning. Pure authoring is a fixed level. PCG lives in between, using structured algorithms whose randomness is *shaped* by constraints.

## Seeding and Determinism

Before any algorithm, decide your source of randomness. A **pseudo-random number generator (PRNG)** maps an integer *seed* deterministically to a stream of numbers. The same seed always reproduces the same stream — this is what makes a *seed code* sharable between players and what makes bugs reproducible.

### Why `rand()` Is Not Enough

The C standard `rand()` and many language defaults are poor for PCG: short periods, weak low-bit randomness, and global hidden state shared with the rest of the program. For generation you want an explicit, fast, well-distributed generator. A common choice is a small **counter-based / hash PRNG** such as **PCG** (Permuted Congruential Generator), **xorshift**, or **SplitMix64**.

The core of SplitMix64 — fast, stateless enough to splat across threads, and good distribution:

```python
def splitmix64(state):
    """One step of SplitMix64. Returns (random_u64, new_state)."""
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z, state
```

### Hierarchical Seeding

Large worlds need *positional determinism*: chunk (10, 7) must generate identically regardless of which chunks were visited first or in what order. The standard trick is to derive a local seed by **hashing the world seed together with coordinates**, so each chunk/feature gets its own independent stream:

```python
def chunk_seed(world_seed, cx, cy):
    h = world_seed
    h = hash_combine(h, cx)
    h = hash_combine(h, cy)
    return h

def hash_combine(h, v):
    # 64-bit mix (boost-style)
    h ^= (v + 0x9E3779B97F4A7C15 + ((h << 6) & 0xFFFFFFFFFFFFFFFF) + (h >> 2))
    return h & 0xFFFFFFFFFFFFFFFF
```

This gives **order-independent, position-addressable** randomness — the foundation of infinite open worlds. A feature at a location can be decided by hashing its coordinates without generating its neighbors first.

### Determinism Pitfalls

- **Floating-point non-determinism.** Results can differ across CPUs/compilers (x87 vs SSE, FMA contraction, `-ffast-math`). For lockstep multiplayer or shared seeds, prefer integer/fixed-point math or pin floating-point behavior.
- **Iteration order.** Iterating a hash map/set in nondeterministic order will desync generation. Sort or use ordered containers.
- **Consuming the stream conditionally.** If a branch consumes random numbers only sometimes, two runs diverge. Draw all randomness up front, or per-feature with hierarchical seeds.

## Noise Functions

**Coherent noise** is the workhorse of PCG: a function `noise(x, y, ...)` that is *random-looking but smooth* — nearby inputs give nearby outputs. This smoothness is what makes it usable for terrain (no jagged single-pixel spikes) while still being unpredictable.

### Value Noise

The simplest coherent noise. Assign a random value to each integer lattice point, then **interpolate** between them for fractional coordinates.

```python
def value_noise_1d(x, seed):
    x0 = floor(x)
    x1 = x0 + 1
    t = x - x0
    v0 = hash_to_unit(x0, seed)   # random value in [0, 1]
    v1 = hash_to_unit(x1, seed)
    return lerp(v0, v1, smoothstep(t))
```

The interpolation curve matters. Linear interpolation produces visible creases at lattice lines because the *derivative* is discontinuous. Use a smoothstep (Hermite) fade so the slope is continuous:

$$
s(t) = 3t^2 - 2t^3
$$

Ken Perlin's improved "quintic" fade goes further, making the *second* derivative continuous too (important for lighting/normals derived from the height):

$$
s(t) = 6t^5 - 15t^4 + 10t^3
$$

Value noise is cheap but tends to look "blobby" and shows lattice-aligned artifacts.

### Perlin Noise (Gradient Noise)

Perlin noise (Ken Perlin, 1985; improved 2002) stores a random *gradient vector* (not a value) at each lattice point. The contribution of a corner is the dot product of its gradient with the offset vector from the corner to the sample point, and these are interpolated with the quintic fade.

For a 2D sample at point **p** inside a unit cell with corner **c** holding gradient **g(c)**:

$$
\text{contribution}(c) = g(c) \cdot (p - c)
$$

The four corner contributions are blended with the fade function in x and y:

$$
n(p) = \text{lerp}\big(\text{lerp}(n_{00}, n_{10}, s(u)),\ \text{lerp}(n_{01}, n_{11}, s(u)),\ s(v)\big)
$$

where $u, v$ are the fractional coordinates within the cell and $s$ is the quintic fade. Because the value at every lattice point is exactly zero (the offset vector is zero there), Perlin noise has no value-noise blobbiness, but it has a subtle **axis-aligned directional bias** — features line up with the grid.

```python
def perlin2d(x, y, perm, grad):
    xi, yi = floor(x) & 255, floor(y) & 255
    xf, yf = x - floor(x), y - floor(y)
    u, v = fade(xf), fade(yf)          # quintic fade

    # Hash the four corners into gradient indices
    aa = perm[perm[xi]     + yi]
    ab = perm[perm[xi]     + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    # Dot each corner gradient with the offset to the sample point
    n00 = dot(grad[aa], (xf,     yf))
    n10 = dot(grad[ba], (xf - 1, yf))
    n01 = dot(grad[ab], (xf,     yf - 1))
    n11 = dot(grad[bb], (xf - 1, yf - 1))

    return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
```

### Simplex Noise

Simplex noise (Perlin, 2001) was designed to fix Perlin's weaknesses. Instead of a hypercube grid it tessellates space into **simplices** (triangles in 2D, tetrahedra in 3D) — the simplest shapes that fill *n*-dimensional space. In *n* dimensions a hypercube has $2^n$ corners but a simplex has only $n + 1$, so the cost scales much better:

$$
\text{corners}_{\text{Perlin}} = 2^n
\qquad
\text{corners}_{\text{Simplex}} = n + 1
$$

In 3D that is 8 corners vs. 4; in 4D, 16 vs. 5. Simplex also uses a **radial summation** with a smooth falloff per corner rather than separable interpolation, which removes Perlin's axis-aligned bias and gives continuous gradients everywhere — better for normal mapping and animation. The summed contribution of the surrounding simplex corners is:

$$
n(p) = \sum_{i} \max\!\big(0,\ r^2 - |d_i|^2\big)^4 \,\big(g_i \cdot d_i\big)
$$

where $d_i$ is the offset from corner $i$ to the sample, $g_i$ is that corner's gradient, and $r^2$ is the squared radius of the kernel (commonly $0.5$). The fourth power gives a smooth falloff to zero at the kernel edge.

> **Patent note.** Simplex noise in 3D+ was patented (now expired in most jurisdictions). To avoid any ambiguity many engines ship **OpenSimplex** / **OpenSimplex2**, which are unencumbered and have similar properties.

### Noise Comparison

| Property | Value | Perlin | Simplex |
|----------|-------|--------|---------|
| Cost per sample | Lowest | Medium ($2^n$ corners) | Lower in high-D ($n{+}1$ corners) |
| Visual quality | Blobby, lattice artifacts | Good, slight axis bias | Best, isotropic |
| Gradient continuity | Depends on fade | Continuous (quintic) | Continuous everywhere |
| Implementation | Trivial | Simple | More involved |
| Best use | Quick/cheap effects | General terrain, 2D | High-D, animated, normals |

### Worley (Cellular) Noise

A different family: scatter feature points, then for each sample compute the distance to the nearest point(s). Using the distance to the nearest point $F_1$ gives a "cellular" / Voronoi look; $F_2 - F_1$ gives crack/web patterns. Worley noise is ideal for stone, water caustics, scales, and biome/region partitioning.

$$
W(p) = \min_{i} \, \lVert p - q_i \rVert
$$

where the $q_i$ are the scattered feature points (typically one per cell of a jittered grid, searched over the 3x3 neighborhood).

## Fractals: Stacking Noise into Terrain

A single octave of noise has one characteristic feature size. Real terrain has detail at *every* scale — mountain ranges, hills, boulders, gravel. **Fractal Brownian motion (fBm)** sums multiple **octaves** of the same noise, each at higher frequency and lower amplitude:

$$
f(p) = \sum_{i=0}^{N-1} a^{\,i}\, \text{noise}\!\big(\,l^{\,i}\, p\big)
$$

Two parameters control the look:

- **Lacunarity** $l$ — how much the *frequency* multiplies per octave (typically 2.0). Higher lacunarity spreads detail across more distinct scales.
- **Gain / persistence** $a$ — how much the *amplitude* multiplies per octave (typically 0.5). Lower gain makes higher octaves fainter (smoother terrain); higher gain makes them rougher.

```python
def fbm(x, y, octaves=6, lacunarity=2.0, gain=0.5):
    amplitude = 1.0
    frequency = 1.0
    total = 0.0
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * noise(x * frequency, y * frequency)
        norm  += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm    # normalize to keep range stable
```

Variations give characteristic landforms:

- **Ridged multifractal** — apply $1 - |\,\text{noise}\,|$ per octave to turn smooth hills into sharp ridges and canyons. Great for mountain ranges.
- **Billowy / turbulence** — use $|\,\text{noise}\,|$, producing puffy, cloud-like or rolling-dune shapes.
- **Domain warping** — feed the *output* of one noise field into the *input coordinates* of another: `fbm(x + fbm(x,y), y + fbm(x,y))`. This warps the field along itself, producing organic, flowing, marbled structure that pure fBm cannot.

The expected variance of fBm is geometric in the gain, so the sum stays bounded as long as $a < 1$:

$$
\sum_{i=0}^{\infty} a^{\,i} = \frac{1}{1 - a}, \qquad 0 < a < 1
$$

## Cellular Automata

A **cellular automaton** (CA) is a grid of cells whose next state is a function of its current state and its neighbors' states. Despite trivially simple local rules, CAs produce rich global structure — making them excellent, cheap shape generators for caves and organic regions.

### Cave Generation

The classic recipe: randomly fill the grid with wall/floor, then run a **smoothing** rule for a few iterations. The rule is essentially "become a wall if surrounded by walls" — a majority vote over the 8 neighbors (a *Moore* neighborhood):

```python
def generate_cave(width, height, fill_prob, iterations, rng):
    # 1. Random fill
    grid = [[1 if rng.random() < fill_prob else 0
             for _ in range(width)] for _ in range(height)]

    # 2. Smoothing iterations
    for _ in range(iterations):
        new = [[0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                walls = count_wall_neighbors(grid, x, y)  # 8-neighborhood
                if walls > 4:        new[y][x] = 1   # majority wall -> wall
                elif walls < 4:      new[y][x] = 0   # majority floor -> floor
                else:                new[y][x] = grid[y][x]  # tie -> hold
        grid = new
    return grid
```

A common choice is `fill_prob ~ 0.45` and 4-5 iterations. Treat the border as wall so caves are enclosed.

**Post-processing is essential.** Raw CA output often has disconnected pockets. Run a **flood fill** to find connected regions, keep the largest as the main cave (or carve corridors to connect smaller ones), and fill the rest. This guarantees a *traversable* level rather than just a pretty texture.

### Conway's Game of Life

The famous CA with the rule set: a live cell survives with 2 or 3 live neighbors; a dead cell becomes alive with exactly 3. While not a level generator itself, it demonstrates that simple rules can be Turing-complete and is a useful mental model for CA-based generation. The general CA update is:

$$
c_{t+1}(x) = R\big(c_t(x),\ \textstyle\sum_{j \in N(x)} c_t(j)\big)
$$

where $N(x)$ is the neighborhood and $R$ the transition rule.

## L-Systems

**Lindenmayer systems** (L-systems) are formal grammars for modeling growth — originally plant and algae development. You start with an **axiom** (initial string) and repeatedly apply **production rules** that rewrite symbols in parallel. The resulting string is then interpreted as **turtle graphics** commands to draw a structure.

### Anatomy

- **Alphabet** — the symbols, e.g. `F` (draw forward), `+`/`-` (turn), `[`/`]` (push/pop turtle state).
- **Axiom** — the starting string.
- **Productions** — rewrite rules applied simultaneously each generation.

A classic plant grammar:

```
Axiom:      X
Rules:      X -> F+[[X]-X]-F[-FX]+X
            F -> FF
Angle:      25 degrees

Turtle interpretation:
  F  -> move forward, drawing a segment
  +  -> turn left by angle
  -  -> turn right by angle
  [  -> push position + heading onto a stack
  ]  -> pop position + heading from the stack
```

Each `[ ... ]` pair is a branch: push state, draw a sub-branch, pop back to continue the trunk. After a few rewrite generations the string encodes a full bush; rendering the turtle path draws it.

```python
def expand(axiom, rules, generations):
    s = axiom
    for _ in range(generations):
        s = "".join(rules.get(ch, ch) for ch in s)  # parallel rewrite
    return s
```

### Variants

- **Stochastic L-systems** — a symbol has several productions chosen by probability, so every plant is unique (seed the RNG for determinism).
- **Context-sensitive** — a rule fires only given specific neighbors, modeling signal propagation along the plant.
- **Parametric** — symbols carry numeric parameters (length, age), enabling continuous growth and realistic tapering.

L-systems excel at **self-similar branching structures**: plants, trees, river networks, lightning, road/cave networks, and even some building/street layouts.

## Wave Function Collapse

**Wave Function Collapse** (WFC, Maxim Gumin, 2016) is a constraint-solving generator that produces output *locally similar* to an example. It is borrowed-by-analogy from quantum mechanics: every cell starts in a *superposition* of all possible tiles, and the algorithm repeatedly **collapses** the most-constrained cell to a single tile, then **propagates** that decision to its neighbors.

### The Algorithm

1. **Define tiles and adjacency.** Either author tiles with edge-compatibility rules, or extract the rules automatically from an example bitmap (which N×N patterns appear, and which may sit next to which).
2. **Initialize.** Every cell's domain = all tiles (full superposition).
3. **Observe (collapse).** Pick the cell with the **lowest entropy** (fewest remaining options — the most constrained) and collapse it to a single tile, chosen by weighted random from its domain.
4. **Propagate (constraint propagation).** Remove from neighboring cells any tile that is now incompatible with the collapsed choice; cascade these removals outward until the grid is consistent.
5. **Repeat** until all cells are collapsed, or **backtrack/restart** on a contradiction (a cell whose domain became empty).

The entropy used for the "most constrained" heuristic is the Shannon entropy of the cell's remaining tile probabilities (with the tile weights $w_t$ over the remaining domain $D$):

$$
H = -\sum_{t \in D} p_t \log p_t,
\qquad
p_t = \frac{w_t}{\sum_{u \in D} w_u}
$$

Collapsing low-entropy cells first minimizes the chance of painting yourself into a contradiction.

```python
def wfc_step(grid):
    cell = min_entropy_cell(grid)        # most constrained, undecided cell
    if cell is None:
        return DONE                      # everything collapsed
    tile = weighted_choice(cell.domain)  # observe
    cell.collapse_to(tile)
    if not propagate(grid, cell):        # remove now-illegal neighbor options
        return CONTRADICTION             # backtrack or restart
    return CONTINUE
```

### Strengths and Costs

WFC is superb for **tile-based** content with strong local constraints: dungeon rooms, pipe/circuit layouts, texture synthesis, and stylized levels that must "look hand-made." Its weaknesses:

- **No global guarantees.** WFC enforces *local* adjacency only. It will happily produce a level that is locally valid but globally unsolvable (no path to the exit). You typically add post-checks or extra constraints (pre-place start/exit, verify connectivity).
- **Contradictions.** Dense constraints can make the solver fail and restart. Mitigations: backtracking, careful tile/weight design, or constraint relaxation.
- **Cost.** Propagation can be expensive on large grids; chunked or hierarchical WFC helps.

## Procedural Terrain

Terrain ties the pieces together: a **heightmap** $h(x, y)$ generated by fBm noise, then shaped and interpreted.

### From Noise to Landscape

```python
def terrain_height(x, y, seed):
    # Continent shape: low-frequency, large amplitude
    base = fbm(x * 0.001, y * 0.001, octaves=4) * 1000

    # Mountains: ridged noise, masked to the high-base regions
    mountains = ridged_fbm(x * 0.005, y * 0.005, octaves=6) * 800
    mask = smoothstep(0.4, 0.7, normalize(base))
    base += mountains * mask

    # Erosion / detail
    base += fbm(x * 0.05, y * 0.05, octaves=3) * 20
    return base
```

Common shaping steps:

- **Continent/ocean mask** — a separate low-frequency field decides land vs. sea; multiply or threshold the height by it.
- **Falloff map** — for islands, subtract a radial gradient so edges drop into the ocean.
- **Redistribution** — raising the normalized height to a power $h^k$ flattens valleys and sharpens peaks: $h' = h^k$.
- **Hydraulic / thermal erosion** — simulate water/sediment flow to carve realistic valleys and deposit sediment; the single biggest quality win for noise terrain.

### Biomes

Biomes are usually a 2D lookup on two more noise fields, **temperature** and **moisture** (the Whittaker-diagram approach):

```python
def biome(temperature, moisture):
    if moisture < 0.2:  return DESERT if temperature > 0.6 else TUNDRA
    if temperature > 0.7: return RAINFOREST if moisture > 0.6 else SAVANNA
    if temperature > 0.4: return FOREST if moisture > 0.4 else GRASSLAND
    return TAIGA if moisture > 0.4 else SNOW
```

### Chunking for Infinite Worlds

Open worlds generate terrain in **chunks** on demand, seeded by chunk coordinates (see [Hierarchical Seeding](#hierarchical-seeding)). Because the height function is *positional and deterministic*, chunk boundaries match seamlessly — neighboring chunks evaluate the same continuous field, so no edge-stitching is required.

## Procedural Dungeons

Dungeon generation is a distinct discipline because the output must be **playable**: connected, fair, and paced. Several families dominate.

### Room-and-Corridor

Place rectangular rooms (rejecting overlaps), then connect them with corridors. Connection strategy controls the feel:

- **Random connections** — chaotic, mazey.
- **Minimum spanning tree (MST)** — guarantees full connectivity with no redundancy; add a few extra edges back for loops so the dungeon isn't a pure tree (loops make exploration less frustrating).
- **Delaunay triangulation + MST** — triangulate room centers, take the MST for guaranteed connectivity, then re-add a fraction of the remaining Delaunay edges for loops. This is the *Spelunky / TinyKeep* approach and gives natural-feeling layouts.

```python
def generate_dungeon(rng):
    rooms = place_non_overlapping_rooms(rng)
    graph = delaunay(centers(rooms))         # candidate connections
    corridors = mst(graph)                   # guaranteed connectivity
    corridors += sample(graph - corridors, fraction=0.15, rng)  # add loops
    carve(rooms, corridors)
    return rooms, corridors
```

### BSP Trees

**Binary Space Partitioning** recursively splits the dungeon rectangle into sub-rectangles, placing one room per leaf and connecting sibling rooms back up the tree. The recursion guarantees connectivity and gives tunable room density. This is a classic *Rogue*-style approach.

```
Split the map -> two halves -> recurse until cells are small enough
Each leaf -> place a room inside it
Each internal node -> connect the rooms/regions of its two children
```

### Drunkard's Walk and Cellular Automata

For **organic** caves rather than rectilinear rooms, carve a path with a random walk ("drunkard's walk") until a target floor percentage is reached, or use the [cellular-automata cave](#cave-generation) method above. Both produce natural, non-grid layouts; combine with connectivity post-processing.

### Grammar / Mission Graphs

The most *design-controllable* approach separates **mission** (the logical sequence — get key, fight boss, find treasure) from **space** (the physical layout). A graph grammar generates the mission graph; a separate step embeds it into geometry. This is how generators guarantee **lock-and-key** progression and pacing rather than leaving them to chance (*Unexplored* is the canonical example).

## Control vs. Randomness

The recurring theme of PCG is steering a random process toward *good* content. The techniques:

| Technique | What it does |
|-----------|--------------|
| Seeded determinism | Reproducibility; sharable/debuggable worlds |
| Constraints | Forbid invalid output (WFC adjacency, no-overlap, solvability checks) |
| Weighting | Bias choices toward desirable outcomes (rare loot weights, biome frequency) |
| Authored set-pieces | Hand-made rooms/vistas dropped into procedural space ("set-piece injection") |
| Grammars / templates | Guarantee structure and pacing (lock-and-key, mission graphs) |
| Generate-and-test | Produce many candidates, score them, keep the best |

### Generate-and-Test

When direct control is hard, generate many candidates and **evaluate** them against metrics (path length, difficulty, symmetry, reachability), keeping or evolving the best. The fitness function encodes "what good looks like":

$$
\text{fitness}(L) = \sum_{k} w_k \, m_k(L)
$$

where each $m_k$ is a metric (e.g. solvability, branching factor, item spread) and $w_k$ its designer weight. **Search-based PCG** pushes this further with evolutionary algorithms or constraint solvers (ASP / answer-set programming) that generate *only* content satisfying hard constraints while optimizing soft ones.

### Designer-Facing Controls

In production, PCG is rarely fully autonomous. Designers expose **knobs** — density sliders, biome weights, difficulty curves, "spice" parameters — and the generator maps them to its internal parameters. The goal is a generator a *designer* can direct, not a black box. A useful rule of thumb:

> Make the generator produce *plausible* content by default, then give designers controls to bias it. Validate hard requirements (solvability, fairness) with post-checks; tune soft qualities (variety, pacing) with weights and metrics.

## Key Takeaways

- **Seed everything.** A PRNG seeded from a world seed plus hashed coordinates gives order-independent, reproducible, position-addressable worlds — the basis of infinite open worlds and sharable seed codes.
- **Noise is the raw material.** Value noise is cheapest; Perlin adds gradient quality; Simplex/OpenSimplex scale to higher dimensions without axis bias. Stack octaves with fBm (lacunarity + gain) to get multi-scale terrain.
- **Local rules, global structure.** Cellular automata grow organic caves, L-systems grow branching plants, and Wave Function Collapse solves local adjacency constraints into coherent tile maps.
- **Playability is a separate problem.** Dungeons need connectivity (MST + loops, BSP) and pacing (mission grammars). Always post-process for reachability — pretty is not the same as solvable.
- **PCG is control over randomness,** not the absence of it: constraints, weighting, grammars, set-piece injection, and generate-and-test steer a random process toward content that is both varied and fun.

## See Also

- [Game Development](./) - Engines, core systems, and design principles
- [Game AI Systems](../ai-ml/game-ai.html) - Pathfinding and decision-making over generated worlds
- [3D Graphics &amp; Rendering](../graphics/3d-rendering.html) - Rendering the meshes and textures PCG produces
- [Performance Optimization](../optimization/) - Keeping generation within frame and memory budgets
- [Unreal Engine](../technology/unreal.html) - PCG framework, World Partition, and procedural tooling
