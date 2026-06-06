---
layout: docs
title: "Optimization: Platform-Specific Tuning"
permalink: /docs/optimization/platform-tuning.html
toc: true
toc_sticky: true
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Platform-Specific Tuning</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Mobile thermals, console fixed-hardware tuning, PC scalability presets, and compiler PGO</p>
</div>

[Performance Optimization](./) &raquo; Platform-Specific Tuning

<div class="code-example" markdown="1">
A frame-time budget is not a single number — it is three different numbers on three different classes of hardware. A 16.67 ms/frame target that is comfortable on a desktop GPU is a thermal-throttle death sentence on a phone, a known-quantity engineering problem on a console, and a scalability matrix on PC. This page covers how to tune for each: managing **power and thermal** envelopes on mobile, exploiting **fixed hardware** on consoles, exposing **scalable quality presets** on PC, and squeezing the last few percent out of every platform with **compiler optimizations and profile-guided optimization (PGO)**.
</div>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Why Platform Matters

The same engine, the same scene, and the same render path behave radically differently across hardware classes because the binding constraint is different in each case:

| Platform | Binding constraint | Optimization posture |
|----------|--------------------|----------------------|
| **Mobile** | Energy and heat (sustained power budget) | Throttle-aware; minimize joules per frame |
| **Console** | Fixed, known silicon | Optimize to the exact spec; no driver variance |
| **PC** | Unknown, wildly variable hardware | Scale gracefully across a 50x performance range |

A useful mental model: on a desktop you optimize for the *instantaneous* frame, on a phone you optimize for the *thousandth consecutive* frame (after the SoC has heated up), on a console you optimize for the *exact* frame the hardware will always produce, and on PC you optimize for the *worst plausible* frame your customer will run.

---

## Mobile Optimization

Mobile SoCs (smartphones, tablets, handhelds, standalone VR) share a defining characteristic: there is no fan, or only a small one, so **all dissipated power eventually becomes heat the chassis cannot shed**. The hardware will run flat-out for a few seconds, then the on-die thermal governor will quietly reduce clocks to protect the silicon and the user's hand. Mobile optimization is therefore fundamentally about *sustained* performance, not peak.

### Power and Thermal

The original guidance is simple and remains the core of the discipline:

- **Reduce GPU load to prevent throttling** — the GPU is usually the dominant power consumer in a graphics workload.
- **Target 30 FPS for better battery life** when smoothness allows; halving the frame rate roughly halves render-related energy draw.
- **Minimize background processing** — wake locks, polling loops, and unbatched network calls drain the battery even when nothing is on screen.
- **Use platform power management APIs** to react to the OS's own thermal and battery signals rather than guessing.

#### The thermal curve

A mobile device does not run at a constant performance level. Its behaviour over a long session follows a predictable curve:

```
Performance
    ^
    |  ____ boost clocks (first ~30-90 s)
    | /    \
    |/      \___ throttle step 1
    |           \____ throttle step 2 (sustained)
    |                \________________________
    +------------------------------------------> time
       "burst"        "thermal cliff"   "sustained floor"
```

Benchmarking only the first few seconds (the "burst" region) produces numbers your players will never see. **Always measure sustained performance**: run the workload for 10–20 minutes with the device thermally soaked, ideally in a warm room and inside its real case.

#### Energy is power x time

The quantity that actually drains the battery and heats the chassis is energy, not power. The work done per frame, integrated over the session, is:

$$ E = \int_0^T P(t)\, dt \approx \sum_{i=1}^{N} P_i \, \Delta t_i $$

Two corollaries fall directly out of this:

- **Race to idle.** A CPU core that finishes its work quickly and returns to a deep sleep state often uses *less* total energy than one that dawdles at a low clock, because static leakage and the fixed cost of being awake dominate. Bursty, then-asleep beats slow-but-steady on most mobile cores.
- **Frame pacing beats frame count.** Rendering at a steady, capped 30 FPS draws far less energy than rendering an uncapped 45 FPS that the display cannot even show. Always cap to the display refresh (or a clean divisor of it) so you do not burn joules on frames that are discarded.

#### Frame rate vs. battery, quantified

A rough first-order model: if rendering one frame costs a fixed energy $e$ and the device draws a baseline idle power $P_0$, then the battery life $L$ for a battery of capacity $Q$ at frame rate $f$ is:

$$ L = \frac{Q}{P_0 + e \cdot f} $$

Dropping from 60 FPS to 30 FPS does not double battery life (the constant $P_0$ — screen backlight, radio, OS — does not change), but for GPU-heavy games where $e \cdot f \gg P_0$ it gets close. This is why a "30 FPS battery saver" mode is the single highest-leverage power option a mobile title can ship.

#### Tile-based deferred rendering (TBDR)

Most mobile GPUs (Apple, ARM Mali, Qualcomm Adreno, Imagination PowerVR) are **tile-based deferred renderers**. The screen is split into small on-chip tiles; all geometry for a tile is binned, then shaded entirely in fast on-chip tile memory before being written out to main memory once. This architecture rewards a very different style of rendering than a desktop immediate-mode GPU:

- **Avoid mid-frame framebuffer reads.** Reading back a render target you just wrote forces the tile to be flushed to main memory and reloaded — the single most expensive mistake on mobile. Structure post-processing to stay within one render pass where possible (subpasses in Vulkan, programmable blending on Apple).
- **Use `LOAD`/`STORE`/`DONT_CARE` correctly.** Tell the driver when a render target's previous contents are not needed (`DONT_CARE` on load) and when its results need not be kept (`DONT_CARE` on store). A depth buffer almost never needs to be stored to main memory; marking it `DONT_CARE` on store saves a large chunk of bandwidth, which is energy.
- **MSAA is cheap, bandwidth is not.** Because resolve happens in tile memory, MSAA is far cheaper on TBDR hardware than on desktop. Conversely, large G-buffers (deferred shading) are *more* expensive because the fat per-pixel data must spill out of tile memory. Forward / forward+ rendering is usually the better fit on mobile.

#### Reacting to thermal state in code

Modern mobile OSes expose the current thermal pressure so the app can shed load *before* the OS forces a hard throttle. The pattern is to map a thermal state to a quality tier and adjust dynamically:

```cpp
// Pseudocode: cross-platform thermal-aware quality governor.
// On Android, read PowerManager.getCurrentThermalStatus();
// on iOS, observe ProcessInfo.thermalState.

enum class ThermalState { Nominal, Fair, Serious, Critical };

void update_quality_for_thermal(ThermalState state, RenderSettings& s) {
    switch (state) {
        case ThermalState::Nominal:
            s.target_fps      = 60;
            s.render_scale    = 1.0f;
            s.shadow_quality  = ShadowQuality::High;
            break;
        case ThermalState::Fair:
            s.target_fps      = 60;
            s.render_scale    = 0.85f;   // dynamic-resolution cushion
            s.shadow_quality  = ShadowQuality::Medium;
            break;
        case ThermalState::Serious:
            s.target_fps      = 30;      // halve render energy
            s.render_scale    = 0.7f;
            s.shadow_quality  = ShadowQuality::Low;
            break;
        case ThermalState::Critical:
            s.target_fps      = 30;
            s.render_scale    = 0.6f;
            s.shadow_quality  = ShadowQuality::Off;
            s.post_effects    = false;   // strip the most expensive passes
            break;
    }
}
```

The goal is graceful, *gradual* degradation: small render-scale and shadow concessions long before the OS would otherwise impose a sudden, jarring 50% clock cut.

### Memory Constraints

Mobile memory is shared between the OS, the GPU, and every backgrounded app, and the OS will kill your process without warning if it exceeds its share. The original guidance applies in full:

- **Aggressive texture compression** — use ASTC (variable 1–8 bpp) on modern hardware, ETC2 (4–8 bpp) as a baseline. Never ship uncompressed RGBA8 textures to a phone.
- **Stream assets from storage** rather than loading the whole world resident.
- **Unload unused assets quickly** — return memory the moment a level or screen is left.
- **Monitor memory warnings** — both Android (`onTrimMemory`) and iOS (`didReceiveMemoryWarning`) signal pressure; treat these as a hard instruction to free caches immediately, not advice.

A practical budget on a mid-tier device is to keep total resident memory well under ~50–60% of the device RAM, because the OS, the compositor, and the GPU framebuffer claim a large fixed slice you never see. Profile on the *cheapest* device you intend to support, not the flagship.

---

## Console Optimization

Consoles are the opposite of mobile's uncertainty: a console is a single, frozen hardware specification that every unit shares for the lifetime of the generation. This is an enormous gift to the optimizer.

### Fixed Hardware Benefits

- **Known performance characteristics** — you know the exact CPU core count and clock, the exact GPU compute-unit count, the memory bandwidth, and the storage throughput. There is no "minimum spec" to guess at.
- **Can optimize to exact specs** — a constant you tune for one console is correct for *every* unit of that console, forever.
- **No driver variation** — the graphics driver is fixed and ships with the OS; a shader that works today works identically on every machine.
- **Predictable memory budget** — you are given a precise, contractual amount of memory and can fill it to the brim without the safety margin a PC requires.

The single biggest consequence: **you can profile once and trust the result.** A frame that fits the budget on a dev kit fits the budget for every player. This lets console teams chase the last 1–2 ms of a frame with confidence that would be wasted effort on PC.

### Techniques

The original techniques remain the backbone:

- **SPU / compute-shader offloading** — move parallelizable CPU work (culling, particle simulation, skinning, decompression) onto the GPU's compute units or dedicated coprocessors, which fixed hardware lets you schedule precisely.
- **Platform-specific APIs** — consoles expose lower-level, lower-overhead graphics APIs (GNM/GNMX on PlayStation, the Xbox flavour of D3D12) than any cross-platform abstraction. Bypassing the portable layer recovers real CPU time.
- **Hardware-specific features** — dedicated decompression blocks, hardware ray-tracing units, and fixed-function tessellators can be used unconditionally because every unit has them.
- **Memory layout optimization** — with a known cache hierarchy and a unified memory architecture, you can hand-place data structures for the exact cache-line and bank layout of the silicon.

#### Unified memory and the bandwidth budget

Current-generation consoles use a **unified memory architecture (UMA)**: CPU and GPU share one physical pool, so there is no PCIe copy to move data from system to video memory — but they also *contend* for the same memory bus. The optimization problem shifts from "how do I get data to the GPU" to "how do I stay within the shared bandwidth budget".

You can reason about it directly. If the console offers $B$ bytes/second of memory bandwidth and you target $f$ frames/second, the bandwidth budget per frame is:

$$ B_{\text{frame}} = \frac{B}{f} $$

For a console with 448 GB/s of bandwidth targeting 60 FPS, that is roughly 7.46 GB of memory traffic per frame. Every G-buffer write, texture fetch, and framebuffer resolve draws against that number. Counting bytes per frame — not just GPU math — is often the real console optimization loop.

#### Fixed-spec constants

Because the hardware never changes, magic numbers that would be reckless on PC are correct and permanent on console:

```cpp
// On a fixed console spec these are not heuristics — they are facts.
constexpr int   kCpuCores          = 8;     // known core count
constexpr int   kReservedForOs     = 2;     // OS keeps these; game gets 6
constexpr int   kGameWorkerThreads = kCpuCores - kReservedForOs;
constexpr float kGpuClockGHz       = 2.23f; // exact, every unit
constexpr int   kCacheLineBytes    = 64;    // tune data layout to this

// Hand-place hot data to one cache line, knowing the exact line size.
struct alignas(kCacheLineBytes) HotEntityData {
    Vector3 position;
    Vector3 velocity;
    float   health;
    // ... packed to fill, not straddle, a 64-byte line
};
```

---

## PC Scalability

PC is the hardest target precisely because it is the *least* like a console: an integrated laptop GPU and a flagship desktop card can differ by more than 50x in throughput, and both run your binary. The strategy is not to pick one target but to **scale gracefully** across the whole range, exposing knobs the user (or an auto-detector) can set.

### Graphics Options

The original taxonomy of settings is the canonical menu:

```
Resolution: 720p to 4K+
Quality Presets: Low, Medium, High, Ultra
Individual Settings:
├── Texture Quality
├── Shadow Quality
├── Anti-Aliasing
├── Post-Processing
├── Draw Distance
├── LOD Bias
└── Effect Quality
```

#### Designing quality presets

The "Low / Medium / High / Ultra" presets are not arbitrary — each is a curated bundle of the individual settings, chosen so that each step up costs *roughly* proportional GPU time for the visual gain. Good preset design follows a few rules:

- **Order settings by cost-per-quality.** Spend the first quality budget on the settings with the highest visual return per millisecond (texture resolution, shadow resolution) before the diminishing-returns settings (ultra-distance shadows, screen-space reflections at full resolution).
- **Make Low actually run.** "Low" must hit playable frame rates on the *minimum* supported GPU, not merely look worse on a good one. Validate it on real low-end silicon.
- **Make Ultra a true headroom setting.** "Ultra" exists for high-end and future hardware; it is acceptable for it to be unattainable on mid-range cards today.
- **Detect and default sensibly.** Auto-detect the GPU on first launch and pick a preset, so the median player never sees a slideshow before they find the settings menu.

| Setting | Cheapest tier | Most expensive tier | Primary cost class |
|---------|---------------|---------------------|--------------------|
| Texture Quality | Lower-resolution mips | Full-resolution textures | Memory / bandwidth |
| Shadow Quality | Low-res, short-range maps | High-res cascaded maps | Fill rate + draw calls |
| Anti-Aliasing | None / FXAA | TAA / MSAA / DLAA | Fill rate / bandwidth |
| Draw Distance | Near cull plane | Far cull plane | Geometry + draw calls |
| LOD Bias | Aggressive (low detail) | Conservative (high detail) | Geometry throughput |
| Effect Quality | Half-res particles/SSR | Full-res, more samples | Fill rate / shader ALU |

### Dynamic Resolution

When a fixed preset still cannot guarantee the target frame rate (the worst-case scene is always heavier than the average), **dynamic resolution** closes the gap automatically. The original guidance is the heart of it:

- **Target frame time** — pick a millisecond budget (e.g. 16.67 ms for 60 FPS).
- **Scale resolution to maintain FPS** — when a frame runs long, shrink the internal render resolution next frame; when it runs short, grow it back toward native.
- **Temporal upscaling to hide changes** — reconstruct the lost detail with TAA-based upscalers (TAAU, DLSS, FSR, XeSS) so the resolution change is imperceptible.

The control loop is a simple feedback controller. Letting $t$ be the measured GPU frame time, $t^\*$ the target, and $s$ the current render-scale factor:

$$ s_{n+1} = \text{clamp}\left( s_n \cdot \sqrt{\frac{t^\*}{t_n}},\; s_{\min},\; s_{\max} \right) $$

The square root appears because GPU cost scales roughly with pixel *count*, which is the square of the linear scale factor; correcting in linear space therefore needs the square root of the time ratio. Clamping prevents the resolution from collapsing to an unusable blur during a momentary spike.

```cpp
// Dynamic-resolution controller, run once per frame.
float update_render_scale(float last_gpu_ms, float target_ms,
                          float scale, float min_s, float max_s) {
    float ratio    = target_ms / std::max(last_gpu_ms, 0.01f);
    float proposed = scale * std::sqrt(ratio);
    // Damp toward the proposed value to avoid oscillation/visible pumping.
    float damped   = scale + 0.25f * (proposed - scale);
    return std::clamp(damped, min_s, max_s);
}
```

A small damping factor (here 0.25) is important: snapping resolution to the exact proposed value every frame produces visible "resolution pumping". Easing toward the target keeps the change below the perceptual threshold.

---

## Compiler Optimizations and PGO

The platform discussion so far has been about runtime behaviour, but a large, free performance margin lives in *how the binary is built*. The compiler and linker can transform the same source into substantially faster machine code — and profile-guided optimization (PGO) lets them do it using evidence from real runs instead of static guesses.

### Optimization levels and flags

Every production binary should be built with optimizations on. The common levels:

| Flag (GCC/Clang) | Meaning | Typical use |
|------------------|---------|-------------|
| `-O0` | No optimization | Debug builds only |
| `-O1` | Light optimization | Rare; faster builds with some speedup |
| `-O2` | Full optimization, no size/speed tradeoff risk | **Default for shipping** |
| `-O3` | Aggressive (more inlining, vectorization) | Hot, math-heavy code; measure — can regress |
| `-Os` | Optimize for size | Memory-constrained / embedded |
| `-Ofast` | `-O3` plus relaxed IEEE math | Only where strict FP is not required |

> **Never profile or ship a debug build.** A `-O0` build can be several times slower than `-O2` and will mislead every profiling decision — the original "Avoiding Common Pitfalls" warning that *"debug builds hide real performance"* is largely a statement about optimization flags.

Two structural flags deserve special mention:

- **Link-Time Optimization (`-flto`)** lets the optimizer see across translation-unit boundaries at link time, enabling cross-file inlining and dead-code elimination that per-file compilation cannot. It often yields a few percent for modest build-time cost.
- **Architecture targeting (`-march=native`, or a specific `-march=x86-64-v3`)** lets the compiler emit newer instructions (AVX2, etc.). On a console with fixed hardware this is free and unconditional; on PC it must be balanced against the minimum CPU you support (or shipped as multiple dispatched code paths).

### Profile-Guided Optimization (PGO)

Many of a compiler's most consequential decisions — which branch is the likely one, which function is hot enough to inline, how to lay out basic blocks so the common path stays in the instruction cache — are *guesses* when made from source alone. **PGO replaces those guesses with measured truth** by feeding the compiler an execution profile gathered from a real run.

#### The three-step workflow

PGO is an instrument-run-rebuild loop:

```
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ 1. INSTRUMENT     │     │ 2. TRAIN          │     │ 3. OPTIMIZE       │
   │ compile with      │ --> │ run on a          │ --> │ recompile using   │
   │ profiling probes  │     │ representative    │     │ the collected     │
   │ (-fprofile-       │     │ workload; emit    │     │ profile           │
   │  generate)        │     │ .profraw data     │     │ (-fprofile-use)   │
   └──────────────────┘     └──────────────────┘     └──────────────────┘
```

With Clang the concrete sequence is:

```bash
# 1. Build an instrumented binary.
clang++ -O2 -fprofile-generate -o app.inst app.cpp

# 2. Run it on a REPRESENTATIVE workload (a real level, a real benchmark).
#    This drops .profraw files recording which branches/functions ran hot.
./app.inst --run-typical-workload

# 3. Merge the raw profile, then rebuild using it.
llvm-profdata merge -output=app.profdata *.profraw
clang++ -O2 -fprofile-use=app.profdata -o app app.cpp
```

GCC uses the analogous `-fprofile-generate` / `-fprofile-use` pair.

#### What PGO actually improves

- **Branch layout** — the hot side of each `if` is made the fall-through, keeping the common path in straight-line code.
- **Inlining decisions** — functions that are *actually* called frequently get inlined even if they look large; functions that are rarely called are left out-of-line to keep hot code dense.
- **Basic-block ordering and code locality** — hot blocks are clustered so the CPU instruction cache and the branch predictor stay warm; cold paths (error handling, rare cases) are exiled to the end of the function or to separate "cold" sections.
- **Devirtualization and switch lowering** — a profile that shows one virtual target or one `switch` case dominates lets the compiler speculate on it directly.

Typical real-world gains are in the **5–20%** range on branch-heavy code (interpreters, parsers, game logic, compilers themselves), with essentially no source changes. Because the win comes entirely from the build, it stacks on top of every algorithmic and data-layout optimization done elsewhere.

#### The representativeness rule

PGO is only as good as the training run. **The profile must reflect how the program is actually used**; optimizing for a workload nobody runs can pessimize the workload everybody runs. Practical guidance:

- Train on a realistic, worst-case-inclusive scenario, not a trivial smoke test.
- For games, train on representative gameplay (a real level, real input), not the main menu.
- Re-collect the profile when the code or the dominant workload changes meaningfully; a stale profile slowly drifts from reality.
- In CI, automate the instrument → run-benchmark → rebuild loop so every release ships with a fresh, representative profile.

A related, lower-effort variant is **AutoFDO / sampling-based PGO**, which gathers the profile from low-overhead hardware sampling (e.g. Linux `perf`) on an ordinary optimized build rather than a special instrumented one. It trades a little accuracy for the ability to profile production binaries continuously, which makes keeping the profile representative far easier.

---

## Putting It Together

| Platform | First thing to measure | First lever to pull | Key build flag |
|----------|------------------------|---------------------|----------------|
| **Mobile** | Sustained (thermally soaked) frame time | Cap FPS; cut GPU energy; stay in tile memory | `-O2` + size-aware; ASTC textures |
| **Console** | Bytes of memory traffic per frame | Offload to compute; native low-level API | `-O3`/`-march=<exact>` + PGO (fixed spec) |
| **PC** | Worst-case frame on minimum spec | Quality presets + dynamic resolution | `-O2 -flto` + PGO; multi-arch dispatch |

The unifying discipline is unchanged from the rest of this section: **profile on the real target, optimize the binding constraint, and verify the win.** Platform tuning simply changes *which* constraint binds — energy on mobile, shared bandwidth on console, the long tail of unknown hardware on PC — and adds the compiler itself as a final, almost-free source of speed through optimization flags and PGO.

## See Also

- [Performance Optimization](./) — the optimization hub and overview
- [3D Graphics & Rendering](../graphics/3d-rendering.html) — rendering pipelines and GPU architecture that platform tuning targets
- [VR/AR Development](../vr-ar/) — standalone-headset performance shares mobile's thermal and battery constraints
- [Unreal Engine](../technology/unreal.html) — engine scalability settings and platform profiling tools
- [Game Development](../gamedev/) — game development workflows and frame-budget planning
