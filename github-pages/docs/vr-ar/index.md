---
layout: docs
title: VR & AR Development
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "vr-cardboard"
---

Virtual Reality (VR) and Augmented Reality (AR) represent transformative technologies that create immersive experiences by either fully replacing or enhancing the user's perception of reality. These technologies demand specialized development approaches, strict performance requirements, and careful attention to human factors to create comfortable, engaging experiences.

## Understanding XR Technologies

### The Reality-Virtuality Continuum

```
Real          │ Augmented    │ Mixed        │ Virtual
Environment   │ Reality (AR) │ Reality (MR) │ Reality (VR)
──────────────┼──────────────┼──────────────┼──────────────
Physical      │ Digital      │ Digital      │ Fully
world only    │ overlays on  │ interacts    │ synthetic
              │ physical     │ with physical│ environment
              │              │              │
Examples:     │ Pokemon GO   │ HoloLens     │ Beat Saber
              │ Snapchat     │ Magic Leap   │ Half-Life Alyx
              │ Google Maps  │ Quest MR     │ VRChat
```

### XR Hardware Landscape

**VR Headsets:**

| Device | Type | Resolution (per eye) | Refresh Rate | Tracking |
|--------|------|---------------------|--------------|----------|
| Meta Quest 3 | Standalone | 2064x2208 | 120Hz | Inside-out |
| Valve Index | PC VR | 1440x1600 | 144Hz | Outside-in |
| PlayStation VR2 | Console | 2000x2040 | 120Hz | Inside-out |
| Apple Vision Pro | Standalone | 3660x3200 | 90Hz | Inside-out |
| HTC Vive XR Elite | Standalone/PC | 1920x1920 | 90Hz | Inside-out |

**AR Devices:**
- **Meta Ray-Ban**: Consumer smart glasses
- **Microsoft HoloLens 2**: Enterprise AR
- **Magic Leap 2**: Enterprise/industrial
- **Smartphones**: ARKit (iOS) / ARCore (Android)

## VR Development Fundamentals

### Rendering Requirements

VR demands exceptional performance:

```
Target Metrics:
- Frame Rate: 90Hz minimum (72-120Hz typical)
- Latency: <20ms motion-to-photon
- Resolution: 2K+ per eye
- Stereo Rendering: 2x geometry passes

Effective load:
90 FPS × 2 eyes × 4MP per eye = 720M pixels/second
(vs 60 FPS × 1 view × 2MP = 120M pixels/second for traditional)
```

**Stereo Rendering Techniques:**

| Technique | Description | Performance |
|-----------|-------------|-------------|
| Multi-pass | Render scene twice | Baseline |
| Instanced Stereo | Single draw, dual viewports | 1.5-2x faster |
| Single Pass Stereo | Geometry shader duplication | Variable |
| Variable Rate Shading | Lower resolution periphery | 20-30% savings |

### Motion Sickness Prevention

Causes and mitigations:

**Vestibular Mismatch:**
- Visual motion without physical motion
- Worse with acceleration, rotation
- Accumulates over time

**Mitigation Strategies:**
```
1. Locomotion Design
   - Teleportation (safest)
   - Snap turning (reduce smooth rotation)
   - Comfort vignette during movement
   - User-controlled movement speed

2. Technical Requirements
   - Maintain target frame rate always
   - Minimize latency
   - Lock horizon/reference points
   - Avoid camera shake

3. User Options
   - Comfort settings exposed
   - Gradual exposure modes
   - Session length recommendations
```

### Interaction Design

VR-specific input paradigms:

**Hand Tracking:**
```
Gestures:
- Pinch: Select/grab
- Point: Direct manipulation
- Open palm: Menu/UI
- Fist: Grip objects
- Thumbs up: Confirm

Hand presence adds:
- Natural object manipulation
- Social expression
- Accessibility without controllers
```

**Controller Interactions:**
- Trigger: Primary action
- Grip: Grab/hold objects
- Thumbstick: Locomotion/menu navigation
- Buttons: Context actions
- Haptics: Tactile feedback

**Gaze-Based:**
- Dwell time selection
- Head-tracked cursor
- Eye tracking for foveated rendering
- Natural social cues

## AR Development

### Spatial Understanding

AR requires understanding the physical world:

**Plane Detection:**
```
Types:
- Horizontal (floors, tables)
- Vertical (walls)
- Arbitrary angle (stairs)

Quality:
- Boundaries refinement over time
- Hole detection
- Classification (floor vs table)
```

**Scene Understanding:**
- Mesh reconstruction (real-time 3D scanning)
- Semantic segmentation (identify objects)
- Occlusion (virtual behind real)
- Light estimation (match virtual to real lighting)

### ARKit vs ARCore

**Apple ARKit (iOS):**
```
Features:
- World Tracking (6DoF)
- Plane Detection
- Image/Object Recognition
- Face Tracking (TrueDepth)
- Body Tracking
- LiDAR (Pro devices)
- RoomPlan API
```

**Google ARCore (Android):**
```
Features:
- Motion Tracking
- Environmental Understanding
- Light Estimation
- Cloud Anchors (shared experiences)
- Augmented Faces
- Depth API
- Geospatial API (outdoor positioning)
```

### WebXR

Browser-based XR experiences:

```javascript
// Basic WebXR session
async function startXR() {
    if (navigator.xr) {
        const session = await navigator.xr.requestSession(
            'immersive-vr',  // or 'immersive-ar'
            {
                requiredFeatures: ['local-floor'],
                optionalFeatures: ['hand-tracking']
            }
        );

        // Set up render loop
        session.requestAnimationFrame(onXRFrame);
    }
}

function onXRFrame(time, frame) {
    const pose = frame.getViewerPose(referenceSpace);

    if (pose) {
        for (const view of pose.views) {
            renderView(view);
        }
    }

    frame.session.requestAnimationFrame(onXRFrame);
}
```

## XR Development Platforms

### Unity XR

Cross-platform XR development:

**XR Interaction Toolkit:**
- Standardized input handling
- Locomotion systems
- Grab/socket interactions
- UI interaction

**Supported Platforms:**
- Meta Quest (native, PC VR)
- SteamVR (Index, Vive)
- PlayStation VR2
- Apple Vision Pro (visionOS)
- ARKit/ARCore

### Unreal Engine VR

Enterprise-grade XR:

**Features:**
- OpenXR support
- Motion controller support
- VR Template projects
- Stereo instancing
- Forward renderer (better for VR)

See our [Unreal Engine Guide](../technology/unreal.html) for details.

### Native Development

Platform-specific SDKs:

**Meta Quest (Oculus SDK):**
- Native Android development
- Best performance on Quest hardware
- Access to all platform features

**Apple Vision Pro (visionOS):**
- SwiftUI for spatial computing
- RealityKit for 3D
- ARKit for world understanding

**SteamVR (OpenVR):**
- C++ SDK
- Works with any SteamVR headset
- Room-scale and seated configurations

## Performance Optimization for XR

### Frame Budget

Strict timing requirements:

```
At 90 Hz: 11.1ms per frame budget

Typical breakdown:
├── Game logic: 2-3ms
├── Physics: 1-2ms
├── Animation: 1ms
├── Rendering (CPU): 3-4ms
├── Rendering (GPU): 8-10ms
└── OS/Driver overhead: 1-2ms

Note: GPU renders previous frame while
CPU prepares next (pipelining)
```

### Foveated Rendering

Reduce peripheral detail:

```
Foveation levels:
- Full resolution: Central 10-15°
- Medium: 15-45° from center
- Low: 45°+ periphery

Types:
- Fixed: Static regions (most compatible)
- Dynamic: Follows gaze (requires eye tracking)
- Application-based: Content-aware

Savings: 30-50% GPU time
```

### Application SpaceWarp (ASW) / Reprojection

Frame synthesis when performance drops:

```
Normal: Render at 90 FPS, display at 90 FPS
ASW: Render at 45 FPS, synthesize to 90 FPS

How it works:
1. Detect missed frame deadline
2. Take previous frame
3. Apply motion vectors
4. Reproject to new head position
5. Display synthesized frame

Artifacts:
- Edge shimmer
- Disocclusion errors
- Latency perception
```

### Optimization Techniques

**Rendering:**
- Single-pass stereo rendering
- Aggressive LOD
- Occlusion culling
- Baked lighting where possible
- Forward rendering (simpler, faster)

**Assets:**
- Mobile-quality textures
- Simplified materials
- Instance static meshes
- Compress and stream textures

**Code:**
- Object pooling
- Avoid GC allocations
- Multithreaded physics
- Job system for parallel work

## UX Design for XR

### Spatial UI Design

UI in 3D space:

```
Placement:
- Comfortable viewing: 1-2m distance
- Avoid extremes of vision
- World-locked vs head-locked
- Hand-anchored for tools

Sizing:
- Minimum text: 1.5° of visual angle
- Comfortable text: 2-3°
- Touch targets: 2cm minimum

Depth:
- Avoid UI at arm's length (focus conflict)
- Match UI depth to content
- Use subtle depth cues
```

### Comfort Guidelines

**Physical Comfort:**
- Session length recommendations
- Break reminders
- Adjust for IPD (interpupillary distance)
- Standing vs seated options

**Visual Comfort:**
- Avoid vergence-accommodation conflict
- Maintain stable frame rate
- Limit bright flashing
- Provide reference points

**Motion Comfort:**
- Offer locomotion options
- Gradual exposure for new users
- User-controlled speed
- Comfort vignettes

### Accessibility

Inclusive XR design:

- **Seated play mode**: For mobility limitations
- **One-handed mode**: Controller remapping
- **Subtitles**: Spatial audio visualization
- **Colorblind modes**: Visual indicators
- **Eye tracking alternatives**: Head gaze fallback
- **Adjustable text size**: Readability options

## Mixed Reality Features

### Passthrough

Seeing the real world in VR:

**Types:**
- Grayscale (older devices)
- Color (Quest 3, Vision Pro)
- High-resolution (Vision Pro)

**Uses:**
- Room awareness
- Mixed reality games
- AR mode in VR headset
- Safety boundaries

### Spatial Anchors

Persistent placement in space:

```
Local Anchors:
- Saved to device
- Persist across sessions
- Limited to original location

Cloud Anchors:
- Shared across devices
- Multi-user experiences
- Azure Spatial Anchors, ARCore Cloud Anchors

World Anchors:
- GPS + visual positioning
- Outdoor AR experiences
- City-scale applications
```

### Hand and Body Tracking

Natural interaction:

```
Hand Tracking Capabilities:
- 26 joints per hand
- Finger gestures
- Hand poses
- Pinch detection

Body Tracking:
- Full body (Quest 3 experimental)
- Upper body (common)
- Inverse kinematics for legs
- Social presence in VR
```

## Development Workflow

### Testing and Iteration

XR-specific challenges:

**Device Testing:**
- Regular on-device testing essential
- Simulator has limitations
- Performance differs significantly
- Comfort only testable in VR

**Play Testing:**
- Recruit diverse testers
- Track comfort metrics
- Observe natural behaviors
- Iterate on feedback

### Tools and Debugging

**In-VR Tools:**
- Performance HUD overlays
- Debug visualization
- Console access in VR
- Screenshot/recording

**External Tools:**
- OVR Metrics Tool (Quest)
- RenderDoc (GPU debugging)
- PIX (Windows)
- Unity Profiler

## Future Directions

### Emerging Technologies

- **Neural interfaces**: Direct brain input
- **Haptic suits**: Full-body feedback
- **Varifocal displays**: Natural focus depth
- **Wider FOV**: 200°+ field of view
- **Higher resolution**: 8K+ per eye
- **Lighter form factors**: Glasses-like devices

### Market Trends

- Enterprise adoption accelerating
- Consumer VR gaming maturing
- AR glasses approaching viability
- Spatial computing as new paradigm
- AI integration for scene understanding

## Related Documentation

- [Game Development](../gamedev/index.html) - Game development fundamentals
- [3D Graphics & Rendering](../graphics/3d-rendering.html) - Rendering pipeline
- [Unreal Engine](../technology/unreal.html) - UE5 VR development
- [Performance Optimization](../optimization/index.html) - Optimization techniques
