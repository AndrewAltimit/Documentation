---
layout: docs
title: Game AI Systems
parent: AI/ML Documentation
nav_order: 8
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "robot"
hide_title: true
---

<div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">Game AI Systems</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">Techniques and systems for creating intelligent, responsive NPCs and game behaviors that prioritize engaging gameplay over optimal decision-making.</p>
</div>

Game AI encompasses the techniques and systems that create intelligent, responsive, and believable non-player characters (NPCs) and game behaviors. Unlike traditional AI research focused on optimal decision-making, game AI prioritizes creating engaging, entertaining, and appropriately challenging experiences while maintaining performance constraints of real-time applications.

## Foundations of Game AI

### Goals of Game AI

Game AI differs from academic AI in key ways:

| Academic AI | Game AI |
|-------------|---------|
| Optimal solutions | Entertaining solutions |
| Unlimited computation | Real-time constraints |
| Perfect play | Believable play |
| Win at all costs | Create fun experiences |
| Single agent focus | Many agents simultaneously |

### AI Architecture Overview

Typical game AI system layers:

```
Decision Layer (What to do)
├── Goal Selection
├── Planning
└── Behavior Trees / FSMs
    │
    ▼
Steering Layer (How to move)
├── Path Following
├── Obstacle Avoidance
└── Formation Movement
    │
    ▼
Animation Layer (How to look)
├── Animation State Machine
├── IK / Procedural Animation
└── Facial Expressions
```

## Pathfinding

### Navigation Meshes (NavMesh)

Industry standard for 3D environments:

**NavMesh Generation:**
1. Voxelize walkable geometry
2. Identify walkable surfaces
3. Build regions from voxels
4. Create polygon mesh from regions
5. Add connectivity data

**Properties:**
- Efficient storage and queries
- Dynamic updates possible
- Supports different agent sizes
- Handles multi-level environments

### A* Algorithm

The foundation of game pathfinding:

```python
def a_star(start, goal, graph):
    open_set = PriorityQueue()
    open_set.put(start, 0)

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.neighbors(current):
            tentative_g = g_score[current] + cost(current, neighbor)

            if tentative_g < g_score.get(neighbor, infinity):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    open_set.put(neighbor, f_score[neighbor])

    return None  # No path found
```

**Heuristics:**
- **Euclidean**: Straight-line distance (any angle movement)
- **Manhattan**: Grid distance (4-directional)
- **Chebyshev**: Grid distance (8-directional)
- **Octile**: Weighted diagonal movement

### Hierarchical Pathfinding

For large worlds:

```
Level 3: Region graph (countries/zones)
    │
Level 2: Cluster graph (neighborhoods)
    │
Level 1: NavMesh polygons (rooms)
    │
Level 0: Detailed path (within polygon)

Search: Top-down refinement
1. Find path through regions
2. Refine through clusters
3. Detailed path in NavMesh
```

**Benefits:**
- O(log n) instead of O(n) searches
- Memory-efficient for huge worlds
- Natural for streaming worlds

### Path Smoothing

Post-processing for natural movement:

- **String Pulling**: Funnel algorithm for shortest path
- **Bezier Curves**: Smooth corners
- **Catmull-Rom Splines**: Natural curves through waypoints
- **Runtime Smoothing**: Adjust path during movement

## Steering Behaviors

### Reynolds' Steering Behaviors

Classic autonomous movement algorithms:

**Basic Behaviors:**

```python
def seek(agent, target):
    desired = normalize(target - agent.position) * max_speed
    return desired - agent.velocity

def flee(agent, target):
    return -seek(agent, target)

def arrive(agent, target, slowing_radius):
    to_target = target - agent.position
    distance = length(to_target)

    if distance < slowing_radius:
        desired_speed = max_speed * (distance / slowing_radius)
    else:
        desired_speed = max_speed

    desired = normalize(to_target) * desired_speed
    return desired - agent.velocity
```

**Group Behaviors:**
- **Separation**: Avoid crowding neighbors
- **Alignment**: Steer toward average heading
- **Cohesion**: Steer toward average position
- **Flocking**: Combination of above three

### Obstacle Avoidance

Real-time collision prevention:

**Context Steering:**
```
1. Create interest map (directions toward goal)
2. Create danger map (directions toward obstacles)
3. Combine: interest - danger
4. Select highest-scoring direction
```

**Velocity Obstacles (VO):**
- Project obstacle's future positions
- Calculate collision cone
- Choose velocity outside all cones
- ORCA variant for multi-agent

### Local Avoidance

For crowds and traffic:

- **RVO2**: Reciprocal Velocity Obstacles
- **Social Forces**: Crowd simulation model
- **Flow Fields**: Precomputed direction vectors
- **Continuum Crowds**: Density-based movement

## Decision Making

### Finite State Machines (FSM)

Simple, reliable decision structure:

```
States and Transitions:

[Patrol] ──see enemy──► [Chase]
    ▲                      │
    │                      ▼
    └──lost enemy──── [Search]
                          │
                     timeout
                          ▼
                      [Patrol]

[Chase] ──in range──► [Attack]
    │                     │
    │                     ▼
    └──enemy dead──► [Celebrate]
```

**Advantages:**
- Easy to understand and debug
- Predictable behavior
- Low runtime cost

**Disadvantages:**
- State explosion with complexity
- Hard to reuse across agents
- Difficult to handle interrupts

### Hierarchical FSMs (HFSM)

Nested states reduce complexity:

```
[Combat]
├── [Melee]
│   ├── [Approach]
│   ├── [Attack]
│   └── [Retreat]
└── [Ranged]
    ├── [Find Cover]
    ├── [Aim]
    └── [Fire]

[Non-Combat]
├── [Patrol]
└── [Idle]
```

### Behavior Trees

Modern industry standard:

**Node Types:**
- **Composite**: Sequence, Selector, Parallel
- **Decorator**: Inverter, Repeater, Succeeder
- **Leaf**: Action, Condition

**Example Tree:**

```
Selector (try until success)
├── Sequence (all must succeed)
│   ├── [Condition] Is Enemy Visible?
│   ├── [Condition] Has Ammo?
│   └── [Action] Shoot Enemy
├── Sequence
│   ├── [Condition] Is Enemy Visible?
│   ├── [Condition] Is Low Health?
│   └── [Action] Flee
└── [Action] Patrol
```

**Execution:**
- Tick tree every frame
- Return: Success, Failure, or Running
- Parent nodes respond to child results

### Utility AI

Score-based decision making:

```python
def select_action(agent, actions):
    best_action = None
    best_score = -infinity

    for action in actions:
        score = evaluate_utility(agent, action)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action

def evaluate_utility(agent, action):
    # Combine multiple considerations
    score = 1.0
    score *= health_consideration(agent, action)
    score *= distance_consideration(agent, action)
    score *= threat_consideration(agent, action)
    return score
```

**Response Curves:**
- Linear, polynomial, logistic
- Custom curves per consideration
- Normalize to [0,1] range

**Advantages:**
- Smooth, nuanced decisions
- Easy to tune and balance
- Natural prioritization
- No explicit state transitions

### Goal-Oriented Action Planning (GOAP)

AI plans sequences of actions:

```
World State: {has_weapon: false, enemy_dead: false, in_cover: false}
Goal State: {enemy_dead: true}

Actions:
- pickup_weapon: {pre: {}, post: {has_weapon: true}, cost: 1}
- find_cover: {pre: {}, post: {in_cover: true}, cost: 2}
- attack: {pre: {has_weapon: true}, post: {enemy_dead: true}, cost: 3}

Planner finds: pickup_weapon → attack
```

**Benefits:**
- Emergent complex behaviors
- Reusable action library
- Handles novel situations

**Used in:**
- F.E.A.R. (original implementation)
- Shadow of Mordor
- Tomb Raider (2013+)

## Tactical and Strategic AI

### Influence Maps

Spatial reasoning for AI:

```
For each cell in grid:
    influence = 0
    for each entity:
        distance = dist(cell, entity)
        influence += entity.strength / (1 + distance * decay)

Uses:
- Find safe areas (low enemy influence)
- Identify strategic positions
- Territory control visualization
```

### Cover System

Finding and using cover:

```python
def evaluate_cover_point(cover, agent, threats):
    score = 0

    # Protection from threats
    for threat in threats:
        if not has_line_of_sight(cover, threat):
            score += 10

    # Distance to agent (prefer closer)
    score -= distance(cover, agent.position) * 0.5

    # Flanking opportunity
    if can_flank_from(cover, threats):
        score += 5

    # Escape routes
    score += count_exit_routes(cover) * 2

    return score
```

### Squad Tactics

Coordinated group behavior:

**Formation Movement:**
- Leader-follower patterns
- Slot-based formations
- Dynamic reformation around obstacles

**Role Assignment:**
- Point man (first in)
- Flankers (side attack)
- Support (covering fire)
- Medic (heal priority)

**Communication:**
- Shared blackboard for knowledge
- Signal system for coordination
- Priority-based task allocation

## Machine Learning in Games

### Reinforcement Learning

Training agents through rewards:

**Applications:**
- Racing game AI (learn optimal racing lines)
- Fighting game opponents (adapt to player style)
- Strategy game opponents (learn build orders)
- Procedural animation (physics-based movement)

**Challenges:**
- Training time requirements
- Unpredictable emergent behaviors
- Difficulty balancing for fun
- Reproducibility issues

### Imitation Learning

Learn from human demonstrations:

```
1. Record human player sessions
2. Extract state-action pairs
3. Train model to predict actions
4. Fine-tune with reinforcement learning
```

**Used in:**
- Racing games (ghost opponents)
- Sports games (player movement)
- Driving simulations

### Neural Network NPCs

Deep learning for game AI:

**Pros:**
- Can learn complex behaviors
- Adapts to player patterns
- Emergent interesting behaviors

**Cons:**
- "Black box" debugging
- Inconsistent behavior
- High computational cost
- Requires training data

## Perception Systems

### Sensory Simulation

What AI can "see" and "hear":

**Vision:**
- Field of view cone
- Line of sight checks
- Distance falloff
- Peripheral vs focused vision

**Hearing:**
- Sound propagation
- Occlusion by geometry
- Sound priority/type
- Memory of heard sounds

**Knowledge:**
- Last known position
- Memory decay over time
- Shared team knowledge

### Awareness System

```python
class AwarenessComponent:
    def __init__(self):
        self.detection_level = 0  # 0-100
        self.last_known_position = None
        self.last_seen_time = 0

    def update(self, target, dt):
        if can_see(target):
            # Increase detection based on visibility
            visibility = calculate_visibility(target)
            self.detection_level += visibility * dt * detection_rate

            if self.detection_level >= 100:
                self.state = ALERT
                self.last_known_position = target.position
        else:
            # Decay detection when not visible
            self.detection_level -= dt * decay_rate
            self.detection_level = max(0, self.detection_level)
```

## Performance Optimization

### LOD for AI

Scale AI complexity with importance:

| Distance/Importance | AI Complexity |
|---------------------|---------------|
| On-screen, close | Full behavior tree, full perception |
| On-screen, far | Simplified decisions, reduced updates |
| Off-screen | Minimal simulation, time-sliced |
| Very far | Statistical simulation only |

### Time Slicing

Spread computation across frames:

```python
class AIManager:
    def update(self):
        # Process subset of agents each frame
        budget = 2.0  # ms

        while budget > 0 and self.update_queue:
            agent = self.update_queue.pop(0)
            start = time.now()
            agent.update()
            budget -= time.now() - start
            self.update_queue.append(agent)  # Re-add to end
```

### Spatial Partitioning

Efficient queries:

- **Grids**: Simple, fast for uniform distribution
- **Quadtree/Octree**: Adaptive subdivision
- **BVH**: Hierarchical bounding volumes
- **Spatial hashing**: O(1) neighbor lookup

## Related Documentation

- [Game Development](../gamedev/) - Game development fundamentals
- [AI Fundamentals](../technology/ai.html) - Machine learning foundations
- [Unreal Engine](../technology/unreal.html) - UE5 AI systems
- [Performance Optimization](../optimization/) - Optimization techniques

---

## See Also
- [Stable Diffusion Fundamentals](stable-diffusion-fundamentals.html) - Core concepts explained
- [ComfyUI Guide](comfyui-guide.html) - Visual workflow creation
- [Model Types](model-types.html) - Understanding LoRAs, VAEs, embeddings
- [Base Models Comparison](base-models-comparison.html) - SD 1.5, SDXL, FLUX compared
- [Advanced Techniques](advanced-techniques.html) - Cutting-edge workflows
- [Game Development](../gamedev/) - Game development fundamentals
- [AI Fundamentals](../technology/ai.html) - Core AI/ML concepts
- [AI/ML Documentation Hub](./) - Complete AI/ML documentation index
