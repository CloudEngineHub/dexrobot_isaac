# Control_dt vs Physics_dt Illustration Documentation

## Problem Statement

The control_dt measurement and two-stage initialization system is a core architectural concept that is poorly understood. Current documentation lacks visual explanation of the parallel simulation constraint that drives this design.

## Current Understanding Issues

- Users assume physics_steps_per_control_step is configurable (it's measured)
- Confusion about why measurement is necessary (parallel GPU simulation constraint)
- Misunderstanding that stepping varies per control cycle (it's deterministic after measurement - ALL control steps have the SAME number of physics steps)

## Documentation Goals

Create comprehensive SVG timeline illustration showing:

### 1. Parallel Simulation Constraint (Core Concept)
- Timeline showing all N environments must step together on GPU
- Demonstrate why individual environment stepping is impossible
- Show how worst-case reset logic determines physics step count for ALL control steps

### 2. Consistent Physics Step Count (Key Insight)
Example: 4 Physics Steps Per Control Step (measured during initialization)
```
Physics Step Breakdown:
├── P₁: Standard env.step() call (always required)
├── P₂: Reset logic - moving hand to new position
├── P₃: Reset logic - placing/repositioning object
└── P₄: Reset logic - final stabilization after setup

Result: ALL control steps use 4 physics steps (physics_steps_per_control_step = 4)
        control_dt = physics_dt × 4
```

### 3. Deterministic Operation (Post-Measurement)
- EVERY control step takes exactly 4 physics steps (regardless of whether individual environments need reset)
- Fixed control_dt ensures consistent action scaling and timing
- Timeline shows: Control Step 1 [P₁|P₂|P₃|P₄], Control Step 2 [P₁|P₂|P₃|P₄], etc.

### 4. Impact on Action Scaling
- position_delta mode requires control_dt for velocity-to-delta conversion
- max_delta = control_dt × max_velocity
- Action scaling coefficients computed during finalize_setup()

## Implementation Plan

### Documentation Organization
- **Main document**: `docs/control-dt-timing-diagram.md` (conceptual understanding)
- **SVG timeline**: `docs/assets/control-dt-timeline.svg` (visual diagrams)
- **Cross-references**: Links to existing `guide-component-initialization.md` and `reference-physics-implementation.md`

### SVG Timeline Specifications
**Dimensions**: ~800px × 400px
**Structure**:
- **Horizontal axis**: Time progression (Control Step 1, 2, 3...)
- **Vertical axis**: Environment timelines (Env 0, Env 1, Env 2, Env 3)
- **Control step containers**: Large boxes spanning 4 physics steps each
- **Physics step subdivisions**: P₁, P₂, P₃, P₄ within each control step

**Timeline Sequence to Illustrate**:
1. **Control Step 1**: 4 physics steps [P₁|P₂|P₃|P₄] across all environments
2. **Control Step 2**: 4 physics steps [P₁|P₂|P₃|P₄] (highlight which env is driving reset)
3. **Control Step 3**: 4 physics steps [P₁|P₂|P₃|P₄] (consistent timing)

**Visual Elements**:
- **Color coding**: Blue (standard step), Red (reset-driven steps P₂,P₃,P₄)
- **Reset highlighting**: Show which environment needs reset, but ALL environments take 4 steps
- **Synchronization emphasis**: Vertical alignment showing parallel constraint
- **Callouts**: Explain physics step breakdown (env.step + 3 reset steps)

### Cross-Reference Strategy
**FROM existing docs TO new timing diagram**:
- `guide-component-initialization.md` → Add link in "Why Two-Stage is Necessary" section
- `reference-physics-implementation.md` → Add link in "Physics Step Management" section

**FROM new timing diagram TO existing docs**:
- Reference component-initialization for two-stage implementation details
- Reference physics-implementation for technical stepping specifics
- Reference action pipeline for control_dt scaling impact

### Key Messages
1. **Parallel Constraint**: GPU simulation requires ALL environments step together
2. **Worst-Case Measurement**: System measures maximum physics steps needed (e.g., 4)
3. **Deterministic Result**: ALL control steps use same physics step count forever
4. **Reset Logic Breakdown**: Show specific physics steps (hand move, object place, stabilize)

## Success Criteria

- Timeline clearly shows ALL control steps have identical physics step count (4)
- Parallel environment constraint visually obvious through vertical alignment
- Reset logic breakdown clearly explains where extra physics steps come from
- Readers understand deterministic timing (no variation between control steps)
- Cross-references create logical documentation flow from concept → implementation → technical details
