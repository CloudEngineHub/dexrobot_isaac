# doc-003-action-processing-illustration.md

Add action processing timing illustration to existing guide-action-pipeline.md

## Background
The refactor-006 split action processing timing to align with RL rollout patterns:
- **post_physics (step N-1)**: Observation computation + pre-action rule (pipeline stage 1)
- **pre_physics (step N)**: Action rule + post-action filters + coupling (pipeline stages 2-4)

## Implementation Plan

### Phase 1: Enhance Existing Documentation
**File**: `docs/guide-action-pipeline.md` (modify existing)
- **Add new section**: "Timing and Execution Flow"
- **Explain timing split**: WHY post_physics vs pre_physics phases exist
- **Stage mapping**: How 4-stage pipeline maps to execution phases
- **RL alignment**: Why this timing benefits RL framework patterns
- **Integration**: Keep all action processing concepts in single document

### Phase 2: Visual Diagram - Content and Layout Specifications

**File**: `docs/assets/action-processing-timeline.svg` (new)

#### Primary Content Structure:
- **Two Control Steps**: Step N-1 and Step N showing temporal relationship
- **Four Pipeline Stages**: Stage 1 (Pre-Action Rule), Stage 2 (Action Rule), Stage 3 (Post-Action Filters), Stage 4 (Coupling Rule)
- **Timing Phase Mapping**: Stage 1 → post_physics phase, Policy forward + Stages 2-4 → pre_physics phase
- **Data Flow Elements**: Specific tensor variables labeled on arrows (active_prev_targets, active_rule_targets, actions, active_raw_targets, active_next_targets, full_dof_targets)

#### Layout Organization:
- **Linear Pipeline Flow**: Horizontal arrangement Stage 1 → Policy → Stage 2 → Stage 3 → Stage 4 (left to right progression)
- **Phase Context**: Subtle background zones indicating post_physics (Step N-1) and pre_physics (Step N) timing without overwhelming stage flow
- **Clean Staging**: Each stage as distinct box (~120-140px width) with clear functional purpose
- **Policy Integration**: Policy network as natural bridge between Stage 1 (observations) and Stage 2 (actions)
- **Directional Flow**: Prominent arrows showing data progression through pipeline stages

#### Visual Hierarchy:
1. **Primary**: Linear stage sequence showing functional pipeline progression
2. **Secondary**: Phase timing context as subtle background information
3. **Supporting**: Data flow arrows and timing labels (Step N-1, Step N)

#### Content Approach (Descriptive, Not Promotional):
- **Architecture Description**: Focus on WHAT the timing pattern accomplishes
- **Timing Context**: Clear temporal labels without "benefits" language
- **Functional Focus**: Describe data flow and stage purposes rather than advantages
- **No Architecture Summary Box**: Keep diagram focused on visual flow, move architectural description to text documentation

#### Educational Focus:
- Show WHEN each stage executes through clear timing phases
- Show WHAT data flows between stages with prominent arrows and specific tensor labels
- Show HOW the 4-stage pipeline maps to 2 timing phases
- Include all data dependencies (e.g., Stage 2 receives both actions AND active_prev_targets)
- Clarify policy forward pass happens in pre_physics phase

#### Additional Requirements:
- **Policy Interpretation Note**: Text documentation should clarify that policy output can have any meaning; the action rule determines how to interpret policy output for DOF target updates
- **Complete Data Flow**: Show all inputs to each stage, not just primary flow (e.g., Stage 2 needs active_prev_targets, active_rule_targets, AND actions)
- **Variable Clarity**: Label arrows with exact tensor variable names from implementation to avoid confusion between similar-sounding targets

### Phase 3: Cross-References
- Update any existing links to guide-action-pipeline.md
- No new documentation file needed

## Quality Standards
- Follow CLAUDE.md documentation development protocol
- Maintain existing document structure and flow
- Verify technical accuracy against refactor-006-action-processing.md
- Single source of truth for all action processing concepts
