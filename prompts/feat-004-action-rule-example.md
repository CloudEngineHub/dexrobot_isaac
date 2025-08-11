# Action Rule Use Cases Documentation

## Status
- **Example Implementation**: Not needed at this time
- **Documentation**: Required - create conceptual guide

## Problem
The existing `guide-action-pipeline.md` provides comprehensive technical documentation, but lacks conceptual understanding through elegant examples that demonstrate the intellectual beauty of the 4-stage pipeline approach.

## Solution
Create `docs/guide-action-rule-use-cases.md` - a conceptual companion guide focusing on elegant examples and use cases rather than technical implementation details.

## Documentation PRD

**Document**: `docs/guide-action-rule-use-cases.md`

**Purpose**: Demonstrate the intellectual beauty and natural problem decomposition enabled by the action rule pipeline through clean, elegant examples.

**Target Audience**:
- Engineers who want to understand and extend standard control modes
- Researchers who want to see clean examples of pipeline-based problem decomposition

**Key Insight**: Standard control modes (`position`/`position_delta`) are elegant action rule implementations, not separate control pathways. This provides a natural bridge from familiar concepts to advanced research applications.

**Structure**:

### 1. Standard Control Modes as Action Rules (2 paragraphs)
- **Concept**: Show how `position` and `position_delta` modes are implemented as action rules
- **Purpose**: Demystify the abstraction - "When you use position control, you're already using action rules"
- **Content**: Brief pseudocode showing clean separation of concerns in standard modes
- **Focus**: Familiar control modes as showcases of good pipeline design

### 2. Pipeline Philosophy (1 paragraph)
- **Concept**: Why 4-stage separation creates intellectual elegance
- **Content**: Each stage has distinct responsibility, enabling natural problem decomposition
- **Focus**: Conceptual benefits of the pipeline approach

### 3. Research Use Cases (2-3 elegant examples)

#### 3.1 Residual Learning
- **Pre-action**: Set DOF targets to dataset values (baseline)
- **Action rule**: Add scaled policy output to previous targets (correction)
- **Post-action**: Clip targets to physical limits (constraint)
- **Beauty**: Clean separation of baseline, correction, and constraint

#### 3.2 Confidence-Based Selective DOF Control
- **Pre-action**: Fallback controller computes complete safe baseline targets (not using policy)
- **Action rule**: Per-DOF selective replacement - if confidence[i] > threshold, use policy target[i], else keep fallback target[i]
- **Post-action**: Sanity-based selective reversion - check mixed targets against safety rules, revert problematic DOFs to fallback
- **Beauty**: Heterogeneous control architecture with dual-layer safety (confidence + sanity validation)

### 4. Implementation Notes (brief)
- **Function signatures**: Reference existing technical guide
- **Registration patterns**: Basic examples
- **Keep minimal**: Focus readers on conceptual understanding

**Writing Principles**:
- **Concise**: Each example focuses on data flow and conceptual beauty
- **Objective**: Clear, descriptive language without promotional tone
- **Educational**: Show natural problem decomposition through pipeline stages
- **Complementary**: References technical guide for implementation details
- **Elegant**: Examples chosen for intellectual beauty and clean separation of concerns

**Key Messages**:
1. Standard control modes demonstrate good pipeline design
2. Complex research problems become elegant when properly decomposed
3. Each pipeline stage serves a distinct, focused purpose
4. The 4-stage approach enables natural problem decomposition
5. Pipeline supports both uniform correction (residual learning) and selective control (confidence switching) patterns

**Success Criteria**:
- Readers understand how standard modes work as action rules
- Readers see the conceptual elegance of pipeline-based problem decomposition
- Researchers can envision how to apply the pattern to their own problems
- Engineers understand how to extend familiar control modes
