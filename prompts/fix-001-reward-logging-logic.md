# fix-001-reward-logging-logic.md

Fix RewardComponentObserver logging cumulative averages instead of windowed statistics.

## Context

RewardComponentObserver currently logs cumulative averages from training start instead of meaningful windowed statistics. This creates slowly-changing metrics that mask recent performance trends and provide poor insight into training progress.

## Current State

**Flawed cumulative logic:**
```python
# Accumulates forever (never resets except after_clear_stats)
self.cumulative_sums["all"][component_name]["rewards"] += total_reward
self.cumulative_sums["all"][component_name]["steps"] += total_steps

# Logs cumulative average since training start
step_mean = cum_rewards / max(cum_steps, 1)
```

This produces metrics like "average reward per step since training began" instead of "average reward per step over recent episodes."

## Desired Outcome

Replace cumulative statistics with windowed statistics that reset after each logging interval, providing meaningful trending data.

## Implementation Notes

**Windowed statistics approach:**
```python
def _log_to_tensorboard(self):
    # Calculate window averages for current interval
    for component_name in self.discovered_components:
        window_rewards = self.cumulative_sums["all"][component_name]["rewards"]
        window_steps = self.cumulative_sums["all"][component_name]["steps"]
        step_mean = window_rewards / max(window_steps, 1)

        # Log windowed average
        self.writer.add_scalar(f"reward_breakdown/all/raw/step/{component_name}", step_mean, frame)

    # Reset window for next interval
    for term_type in self.cumulative_sums:
        for component_name in self.cumulative_sums[term_type]:
            self.cumulative_sums[term_type][component_name] = {"rewards": 0.0, "steps": 0}
```

**Benefits:**
- Meaningful trending: shows performance changes over time
- Better training insights: recent performance vs long-term averages
- Fewer redundant data points: values change meaningfully between logs

## Constraints

- Keep episode meters (rolling averages) unchanged
- Maintain log_interval in episodes (makes sense for parallel environments)
- Preserve component responsibility separation
