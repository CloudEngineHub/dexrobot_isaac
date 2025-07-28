# fix-000-tb-metrics.md

Fix TensorBoard data point sampling limit causing old reward breakdown data to disappear in long runs.

## Context

During long training runs, TensorBoard displays only the most recent ~780.5M steps for custom reward breakdown metrics, while RL Games built-in metrics show complete training history. This is due to TensorBoard's default 1000 data point limit per scalar tag with reservoir sampling.

Custom reward breakdown logs more frequently (every 10 finished episodes) compared to built-in metrics (every epoch), causing it to hit the 1000-point limit faster.

## Current State

- TensorBoard default: 1000 scalar data points per tag
- RewardComponentObserver logs every `log_interval=10` finished episodes
- Built-in metrics log every epoch (much less frequent)

## Desired Outcome

Reduce logging frequency of reward breakdown metrics to stay within TensorBoard's data point limits for longer training runs.

## Implementation Notes

**Solution: Increase log_interval**
- Change default `log_interval` from 10 to 100+ finished episodes
- Parameter meaning: "Write to TensorBoard once per X finished episodes"
- This reduces data points by 10x, extending visible history from 780.5M to 7.8B+ steps

**Code change:**
```python
# In train.py or config
observer = RewardComponentObserver(log_interval=100)  # Was 10
```
