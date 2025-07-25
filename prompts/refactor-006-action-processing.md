MAJOR refactoring in the action processing logic; make sure to use EXTRA high thinking effort!

- Pre-action of step N: run in post_physics of step N-1. Specific order:
    1. Compute other observations for step N.
    2. Run pre-action of step N, using these observations.
    3. Add the result dof targets to the observation.
- Post-action of step N: run in pre_physics of step N (after policy takes decision).
