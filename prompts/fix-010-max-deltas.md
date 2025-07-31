`max_deltas` scaled wrongly.

For the `BlindGrasping` env, it should be 0.01 * 2 * 1.0 = 0.02 for finger DOF 0 for example (control_dt=0.01*2=0.02); seems related to bug with control_dt vs physics_dt initialization.
