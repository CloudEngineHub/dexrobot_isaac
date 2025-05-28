# 🚨 Important Design Decisions & Caveats

**READ THIS BEFORE MODIFYING THE PHYSICS OR MODEL CONFIGURATION**

## Quick Reference

### 1. **Fixed Hand Base** 
- Hand uses `fix_base_link = True` - it won't fall under gravity
- Base movement controlled via ARTx/y/z/Rx/y/z DOFs, not actor translation

### 2. **ARTz is RELATIVE Motion**
- ARTz=0.0 → stay at initial Z position (spawn point)
- ARTz=+0.1 → move up 0.1 units from initial position
- ARTz=-0.1 → move down 0.1 units from initial position
- **NOT absolute world coordinates!**

### 3. **Isaac Gym Requires `limited="true"`**
```xml
<!-- WRONG - limits will be ignored -->
<joint name="finger_joint" range="0 1.3" />

<!-- CORRECT - limits will work -->  
<joint name="finger_joint" range="0 1.3" limited="true" />
```

### 4. **MJCF Values Used Directly**
- No hardcoded stiffness/damping overrides in code
- Actuator `kp` → stiffness, joint `damping` → damping
- Edit MJCF files to change joint properties

### 5. **Model Regeneration Required**
- After MJCF changes: `cd assets/dexrobot_mujoco/scripts && ./regenerate_all.sh`
- Scripts automatically add `limited="true"` to joints with ranges

---

**Full details**: See [physics_caveats.md](physics_caveats.md)